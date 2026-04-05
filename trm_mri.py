"""
models/recursive_reasoning/trm_mri.py

TRM adapted for MRI k-space reconstruction.
Based directly on models/recursive_reasoning/trm.py.

Label format (fundamental fix):
    Labels are stored as complex images -- (2, H*W) [real|imag] normalised by
    scale = max(|kspace|).  This guarantees:
        FFT(label_real + j*label_imag) == observed_kspace / scale
    so data_consistency operates on exactly matching scales.
    The loss is MSE on the MAGNITUDE of the complex prediction.
    PSNR is computed on the magnitude image.

Changes from trm.py:
    1. embed_tokens removed        -- continuous float inputs
    2. puzzle_emb removed          -- no puzzle identity for MRI
    3. lm_head replaced            -- MRIDecoder: outputs 2 channels (real+imag)
    4. MRIEncoder added            -- CNN (B,3,H,W) -> (B,H*W,D), 3ch: real,imag,mask
    5. data_consistency() fixed    -- operates on complex domain; exact scale match
    6. Q-head uses mean pool       -- no special summary token in MRI
    7. seq_len compat field added  -- pretrain.py passes it; stored but unused
    8. reset_carry fixed           -- explicit expand before torch.where (Bug #3)

Everything else identical to trm.py:
    - z_H / z_L dual-latent carry
    - H_cycles-1 no-grad + 1 grad pass
    - L_cycles inner loop
    - ACT halting with Q-head
    - no_ACT_continue, exploration, EMA compatible
"""

from typing import Tuple, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, SwiGLU, Attention, RotaryEmbedding,
    CosSin, CastedLinear,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TinyRecursiveReasoningModel_MRIConfig(BaseModel):
    height: int
    width:  int

    hidden_size:  int   = 512
    num_heads:    int   = 8
    expansion:    float = 4.0
    L_layers:     int   = 2
    H_cycles:     int   = 3
    L_cycles:     int   = 4
    cnn_channels: int   = 32

    pos_encodings: str   = "rope"
    rope_theta:    float = 10000.0
    rms_norm_eps:  float = 1e-5

    halt_max_steps:        int   = 16
    halt_exploration_prob: float = 0.1
    no_ACT_continue:       bool  = True

    forward_dtype: str = "bfloat16"

    # Compat fields -- passed by pretrain.py, not used internally
    seq_len:                int  = 0
    puzzle_emb_ndim:        int  = 0
    puzzle_emb_len:         int  = 0
    mlp_t:                  bool = False
    vocab_size:             int  = 1
    num_puzzle_identifiers: int  = 1
    batch_size:             int  = 1
    causal:                 bool = False
    H_layers:               int  = 0


# ---------------------------------------------------------------------------
# Carry dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TinyRecursiveReasoningModel_MRIInnerCarry:
    z_H: torch.Tensor   # (B, seq_len, hidden_size)
    z_L: torch.Tensor   # (B, seq_len, hidden_size)


@dataclass
class TinyRecursiveReasoningModel_MRICarry:
    inner_carry:  TinyRecursiveReasoningModel_MRIInnerCarry
    steps:        torch.Tensor
    halted:       torch.Tensor
    current_data: Dict[str, torch.Tensor]


# ---------------------------------------------------------------------------
# CNN Encoder
# ---------------------------------------------------------------------------

class MRIEncoder(nn.Module):
    """
    (B, 3, H, W) -> (B, H*W, hidden_size)
    Channel 0: real part of normalised masked k-space
    Channel 1: imag part of normalised masked k-space
    Channel 2: undersampling mask (1=measured, 0=zero-filled)
    """

    def __init__(self, hidden_size: int, cnn_channels: int):
        super().__init__()
        C = cnn_channels
        self.conv1 = nn.Conv2d(3, C,   kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(C, C*2, kernel_size=3, padding=1, bias=True)
        self.proj  = nn.Linear(C*2, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class MRIDecoder(nn.Module):
    """
    (B, H*W, hidden_size) -> (B, 2, H*W)
    Outputs 2 channels: predicted real and imaginary parts of the complex image.
    No activation -- unbounded output needed for data_consistency.
    Loss is computed on the magnitude: sqrt(real^2 + imag^2).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 2, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, seq_len, hidden_size)
        returns (B, 2, seq_len)
        """
        out = self.proj(x)                  # (B, seq_len, 2)
        return out.permute(0, 2, 1)         # (B, 2, seq_len)


# ---------------------------------------------------------------------------
# Data consistency
# ---------------------------------------------------------------------------

def data_consistency(pred_complex:      torch.Tensor,
                     kspace_observed:   torch.Tensor,
                     mask:              torch.Tensor,
                     height:            int,
                     width:             int) -> torch.Tensor:
    """
    Replace the model's predicted k-space with real measurements where measured.

    Scale correctness:
        Both pred_complex and kspace_observed are normalised by the same
        per-slice scale factor (max(|kspace_raw|)).  Therefore:
            FFT(label_real + j*label_imag) == kspace_observed exactly
        and the model is trained to predict the normalised complex image,
        so FFT(pred_complex) and kspace_observed are on matching scales.

    Args:
        pred_complex    : (B, 2, H*W) float32  predicted [real|imag] image
        kspace_observed : (B, 2, H*W) float32  normalised [real|imag] k-space
        mask            : (B, W)      float32  1=measured, 0=zero-filled
        height, width   : spatial dimensions

    Returns:
        (B, 2, H*W) float32  complex image after data consistency
    """
    B = pred_complex.shape[0]

    # Build complex predicted image
    pred_2d  = pred_complex.view(B, 2, height, width).to(torch.float32)
    pred_cplx = torch.complex(pred_2d[:, 0], pred_2d[:, 1])        # (B, H, W)

    # FFT predicted image into k-space
    pred_k = torch.fft.fft2(pred_cplx)                             # (B, H, W) complex

    # Reconstruct observed k-space
    obs_2d = kspace_observed.view(B, 2, height, width).to(torch.float32)
    obs_k  = torch.complex(obs_2d[:, 0], obs_2d[:, 1])             # (B, H, W) complex

    # Broadcast mask: (B, W) -> (B, 1, W)
    mask_2d = mask.view(B, 1, width).to(torch.float32)

    # Data consistency
    dc_k = mask_2d * obs_k + (1.0 - mask_2d) * pred_k             # (B, H, W) complex

    # iFFT back to complex image domain
    dc_img = torch.fft.ifft2(dc_k)                                 # (B, H, W) complex

    # Pack back to (B, 2, H*W)
    dc_real = dc_img.real.view(B, 1, -1)
    dc_imag = dc_img.imag.view(B, 1, -1)
    return torch.cat([dc_real, dc_imag], dim=1)                     # (B, 2, H*W)


# ---------------------------------------------------------------------------
# Transformer block  (identical to trm.py)
# ---------------------------------------------------------------------------

class TinyRecursiveReasoningModel_MRIBlock(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_MRIConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp      = SwiGLU(hidden_size=config.hidden_size, expansion=config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps,
        )
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )
        return hidden_states


# ---------------------------------------------------------------------------
# Reasoning module  (identical to trm.py)
# ---------------------------------------------------------------------------

class TinyRecursiveReasoningModel_MRIReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_MRIBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor,
                input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# ---------------------------------------------------------------------------
# Inner model
# ---------------------------------------------------------------------------

class TinyRecursiveReasoningModel_MRI_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_MRIConfig) -> None:
        super().__init__()
        self.config        = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.seq_len       = config.height * config.width

        self.encoder = MRIEncoder(config.hidden_size, config.cnn_channels)
        self.decoder = MRIDecoder(config.hidden_size)
        self.q_head  = CastedLinear(config.hidden_size, 2, bias=True)

        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=self.seq_len,
                base=config.rope_theta,
            )

        self.L_level = TinyRecursiveReasoningModel_MRIReasoningModule(
            layers=[TinyRecursiveReasoningModel_MRIBlock(config)
                    for _ in range(config.L_layers)]
        )

        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def empty_carry(self, batch_size: int) -> TinyRecursiveReasoningModel_MRIInnerCarry:
        return TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=torch.empty(batch_size, self.seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.seq_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(self, reset_flag: torch.Tensor,
                    carry: TinyRecursiveReasoningModel_MRIInnerCarry,
                    ) -> TinyRecursiveReasoningModel_MRIInnerCarry:
        """
        Bug #3 fix: explicitly expand H_init/L_init to full (B, seq_len, D)
        before torch.where.  Naive (D,) + (B,1,1) broadcasts to (B,1,D)
        not (B, seq_len, D).
        """
        B    = carry.z_H.shape[0]
        flag = reset_flag.view(B, 1, 1)
        H_e  = self.H_init.view(1, 1, -1).expand(B, self.seq_len, self.config.hidden_size)
        L_e  = self.L_init.view(1, 1, -1).expand(B, self.seq_len, self.config.hidden_size)
        return TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=torch.where(flag, H_e, carry.z_H),
            z_L=torch.where(flag, L_e, carry.z_L),
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_MRIInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        TinyRecursiveReasoningModel_MRIInnerCarry,
        torch.Tensor,                           # pred_complex (B, 2, H*W)
        Tuple[torch.Tensor, torch.Tensor],      # q_halt, q_continue
    ]:
        seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)

        B = batch["inputs"].shape[0]

        # Build 3-channel encoder input: [real | imag | mask]
        inputs_2d = batch["inputs"].view(B, 2, self.config.height, self.config.width)
        mask_2d   = (batch["masks"]
                     .view(B, 1, 1, self.config.width)
                     .expand(B, 1, self.config.height, self.config.width))
        enc_in = torch.cat([inputs_2d, mask_2d], dim=1).float()    # (B, 3, H, W)

        input_embeddings = self.encoder(enc_in).to(self.forward_dtype)  # (B, seq_len, D)

        # Recursive reasoning loop (identical to trm.py)
        z_H, z_L = carry.z_H, carry.z_L

        with torch.no_grad():
            for _H in range(self.config.H_cycles - 1):
                for _L in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        for _L in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Decode: (B, seq_len, D) -> (B, 2, seq_len)
        pred_complex = self.decoder(z_H.float())                    # (B, 2, H*W)

        # Data consistency
        pred_complex = data_consistency(
            pred_complex=pred_complex,
            kspace_observed=batch["inputs"],                        # (B, 2, H*W) normalised
            mask=batch["masks"],                                    # (B, W)
            height=self.config.height,
            width=self.config.width,
        )

        # Q-head: mean pool over spatial positions
        z_H_mean = z_H.mean(dim=1)                                  # (B, D)
        q_logits = self.q_head(z_H_mean).to(torch.float32)          # (B, 2)

        new_carry = TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=z_H.detach(), z_L=z_L.detach()
        )

        return new_carry, pred_complex, (q_logits[..., 0], q_logits[..., 1])


# ---------------------------------------------------------------------------
# Outer ACT wrapper  (identical structure to trm.py)
# ---------------------------------------------------------------------------

class TinyRecursiveReasoningModel_MRI(nn.Module):

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_MRIConfig(**config_dict)
        self.inner  = TinyRecursiveReasoningModel_MRI_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_MRICarry:
        B = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModel_MRICarry(
            inner_carry=self.inner.empty_carry(B),
            steps=torch.zeros((B,), dtype=torch.int32),
            halted=torch.ones((B,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_MRICarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_MRICarry, Dict[str, torch.Tensor]]:

        new_inner_carry  = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps        = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        new_inner_carry, pred_complex, (q_halt_logits, q_continue_logits) = \
            self.inner(new_inner_carry, new_current_data)

        outputs = {
            "pred_complex":      pred_complex,
            "q_halt_logits":     q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps    = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted       = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    _, _, (nqh, nqc) = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(is_last_step, nqh, torch.maximum(nqh, nqc))
                    )

        return (
            TinyRecursiveReasoningModel_MRICarry(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
