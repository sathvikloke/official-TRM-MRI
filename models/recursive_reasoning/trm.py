"""
models/recursive_reasoning/trm_mri.py

TRM adapted for MRI k-space reconstruction.

Faithful to trm.py — every piece of the TRM logic is preserved exactly:
  - z_H / z_L dual-latent carry
  - H_cycles-1 no-gradient passes + 1 gradient pass (inner loop)
  - L_cycles inner loop structure
  - ACT halting with Q-head and exploration
  - no_ACT_continue flag AND the full else-branch when it is False
  - reset_carry on halted sequences
  - initial_carry / empty_carry pattern

MRI-specific changes (only these, nothing else):
  1. embed_tokens / lm_head removed — inputs are continuous k-space floats.
  2. MRIEncoder  : (B, 3, H, W) -> (B, H*W, hidden_size)
                   ch0 = real(k_masked), ch1 = imag(k_masked), ch2 = mask
  3. MRIDecoder  : (B, H*W, hidden_size) -> (B, H*W)  -- no sigmoid (unbounded).
  4. data_consistency() called after every decoder step.
  5. Q-head uses mean-pool over all positions (not position-0 summary token).
  6. reset_carry uses explicit expand() to fix the (D,)->(B,1,D) broadcast bug.
  7. puzzle_emb_ndim = 0 always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn

from models.common import trunc_normal_init_
from models.layers import (
    Attention,
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
)


# ──────────────────────────────────────────────────────────────────────────────
# Carry dataclasses  (mirror trm.py naming / structure exactly)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TinyRecursiveReasoningModel_MRIInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_MRICarry:
    inner_carry:  TinyRecursiveReasoningModel_MRIInnerCarry
    steps:        torch.Tensor
    halted:       torch.Tensor
    current_data: Dict[str, torch.Tensor]


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class TinyRecursiveReasoningModel_MRIConfig(BaseModel):
    # Spatial size -- read from dataset.json by pretrain.py and passed in
    height: int
    width:  int

    # Compat fields injected by pretrain.py (accepted but unused for MRI)
    seq_len:                int  = 0      # derived at runtime as height*width
    vocab_size:             int  = 1      # unused
    num_puzzle_identifiers: int  = 1      # unused
    puzzle_emb_ndim:        int  = 0      # must stay 0 -- no puzzle embeddings
    causal:                 bool = False  # unused; accepted so pretrain.py does not crash
    batch_size:             int  = 1      # overridden by pretrain.py

    # TRM structure (same fields as trm.py)
    H_cycles: int
    L_cycles: int
    H_layers: int = 0   # kept for compat; TRM uses a single shared L_level module
    L_layers: int

    # Transformer
    hidden_size:   int
    expansion:     float
    num_heads:     int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta:   float = 10000.0

    # ACT halting (same as trm.py)
    halt_max_steps:        int
    halt_exploration_prob: float
    no_ACT_continue:       bool = True

    forward_dtype: str = "bfloat16"

    # MRI CNN encoder channels
    cnn_channels: int = 32


# ──────────────────────────────────────────────────────────────────────────────
# MRI Encoder  (B, 3, H, W) -> (B, H*W, hidden_size)
# Replaces embed_tokens from trm.py.
# ──────────────────────────────────────────────────────────────────────────────

class MRIEncoder(nn.Module):
    """
    Two Conv2d layers followed by a linear projection.
    Input channels:
        0 -- real(k_masked)
        1 -- imag(k_masked)
        2 -- mask (tells model which lines are observed vs zero-filled)
    """
    def __init__(self, hidden_size: int, cnn_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.proj  = nn.Linear(cnn_channels, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        h = F.gelu(self.conv1(x))                              # (B, C, H, W)
        h = F.gelu(self.conv2(h))                              # (B, C, H, W)
        B, C, H, W = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)        # (B, H*W, C)
        return self.proj(h)                                    # (B, H*W, hidden_size)


# ──────────────────────────────────────────────────────────────────────────────
# MRI Decoder  (B, H*W, hidden_size) -> (B, H*W)
# Replaces lm_head from trm.py.
# ──────────────────────────────────────────────────────────────────────────────

class MRIDecoder(nn.Module):
    """
    Linear projection hidden_size -> 1 per position. No sigmoid: unbounded output
    is correct because data_consistency can push values outside [0,1].
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H*W, hidden_size)
        return self.proj(x).squeeze(-1)   # (B, H*W)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer block -- identical to trm.py
# ──────────────────────────────────────────────────────────────────────────────

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
        return rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps,
        )


class TinyRecursiveReasoningModel_MRIReasoningModule(nn.Module):
    """Identical to TinyRecursiveReasoningModel_ACTV1ReasoningModule in trm.py."""
    def __init__(self, layers: List[TinyRecursiveReasoningModel_MRIBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Data consistency  (MRI-specific step added after each decoder pass)
# ──────────────────────────────────────────────────────────────────────────────

def data_consistency(
    image:        torch.Tensor,   # (B, H*W)    predicted image, float32
    kspace_input: torch.Tensor,   # (B, 2, H*W) [real, imag] raw k-space input
    mask:         torch.Tensor,   # (B, W)       1-D Cartesian undersampling mask
    height:       int,
    width:        int,
) -> torch.Tensor:
    """
    Classic MRI data-consistency step:
        pred_k = FFT(predicted_image)
        dc_k   = mask * k_observed  +  (1 - mask) * pred_k
        output = real(iFFT(dc_k))

    All ops in float32 for FFT numerical stability.

    Scale note: kspace_input is raw physical k-space; pred_image is in the
    normalised label domain [0,1]. We equalise scales on-the-fly by matching
    RMS energy over the observed lines before blending.
    """
    B = image.shape[0]

    # Predicted image -> predicted k-space
    pred_image_2d = image.view(B, height, width).to(torch.float32)
    pred_k = torch.fft.fft2(pred_image_2d)                             # (B, H, W) complex

    # Reconstruct observed k-space from [real, imag] channels
    kspace_2d = kspace_input.view(B, 2, height, width).to(torch.float32)
    obs_k = torch.complex(kspace_2d[:, 0], kspace_2d[:, 1])            # (B, H, W) complex

    # Normalise obs_k to match pred_k energy on observed lines
    with torch.no_grad():
        mask_hw = mask.view(B, 1, width).to(torch.float32)
        obs_energy  = (obs_k.abs()  * mask_hw).pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
        pred_energy = (pred_k.abs() * mask_hw).pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
        energy_ratio = pred_energy / obs_energy                         # (B, 1, 1)

    obs_k_scaled = obs_k * energy_ratio                                 # (B, H, W)

    # Blend: keep observed lines from (scaled) ground truth
    mask_2d = mask_hw.expand(B, height, width)                          # (B, H, W)
    dc_k    = mask_2d * obs_k_scaled + (1.0 - mask_2d) * pred_k        # (B, H, W)

    # iFFT -> real image
    dc_image = torch.fft.ifft2(dc_k).real                              # (B, H, W)
    return dc_image.reshape(B, height * width)                          # (B, H*W)


# ──────────────────────────────────────────────────────────────────────────────
# Inner model  (mirrors TinyRecursiveReasoningModel_ACTV1_Inner from trm.py)
# ──────────────────────────────────────────────────────────────────────────────

class TinyRecursiveReasoningModel_MRI_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_MRIConfig) -> None:
        super().__init__()
        self.config        = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.seq_len       = config.height * config.width

        # I/O -- CNN encoder replaces embed_tokens; decoder replaces lm_head
        self.encoder = MRIEncoder(config.hidden_size, config.cnn_channels)
        self.decoder = MRIDecoder(config.hidden_size)

        # Q-head (identical to trm.py: CastedLinear, initialised near zero)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # Positional encodings
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=self.seq_len,
                base=config.rope_theta,
            )
        # "none": rotary_emb will not exist; cos_sin=None passed to blocks

        # Reasoning module (single shared L_level, same as trm.py)
        self.L_level = TinyRecursiveReasoningModel_MRIReasoningModule(
            [TinyRecursiveReasoningModel_MRIBlock(config) for _ in range(config.L_layers)]
        )

        # Learned initial states for z_H and z_L (same pattern as trm.py)
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Q-head init near zero for stable early bootstrapping (same as trm.py)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)   # type: ignore

    # ── Carry helpers ──────────────────────────────────────────────────────────

    def empty_carry(self, batch_size: int) -> TinyRecursiveReasoningModel_MRIInnerCarry:
        return TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=torch.empty(batch_size, self.seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.seq_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,
        carry:      TinyRecursiveReasoningModel_MRIInnerCarry,
    ) -> TinyRecursiveReasoningModel_MRIInnerCarry:
        """
        Reset halted sequences to the learned initial state.
        Uses explicit expand() to fix the trm.py broadcast bug:
        H_init shape (D,) would broadcast to (B,1,D), not (B,seq_len,D).
        """
        B    = carry.z_H.shape[0]
        flag = reset_flag.view(B, 1, 1)

        H_init_exp = self.H_init.view(1, 1, -1).expand(B, self.seq_len, self.config.hidden_size)
        L_init_exp = self.L_init.view(1, 1, -1).expand(B, self.seq_len, self.config.hidden_size)

        return TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=torch.where(flag, H_init_exp, carry.z_H),
            z_L=torch.where(flag, L_init_exp, carry.z_L),
        )

    # ── Input encoding ─────────────────────────────────────────────────────────

    def _encode_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode (B, 2, H*W) k-space + (B, W) mask -> (B, H*W, hidden_size).
        Replaces _input_embeddings() from trm.py.
        """
        B = batch["inputs"].shape[0]
        H, W = self.config.height, self.config.width

        kspace_2d = batch["inputs"].view(B, 2, H, W).to(self.forward_dtype)           # (B, 2, H, W)
        mask_2d   = batch["masks"].view(B, 1, 1, W).expand(B, 1, H, W).to(self.forward_dtype)  # (B, 1, H, W)
        encoder_input = torch.cat([kspace_2d, mask_2d], dim=1)                         # (B, 3, H, W)
        return self.encoder(encoder_input)                                              # (B, H*W, hidden_size)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_MRIInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_MRIInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding (replaces _input_embeddings in trm.py)
        input_embeddings = self._encode_input(batch)   # (B, H*W, hidden_size)

        z_H, z_L = carry.z_H, carry.z_L

        # H_cycles-1 passes WITHOUT gradient -- identical to trm.py
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles - 1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # Final pass WITH gradient -- identical to trm.py
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # Decode image (replaces lm_head in trm.py)
        pred_image = self.decoder(z_H)   # (B, H*W) -- unbounded float

        # Data consistency (MRI-specific, added after each decode step)
        # Runs in float32 for FFT numerical stability even inside bfloat16 training.
        pred_image = data_consistency(
            image=pred_image.to(torch.float32),
            kspace_input=batch["inputs"],
            mask=batch["masks"],
            height=self.config.height,
            width=self.config.width,
        ).to(self.forward_dtype)

        # Q-head: mean-pool over spatial positions
        # trm.py uses z_H[:, 0] (dedicated summary token); here pos-0 is a pixel,
        # so we mean-pool to get a global representation for the halt decision.
        q_logits = self.q_head(z_H.mean(dim=1)).to(torch.float32)   # (B, 2)

        # Detach carry for next ACT step (same as trm.py)
        new_carry = TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
        )

        return new_carry, pred_image, (q_logits[..., 0], q_logits[..., 1])


# ──────────────────────────────────────────────────────────────────────────────
# Outer ACT wrapper  (identical control flow to TinyRecursiveReasoningModel_ACTV1)
# ──────────────────────────────────────────────────────────────────────────────

class TinyRecursiveReasoningModel_MRI(nn.Module):
    """
    ACT wrapper. Control flow is identical to TinyRecursiveReasoningModel_ACTV1
    in trm.py — including the full no_ACT_continue=False branch with target_q_continue.
    Only the I/O format differs (continuous images instead of discrete token sequences).
    """

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = TinyRecursiveReasoningModel_MRIConfig(**config_dict)
        self.inner  = TinyRecursiveReasoningModel_MRI_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_MRICarry:
        """Identical to trm.py: empty carry, all sequences start as halted=True."""
        B = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModel_MRICarry(
            inner_carry=self.inner.empty_carry(B),
            steps=torch.zeros(B, dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones(B,  dtype=torch.bool,  device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_MRICarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_MRICarry, Dict[str, torch.Tensor]]:

        # Reset halted sequences -- identical to trm.py
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps       = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }

        # Inner forward
        new_inner_carry, pred_image, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "pred_image":        pred_image,
            "q_halt_logits":     q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Step counter -- identical to trm.py
            new_steps    = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted       = is_last_step

            # ACT halting -- identical to trm.py
            # NOTE: During evaluation always use max steps so all sequences in a
            # batch halt together (same guarantee as trm.py).
            if self.training and self.config.halt_max_steps > 1:

                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: enforce a minimum number of steps -- identical to trm.py
                min_halt_steps = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt_steps)

                # Compute target Q for continue-path -- identical to trm.py else-branch
                if not self.config.no_ACT_continue:
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(
                        new_inner_carry, new_current_data
                    )
                    outputs["target_q_continue"] = torch.sigmoid(
                        torch.where(
                            is_last_step,
                            next_q_halt_logits,
                            torch.maximum(next_q_halt_logits, next_q_continue_logits),
                        )
                    )

        return (
            TinyRecursiveReasoningModel_MRICarry(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
