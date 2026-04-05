"""
models/recursive_reasoning/trm_mri.py

TRM adapted for MRI k-space reconstruction.

Key differences from trm.py
────────────────────────────
1. No embed_tokens / lm_head — inputs are continuous k-space floats, not discrete tokens.
2. MRIEncoder  : (B, 3, H, W) → (B, H*W, hidden_size)
                 channel 0 = real(k_masked), channel 1 = imag(k_masked), channel 2 = mask
3. MRIDecoder  : (B, H*W, hidden_size) → (B, H*W)  unbounded real image
4. data_consistency() called after every decoder step to enforce k-space fidelity.
5. Q-head uses mean-pool over all spatial positions instead of position-0 peek,
   because position 0 is now a spatial pixel (no longer a dedicated summary token).
6. reset_carry uses expand() before torch.where() to avoid the broadcasting bug
   where (D,) expands to (B,1,D) rather than (B,seq_len,D).

Everything else — z_H/z_L dual-latent carry, H_cycles/L_cycles inner loop,
ACT halting with exploration, no_ACT_continue flag — is untouched from trm.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    CastedLinear,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    rms_norm,
    Attention,
)


# ──────────────────────────────────────────────────────────────────────────────
# Carry dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TinyRecursiveReasoningModel_MRIInnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_MRICarry:
    inner_carry: TinyRecursiveReasoningModel_MRIInnerCarry
    steps:       torch.Tensor
    halted:      torch.Tensor
    current_data: Dict[str, torch.Tensor]


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

class TinyRecursiveReasoningModel_MRIConfig(BaseModel):
    # Spatial size — passed from dataset metadata via pretrain.py
    height: int
    width:  int

    # Compat fields passed by pretrain.py but unused for MRI
    seq_len:               int = 0     # overridden by height*width at runtime
    vocab_size:            int = 1     # unused
    num_puzzle_identifiers: int = 1    # unused
    puzzle_emb_ndim:       int = 0     # must be 0 — no puzzle embeddings for MRI
    causal:                bool = False  # unused; accepted so pretrain.py doesn't crash

    # Training
    batch_size: int = 1   # overridden by pretrain.py

    # TRM structure
    H_cycles: int
    L_cycles: int
    H_layers: int = 0   # ignored (TRM shares one module)
    L_layers: int

    # Transformer
    hidden_size: int
    expansion:   float
    num_heads:   int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta:   float = 10000.0

    # ACT halting
    halt_max_steps:        int
    halt_exploration_prob: float
    no_ACT_continue:       bool = True

    forward_dtype: str = "bfloat16"

    # MRI CNN encoder/decoder channels
    cnn_channels: int = 32


# ──────────────────────────────────────────────────────────────────────────────
# CNN Encoder  (B, 3, H, W) → (B, H*W, hidden_size)
# ──────────────────────────────────────────────────────────────────────────────

class MRIEncoder(nn.Module):
    """
    Two depthwise-separable Conv2d layers followed by a linear projection.
    Input channels:
        0 — real(k_masked)
        1 — imag(k_masked)
        2 — mask (broadcast over H, so shape is (B,1,H,W) at call site)
    """
    def __init__(self, hidden_size: int, cnn_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, cnn_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.proj  = nn.Linear(cnn_channels, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        h = F.gelu(self.conv1(x))          # (B, C, H, W)
        h = F.gelu(self.conv2(h))          # (B, C, H, W)
        B, C, H, W = h.shape
        h = h.permute(0, 2, 3, 1).reshape(B, H * W, C)   # (B, H*W, C)
        return self.proj(h)                # (B, H*W, hidden_size)


# ──────────────────────────────────────────────────────────────────────────────
# CNN Decoder  (B, H*W, hidden_size) → (B, H*W)
# ──────────────────────────────────────────────────────────────────────────────

class MRIDecoder(nn.Module):
    """
    Linear projection from hidden_size → 1 per position.
    No sigmoid — output is unbounded; data_consistency can push values outside [0,1]
    so clamping here would break that loop.
    """
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H*W, hidden_size)
        return self.proj(x).squeeze(-1)   # (B, H*W)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer block (identical to TRM)
# ──────────────────────────────────────────────────────────────────────────────

class TRMBlock(nn.Module):
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
        return rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)


class ReasoningModule(nn.Module):
    def __init__(self, layers: List[TRMBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Data consistency layer
# ──────────────────────────────────────────────────────────────────────────────

def data_consistency(
    image:          torch.Tensor,   # (B, H*W)  predicted image, float32
    kspace_input:   torch.Tensor,   # (B, 2, H*W)  [real, imag] raw k-space input
    mask:           torch.Tensor,   # (B, W)  1-D Cartesian mask
    height:         int,
    width:          int,
) -> torch.Tensor:
    """
    Classic MRI data consistency in k-space:
        pred_k = FFT(predicted_image)
        dc_k   = mask * k_observed  +  (1-mask) * pred_k
        output = real(iFFT(dc_k))

    All ops in float32 for numerical stability.
    Input and output shapes: (B, H*W).

    Scale consistency:
        kspace_input is raw (un-normalised) physical k-space.
        image is in the same physical scale as label (normalised by RSS max).
        The RSS label satisfies: label = iFFT(k_full) / scale,
        so FFT(label) = k_full / scale.
        We therefore must also divide kspace_input by the per-slice scale before
        blending.  However, the scale is not passed here — so instead we normalise
        kspace_input on the fly to match the energy of pred_k.  This is equivalent
        to operating in the label's normalised domain.
    """
    B = image.shape[0]

    # Reshape predicted image to (B, H, W) for FFT
    pred_image_2d = image.view(B, height, width).to(torch.float32)

    # FFT of predicted image  →  predicted k-space (complex)
    pred_k = torch.fft.fft2(pred_image_2d)           # (B, H, W) complex64

    # Reconstruct complex observed k-space from stored [real, imag] channels
    kspace_2d = kspace_input.view(B, 2, height, width).to(torch.float32)
    obs_k = torch.complex(kspace_2d[:, 0], kspace_2d[:, 1])   # (B, H, W)

    # Normalise observed k-space to match pred_k energy so they can be blended.
    # We compute the scale as the ratio of RMS magnitudes over the observed lines.
    # This is safe because there is always at least one observed line (center kept).
    with torch.no_grad():
        mask_2d = mask.view(B, 1, width).to(torch.float32)          # (B, 1, W)
        obs_energy  = (obs_k.abs() * mask_2d).pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
        pred_energy = (pred_k.abs() * mask_2d).pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-8)
        energy_ratio = (pred_energy / obs_energy)    # (B, 1, 1)

    # Scale obs_k to match pred_k magnitude
    obs_k_scaled = obs_k * energy_ratio              # (B, H, W)

    # Data consistency blend
    mask_2d_hw = mask_2d.expand(B, height, width)    # (B, H, W)
    dc_k = mask_2d_hw * obs_k_scaled + (1.0 - mask_2d_hw) * pred_k   # (B, H, W)

    # iFFT  →  real image
    dc_image = torch.fft.ifft2(dc_k).real            # (B, H, W)
    return dc_image.reshape(B, height * width)        # (B, H*W)


# ──────────────────────────────────────────────────────────────────────────────
# Inner model
# ──────────────────────────────────────────────────────────────────────────────

class TinyRecursiveReasoningModel_MRI_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_MRIConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.seq_len = config.height * config.width

        # CNN encoder/decoder
        self.encoder = MRIEncoder(config.hidden_size, config.cnn_channels)
        self.decoder = MRIDecoder(config.hidden_size)

        # Q-head (halt/continue classifier)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        # RoPE
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=self.seq_len,
                base=config.rope_theta,
            )

        # Reasoning module (shared for both H and L passes, as in TRM)
        self.L_level = ReasoningModule(
            [TRMBlock(config) for _ in range(config.L_layers)]
        )

        # Learned initial states for z_H and z_L  — shape (hidden_size,)
        # reset_carry expands these correctly to (B, seq_len, hidden_size).
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        # Init Q-head weights near zero for stable early training
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    # ── Carry helpers ──────────────────────────────────────────────────────

    def empty_carry(self, batch_size: int) -> TinyRecursiveReasoningModel_MRIInnerCarry:
        return TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=torch.empty(batch_size, self.seq_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.seq_len, self.config.hidden_size, dtype=self.forward_dtype),
        )

    def reset_carry(
        self,
        reset_flag: torch.Tensor,                          # (B,) bool
        carry: TinyRecursiveReasoningModel_MRIInnerCarry,
    ) -> TinyRecursiveReasoningModel_MRIInnerCarry:
        """
        For sequences where reset_flag is True, replace carry with the learned
        initial state.  Uses expand() to avoid the broadcasting shape bug where
        (D,) would expand to (B,1,D) rather than (B,seq_len,D).
        """
        B = carry.z_H.shape[0]
        flag = reset_flag.view(B, 1, 1)    # (B, 1, 1)

        H_init_exp = self.H_init.view(1, 1, -1).expand(B, self.seq_len, self.config.hidden_size)
        L_init_exp = self.L_init.view(1, 1, -1).expand(B, self.seq_len, self.config.hidden_size)

        return TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=torch.where(flag, H_init_exp, carry.z_H),
            z_L=torch.where(flag, L_init_exp, carry.z_L),
        )

    # ── Input encoding ──────────────────────────────────────────────────────

    def _encode_input(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Encode (B, 2, H*W) k-space + (B, W) mask  →  (B, H*W, hidden_size).

        Mask is broadcast to (B, 1, H, W) and concatenated as a 3rd channel,
        giving the encoder explicit access to which k-space lines are observed.
        This is a standard trick in learned MRI reconstruction.
        """
        B = batch["inputs"].shape[0]
        H, W = self.config.height, self.config.width

        # Reshape flat inputs to 2D spatial
        kspace_2d = batch["inputs"].view(B, 2, H, W).to(self.forward_dtype)  # (B,2,H,W)

        # Mask (B, W) → (B, 1, H, W)
        mask_2d = batch["masks"].view(B, 1, 1, W).expand(B, 1, H, W).to(self.forward_dtype)

        # Concatenate along channel dim
        encoder_input = torch.cat([kspace_2d, mask_2d], dim=1)   # (B, 3, H, W)

        return self.encoder(encoder_input)   # (B, H*W, hidden_size)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_MRIInnerCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_MRIInnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Encode input (always computed fresh — no gradient through encoder at no-grad steps)
        input_embeddings = self._encode_input(batch)   # (B, H*W, hidden_size)

        z_H, z_L = carry.z_H, carry.z_L

        # ── H_cycles-1 passes without gradient ──────────────────────────────
        with torch.no_grad():
            for _h in range(self.config.H_cycles - 1):
                for _l in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)

        # ── Final pass WITH gradient ──────────────────────────────────────
        for _l in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # ── Decode image ─────────────────────────────────────────────────
        pred_image = self.decoder(z_H)   # (B, H*W)  — unbounded float

        # ── Data consistency ─────────────────────────────────────────────
        # Runs in float32 even inside bfloat16 training for numerical stability.
        pred_image = data_consistency(
            image=pred_image.to(torch.float32),
            kspace_input=batch["inputs"],          # (B, 2, H*W) raw k-space
            mask=batch["masks"],                   # (B, W)
            height=self.config.height,
            width=self.config.width,
        ).to(self.forward_dtype)                   # back to training dtype

        # ── Q-head (mean pool over spatial positions) ────────────────────
        # Mean pool is correct here — position 0 is a spatial pixel, not a
        # special summary token as in the original TRM.
        q_logits = self.q_head(z_H.mean(dim=1)).to(torch.float32)   # (B, 2)

        # Detach carry for next step
        new_carry = TinyRecursiveReasoningModel_MRIInnerCarry(
            z_H=z_H.detach(),
            z_L=z_L.detach(),
        )

        return new_carry, pred_image, (q_logits[..., 0], q_logits[..., 1])


# ──────────────────────────────────────────────────────────────────────────────
# Outer ACT wrapper  (matches the interface expected by losses_mri.py / pretrain.py)
# ──────────────────────────────────────────────────────────────────────────────

class TinyRecursiveReasoningModel_MRI(nn.Module):
    """ACT wrapper — identical control flow to TRM, adapted for MRI I/O."""

    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        self.config = TinyRecursiveReasoningModel_MRIConfig(**config_dict)
        self.inner  = TinyRecursiveReasoningModel_MRI_Inner(self.config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> TinyRecursiveReasoningModel_MRICarry:
        B = batch["inputs"].shape[0]
        return TinyRecursiveReasoningModel_MRICarry(
            inner_carry=self.inner.empty_carry(B),
            steps=torch.zeros(B, dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones(B, dtype=torch.bool,  device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()},
        )

    def forward(
        self,
        carry: TinyRecursiveReasoningModel_MRICarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[TinyRecursiveReasoningModel_MRICarry, Dict[str, torch.Tensor]]:

        # ── Reset halted sequences to initial carry ───────────────────────
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps       = torch.where(carry.halted, torch.zeros_like(carry.steps), carry.steps)
        new_current_data = {
            k: torch.where(
                carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                batch[k],
                v,
            )
            for k, v in carry.current_data.items()
        }

        # ── Inner forward ─────────────────────────────────────────────────
        new_inner_carry, pred_image, (q_halt_logits, q_continue_logits) = self.inner(
            new_inner_carry, new_current_data
        )

        outputs = {
            "pred_image":       pred_image,         # (B, H*W)
            "q_halt_logits":    q_halt_logits,       # (B,)
            "q_continue_logits":q_continue_logits,   # (B,)
        }

        with torch.no_grad():
            new_steps    = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted       = is_last_step

            if self.training and self.config.halt_max_steps > 1:
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration: enforce a minimum number of steps
                min_halt = (
                    (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob)
                    * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                )
                halted = halted & (new_steps >= min_halt)

        return (
            TinyRecursiveReasoningModel_MRICarry(new_inner_carry, new_steps, halted, new_current_data),
            outputs,
        )
