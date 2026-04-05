"""
models/losses_mri.py

MRI loss head for the TRM-MRI pipeline.
Modelled after ACTLossHead in models/losses.py.

Label format:
    Labels are (B, 2, H*W) complex images [real|imag] normalised by scale.
    Model outputs pred_complex of same shape.
    Loss = MSE on magnitude: |pred| vs |label|
    PSNR = 20 * log10(1 / sqrt(MSE_magnitude + 1e-8))

Fixes applied:
    - q_halt_loss only supervises HALTED elements (weight=0 for non-halted)
    - import math removed (was unused)
    - Loss/metric naming consistent with pretrain.py logging conventions
"""

from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn


class MRILossHead(nn.Module):
    """
    Wraps TinyRecursiveReasoningModel_MRI.
    Matches ACTLossHead calling convention for pretrain.py compatibility:

        new_carry, loss, metrics, preds, all_finish = loss_head(
            carry=carry, batch=batch, return_keys=return_keys
        )
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    def forward(
        self,
        carry:       Any,
        batch:       Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:

        # Model forward
        new_carry, outputs = self.model(carry=carry, batch=batch)

        # labels: (B, 2, H*W) complex image [real|imag]
        # pred:   (B, 2, H*W) complex image [real|imag]
        labels       = new_carry.current_data["labels"].float()   # (B, 2, H*W)
        pred_complex = outputs["pred_complex"].float()             # (B, 2, H*W)
        halted       = new_carry.halted                           # (B,) bool

        # Magnitude of complex prediction and label
        # magnitude = sqrt(real^2 + imag^2)  (B, H*W)
        pred_mag  = torch.sqrt(pred_complex[:, 0] ** 2 + pred_complex[:, 1] ** 2 + 1e-12)
        label_mag = torch.sqrt(labels[:, 0] ** 2 + labels[:, 1] ** 2 + 1e-12)

        # ── MSE on magnitude ───────────────────────────────────────────────
        mse_per_sample = F.mse_loss(pred_mag, label_mag, reduction="none").mean(dim=-1)  # (B,)
        mse_loss       = mse_per_sample.mean()

        # ── Q-halt loss  (halted elements only) ────────────────────────────
        q_halt_logits = outputs["q_halt_logits"]    # (B,)

        with torch.no_grad():
            median_mse  = mse_per_sample.median()
            q_target    = (mse_per_sample < median_mse).float()
            halt_weight = halted.float()            # 0 for non-halted, 1 for halted

        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits,
            q_target,
            weight=halt_weight,
            reduction="sum",
        ) / halt_weight.sum().clamp_min(1.0)

        # ── Optional Q-continue loss ────────────────────────────────────────
        q_continue_loss = torch.tensor(0.0, device=mse_loss.device)
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="mean",
            )

        # ── Total loss ─────────────────────────────────────────────────────
        loss = mse_loss + 0.5 * (q_halt_loss + q_continue_loss)

        # ── Metrics ────────────────────────────────────────────────────────
        with torch.no_grad():
            valid = halted & (label_mag.sum(dim=-1) > 0)
            count = valid.sum().float().clamp_min(1.0)

            mse_valid  = torch.where(valid, mse_per_sample, torch.zeros_like(mse_per_sample))

            # PSNR = 20 * log10(1 / sqrt(MSE + eps)) = -10 * log10(MSE + eps)
            psnr_valid = torch.where(
                valid,
                -10.0 * torch.log10(mse_per_sample.clamp_min(1e-8)),
                torch.zeros_like(mse_per_sample),
            )

            metrics = {
                "count":       count,
                "mse":         mse_valid.sum(),
                "psnr":        psnr_valid.sum(),
                "steps":       torch.where(halted, new_carry.steps.float(),
                                           torch.zeros_like(new_carry.steps.float())).sum(),
                "mse_loss":    mse_loss.detach(),
                "q_halt_loss": q_halt_loss.detach(),
            }

        # ── Filter outputs for evaluators ──────────────────────────────────
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, loss, metrics, detached_outputs, new_carry.halted.all()
