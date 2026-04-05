"""
models/losses_mri.py

MRI loss head wrapping TinyRecursiveReasoningModel_MRI.

Differences from ACTLossHead in losses.py:
  - Reconstruction loss  : MSE instead of cross-entropy
  - Accuracy metric      : PSNR instead of exact-accuracy
  - Q-halt target        : "is this prediction better than median?" (per-batch)
  - Q-halt loss          : only computed over halted elements (correct supervision)
  - No q_continue loss   : no_ACT_continue=True is assumed
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class MRILossHead(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    # ── Carry delegation ────────────────────────────────────────────────────

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)

    # ── Forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        carry: Any,
        batch: Dict[str, torch.Tensor],
        return_keys: Sequence[str] = (),
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns
        -------
        new_carry, total_loss, metrics, detached_outputs, all_finished
        """
        # ── Model forward ────────────────────────────────────────────────
        new_carry, outputs = self.model(carry=carry, batch=batch)

        pred_image   = outputs["pred_image"]      # (B, H*W)  float (training dtype)
        target_image = new_carry.current_data["labels"].to(pred_image.dtype)  # (B, H*W)

        # ── Reconstruction loss (MSE, sum over batch for gradient scaling) ─
        # pretrain.py scales by (1/global_batch_size), so we sum here.
        mse_per_sample = F.mse_loss(pred_image, target_image, reduction="none").mean(dim=-1)  # (B,)
        mse_loss       = mse_per_sample.sum()   # scalar

        # ── PSNR metric (halted elements only) ───────────────────────────
        halted      = new_carry.halted                     # (B,) bool
        valid_count = halted.sum().clamp_min(1)

        with torch.no_grad():
            psnr_per_sample = 20.0 * torch.log10(
                1.0 / (mse_per_sample.detach().clamp_min(1e-8)).sqrt()
            )                                              # (B,)

            psnr_sum = torch.where(halted, psnr_per_sample, torch.zeros_like(psnr_per_sample)).sum()
            mse_sum  = torch.where(halted, mse_per_sample.detach(), torch.zeros_like(mse_per_sample)).sum()
            steps_sum = torch.where(halted, new_carry.steps.float(), torch.zeros_like(new_carry.steps, dtype=torch.float32)).sum()

        # ── Q-halt loss (halted elements only) ───────────────────────────
        # Target: 1 if this sample is below-median MSE (good prediction), else 0.
        # Only halted samples contribute — they have produced a final answer.
        q_halt_logits = outputs["q_halt_logits"]           # (B,)

        with torch.no_grad():
            median_mse = mse_per_sample.detach().median()
            q_target   = (mse_per_sample.detach() < median_mse).float()   # (B,)
            weight     = halted.float()                    # (B,)  0 or 1

        q_halt_loss = F.binary_cross_entropy_with_logits(
            q_halt_logits, q_target, weight=weight, reduction="sum"
        ) / weight.sum().clamp_min(1)

        # ── Total loss ───────────────────────────────────────────────────
        total_loss = mse_loss + 0.5 * q_halt_loss

        # ── Metrics dict (all tensors, reduced in pretrain.py) ───────────
        metrics: Dict[str, torch.Tensor] = {
            "count":      valid_count.float(),
            "mse":        mse_sum,
            "psnr":       psnr_sum,
            "steps":      steps_sum,
            "mse_loss":   mse_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        }

        # ── Detach selected outputs for evaluators / logging ─────────────
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        all_finished: torch.Tensor = new_carry.halted.all()

        return new_carry, total_loss, metrics, detached_outputs, all_finished
