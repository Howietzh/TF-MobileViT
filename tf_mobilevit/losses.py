from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        valid = target != self.ignore_index
        safe_target = target.clamp(0, num_classes - 1)
        probs = logits.softmax(dim=1)
        one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, gamma: float = 2.0) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        valid = target != self.ignore_index
        loss = ((1 - pt) ** self.gamma) * ce
        return loss[valid].mean() if valid.any() else loss.mean()


class BoundaryLoss(nn.Module):
    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    @staticmethod
    def _soft_boundary(mask: torch.Tensor) -> torch.Tensor:
        dx = F.pad((mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs(), (1, 0, 0, 0))
        dy = F.pad((mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs(), (0, 0, 1, 0))
        return (dx + dy).clamp(0, 1)

    @staticmethod
    def _hard_boundary(mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(1)
        dx = F.pad((mask[:, :, :, 1:] != mask[:, :, :, :-1]).float(), (1, 0, 0, 0))
        dy = F.pad((mask[:, :, 1:, :] != mask[:, :, :-1, :]).float(), (0, 0, 1, 0))
        return (dx + dy).clamp(0, 1)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        foreground_prob = probs[:, 1:].sum(dim=1, keepdim=True) if probs.shape[1] > 1 else probs
        valid_target = target.clone()
        valid_target[target == self.ignore_index] = 0
        pred_boundary = self._soft_boundary(foreground_prob)
        target_boundary = self._hard_boundary((valid_target > 0).long())
        valid = (target != self.ignore_index).unsqueeze(1).float()
        return F.binary_cross_entropy(pred_boundary * valid, target_boundary * valid)


class CombinedLoss(nn.Module):
    def __init__(self, config: dict[str, Any], ignore_index: int = 255) -> None:
        super().__init__()
        weights = config.get("train", {}).get("loss", {})
        self.weights = {
            "dice": float(weights.get("dice", 1.0)),
            "focal": float(weights.get("focal", 1.0)),
            "boundary": float(weights.get("boundary", 0.0)),
        }
        self.dice = DiceLoss(ignore_index)
        self.focal = FocalLoss(ignore_index)
        self.boundary = BoundaryLoss(ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = logits.new_tensor(0.0)
        if self.weights["dice"]:
            loss = loss + self.weights["dice"] * self.dice(logits, target)
        if self.weights["focal"]:
            loss = loss + self.weights["focal"] * self.focal(logits, target)
        if self.weights["boundary"]:
            loss = loss + self.weights["boundary"] * self.boundary(logits, target)
        return loss
