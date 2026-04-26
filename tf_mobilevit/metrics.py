from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SegmentationMetrics:
    num_classes: int
    ignore_index: int = 255

    def __post_init__(self) -> None:
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float64)

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(dim=1).detach().cpu()
        target = target.detach().cpu()
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        t = target[valid].reshape(-1)
        p = pred[valid].reshape(-1).clamp(0, self.num_classes - 1)
        if t.numel() == 0:
            return
        idx = t * self.num_classes + p
        bincount = torch.bincount(idx, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        self.confusion += bincount.double()

    def compute(self) -> dict[str, float]:
        cm = self.confusion
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp_min(1)
        dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        foreground = torch.arange(self.num_classes) > 0
        return {
            "mean_iou": float(iou.mean()),
            "mean_dice": float(dice.mean()),
            "mean_recall": float(recall.mean()),
            "foreground_iou": float(iou[foreground].mean()) if foreground.any() else float(iou.mean()),
            "foreground_dice": float(dice[foreground].mean()) if foreground.any() else float(dice.mean()),
            "foreground_recall": float(recall[foreground].mean()) if foreground.any() else float(recall.mean()),
        }

    def per_class(self, class_names: list[str]) -> list[dict[str, float | str]]:
        cm = self.confusion
        tp = cm.diag()
        fp = cm.sum(0) - tp
        fn = cm.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp_min(1)
        dice = (2 * tp) / (2 * tp + fp + fn).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        return [
            {"class": class_names[i], "iou": float(iou[i]), "dice": float(dice[i]), "recall": float(recall[i])}
            for i in range(self.num_classes)
        ]

