import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.eps           = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W) float32   targets: (B, H, W) int64
        logits = logits.float()  # guard NaN under float16 AMP

        valid = targets != self.ignore_index
        targets_clamped = targets.clone()
        targets_clamped[~valid] = 0

        probs = F.softmax(logits, dim=1)                             # (B, C, H, W)
        one_hot = F.one_hot(targets_clamped, self.num_classes)       # (B, H, W, C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()               # (B, C, H, W)

        mask = valid.unsqueeze(1).expand_as(probs)
        probs   = probs   * mask
        one_hot = one_hot * mask

        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        union        = (probs + one_hot).sum(dim=(0, 2, 3))
        dice_per_class = (2.0 * intersection + self.eps) / (union + self.eps)
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """CrossEntropy + DiceLoss, both guarded against NaN under AMP float16."""

    def __init__(self, num_classes: int, ignore_index: int = 255,
                 ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce   = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)
        self.ce_w   = ce_weight
        self.dice_w = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()  # ensure float32 before loss computation
        ce_loss   = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_w * ce_loss + self.dice_w * dice_loss
