import torch
import numpy as np


class SegmentationMetrics:
    """Accumulates confusion matrix across batches, then computes mIoU."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = logits.argmax(dim=1)          # (B, H, W)
        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        valid = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Fast confusion matrix via np.bincount
        indices = self.num_classes * targets.astype(np.int64) + preds.astype(np.int64)
        mat = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion += mat.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        cm = self.confusion.astype(np.float64)
        tp  = np.diag(cm)
        fp  = cm.sum(axis=0) - tp
        fn  = cm.sum(axis=1) - tp
        iou = tp / (tp + fp + fn + 1e-10)

        valid_classes = (tp + fp + fn) > 0
        miou = iou[valid_classes].mean()

        return {
            'mIoU':          float(miou),
            'per_class_iou': iou.tolist(),
        }
