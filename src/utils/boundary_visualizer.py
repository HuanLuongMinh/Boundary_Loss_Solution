"""Diagnostic visualizer for the boundary regions of GT vs. predicted masks.

"Boundary region" here is extracted with the same 3x3 Laplacian kernel used by
`EdgeLoss.get_boundary()` in the reference GeoSeg codebase
(https://github.com/Sjyhne/ContrastiveGeoSeg/blob/main/geoseg/losses/useful_loss.py):
a pixel is a boundary pixel when it disagrees with its 8-neighbourhood (label
changes class within the 3x3 window). This is a purely qualitative diagnostic
for the baseline (no boundary loss is used during training) — it shows where
the model's predicted region edges line up with (or drift from) the
ground-truth edges after each validation pass.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .visualizer import denormalize


def get_boundary(mask: np.ndarray, thresh: float = 0.1) -> np.ndarray:
    """(H, W) int label mask -> (H, W) uint8 {0,1} boundary mask.

    Equivalent to conv2d(mask, [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], padding=1)
    clamped to >=0 and thresholded, with zero-padding (matches the reference
    implementation, including its minor border artifact at the image edge).
    """
    m = mask.astype(np.float32)
    h, w = m.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.float32)
    padded[1:-1, 1:-1] = m

    lap = 8.0 * m
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            lap = lap - padded[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]

    lap = np.clip(lap, 0, None)
    return (lap >= thresh).astype(np.uint8)


def save_boundary_visualization(image: np.ndarray, gt_mask: np.ndarray,
                                pred_mask: np.ndarray, save_path: str,
                                denormalize_img: bool = True,
                                title: str = ''):
    """Save a 2-column figure: Original | GT/Pred boundary overlay.

    Overlay colours: GT boundary only = cyan, Pred boundary only = magenta,
    both agree (overlap) = white.

    Args:
        image:          (H, W, 3) float32 (normalized) or uint8.
        gt_mask:        (H, W) int ground-truth label map.
        pred_mask:      (H, W) int predicted label map.
        save_path:      Destination .png path (parent dirs created automatically).
        denormalize_img: Reverse ImageNet normalization when image is float.
        title:          Optional suptitle, e.g. "Iter 4000 | mIoU=0.4231".
    """
    if denormalize_img and image.dtype != np.uint8:
        image = denormalize(image)

    gt_boundary   = get_boundary(gt_mask)
    pred_boundary = get_boundary(pred_mask)
    overlap       = (gt_boundary & pred_boundary).astype(bool)
    gt_only       = (gt_boundary.astype(bool)) & ~overlap
    pred_only     = (pred_boundary.astype(bool)) & ~overlap

    overlay = image.copy()
    overlay[gt_only]   = [0,   255, 255]   # cyan   — GT boundary only
    overlay[pred_only] = [255, 0,   255]   # magenta— Pred boundary only
    overlay[overlap]   = [255, 255, 255]   # white  — both agree

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image);   axes[0].set_title('Original',          fontsize=12)
    axes[1].imshow(overlay); axes[1].set_title('Boundary (GT vs Pred)', fontsize=12)
    for ax in axes:
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)

    patches = [
        mpatches.Patch(color=(0, 1, 1),    label='GT boundary only'),
        mpatches.Patch(color=(1, 0, 1),    label='Pred boundary only'),
        mpatches.Patch(color=(1, 1, 1),    label='Overlap (agree)'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='black')
    plt.close(fig)
