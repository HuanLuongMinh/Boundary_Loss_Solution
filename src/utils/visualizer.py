import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# OpenEarthMap 9-class palette (RGB, 0-255) — chính thức theo bảng màu OEM
# (class 0 "Background"/unlabeled dùng đen 000000, không có trong bảng gốc
# vì OEM chỉ định nghĩa màu cho 8 lớp có nhãn).
OEM_PALETTE = np.array([
    [0,   0,   0  ],   # 0 Background      (000000)
    [128, 0,   0  ],   # 1 Bareland        (800000)
    [0,   255, 36 ],   # 2 Rangeland       (00FF24)
    [148, 148, 148],   # 3 Developed space (949494)
    [255, 255, 255],   # 4 Road            (FFFFFF)
    [34,  97,  38 ],   # 5 Tree            (226126)
    [0,   69,  255],   # 6 Water           (0045FF)
    [75,  181, 73 ],   # 7 Agriculture land(4BB549)
    [222, 31,  7  ],   # 8 Building        (DE1F07)
], dtype=np.uint8)

CLASS_NAMES = [
    'Background', 'Bareland', 'Rangeland', 'Developed',
    'Road', 'Tree', 'Water', 'Agriculture', 'Building',
]

# ImageNet stats (same as transforms.py) for denormalization
_MEAN = np.array([123.675, 116.28,  103.53], dtype=np.float32)
_STD  = np.array([ 58.395,  57.12,   57.375], dtype=np.float32)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) int label mask -> (H, W, 3) RGB using vectorized lookup."""
    idx = np.clip(mask, 0, len(OEM_PALETTE) - 1)
    return OEM_PALETTE[idx]


def denormalize(image: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalization: float (H,W,3) -> uint8 (H,W,3)."""
    return (image * _STD + _MEAN).clip(0, 255).astype(np.uint8)


def save_visualization(image: np.ndarray, gt_mask: np.ndarray,
                       pred_mask: np.ndarray, save_path: str,
                       denormalize_img: bool = True,
                       title: str = ''):
    """Save a 3-column figure: Original | Ground Truth | Prediction.

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

    gt_rgb   = mask_to_rgb(gt_mask)
    pred_rgb = mask_to_rgb(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image);    axes[0].set_title('Original',      fontsize=12)
    axes[1].imshow(gt_rgb);   axes[1].set_title('Ground Truth',  fontsize=12)
    axes[2].imshow(pred_rgb); axes[2].set_title('Prediction',    fontsize=12)
    for ax in axes:
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)

    patches = [
        mpatches.Patch(color=OEM_PALETTE[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(len(CLASS_NAMES))
    ]
    fig.legend(handles=patches, loc='lower center', ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
