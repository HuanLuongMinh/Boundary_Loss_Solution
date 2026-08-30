"""
Training script — "+Static Boundary" (Run 3 của Bảng 1, docs/idea_research.md):
L_total = L_region + alpha * (lambda1 * L_BCE_edge + lambda2 * L_Affinity),
lambda1 = lambda2 = 1 CỐ ĐỊNH (static — dynamic lambda1(t)/lambda2(t) là Run 4
"Ours", NGOÀI PHẠM VI file này). L_region = CombinedLoss (CE+Dice, y hệt
baseline/Run 2). L_BCE_edge = BalancedBCEEdgeLoss (src/losses/boundary_bce.py,
y hệt Run 2). L_Affinity = AffinityLoss (src/losses/affinity.py, mới cho Run
3). Cả 3 thành phần được gộp qua StaticBoundaryTotalLoss
(src/losses/total_loss.py).

File này ĐỘC LẬP HOÀN TOÀN với src/train_unet_former_resnet18.py (baseline) và
src/train_bce_edge.py (Run 2) — KHÔNG import 2 file đó, KHÔNG sửa 2 file đó,
cả hai vẫn chạy lại y hệt bất kỳ lúc nào. Chỉ dùng chung (import, không sửa):
src/data (dataset/transforms), src/utils (losses/metrics/callbacks/
visualizer/boundary_visualizer/boundary_metrics), src/models/
unet_former_resnet18.py (build_model, return_fused_feature=True),
src/losses/boundary_bce.py + src/losses/affinity.py + src/losses/total_loss.py
(riêng cho thực nghiệm này).

Khác biệt so với Run 2 (train_bce_edge.py):
    - Model: StaticBoundaryUNetFormer (wrapper cục bộ) = UNetFormer
      (build_model) + BoundaryHead trên Fused Feature -> (logits, edge_logits,
      fused_feature) — trả về CẢ fused_feature (Run 2 chỉ trả 2 giá trị) vì
      L_Affinity cần feature CHƯA upsample, đúng stride 4.
    - Loss: 1 instance StaticBoundaryTotalLoss duy nhất thay cho cặp
      seg_criterion/edge_criterion riêng lẻ — alpha đọc từ
      BOUNDARY_LOSS.ALPHA (mặc định 0.4, có thể ghi đè qua --alpha để
      ablation nhanh — WORK_DIR tự thêm hậu tố _alphaX.XX khi đó, mirror đúng
      cách --lambda-edge của Run 2).
    - pos_weight (balanced BCE, KHÔNG dùng cho L_Affinity): logic tính/resume
      giữ NGUYÊN y hệt Run 2 (EdgeStatsAccumulator/compute_pos_weight/resume
      từ checkpoint).
    - validate(): tích hợp thêm BoundaryMetrics (src/utils/boundary_metrics.py,
      đã có sẵn từ nhiệm vụ trước) song song SegmentationMetrics — Run 3 log
      Boundary IoU@1/@2/@4 + BF-Score + ASD THẬT ngay từ đầu, không để N/A như
      Run 2. BoundaryMetrics không có sẵn cơ chế all-reduce DDP (khác
      SegmentationMetrics/EdgeStatsAccumulator) nên đã bổ sung THÊM (không đổi
      hành vi cũ) 2 method counts_tensor()/load_counts_tensor() vào
      BoundaryMetrics, mirror đúng pattern EdgeStatsAccumulator, để tổng hợp
      accumulator đúng qua các rank trước khi compute().
    - summary.txt: điền SỐ THẬT (không phải hậu kỳ patch) vào đúng 4 nhãn dòng
      Run 2 đã có (Best validation BFScore / Best Boundary IoU @1/@2/@4,
      dùng lại đúng style format của Tools/patch_bce_edge_summary.py), thêm 1
      dòng mới "Best validation ASD (pixel)" (chưa có nhãn tương ứng ở Run 2).
      In đủ cả 4 thành phần loss (l_region/l_bce/l_affinity/l_total).
    - Tiêu chí chọn best checkpoint / early stopping GIỮ NGUYÊN mIoU 9-class
      (đúng convention baseline/Run 2) để so sánh apples-to-apples.

Usage (Kaggle, 2x T4):
    torchrun --nproc_per_node=2 src/train_static_boundary.py --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml
    torchrun --nproc_per_node=2 src/train_static_boundary.py --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml --dry-run
    torchrun --nproc_per_node=2 src/train_static_boundary.py --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml --resume <work_dir>/latest_checkpoint.pth
    torchrun --nproc_per_node=2 src/train_static_boundary.py --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml --alpha 0.2   # ablation
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# ── Path so `src.*` imports work regardless of cwd ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet_former_resnet18 import build_model
from src.utils.callbacks import EarlyStopping
from src.utils.metrics import SegmentationMetrics
from src.utils.boundary_metrics import BoundaryMetrics
from src.utils.visualizer import save_visualization, denormalize
from src.utils.boundary_visualizer import save_boundary_visualization
from src.losses.boundary_bce import (
    BoundaryHead, EdgeStatsAccumulator, extract_edge_gt, compute_pos_weight,
)
from src.losses.total_loss import StaticBoundaryTotalLoss

_CLASS_NAMES = ['Background', 'Bareland', 'Rangeland', 'Developed', 'Road',
                'Tree', 'Water', 'Agriculture', 'Building']
_TRAIN_CROP_SIZE = 512  # khớp src/data/transforms.py::get_train_transforms() mặc định


# ── Helpers (duplicated on purpose từ train_bce_edge.py — mỗi script
#    train_*.py trong project này tự chứa, không phụ thuộc lẫn nhau) ──────────

def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def find_data_base(data_root: str) -> str:
    candidates = [
        data_root,
        os.path.join(data_root, 'OpenEarthMap_Mini'),
        os.path.join(data_root, 'OpenEarthMap_flat'),
        os.path.join(data_root, 'OpenEarthMap'),
        os.path.join(data_root, 'openearthmap'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'images', 'train')) or \
           os.path.isdir(os.path.join(c, 'images', 'val')):
            return c
    return data_root


def find_label_subdir(base: str) -> str:
    for name in ('labels', 'label'):
        if os.path.isdir(os.path.join(base, name)):
            return name
    return 'labels'


def resolve_dataset_paths(ds: dict) -> None:
    root_base = find_data_base(ds['ROOT_DIR'])
    if root_base != ds['ROOT_DIR']:
        log(f"Resolved ROOT_DIR: {ds['ROOT_DIR']} -> {root_base}")
        ds['ROOT_DIR'] = root_base

    val_root_in = ds.get('VAL_ROOT_DIR', ds['ROOT_DIR'])
    val_root_base = find_data_base(val_root_in)
    if val_root_base != val_root_in:
        log(f"Resolved VAL_ROOT_DIR: {val_root_in} -> {val_root_base}")
    ds['VAL_ROOT_DIR'] = val_root_base

    label_sub = find_label_subdir(root_base)
    for key, root in (('TRAIN_MASK_DIR', root_base), ('VAL_MASK_DIR', val_root_base)):
        cur = ds[key]
        head, _, tail = cur.partition('/')
        if head in ('labels', 'label') and head != label_sub:
            fixed = label_sub + '/' + tail if tail else label_sub
            log(f"Resolved {key}: '{cur}' -> '{fixed}' (detected '{label_sub}/' under {root})")
            ds[key] = fixed


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def poly_lr(base_lr: float, cur_iter: int, max_iters: int,
            warmup_iters: int = 500, warmup_ratio: float = 1e-6,
            power: float = 0.9, min_lr: float = 0.0) -> float:
    if cur_iter < warmup_iters:
        k = (1 - warmup_ratio) / warmup_iters
        return base_lr * (warmup_ratio + k * cur_iter)
    progress = (cur_iter - warmup_iters) / max(max_iters - warmup_iters, 1)
    scale = (1 - progress) ** power
    return max(min_lr, base_lr * scale)


def set_lr(optimizer, lr: float):
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def is_main() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


_log_file = None  # set by setup_log_file(); rank 0 only


def setup_log_file(path: str):
    global _log_file
    if not is_main():
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _log_file = open(path, 'a', encoding='utf-8')
    _log_file.write(f"\n{'=' * 70}\nRun started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 70}\n")
    _log_file.flush()


def log(msg: str):
    if not is_main():
        return
    print(msg, flush=True)
    if _log_file is not None:
        ts = datetime.now().strftime('%H:%M:%S')
        _log_file.write(f'[{ts}] {msg}\n')
        _log_file.flush()


def load_checkpoint_file(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f'--resume checkpoint not found: {path}')
    return torch.load(path, map_location='cpu', weights_only=False)


def save_checkpoint(path: str, model, optimizer, scaler, iteration: int,
                     early_stopping, best_miou: float, best_per_class,
                     best_iteration: int = 0, best_val_round: int = 0,
                     edge_stats: dict = None, edge_stats_source: str = '',
                     alpha: float = 0.4):
    ckpt = {
        'iteration':            iteration,
        'model_state_dict':     model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':    scaler.state_dict(),
        'early_stopping': {
            'prev':      early_stopping.prev,
            'counter':   early_stopping.counter,
            'triggered': early_stopping.triggered,
        },
        'best_miou':          best_miou,
        'best_per_class':      best_per_class,
        'best_iteration':      best_iteration,
        'best_val_round':      best_val_round,
        # Lưu lại pos_weight/r_edge đã đo để RESUME dùng lại đúng giá trị cũ
        # thay vì đo lại (đo lại có thể ra pos_weight hơi khác vì train_iter
        # đã lệch vị trí so với lần chạy gốc).
        'edge_stats':          edge_stats,
        'edge_stats_source':   edge_stats_source,
        'alpha':               alpha,
    }
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


# ── Model: UNetFormer (không sửa) + BoundaryHead phụ ────────────────────────

class StaticBoundaryUNetFormer(nn.Module):
    """Wrapper cục bộ: UNetFormer (build_model(cfg), model gốc KHÔNG bị sửa)
    + BoundaryHead (src/losses/boundary_bce.py) gắn trên Fused Feature.
    forward() LUÔN gọi cả 2 nhánh — bắt buộc, để an toàn với
    DDP(find_unused_parameters=False) (không nhánh nào được phép bị skip).
    Trả về CẢ 3 giá trị (logits, edge_logits, fused_feature): fused_feature
    (đã có sẵn từ self.base(..., return_fused_feature=True)) cần thiết cho
    L_Affinity, vốn hoạt động TRỰC TIẾP ở stride 4 (KHÔNG dùng edge_logits đã
    upsample về input size)."""

    def __init__(self, cfg: dict):
        super().__init__()
        self.base = build_model(cfg)
        decode_channels = cfg['MODEL'].get('DECODE_CHANNELS', 64)
        self.boundary_head = BoundaryHead(in_channels=decode_channels)

    def forward(self, x):
        logits, fused_feature = self.base(x, return_fused_feature=True)
        edge_logits = self.boundary_head(fused_feature, x.shape[2:])
        return logits, edge_logits, fused_feature


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, loss_fn: StaticBoundaryTotalLoss, pos_weight: float,
             num_classes: int, device: torch.device, amp_enabled: bool,
             boundary_distances=(1, 2, 4), bf_tolerance: int = 2,
             connectivity: int = 4, dilation_radius: int = 0,
             ignore_index: int = 255) -> dict:
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    boundary_metrics = BoundaryMetrics(
        num_classes=num_classes, ignore_index=ignore_index,
        boundary_distances=boundary_distances, bf_tolerance=bf_tolerance,
        connectivity=connectivity, dilation_radius=dilation_radius,
    )
    total_l_region = total_l_bce = total_l_affinity = total_l_total = 0.0
    n_batches = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits, edge_logits, fused_feature = model(images)
            _l_total, parts = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight)

        total_l_region   += parts['l_region']
        total_l_bce      += parts['l_bce']
        total_l_affinity += parts['l_affinity']
        total_l_total    += parts['l_total']
        n_batches += 1
        metrics.update(logits, masks)
        boundary_metrics.update(logits, masks)

    confusion_t = torch.from_numpy(metrics.confusion).to(device)
    dist.all_reduce(confusion_t, op=dist.ReduceOp.SUM)
    metrics.confusion = confusion_t.cpu().numpy()

    # BoundaryMetrics: mỗi rank chỉ tích luỹ trên phần val set của riêng mình
    # (qua DistributedSampler) — all-reduce accumulator thô trước compute(),
    # giống hệt cách confusion matrix ở trên được all-reduce trước
    # metrics.compute().
    boundary_counts = boundary_metrics.counts_tensor(device)
    dist.all_reduce(boundary_counts, op=dist.ReduceOp.SUM)
    boundary_metrics.load_counts_tensor(boundary_counts)

    loss_t = torch.tensor(
        [total_l_region, total_l_bce, total_l_affinity, total_l_total, float(n_batches)],
        device=device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
    n = max(loss_t[4].item(), 1)

    result = metrics.compute()
    result['l_region']   = (loss_t[0] / n).item()
    result['l_bce']      = (loss_t[1] / n).item()
    result['l_affinity'] = (loss_t[2] / n).item()
    result['l_total']    = (loss_t[3] / n).item()
    result['val_loss']   = result['l_total']  # tương thích ngược với cột val_loss của baseline/Run 2

    result['boundary'] = boundary_metrics.compute()
    return result


# ── Visualise a few samples (tái dùng nguyên save_visualization /
#    save_boundary_visualization — chỉ đổi chỗ gọi để unpack 3 giá trị) ──────

@torch.no_grad()
def visualise_samples(model, loader, work_dir: str, iteration: int,
                      device: torch.device, amp_enabled: bool,
                      miou: float = 0.0, n: int = 4):
    model.eval()
    count = 0
    title = f'Iter {iteration:,}  |  mIoU = {miou:.4f}'
    for images, masks in loader:
        for i in range(images.size(0)):
            if count >= n:
                return
            img = images[i].to(device).unsqueeze(0)
            with autocast(enabled=amp_enabled):
                logit, _edge_logit, _fused_feature = model(img)  # edge_logits/fused_feature không dùng ở 2 visualizer này
            pred   = logit.argmax(dim=1).squeeze(0).cpu().numpy()
            gt     = masks[i].cpu().numpy()
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            save_path = os.path.join(work_dir, 'vis', f'iter{iteration:06d}_s{count}.png')
            save_visualization(img_np, gt, pred, save_path,
                               denormalize_img=True, title=title)
            boundary_path = os.path.join(work_dir, 'vis', 'boundary', f'iter{iteration:06d}_s{count}.png')
            save_boundary_visualization(img_np, gt, pred, boundary_path,
                                        denormalize_img=True, title=title)
            count += 1


# ── Sanity check #6: overlay edge_gt (trích từ GT mask thật) lên ảnh gốc ────

def save_edge_gt_overlay_images(loader, work_dir: str, connectivity: int,
                                 dilation_radius: int, ignore_index: int, n: int = 4):
    """Diagnostic thuần định tính (docs/workflow_2.md mục 3.1, sanity check
    #6): lưu WORK_DIR/sanity/edge_gt_overlay_sK.png để tự kiểm tra bằng mắt
    edge_gt có trùng khớp ranh giới đối tượng thật hay không. File MỚI, không
    sửa src/utils/boundary_visualizer.py."""
    sanity_dir = os.path.join(work_dir, 'sanity')
    os.makedirs(sanity_dir, exist_ok=True)
    count = 0
    for images, masks in loader:
        for i in range(images.size(0)):
            if count >= n:
                return
            img_np = images[i].permute(1, 2, 0).cpu().numpy()
            img_u8 = denormalize(img_np) if img_np.dtype != np.uint8 else img_np
            edge_gt, _ = extract_edge_gt(masks[i:i + 1], ignore_index, connectivity, dilation_radius)
            edge_np = edge_gt[0, 0].cpu().numpy().astype(bool)

            overlay = img_u8.copy()
            overlay[edge_np] = [0, 255, 255]  # cyan

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_u8);  axes[0].set_title('Original', fontsize=11)
            axes[1].imshow(overlay); axes[1].set_title('edge_gt overlay (cyan)', fontsize=11)
            for ax in axes:
                ax.axis('off')
            fig.tight_layout()
            save_path = os.path.join(sanity_dir, f'edge_gt_overlay_s{count}.png')
            fig.savefig(save_path, dpi=110, bbox_inches='tight')
            plt.close(fig)
            count += 1
        if count >= n:
            return


# ── summary.txt (mirror cấu trúc write_bce_edge_summary() của Run 2, điền
#    THẬT các dòng boundary-metric mà Run 2 để N/A) ─────────────────────────

def write_static_boundary_summary(path: str, *, experiment_name: str, alpha: float,
                                   affinity_window_size: int, affinity_distance: str,
                                   affinity_margin: float,
                                   best_iteration: int, best_val_round: int, total_val_rounds: int,
                                   best_miou: float, best_per_class, best_boundary: dict,
                                   final_l_region: float, final_l_bce: float,
                                   final_l_affinity: float, final_l_total: float,
                                   hours: int, minutes: int, seconds: int,
                                   num_classes: int, ignore_index: int,
                                   connectivity: int, dilation_radius: int, seed: int,
                                   edge_stats: dict, edge_stats_source: str,
                                   pos_weight_max: float, sanity: dict):
    miou8 = float(np.mean(best_per_class[1:])) if best_per_class is not None else None
    edge_width = 1 if dilation_radius == 0 else 2 * dilation_radius + 1

    def fmt(value) -> str:
        # Cùng convention Tools/patch_bce_edge_summary.py::format_value() — NaN
        # (vd. không có validation nào hoàn thành, hoặc val set không có pixel
        # biên hợp lệ) -> chuỗi N/A rõ ràng thay vì in "nan".
        if value is None or value != value:
            return 'N/A (khong co pixel bien hop le nao trong val set)'
        return f'{value:.4f}'

    def pf(ok: bool) -> str:
        return 'passed' if ok else 'FAILED'

    lines = []
    lines.append('=' * 64)
    lines.append(' EDGE + AFFINITY SUPERVISION STATISTICS')
    lines.append('=' * 64)
    lines.append('')
    lines.append(f'Experiment name            : {experiment_name}')
    lines.append('Dataset                    : OpenEarthMap (OEM)')
    lines.append('Task                       : 8-class land-cover semantic segmentation '
                 '(+ Background, 9 lop trong pipeline nay)')
    lines.append('Model                      : UNetFormer')
    lines.append('Backbone                   : ResNet-18')
    lines.append('Region loss                : CombinedLoss (CrossEntropy + Dice)')
    lines.append('Edge loss                  : BCEWithLogitsLoss (class-balanced, pos_weight)')
    lines.append('Affinity loss              : contrastive feature-distance gan bien '
                 f'(window={affinity_window_size}, distance={affinity_distance}, margin={affinity_margin})')
    lines.append('Total loss                 : L_total = L_region + alpha * (L_BCE_edge + L_Affinity)  [lambda1=lambda2=1]')
    lines.append(f'alpha                      : {alpha:.4f}')
    lines.append('')
    lines.append('=' * 64)
    lines.append(' FINAL EVALUATION -- BEST CHECKPOINT')
    lines.append('=' * 64)
    lines.append('')
    lines.append(f'Best iteration             : {best_iteration}')
    lines.append(f'Best val round              : {best_val_round}/{total_val_rounds}')
    lines.append('Selection metric            : validation mIoU (9-class, gom Background -- dung quy uoc '
                 'baseline hien co)')
    lines.append(f'Best validation mIoU-9      : {best_miou:.4f}')
    lines.append(f'Best validation mIoU-8      : {miou8:.4f}  (Background excluded, tinh hau ky)'
                 if miou8 is not None else 'Best validation mIoU-8      : N/A (khong co validation nao hoan thanh)')
    lines.append(f'Best validation BFScore     : {fmt(best_boundary.get("bf_score") if best_boundary else None)}')
    lines.append(f'Best Boundary IoU @1        : {fmt(best_boundary.get("boundary_iou_d1") if best_boundary else None)}')
    lines.append(f'Best Boundary IoU @2        : {fmt(best_boundary.get("boundary_iou_d2") if best_boundary else None)}')
    lines.append(f'Best Boundary IoU @4        : {fmt(best_boundary.get("boundary_iou_d4") if best_boundary else None)}')
    lines.append(f'Best validation ASD (pixel) : {fmt(best_boundary.get("asd") if best_boundary else None)}')
    lines.append('')
    lines.append('Per-class IoU (9 lop, gom Background):')
    if best_per_class is not None:
        for name, iou_v in zip(_CLASS_NAMES, best_per_class):
            lines.append(f'  {name:<12}: {iou_v:.4f}')
    else:
        lines.append('  (no validation completed)')
    lines.append('')
    lines.append(f'Final validation L_region   : {final_l_region:.4f}')
    lines.append(f'Final validation L_bce      : {final_l_bce:.4f}')
    lines.append(f'Final validation L_affinity : {final_l_affinity:.4f}')
    lines.append(f'Final validation L_total    : {final_l_total:.4f}')
    lines.append(f'Total training time         : {hours:02d}h {minutes:02d}m {seconds:02d}s')
    lines.append('-' * 64)
    lines.append(' LABEL AND VALID-PIXEL POLICY')
    lines.append('-' * 64)
    lines.append('')
    lines.append(f'Number of OEM classes       : {num_classes} (Background + 8 land-cover)')
    lines.append('Class IDs used for mIoU-9   : 0-8 (toan bo)')
    lines.append('Class IDs used for mIoU-8   : 1-8 (loai Background)')
    lines.append('Background handling         : supervised class (existing baseline convention) -- '
                 'KHONG map sang ignore_index')
    lines.append(f'Ignore index                : {ignore_index}  (hien la no-op -- dataset khong sinh pixel gia tri nay)')
    lines.append('Ignore pixels in edge loss  : excluded (qua valid_mask trong extract_edge_gt)')
    lines.append('Ignore pixels in affinity   : excluded (ca pixel trung tam va hang xom, qua valid_mask)')
    lines.append('Ignore pixels in mIoU       : excluded (qua ignore_index trong SegmentationMetrics)')
    lines.append('mIoU reporting convention   : ghi ca mIoU-9 (dung chon best checkpoint) va mIoU-8 (chi bao cao)')
    lines.append('-' * 64)
    lines.append(' EDGE-TARGET DEFINITION (dung chung cho L_BCE_edge, L_Affinity, BoundaryMetrics)')
    lines.append('-' * 64)
    lines.append('')
    lines.append('Edge source                 : semantic ground-truth mask (post-transform), '
                 'downsample NEAREST ve stride-4 cho L_Affinity')
    lines.append(f'Neighborhood connectivity   : {connectivity}-connected')
    lines.append('Edge criterion              : center valid pixel co it nhat 1 hang-xom valid mang nhan khac')
    lines.append(f'Edge dilation radius        : {dilation_radius} pixels')
    lines.append(f'Final edge-target width     : {edge_width} pixel(s)')
    lines.append('Edge target values          : binary {0, 1}')
    lines.append(f'Crop size used for stats    : {_TRAIN_CROP_SIZE} x {_TRAIN_CROP_SIZE}')
    lines.append(f'Statistics sampling         : {edge_stats_source}')
    lines.append(f'Random seed for statistics  : {seed}')
    lines.append('-' * 64)
    lines.append(' DATASET-LEVEL EDGE STATISTICS')
    lines.append('-' * 64)
    lines.append('')
    lines.append(f'Number of analyzed crops    : {edge_stats.get("n_crops_analyzed", "N/A")}')
    lines.append(f'Number of valid pixels      : {edge_stats["N_valid"]:.0f}')
    lines.append(f'Number of edge pixels       : {edge_stats["N_edge"]:.0f}')
    lines.append(f'Number of non-edge pixels   : {edge_stats["N_nonedge"]:.0f}')
    lines.append(f'Edge pixel ratio            : {edge_stats["r_edge"]:.6f}')
    lines.append(f'Non-edge pixel ratio        : {edge_stats["r_nonedge"]:.6f}')
    lines.append('')
    lines.append('Formula:')
    lines.append('  r_edge = N_edge / max(N_valid, 1)')
    lines.append('  N_nonedge = N_valid - N_edge')
    lines.append('-' * 64)
    lines.append(' BCE IMBALANCE CORRECTION (chi ap dung cho L_BCE_edge, khong ap dung cho L_Affinity)')
    lines.append('-' * 64)
    lines.append('')
    lines.append('pos_weight formula          : N_nonedge / max(N_edge, 1)')
    lines.append(f'pos_weight raw              : {edge_stats["pos_weight_raw"]:.4f}')
    lines.append('pos_weight clipping         : enabled')
    lines.append(f'pos_weight clip range       : [1.0, {pos_weight_max}]')
    lines.append(f'pos_weight used             : {edge_stats["pos_weight"]:.4f}')
    lines.append("BCE reduction               : none -> valid-mask multiplication -> sum / valid_mask.sum()")
    lines.append('Logit handling              : raw edge logits; no manual sigmoid')
    lines.append(f'NaN / Inf check             : {pf(sanity.get("pos_weight_finite", False))}')
    lines.append('-' * 64)
    lines.append(' AFFINITY LOSS CONFIGURATION')
    lines.append('-' * 64)
    lines.append('')
    lines.append(f'Feature source              : Fused Feature (decoder, C=64, stride 4, truoc upsample)')
    lines.append(f'Neighborhood window (KxK)   : {affinity_window_size} x {affinity_window_size}')
    lines.append(f'Distance metric             : {affinity_distance}')
    lines.append(f'Margin                      : {affinity_margin}')
    lines.append('Sampling region              : chi cac pixel trung tam trong vung gan bien (extract_edge_gt)')
    lines.append('lambda1 (L_BCE_edge weight) : 1.0 (static, Run 3)')
    lines.append('lambda2 (L_Affinity weight) : 1.0 (static, Run 3)')
    lines.append('-' * 64)
    lines.append(' PRE-FLIGHT VALIDATION')
    lines.append('-' * 64)
    lines.append('')
    lines.append(f'Edge target binary check    : {pf(sanity.get("edge_target_binary_check", False))}')
    lines.append(f'Ignore-pixel mask check     : {pf(sanity.get("ignore_pixel_mask_check", False))}')
    lines.append(f'Edge overlay visual check   : {pf(sanity.get("edge_overlay_visual_check", False))}')
    lines.append(f'Edge-head gradient check    : {pf(sanity.get("edge_head_gradient_check", False))}')
    lines.append(f'Affinity gradient check     : {pf(sanity.get("affinity_gradient_check", False))}')
    lines.append(f'Initial L_region finite     : {pf(sanity.get("initial_l_region_finite", False))}')
    lines.append(f'Initial L_bce finite        : {pf(sanity.get("initial_l_bce_finite", False))}')
    lines.append(f'Initial L_affinity finite   : {pf(sanity.get("initial_l_affinity_finite", False))}')
    lines.append(f'Initial L_total finite      : {pf(sanity.get("initial_l_total_finite", False))}')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  required=True, help='Path to YAML config')
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick smoke-test: 5 iters, 2 val steps, ~4 samples/rank')
    parser.add_argument('--resume', default=None,
                        help='Path to latest_checkpoint.pth to resume training from')
    parser.add_argument('--data-root', default=None,
                        help='Override DATASET.ROOT_DIR/VAL_ROOT_DIR from the config')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Ghi de BOUNDARY_LOSS.ALPHA cua config (vd 0.2) de ablation nhanh — '
                             'khi truyen, WORK_DIR tu them hau to _alphaX.XX de khong de len run mac dinh.')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend='nccl' if use_cuda else 'gloo')
    train_start    = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    local_rank = int(os.environ['LOCAL_RANK'])
    if use_cuda:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    amp_enabled = use_cuda
    world_size = dist.get_world_size()

    cfg = load_config(args.config)
    ds  = cfg['DATASET']
    mdl = cfg['MODEL']
    tr  = cfg['TRAIN']
    opt = cfg['OPTIMIZER']
    out = cfg['OUTPUT']
    bl  = cfg.setdefault('BOUNDARY_LOSS', {})

    # ── alpha override + WORK_DIR suffix (ablation, xem README) ─────────────
    alpha = bl.get('ALPHA', 0.4)
    if args.alpha is not None:
        alpha = args.alpha
        bl['ALPHA'] = alpha
        out['WORK_DIR'] = out['WORK_DIR'].rstrip('/') + f'_alpha{alpha:.2f}'

    ignore_index      = bl.get('IGNORE_INDEX', 255)
    connectivity      = bl.get('CONNECTIVITY', 4)
    dilation_radius   = bl.get('DILATION_RADIUS', 0)
    pos_weight_max    = bl.get('POS_WEIGHT_MAX', 20.0)
    edge_stats_file   = bl.get('EDGE_STATS_FILE') or None
    edge_stats_nbatch = bl.get('EDGE_STATS_NUM_BATCHES', 100)
    affinity_window_size = bl.get('AFFINITY_WINDOW_K', 5)
    affinity_distance    = bl.get('AFFINITY_DISTANCE', 'cosine')
    affinity_margin      = bl.get('AFFINITY_MARGIN', 1.0)

    os.makedirs(out['WORK_DIR'], exist_ok=True)
    setup_log_file(os.path.join(out['WORK_DIR'], 'train_log.txt'))
    if not use_cuda:
        log("WARNING: CUDA not available — running on CPU with backend='gloo', AMP disabled. "
            "This is only intended for local dry-run/dev; real training must run on Kaggle GPUs.")

    experiment_name = os.path.basename(out['WORK_DIR'].rstrip('/'))
    log(f"Experiment: +Static Boundary — {experiment_name}  (alpha={alpha}, "
        f"affinity_window={affinity_window_size}, affinity_distance={affinity_distance})")

    # Auto-resume
    if args.resume is None:
        auto_ckpt = os.path.join(out['WORK_DIR'], 'latest_checkpoint.pth')
        if os.path.exists(auto_ckpt):
            args.resume = auto_ckpt
            log(f"Auto-resume: found existing checkpoint at {auto_ckpt} — resuming from it.")
    resume_ckpt = load_checkpoint_file(args.resume) if args.resume else None

    seed = tr.get('SEED', 19)
    set_seed(seed)
    log(f"Using seed={seed}")

    if args.dry_run:
        tr['MAX_ITERS']    = 5
        tr['VAL_INTERVAL'] = 2
        edge_stats_nbatch  = min(edge_stats_nbatch, 2)

    if args.data_root:
        ds['ROOT_DIR'] = args.data_root
        ds['VAL_ROOT_DIR'] = args.data_root
        log(f"--data-root override: ROOT_DIR/VAL_ROOT_DIR = {args.data_root}")

    resolve_dataset_paths(ds)

    # ── Datasets ─────────────────────────────────────────────────────────────
    tsf = ds.get('TRAIN_SPLIT_FILE')
    train_split = (tsf if os.path.isabs(tsf) else os.path.join(ds['ROOT_DIR'], tsf)) if tsf else None
    val_split   = ds.get('VAL_SPLIT_FILE')

    train_ds = OpenEarthMapDataset(
        root_dir=ds['ROOT_DIR'], img_dir=ds['TRAIN_IMG_DIR'],
        mask_dir=ds['TRAIN_MASK_DIR'], split_file=train_split,
        transform=get_train_transforms(),
    )
    val_root = ds.get('VAL_ROOT_DIR', ds['ROOT_DIR'])
    val_ds = OpenEarthMapDataset(
        root_dir=val_root, img_dir=ds['VAL_IMG_DIR'],
        mask_dir=ds['VAL_MASK_DIR'], split_file=val_split,
        transform=get_val_transforms(),
    )
    log(f"Dataset: {len(train_ds)} train images, {len(val_ds)} val images "
        f"(train_dir={ds['TRAIN_IMG_DIR']}, val_dir={ds['VAL_IMG_DIR']})")

    if args.dry_run:
        from torch.utils.data import Subset
        n_train = max(4, tr['BATCH_SIZE_PER_GPU'] * world_size)
        train_ds = Subset(train_ds, list(range(min(n_train, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(4, len(val_ds)))))

    train_sampler = DistributedSampler(train_ds, shuffle=True, seed=seed)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)

    train_loader = DataLoader(
        train_ds, batch_size=tr['BATCH_SIZE_PER_GPU'],
        sampler=train_sampler, num_workers=tr['NUM_WORKERS'],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=tr['BATCH_SIZE_PER_GPU'],
        sampler=val_sampler, num_workers=tr['NUM_WORKERS'],
        pin_memory=True, drop_last=False,
    )

    def cycle(loader):
        epoch = 0
        while True:
            loader.sampler.set_epoch(epoch)
            epoch += 1
            yield from loader

    train_iter = cycle(train_loader)

    # ── Model ────────────────────────────────────────────────────────────────
    model = StaticBoundaryUNetFormer(cfg).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state_dict'])
        log(f"Resumed model weights from {args.resume} (iteration {resume_ckpt['iteration']})")
    model = DDP(model, device_ids=[local_rank] if use_cuda else None,
                find_unused_parameters=False)

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_cfg = cfg.get('LOSS', {})
    ce_w   = loss_cfg.get('CE_WEIGHT',   1.0)
    dice_w = loss_cfg.get('DICE_WEIGHT', 1.0)
    loss_fn = StaticBoundaryTotalLoss(
        num_classes=tr['NUM_CLASSES'], alpha=alpha, ignore_index=ignore_index,
        ce_weight=ce_w, dice_weight=dice_w,
        connectivity=connectivity, dilation_radius=dilation_radius,
        affinity_window_size=affinity_window_size, affinity_distance=affinity_distance,
        affinity_margin=affinity_margin,
    ).to(device)
    log(f"Loss = CombinedLoss(CE+Dice, ce_weight={ce_w}, dice_weight={dice_w}) "
        f"+ {alpha} * (BalancedBCEEdgeLoss(connectivity={connectivity}, dilation_radius={dilation_radius}) "
        f"+ AffinityLoss(window={affinity_window_size}, distance={affinity_distance}, margin={affinity_margin}))  "
        f"[lambda1=lambda2=1]")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt['BASE_LR'], weight_decay=opt['WEIGHT_DECAY'],
    )
    scaler = GradScaler(enabled=amp_enabled)
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scaler.load_state_dict(resume_ckpt['scaler_state_dict'])

    # ── pos_weight resolution (1 lần lúc khởi động, KHÔNG tính lại mỗi iter) ─
    if resume_ckpt is not None and resume_ckpt.get('edge_stats') is not None:
        edge_stats = resume_ckpt['edge_stats']
        edge_stats_source = resume_ckpt.get('edge_stats_source', 'resumed from checkpoint')
        log(f"pos_weight resumed từ checkpoint: {edge_stats['pos_weight']:.3f} "
            f"(r_edge={edge_stats['r_edge']:.4f})")
    elif edge_stats_file and os.path.exists(edge_stats_file):
        with open(edge_stats_file, encoding='utf-8') as f:
            file_stats = json.load(f)
        n_edge, n_nonedge = file_stats['N_edge'], file_stats['N_nonedge']
        edge_stats = {
            'N_valid': file_stats['N_valid'], 'N_edge': n_edge, 'N_nonedge': n_nonedge,
            'r_edge': file_stats['r_edge'],
            'r_nonedge': file_stats.get('r_nonedge', 1.0 - file_stats['r_edge']),
            'pos_weight_raw': n_nonedge / max(n_edge, 1.0),
            'pos_weight': compute_pos_weight(n_edge, n_nonedge, pos_weight_max),
            'connectivity': connectivity, 'dilation_radius': dilation_radius,
            'n_crops_analyzed': file_stats.get('n_batches', 0) * file_stats.get('batch_size', 1),
        }
        edge_stats_source = (f"EDGE_STATS_FILE={edge_stats_file} "
                             f"(full epoch, do truoc boi Tools/measure_edge_ratio.py, seed={file_stats.get('seed')})")
        log(f"pos_weight từ EDGE_STATS_FILE: r_edge={edge_stats['r_edge']:.4f} "
            f"pos_weight={edge_stats['pos_weight']:.3f}")
    else:
        log(f"Không có EDGE_STATS_FILE hợp lệ — tự ước lượng pos_weight "
            f"(APPROXIMATE, bounded {edge_stats_nbatch} batch/rank x {world_size} rank)")
        acc = EdgeStatsAccumulator(ignore_index, connectivity, dilation_radius)
        for _ in range(edge_stats_nbatch):
            _, m = next(train_iter)
            acc.update(m)
        counts = acc.counts_tensor(device)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        acc.load_counts_tensor(counts)
        edge_stats = acc.result(pos_weight_max)
        edge_stats['n_crops_analyzed'] = edge_stats_nbatch * world_size * tr['BATCH_SIZE_PER_GPU']
        edge_stats_source = (f"bounded auto-estimate ({edge_stats_nbatch} batch/rank x {world_size} rank, "
                             f"APPROXIMATE -- khong phai full epoch)")
        log(f"pos_weight ước lượng: r_edge={edge_stats['r_edge']:.4f} pos_weight={edge_stats['pos_weight']:.3f} "
            f"(N_valid={edge_stats['N_valid']:.0f} N_edge={edge_stats['N_edge']:.0f})")

    pos_weight = edge_stats['pos_weight']

    # Gate cứng (mọi rank tự kiểm/raise — KHÔNG gate is_main() ở đây, để
    # tránh treo collective op phía sau nếu 1 rank raise mà rank khác không).
    if not (0.0 < edge_stats['r_edge'] < 1.0):
        raise RuntimeError(f"Sanity check thất bại: r_edge={edge_stats['r_edge']} không nằm trong (0,1) — "
                           f"kiểm tra lại ignore_index/connectivity hoặc dữ liệu train.")
    pos_weight_finite_ok = math.isfinite(pos_weight) and pos_weight > 0
    if not pos_weight_finite_ok:
        raise RuntimeError(f"Sanity check thất bại: pos_weight={pos_weight} không hữu hạn/dương.")
    log(f"[OK] r_edge trong (0,1) và pos_weight hữu hạn dương (pos_weight={pos_weight:.3f})")

    # ── Sanity checks (mirror docs/workflow_2.md mục 3.1, + 1 check mới cho
    #    affinity gradient) — bắt đầu ──────────────────────────────────────
    sanity = {'pos_weight_finite': pos_weight_finite_ok}

    # #2 Ignore-pixel mask check — synthetic, không cần forward qua model.
    synth_masks  = torch.full((2, 32, 32), ignore_index, dtype=torch.int64, device=device)
    synth_edge_logits = torch.randn(2, 1, 32, 32, device=device)
    synth_edge_loss = loss_fn.bce_edge(synth_edge_logits, synth_masks, pos_weight)
    ok2 = bool(torch.isfinite(synth_edge_loss)) and abs(synth_edge_loss.item()) < 1e-6
    sanity['ignore_pixel_mask_check'] = ok2
    log(f"Sanity #2 (batch toàn ignore -> L_bce=0): {'PASS' if ok2 else 'FAIL'} (loss={synth_edge_loss.item():.6g})")
    if not ok2:
        raise RuntimeError("Sanity check #2 (ignore-pixel mask) thất bại — L_bce trên batch toàn ignore phải =0.")

    # #6 Overlay trực quan (định tính, rank0-only, KHÔNG raise nếu lỗi — chỉ
    # log FAIL — không có collective op nào phụ thuộc block này nên an toàn).
    if is_main():
        try:
            save_edge_gt_overlay_images(val_loader, out['WORK_DIR'], connectivity,
                                        dilation_radius, ignore_index, n=4)
            sanity['edge_overlay_visual_check'] = True
            log(f"Sanity #6 (edge_gt overlay): đã lưu {out['WORK_DIR']}/sanity/ — tự kiểm tra bằng mắt.")
        except Exception as e:
            sanity['edge_overlay_visual_check'] = False
            log(f"Sanity #6 (edge_gt overlay): FAIL — {e}")
    else:
        sanity['edge_overlay_visual_check'] = True  # không đánh giá lại trên rank khác

    # ── Training state ───────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=tr['EARLY_STOPPING_PATIENCE'])
    if resume_ckpt is not None:
        early_stopping.prev      = resume_ckpt['early_stopping']['prev']
        early_stopping.counter   = resume_ckpt['early_stopping']['counter']
        early_stopping.triggered = resume_ckpt['early_stopping']['triggered']

    best_miou      = resume_ckpt['best_miou']      if resume_ckpt is not None else 0.0
    best_per_class = resume_ckpt['best_per_class'] if resume_ckpt is not None else None
    best_iteration = resume_ckpt.get('best_iteration', 0) if resume_ckpt is not None else 0
    best_val_round = resume_ckpt.get('best_val_round', 0) if resume_ckpt is not None else 0
    best_boundary  = None
    best_state_dict = None
    history         = []
    total_val_rounds = -(-tr['MAX_ITERS'] // tr['VAL_INTERVAL'])

    if resume_ckpt is not None and is_main():
        best_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
        if os.path.exists(best_path):
            best_state_dict = torch.load(best_path, map_location='cpu', weights_only=False)
            log(f"Reloaded best_model.pth (mIoU={best_miou:.4f}) for export continuity")
        elif best_miou > 0.0:
            log(f"WARNING: resumed with best_miou={best_miou:.4f} but {best_path} "
                f"was not found — best checkpoint weights are unavailable until a new best is found.")

        csv_path = os.path.join(out['WORK_DIR'], 'benchmark_results.csv')
        if os.path.exists(csv_path):
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    row['iter']      = int(row['iter'])
                    row['val_round'] = int(row['val_round']) if row.get('val_round') else row['iter'] // tr['VAL_INTERVAL']
                    row['mIoU']      = float(row['mIoU'])
                    row['val_loss']  = float(row['val_loss'])
                    for k in ('l_region', 'l_bce', 'l_affinity', 'l_total',
                              'bf_score', 'boundary_iou_d1', 'boundary_iou_d2', 'boundary_iou_d4', 'asd'):
                        if row.get(k):
                            row[k] = float(row[k])
                    history.append(row)
            log(f"Reconstructed {len(history)} history rows from {csv_path}")

    start_iter = resume_ckpt['iteration'] + 1 if resume_ckpt is not None else 1
    if args.dry_run and resume_ckpt is not None and start_iter > tr['MAX_ITERS']:
        log(f"Warning: --dry-run MAX_ITERS={tr['MAX_ITERS']} <= resumed "
            f"start_iter={start_iter}; loop will not execute.")

    # ── Training loop ────────────────────────────────────────────────────────
    log(f"Training {tr['MAX_ITERS']} iterations | encoder={mdl['ENCODER']} | "
        f"loss=CombinedLoss+{alpha}*(BalancedBCEEdgeLoss+AffinityLoss) | "
        f"train_dir={ds['ROOT_DIR']}/{ds['TRAIN_IMG_DIR']}")

    last_l_region = last_l_bce = last_l_affinity = last_l_total = 0.0

    for iteration in range(start_iter, tr['MAX_ITERS'] + 1):
        model.train()

        lr = poly_lr(
            base_lr=opt['BASE_LR'], cur_iter=iteration - 1, max_iters=tr['MAX_ITERS'],
            warmup_iters=tr.get('WARMUP_ITERS', 500),
        )
        set_lr(optimizer, lr)

        images, masks = next(train_iter)
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits, edge_logits, fused_feature = model(images)
            l_total, parts = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight)

        if iteration == 1:
            # #1 shape/binary edge_gt + finite-loss checks — MỌI rank tự kiểm
            # (không is_main-gate) để tránh treo collective backward() sau đây
            # nếu 1 rank raise mà rank khác thì không.
            with torch.no_grad():
                edge_gt_probe, _ = extract_edge_gt(masks, ignore_index, connectivity, dilation_radius)
            shape_ok = tuple(edge_gt_probe.shape) == (masks.shape[0], 1, masks.shape[1], masks.shape[2])
            binary_ok = bool(torch.all((edge_gt_probe == 0) | (edge_gt_probe == 1)))
            sanity['edge_target_binary_check'] = shape_ok and binary_ok
            log(f"Sanity #1 (edge_gt shape={tuple(edge_gt_probe.shape)}, binary={binary_ok}): "
                f"{'PASS' if sanity['edge_target_binary_check'] else 'FAIL'}")
            if not sanity['edge_target_binary_check']:
                raise RuntimeError("Sanity check #1 (shape/binary edge_gt) thất bại.")

            sanity['initial_l_region_finite']   = math.isfinite(parts['l_region'])
            sanity['initial_l_bce_finite']      = math.isfinite(parts['l_bce'])
            sanity['initial_l_affinity_finite'] = math.isfinite(parts['l_affinity'])
            sanity['initial_l_total_finite']    = math.isfinite(parts['l_total'])
            log(f"Sanity (initial losses): l_region={parts['l_region']:.4f} l_bce={parts['l_bce']:.4f} "
                f"l_affinity={parts['l_affinity']:.4f} l_total={parts['l_total']:.4f}")
            for k in ('initial_l_region_finite', 'initial_l_bce_finite',
                      'initial_l_affinity_finite', 'initial_l_total_finite'):
                if not sanity[k]:
                    raise RuntimeError(f"Sanity check '{k}' thất bại (giá trị không hữu hạn ngay iteration 1).")

        scaler.scale(l_total).backward()

        if iteration == 1:
            # #7 gradient khác 0 trên BoundaryHead (L_bce) VÀ trên decoder/frh
            # (L_affinity) — kiểm NGAY SAU backward() thật (không làm pass
            # thừa), mọi rank tự kiểm.
            grad_sum_edge = sum(p.grad.abs().sum().item()
                                for p in model.module.boundary_head.parameters() if p.grad is not None)
            sanity['edge_head_gradient_check'] = grad_sum_edge > 0
            log(f"Sanity #7 (BoundaryHead gradient nonzero, sum|grad|={grad_sum_edge:.6g}): "
                f"{'PASS' if sanity['edge_head_gradient_check'] else 'FAIL'}")
            if not sanity['edge_head_gradient_check']:
                raise RuntimeError("Sanity check #7 (BoundaryHead gradient) thất bại — "
                                   "không có gradient chảy tới boundary_head.")

            grad_sum_affinity = sum(p.grad.abs().sum().item()
                                    for p in model.module.base.frh.parameters() if p.grad is not None)
            sanity['affinity_gradient_check'] = grad_sum_affinity > 0
            log(f"Sanity #8 (decoder/frh gradient nonzero qua L_affinity, sum|grad|={grad_sum_affinity:.6g}): "
                f"{'PASS' if sanity['affinity_gradient_check'] else 'FAIL'}")
            if not sanity['affinity_gradient_check']:
                raise RuntimeError("Sanity check #8 (affinity gradient) thất bại — "
                                   "không có gradient chảy tới decoder qua L_affinity.")

            n_pass = sum(1 for v in sanity.values() if v)
            log(f"{len(sanity)} sanity checks: {n_pass}/{len(sanity)} PASS — bắt đầu training thật.")

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt['GRAD_CLIP'])
        scaler.step(optimizer)
        scaler.update()

        last_l_region, last_l_bce = parts['l_region'], parts['l_bce']
        last_l_affinity, last_l_total = parts['l_affinity'], parts['l_total']

        if iteration % 100 == 0:
            log(f"[{iteration:>6}/{tr['MAX_ITERS']}] l_region={parts['l_region']:.4f} l_bce={parts['l_bce']:.4f} "
                f"l_affinity={parts['l_affinity']:.4f} l_total={parts['l_total']:.4f}  lr={lr:.2e}")

        # ── Validation ───────────────────────────────────────────────────────
        if iteration % tr['VAL_INTERVAL'] == 0 or iteration == tr['MAX_ITERS']:
            val_round = iteration // tr['VAL_INTERVAL']
            val_result = validate(
                model, val_loader, loss_fn, pos_weight, tr['NUM_CLASSES'],
                device, amp_enabled=amp_enabled, boundary_distances=(1, 2, 4),
                connectivity=connectivity, dilation_radius=dilation_radius,
                ignore_index=ignore_index,
            )
            miou = val_result['mIoU']
            last_l_region, last_l_bce = val_result['l_region'], val_result['l_bce']
            last_l_affinity, last_l_total = val_result['l_affinity'], val_result['l_total']
            bnd = val_result['boundary']
            log(f"  [Val round {val_round}/{total_val_rounds}, iter {iteration}] mIoU={miou:.4f}  "
                f"l_region={val_result['l_region']:.4f}  l_bce={val_result['l_bce']:.4f}  "
                f"l_affinity={val_result['l_affinity']:.4f}  l_total={val_result['l_total']:.4f}")
            log(f"    BFScore={bnd['bf_score']:.4f}  BIoU@1={bnd['boundary_iou_d1']:.4f}  "
                f"BIoU@2={bnd['boundary_iou_d2']:.4f}  BIoU@4={bnd['boundary_iou_d4']:.4f}  "
                f"ASD={bnd['asd']:.4f}")

            stop_flag = False
            if is_main():
                per_class = val_result['per_class_iou']
                _abbr = ['Bg', 'Bare', 'Range', 'Dev', 'Road', 'Tree', 'Water', 'Agri', 'Bldg']

                is_new_best = miou > best_miou
                if is_new_best:
                    best_miou       = miou
                    best_per_class  = list(per_class)
                    best_iteration  = iteration
                    best_val_round  = val_round
                    best_boundary   = dict(bnd)
                    best_state_dict = {k: v.cpu().clone() for k, v in model.module.state_dict().items()}
                    best_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
                    torch.save(best_state_dict, best_path)
                    log(f"  ✔ New best mIoU={best_miou:.4f} "
                        f"(val round {val_round}/{total_val_rounds}, iter {iteration}) → saved {best_path}")

                row = {
                    'iter':      iteration,
                    'val_round': val_round,
                    'mIoU':      val_result['mIoU'],
                    'val_loss':  val_result['val_loss'],
                    'l_region':  round(val_result['l_region'], 4),
                    'l_bce':     round(val_result['l_bce'], 4),
                    'l_affinity': round(val_result['l_affinity'], 4),
                    'l_total':   round(val_result['l_total'], 4),
                    'bf_score':        round(bnd['bf_score'], 4),
                    'boundary_iou_d1': round(bnd['boundary_iou_d1'], 4) if bnd['boundary_iou_d1'] == bnd['boundary_iou_d1'] else '',
                    'boundary_iou_d2': round(bnd['boundary_iou_d2'], 4) if bnd['boundary_iou_d2'] == bnd['boundary_iou_d2'] else '',
                    'boundary_iou_d4': round(bnd['boundary_iou_d4'], 4) if bnd['boundary_iou_d4'] == bnd['boundary_iou_d4'] else '',
                    'asd':             round(bnd['asd'], 4) if bnd['asd'] == bnd['asd'] else '',
                    **{f'iou_{n}': round(v, 4) for n, v in zip(_CLASS_NAMES, per_class)},
                    'is_best':   is_new_best,
                }
                history.append(row)
                iou_str = '  '.join(f'{a}:{v:.3f}' for a, v in zip(_abbr, per_class))
                log(f"    {iou_str}")

                csv_path = os.path.join(out['WORK_DIR'], 'benchmark_results.csv')
                write_header = not os.path.exists(csv_path)
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

                visualise_samples(model, val_loader, out['WORK_DIR'],
                                  iteration, device, amp_enabled=amp_enabled, miou=miou)

                stop_flag = early_stopping.step(miou)

                ckpt_path = os.path.join(out['WORK_DIR'], 'latest_checkpoint.pth')
                save_checkpoint(ckpt_path, model, optimizer, scaler, iteration,
                                early_stopping, best_miou, best_per_class,
                                best_iteration=best_iteration, best_val_round=best_val_round,
                                edge_stats=edge_stats, edge_stats_source=edge_stats_source,
                                alpha=alpha)
                log(f"  Saved checkpoint (iter {iteration}) → {ckpt_path}")

            should_stop = torch.tensor(int(stop_flag), device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                log(f"Early stopping at iteration {iteration}.")
                break

    # ── Export artefacts ─────────────────────────────────────────────────────
    if is_main():
        ckpt_path = os.path.join(out['WORK_DIR'], 'best_model.pth')
        if not os.path.exists(ckpt_path):
            if best_state_dict is not None:
                torch.save(best_state_dict, ckpt_path)
                log(f"Saved best model (mIoU={best_miou:.4f}) → {ckpt_path}")
            elif best_miou > 0.0:
                log(f"WARNING: best_miou={best_miou:.4f} but no best checkpoint weights "
                    f"were available to write to {ckpt_path}.")

        try:
            iters  = [h['iter']     for h in history]
            mious  = [h['mIoU']     for h in history]
            losses = [h['val_loss'] for h in history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(iters, mious,  marker='o')
            ax1.set_title('Val mIoU'); ax1.set_xlabel('Iteration'); ax1.grid(True)
            ax2.plot(iters, losses, marker='o', color='orange')
            ax2.set_title('Val L_total'); ax2.set_xlabel('Iteration'); ax2.grid(True)
            fig.tight_layout()
            curve_path = os.path.join(out['WORK_DIR'], 'learning_curves.png')
            fig.savefig(curve_path, dpi=120)
            plt.close(fig)
            log(f"Saved learning curves to {curve_path}")

            if best_per_class is not None:
                colors = [
                    '#000000', '#800000', '#008000', '#808000', '#000080',
                    '#800080', '#008080', '#808080', '#400000',
                ]
                fig2, ax = plt.subplots(figsize=(11, 5))
                bars = ax.bar(_CLASS_NAMES, best_per_class, color=colors, edgecolor='white')
                for bar, val in zip(bars, best_per_class):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel('IoU')
                ax.set_title(f'Per-Class IoU at Best Checkpoint  (mIoU = {best_miou:.4f}, '
                             f'val round {best_val_round}/{total_val_rounds}, iter {best_iteration})',
                             fontweight='bold')
                ax.tick_params(axis='x', rotation=30)
                ax.grid(axis='y', alpha=0.3)
                fig2.tight_layout()
                bar_path = os.path.join(out['WORK_DIR'], 'per_class_iou_best.png')
                fig2.savefig(bar_path, dpi=120)
                plt.close(fig2)
                log(f"Saved per-class IoU chart to {bar_path}")
        except Exception as e:
            log(f"Warning: could not save charts — {e}")

    if is_main():
        elapsed      = time.time() - train_start
        end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        hours        = int(elapsed // 3600)
        minutes      = int((elapsed % 3600) // 60)
        seconds      = int(elapsed % 60)

        num       = os.path.basename(out['WORK_DIR']).split('_')[-1]
        time_path = os.path.join(out['WORK_DIR'], f'time_{num}.txt')
        with open(time_path, 'w', encoding='utf-8') as f:
            f.write(f'Bắt đầu      : {start_datetime}\n')
            f.write(f'Kết thúc     : {end_datetime}\n')
            f.write(f'Tổng thời gian: {hours:02d}h {minutes:02d}m {seconds:02d}s\n')
        print(f'Saved timing → {time_path}', flush=True)

        # final_summary.txt — cùng format baseline/Run 2 (tương thích ngược cho
        # tooling/thói quen đọc cũ), chỉ đổi header cho đúng tên thực nghiệm.
        summary_path = os.path.join(out['WORK_DIR'], 'final_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('=== UNetFormer (ResNet-18) — +Static Boundary — final summary ===\n\n')
            f.write(f'alpha                     : {alpha}\n')
            f.write(f'Total iterations trained  : {history[-1]["iter"] if history else 0} '
                    f'(target MAX_ITERS = {tr["MAX_ITERS"]})\n')
            f.write(f'Validation interval       : every {tr["VAL_INTERVAL"]} iterations\n')
            f.write(f'Validation rounds run     : {len(history)} / {total_val_rounds}\n\n')
            f.write(f'Best mIoU                 : {best_miou:.4f}\n')
            f.write(f'  achieved at val round   : {best_val_round}/{total_val_rounds} '
                    f'(iteration {best_iteration})\n\n')
            f.write('Per-class IoU at best checkpoint:\n')
            if best_per_class is not None:
                for name, iou_v in zip(_CLASS_NAMES, best_per_class):
                    f.write(f'  {name:<12}: {iou_v:.4f}\n')
            else:
                f.write('  (no validation completed)\n')
            f.write(f'\nTotal training time        : {hours:02d}h {minutes:02d}m {seconds:02d}s\n')
            f.write(f'Run started                : {start_datetime}\n')
            f.write(f'Run ended                  : {end_datetime}\n')
        log(f'Saved final summary → {summary_path}')

        # summary.txt — mirror cấu trúc write_bce_edge_summary() của Run 2,
        # điền THẬT các dòng boundary-metric mà Run 2 để N/A.
        write_static_boundary_summary(
            os.path.join(out['WORK_DIR'], 'summary.txt'),
            experiment_name=experiment_name, alpha=alpha,
            affinity_window_size=affinity_window_size, affinity_distance=affinity_distance,
            affinity_margin=affinity_margin,
            best_iteration=best_iteration, best_val_round=best_val_round,
            total_val_rounds=total_val_rounds, best_miou=best_miou,
            best_per_class=best_per_class, best_boundary=best_boundary,
            final_l_region=last_l_region, final_l_bce=last_l_bce,
            final_l_affinity=last_l_affinity, final_l_total=last_l_total,
            hours=hours, minutes=minutes, seconds=seconds,
            num_classes=tr['NUM_CLASSES'], ignore_index=ignore_index,
            connectivity=connectivity, dilation_radius=dilation_radius, seed=seed,
            edge_stats=edge_stats, edge_stats_source=edge_stats_source,
            pos_weight_max=pos_weight_max, sanity=sanity,
        )
        log(f"Saved summary report → {os.path.join(out['WORK_DIR'], 'summary.txt')}")

    dist.destroy_process_group()
    log("Done.")
    if _log_file is not None:
        _log_file.close()


if __name__ == '__main__':
    main()
