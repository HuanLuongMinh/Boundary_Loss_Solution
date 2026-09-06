"""
Training script — Run 4, "Dynamic Boundary-Aware Loss" (ramp -> hold)
(docs/spec-run4-dynamic-weighting.md). Cong thuc:

    L_total = L_region + alpha * (lambda1(t) * L_BCE_edge + lambda2(t) * L_Affinity)

lambda1(t) phang = 1.0 suot run. lambda2(t): 0 trong warmup_frac dau (chi
BCE), ramp tuyen tinh len lambda2_end tu warmup_frac den ramp_end_frac, roi
giu nguyen (hold) toi het. Voi cau hinh mac dinh cua Run 4 (WARMUP_FRAC=0.10,
RAMP_END_FRAC=0.30, LAMBDA2_END=1.0, ALPHA=0.4): alpha_affinity hieu dung di
tu 0.0 (iter 0-4000) len 0.4 (dat tu iter 12000, giu toi het) - dung dung
lieu alpha_aff=0.4 da xac nhan tot nhat o Run 3b.

File nay DOC LAP HOAN TOAN voi src/train_static_boundary.py (Run 3),
src/train_static_boundary_weighted.py (Run 3b) va moi script train_*.py
truoc do - KHONG import, KHONG sua bat ky file nao cua cac run truoc, de cac
run do tai lap lai duoc y het bat ky luc nao. Loss dung
src/losses/total_loss_dynamic.py::DynamicBoundaryTotalLoss (file MOI, tach
biet total_loss.py/total_loss_weighted.py). Chi dung chung (import, khong
sua): src/data, src/utils (losses/metrics/callbacks/visualizer/
boundary_visualizer/boundary_metrics), src/models/unet_former_resnet18.py,
src/losses/boundary_bce.py + src/losses/affinity.py (file dung chung san co
tu Run 3, khong phai file rieng cua Run 3).

Khac biet so voi Run 3b (train_static_boundary_weighted.py):
    - Loss: DynamicBoundaryTotalLoss, doc DYNAMIC_WEIGHTS/SCHEDULE/
      LAMBDA1_CONST/LAMBDA2_END/WARMUP_FRAC/RAMP_END_FRAC tu BOUNDARY_LOSS.
      DYNAMIC_WEIGHTS=false (vd tro thang vao config Run 3 cu) -> dung
      lambda1_static/lambda2_static (mac dinh 1.0/1.0), bo qua schedule hoan
      toan - dung de kiem tuong thich nguoc (muc 5.3 spec).
    - cur_iter truyen vao loss_fn() = iteration - 1 (0-indexed), CUNG QUY UOC
      voi poly_lr(cur_iter=iteration-1, ...) da co san trong file goc - giu
      nhat quan mot quy uoc index duy nhat trong toan bo script.
    - 2 sanity check MOI (muc 5.1/5.4 docs/spec-run4-dynamic-weighting.md):
      #11 schedule dung dang dong (kiem truc tiep lambda_schedule_ramp_hold
      voi 6 moc cua bang, KHONG phu thuoc forward pass that), va #12 warmup
      thuc su vo hieu hoa affinity (alpha_affinity_effective=0 VA l_affinity
      van huu han duong tai iteration 1).
    - lambda_schedule_log.csv (MOI) - ghi 1 hang moi 200 iter: iter, lambda1,
      lambda2, alpha_bce_effective, alpha_affinity_effective, l_region,
      l_bce, l_affinity, l_total (gia tri o BUOC TRAIN, khong phai
      validation) - du lieu cho hinh lich trinh cua bai bao (muc 4/7.4 spec).
    - benchmark_results.csv: cot lambda1/lambda2/alpha/2 gia tri effective
      gio thay doi theo tung vong (truoc day la hang so o Run 3b).
    - summary.txt: khoi "DYNAMIC SCHEDULE" thay cho khoi "AFFINITY LOSS
      CONFIGURATION" gia tri tinh cua Run 3b (muc 4 spec), giu nguyen khoi
      4-checkpoint + mean+/-std 4 vong cuoi.
    - 4 checkpoint + checkpoint_index.json, cach tinh boundary metrics day
      du moi vong (rut gon con 4 vong cuoi neu 1 vong validate > ~5 phut
      tren Kaggle that, ghi ro vao summary.txt) - giu nguyen y het Run 3b.

Usage (Kaggle, 2x T4):
    torchrun --nproc_per_node=2 src/train_dynamic_boundary.py --config configs/unet_former_resnet18_dynamic_boundary/run4_dynamic_ramp_hold.yaml
    torchrun --nproc_per_node=2 src/train_dynamic_boundary.py --config configs/unet_former_resnet18_dynamic_boundary/run4_dynamic_ramp_hold.yaml --dry-run
    torchrun --nproc_per_node=2 src/train_dynamic_boundary.py --config configs/unet_former_resnet18_dynamic_boundary/run4_dynamic_ramp_hold.yaml --resume <work_dir>/latest_checkpoint.pth

    # Kiem tra tuong thich nguoc (muc 5.3 spec): tro thang vao config Run 3 cu
    # (DYNAMIC_WEIGHTS: false) -> phai dung lambda1_static/lambda2_static=1.0/1.0,
    # bo qua schedule hoan toan, tai lap dung dong nhat thuc cua Run 3:
    torchrun --nproc_per_node=2 src/train_dynamic_boundary.py --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml --dry-run
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
from src.losses.total_loss_dynamic import DynamicBoundaryTotalLoss
from src.losses.dynamic_weighting import lambda_schedule_ramp_hold

_CLASS_NAMES = ['Background', 'Bareland', 'Rangeland', 'Developed', 'Road',
                'Tree', 'Water', 'Agriculture', 'Building']
_BARELAND_IDX = _CLASS_NAMES.index('Bareland')
_TRAIN_CROP_SIZE = 512  # khớp src/data/transforms.py::get_train_transforms() mặc định

# 4 checkpoint báo cáo (mục 3 docs/run3b_spec_lambda2_05.md, áp dụng nguyên
# cho Run 4 theo mục 4 docs/spec-run4-dynamic-weighting.md) — key -> tên file.
_CKPT_FILES = {
    'best_miou':    'best_miou.pth',
    'best_bfscore': 'best_bfscore.pth',
    'best_bareland': 'best_bareland.pth',
}

# 6 mốc kiểm schedule đóng — docs/spec-run4-dynamic-weighting.md mục 5.1.
_SCHEDULE_TABLE_MAX_ITERS = 40000
_SCHEDULE_TABLE = [
    (0,     1.0, 0.0),
    (3999,  1.0, 0.0),
    (4000,  1.0, 0.0),
    (8000,  1.0, 0.5),
    (12000, 1.0, 1.0),
    (40000, 1.0, 1.0),
]


# ── Helpers (duplicated on purpose từ train_static_boundary_weighted.py —
#    mỗi script train_*.py trong project này tự chứa, không phụ thuộc lẫn nhau) ─

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
                     early_stopping, best_miou: float,
                     edge_stats: dict = None, edge_stats_source: str = '',
                     alpha: float = 0.4, dynamic_weights: bool = True,
                     schedule_name: str = 'ramp_hold', schedule_kwargs: dict = None,
                     lambda1_static: float = 1.0, lambda2_static: float = 1.0):
    """Checkpoint dùng để RESUME (--resume) — KHÔNG phải 1 trong 4 checkpoint
    báo cáo (những cái đó là raw state_dict riêng biệt, xem
    _save_report_checkpoint()). checkpoint_index.json (ghi riêng, đọc lại lúc
    resume) là nguồn dữ liệu chính cho best_miou/best_bfscore/best_bareland —
    best_miou ở đây chỉ để log nhanh, không phải nguồn sự thật duy nhất."""
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
        # Lưu lại pos_weight/r_edge đã đo để RESUME dùng lại đúng giá trị cũ
        # thay vì đo lại (đo lại có thể ra pos_weight hơi khác vì train_iter
        # đã lệch vị trí so với lần chạy gốc).
        'edge_stats':          edge_stats,
        'edge_stats_source':   edge_stats_source,
        'alpha':               alpha,
        'dynamic_weights':     dynamic_weights,
        'schedule_name':       schedule_name,
        'schedule_kwargs':     schedule_kwargs,
        'lambda1_static':      lambda1_static,
        'lambda2_static':      lambda2_static,
    }
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def _save_report_checkpoint(path: str, model):
    """1 trong 4 checkpoint báo cáo — raw state_dict, KHÔNG bọc thêm key
    'model'/'epoch', đúng format hiện dùng (tương thích ngược với mọi tool
    eval hiện có, vd Tools/eval_boundary_metrics.py)."""
    torch.save(model.module.state_dict(), path)


def _round4(v):
    """round() an toàn với NaN — trả '' (blank CSV cell) cho NaN, giữ nguyên
    convention đã dùng ở Run 3/3b."""
    return round(v, 4) if v == v else ''


# ── Model: UNetFormer (không sửa) + BoundaryHead phụ ────────────────────────

class DynamicBoundaryUNetFormer(nn.Module):
    """Wrapper cục bộ — y hệt StaticBoundaryUNetFormer của Run 3/3b (định
    nghĩa lại ở đây thay vì import, đúng quy ước "mỗi script train_*.py tự
    chứa" của repo này). UNetFormer (build_model(cfg), model gốc KHÔNG bị
    sửa) + BoundaryHead (src/losses/boundary_bce.py) gắn trên Fused Feature.
    forward() LUÔN gọi cả 2 nhánh — bắt buộc, để an toàn với
    DDP(find_unused_parameters=False)."""

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
def validate(model, loader, loss_fn: DynamicBoundaryTotalLoss, pos_weight: float,
             num_classes: int, device: torch.device, amp_enabled: bool, cur_iter: int,
             boundary_distances=(1, 2, 4), bf_tolerance: int = 2,
             connectivity: int = 4, dilation_radius: int = 0,
             ignore_index: int = 255) -> dict:
    """Tính ĐẦY ĐỦ boundary metrics (Boundary IoU/BF-Score/ASD, kể cả 2 chiều
    và per-class) mỗi vòng validate — không rút gọn, xem docstring đầu file.
    cur_iter: vị trí lịch trình dùng CHO CẢ VÒNG validate này (hằng số trong
    1 lần gọi — lambda1(t)/lambda2(t) chỉ phụ thuộc cur_iter, không phụ
    thuộc batch), truyền y hệt cur_iter của bước train gần nhất."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    boundary_metrics = BoundaryMetrics(
        num_classes=num_classes, ignore_index=ignore_index,
        boundary_distances=boundary_distances, bf_tolerance=bf_tolerance,
        connectivity=connectivity, dilation_radius=dilation_radius,
    )
    total_l_region = total_l_bce = total_l_affinity = total_l_total = 0.0
    n_batches = 0
    parts_const = None  # lambda1/lambda2/alpha/*_effective — hằng số trong vòng này (cur_iter cố định)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits, edge_logits, fused_feature = model(images)
            _l_total, parts = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight, cur_iter)
        if parts_const is None:
            parts_const = {k: parts[k] for k in
                           ('lambda1', 'lambda2', 'alpha', 'alpha_bce_effective', 'alpha_affinity_effective')}

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
    # (qua DistributedSampler) — all-reduce accumulator thô trước compute().
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
    result['val_loss']   = result['l_total']  # tương thích ngược với cột val_loss của Run 1/2/3/3b
    result['loss_parts_const'] = parts_const  # lambda1/lambda2/alpha/*_effective tại cur_iter của vòng này

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
    """Diagnostic thuần định tính, y hệt Run 3/3b — lưu
    WORK_DIR/sanity/edge_gt_overlay_sK.png để tự kiểm tra bằng mắt edge_gt có
    trùng khớp ranh giới đối tượng thật hay không."""
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


# ── summary.txt (mirror cấu trúc write_run3b_summary() của Run 3b, thay khối
#    "AFFINITY LOSS CONFIGURATION" tĩnh bằng khối "DYNAMIC SCHEDULE") ────────

def write_run4_summary(path: str, *, experiment_name: str, alpha: float,
                        dynamic_weights: bool, schedule_name: str,
                        lambda1_const: float, lambda2_end: float,
                        warmup_frac: float, ramp_end_frac: float,
                        lambda1_static: float, lambda2_static: float,
                        max_iters: int,
                        affinity_window_size: int, affinity_distance: str,
                        affinity_margin: float,
                        checkpoint_index: dict, history: list, total_val_rounds: int,
                        final_l_region: float, final_l_bce: float,
                        final_l_affinity: float, final_l_total: float,
                        hours: int, minutes: int, seconds: int,
                        num_classes: int, ignore_index: int,
                        connectivity: int, dilation_radius: int, seed: int,
                        edge_stats: dict, edge_stats_source: str,
                        pos_weight_max: float, sanity: dict,
                        boundary_metrics_note: str):
    edge_width = 1 if dilation_radius == 0 else 2 * dilation_radius + 1

    def fmt(value) -> str:
        if value is None or value == '' or value != value:
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
    lines.append('Total loss                 : L_total = L_region + alpha*(lambda1(t)*L_BCE_edge + lambda2(t)*L_Affinity)')
    lines.append(f'alpha                      : {alpha:.4f}')
    lines.append('')
    lines.append('=' * 64)
    lines.append(' DYNAMIC SCHEDULE')
    lines.append('=' * 64)
    lines.append('')
    if dynamic_weights:
        warmup_end_iter = int(round(warmup_frac * max_iters))
        ramp_end_iter = int(round(ramp_end_frac * max_iters))
        lines.append(f'Schedule                   : {schedule_name}')
        lines.append(f'  lambda1 (const)          : {lambda1_const:.4f}')
        lines.append(f'  lambda2_end              : {lambda2_end:.4f}')
        lines.append(f'  warmup_frac              : {warmup_frac:.2f}   (iter 0 - {warmup_end_iter},  lambda2 = 0)')
        lines.append(f'  ramp_end_frac            : {ramp_end_frac:.2f}   (iter {warmup_end_iter} - {ramp_end_iter}, '
                     f'lambda2 0 -> {lambda2_end:.1f})')
        lines.append(f'  hold                     :        (iter {ramp_end_iter} - {max_iters}, lambda2 = {lambda2_end:.1f})')
        lines.append(f'alpha_bce (effective)      : {alpha * lambda1_const:.4f}  (khong doi suot run)')
        lines.append(f'alpha_affinity (effective) : 0.0000 -> {alpha * lambda2_end:.4f}')
    else:
        lines.append('Schedule                   : (tat - DYNAMIC_WEIGHTS=false, dung lambda tinh - kiem tuong thich nguoc)')
        lines.append(f'lambda1 (static)           : {lambda1_static:.4f}')
        lines.append(f'lambda2 (static)           : {lambda2_static:.4f}')
        lines.append(f'alpha_bce (effective)      : {alpha * lambda1_static:.4f}')
        lines.append(f'alpha_affinity (effective) : {alpha * lambda2_static:.4f}')
    lines.append('')
    lines.append('=' * 64)
    lines.append(' EVALUATION -- 4 CHECKPOINTS')
    lines.append('=' * 64)
    lines.append('')
    header = f"{'Checkpoint':<14}{'Iter':>8}{'mIoU-9':>10}{'mIoU-8':>10}{'BFScore':>10}{'BIoU@2':>10}{'BIoU@4':>10}{'ASD':>10}{'Bareland':>10}"
    lines.append(header)
    for key in ('best_miou', 'best_bfscore', 'best_bareland', 'final'):
        entry = checkpoint_index.get(key)
        if entry is None:
            lines.append(f"{key:<14}{'N/A':>8}")
            continue
        m = entry['metrics']
        lines.append(
            f"{key:<14}{entry['iter']:>8}{m['miou9']:>10.4f}{m['miou8']:>10.4f}"
            f"{(m['bf_score'] if m['bf_score'] != '' else float('nan')):>10.4f}"
            f"{(m['biou_d2'] if m['biou_d2'] != '' else float('nan')):>10.4f}"
            f"{(m['biou_d4'] if m['biou_d4'] != '' else float('nan')):>10.4f}"
            f"{(m['asd'] if m['asd'] != '' else float('nan')):>10.4f}"
            f"{m[f'iou_{_CLASS_NAMES[_BARELAND_IDX]}']:>10.4f}"
        )
    lines.append('')
    n_tail = min(4, len(history))
    lines.append('=' * 64)
    lines.append(f' MEAN +/- STD -- {n_tail} VONG CUOI (round {history[-n_tail]["round"] if n_tail else "-"}-'
                 f'{history[-1]["round"] if history else "-"})')
    lines.append('=' * 64)
    lines.append('')
    if n_tail == 0:
        lines.append('(khong co validation nao hoan thanh)')
    else:
        tail = history[-n_tail:]

        def mean_std(key):
            vals = [h[key] for h in tail if h.get(key, '') != '' and h[key] == h[key]]
            if not vals:
                return float('nan'), float('nan')
            arr = np.array(vals, dtype=np.float64)
            return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

        for label, key in (('mIoU-9', 'miou9'), ('mIoU-8', 'miou8'), ('BFScore', 'bf_score'),
                            ('BIoU@2', 'biou_d2'), ('BIoU@4', 'biou_d4'), ('ASD', 'asd')):
            mu, sd = mean_std(key)
            lines.append(f'{label:<12}: {mu:.4f} +/- {sd:.4f}' if mu == mu else f'{label:<12}: N/A')
        mu, sd = mean_std(f'iou_{_CLASS_NAMES[_BARELAND_IDX]}')
        lines.append(f'{"Bareland":<12}: {mu:.4f} +/- {sd:.4f}' if mu == mu else f'{"Bareland":<12}: N/A')
        lines.append('')
        lines.append('Per-class IoU (9 lop, mean +/- std):')
        for name in _CLASS_NAMES:
            mu, sd = mean_std(f'iou_{name}')
            lines.append(f'  {name:<12}: {mu:.4f} +/- {sd:.4f}' if mu == mu else f'  {name:<12}: N/A')
    lines.append('')
    lines.append(f'Boundary metrics coverage   : {boundary_metrics_note}')
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
    lines.append('-' * 64)
    lines.append(' PRE-FLIGHT VALIDATION')
    lines.append('-' * 64)
    lines.append('')
    lines.append(f'Edge target binary check    : {pf(sanity.get("edge_target_binary_check", False))}')
    lines.append(f'Ignore-pixel mask check     : {pf(sanity.get("ignore_pixel_mask_check", False))}')
    lines.append(f'Edge overlay visual check   : {pf(sanity.get("edge_overlay_visual_check", False))}')
    lines.append(f'Edge-head gradient check    : {pf(sanity.get("edge_head_gradient_check", False))}')
    lines.append(f'Affinity gradient check     : {pf(sanity.get("affinity_gradient_check", False))}')
    lines.append(f'Loss-formula identity check : {pf(sanity.get("loss_identity_check", False))}')
    lines.append(f'Initial L_region finite     : {pf(sanity.get("initial_l_region_finite", False))}')
    lines.append(f'Initial L_bce finite        : {pf(sanity.get("initial_l_bce_finite", False))}')
    lines.append(f'Initial L_affinity finite   : {pf(sanity.get("initial_l_affinity_finite", False))}')
    lines.append(f'Initial L_total finite      : {pf(sanity.get("initial_l_total_finite", False))}')
    lines.append(f'Schedule closed-form check  : {pf(sanity.get("schedule_closed_form_check", False))}')
    lines.append(f'Warmup disables affinity    : {pf(sanity.get("warmup_disables_affinity_check", False))}')

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
                        help='Ghi de BOUNDARY_LOSS.ALPHA cua config de ablation nhanh — '
                             'khi truyen, WORK_DIR tu them hau to.')
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

    # ── alpha override + WORK_DIR suffix (ablation) ─────────────────────────
    alpha  = bl.get('ALPHA', 0.4)
    suffix = ''
    if args.alpha is not None:
        alpha = args.alpha
        bl['ALPHA'] = alpha
        suffix += f'_alpha{alpha:.2f}'
    if suffix:
        out['WORK_DIR'] = out['WORK_DIR'].rstrip('/') + suffix

    # Lich trinh luon tinh theo MAX_ITERS THAT cua config (40000), BAT KE
    # --dry-run co rut gon vong lap chay hay khong — --dry-run chi giam SO
    # BUOC CHAY (de nhanh), khong duoc lam lech vi tri tren truc thoi gian
    # cua schedule (cur_iter=0..99 trong dry-run van la 99 buoc dau cua 1 run
    # 40000 buoc that, tuc van nam sau trong warmup — dung y nhu no se the
    # trong run that).
    schedule_max_iters = tr['MAX_ITERS']

    dynamic_weights   = bool(bl.get('DYNAMIC_WEIGHTS', True))
    schedule_name     = bl.get('SCHEDULE', 'ramp_hold')
    lambda1_const     = bl.get('LAMBDA1_CONST', 1.0)
    lambda2_end       = bl.get('LAMBDA2_END', 1.0)
    warmup_frac       = bl.get('WARMUP_FRAC', 0.10)
    ramp_end_frac     = bl.get('RAMP_END_FRAC', 0.30)
    schedule_kwargs   = {
        'lambda1_const': lambda1_const, 'lambda2_end': lambda2_end,
        'warmup_frac': warmup_frac, 'ramp_end_frac': ramp_end_frac,
    } if schedule_name == 'ramp_hold' else {}
    lambda1_static    = bl.get('LAMBDA1_STATIC', 1.0)
    lambda2_static    = bl.get('LAMBDA2_STATIC', 1.0)

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
    log(f"Experiment: Run 4 (dynamic_weights={dynamic_weights}, schedule={schedule_name}) — "
        f"{experiment_name}  (alpha={alpha}, schedule_kwargs={schedule_kwargs}, "
        f"lambda1_static={lambda1_static}, lambda2_static={lambda2_static})")

    # ── Sanity #11 (mục 5.1 spec) — schedule đúng dạng đóng, KHÔNG phụ thuộc
    # forward pass thật, kiểm ngay khi khởi động ─────────────────────────────
    sanity = {}
    ok11 = True
    for cur_iter, exp_lam1, exp_lam2 in _SCHEDULE_TABLE:
        lam1, lam2 = lambda_schedule_ramp_hold(
            cur_iter, _SCHEDULE_TABLE_MAX_ITERS,
            lambda1_const=lambda1_const, lambda2_end=lambda2_end,
            warmup_frac=warmup_frac, ramp_end_frac=ramp_end_frac)
        point_ok = abs(lam1 - exp_lam1) < 1e-6 and abs(lam2 - exp_lam2) < 1e-6
        ok11 = ok11 and point_ok
        log(f"Sanity #11 point (cur_iter={cur_iter}): lambda1={lam1:.4f} (exp {exp_lam1}) "
            f"lambda2={lam2:.4f} (exp {exp_lam2}): {'PASS' if point_ok else 'FAIL'}")
    sanity['schedule_closed_form_check'] = ok11
    log(f"Sanity #11 (schedule dang dong, {len(_SCHEDULE_TABLE)} moc): {'PASS' if ok11 else 'FAIL'}")
    if not ok11:
        raise RuntimeError("Sanity check #11 (schedule closed-form) thất bại — "
                           "lambda_schedule_ramp_hold không khớp bảng mục 5.1.")

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
        tr['MAX_ITERS']    = 100
        tr['VAL_INTERVAL'] = 25
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
    model = DynamicBoundaryUNetFormer(cfg).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state_dict'])
        log(f"Resumed model weights from {args.resume} (iteration {resume_ckpt['iteration']})")
    model = DDP(model, device_ids=[local_rank] if use_cuda else None,
                find_unused_parameters=False)

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_cfg = cfg.get('LOSS', {})
    ce_w   = loss_cfg.get('CE_WEIGHT',   1.0)
    dice_w = loss_cfg.get('DICE_WEIGHT', 1.0)
    loss_fn = DynamicBoundaryTotalLoss(
        num_classes=tr['NUM_CLASSES'], alpha=alpha, max_iters=schedule_max_iters,
        ignore_index=ignore_index, ce_weight=ce_w, dice_weight=dice_w,
        connectivity=connectivity, dilation_radius=dilation_radius,
        affinity_window_size=affinity_window_size, affinity_distance=affinity_distance,
        affinity_margin=affinity_margin,
        dynamic_weights=dynamic_weights, schedule_name=schedule_name, schedule_kwargs=schedule_kwargs,
        lambda1_static=lambda1_static, lambda2_static=lambda2_static,
    ).to(device)
    log(f"Loss = CombinedLoss(CE+Dice, ce_weight={ce_w}, dice_weight={dice_w}) "
        f"+ {alpha} * (lambda1(t)*BalancedBCEEdgeLoss(connectivity={connectivity}, dilation_radius={dilation_radius}) "
        f"+ lambda2(t)*AffinityLoss(window={affinity_window_size}, distance={affinity_distance}, margin={affinity_margin}))  "
        f"[dynamic_weights={dynamic_weights}, schedule={schedule_name}, schedule_max_iters={schedule_max_iters}]")

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

    # ── Sanity checks (mirror Run 3b + #11 (đã chạy) / #12 mới) ──────────────
    sanity['pos_weight_finite'] = pos_weight_finite_ok

    # #2 Ignore-pixel mask check — synthetic, không cần forward qua model.
    synth_masks  = torch.full((2, 32, 32), ignore_index, dtype=torch.int64, device=device)
    synth_edge_logits = torch.randn(2, 1, 32, 32, device=device)
    synth_edge_loss = loss_fn.bce_edge(synth_edge_logits, synth_masks, pos_weight)
    ok2 = bool(torch.isfinite(synth_edge_loss)) and abs(synth_edge_loss.item()) < 1e-6
    sanity['ignore_pixel_mask_check'] = ok2
    log(f"Sanity #2 (batch toàn ignore -> L_bce=0): {'PASS' if ok2 else 'FAIL'} (loss={synth_edge_loss.item():.6g})")
    if not ok2:
        raise RuntimeError("Sanity check #2 (ignore-pixel mask) thất bại — L_bce trên batch toàn ignore phải =0.")

    # #6 Overlay trực quan (định tính, rank0-only, KHÔNG raise nếu lỗi).
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
        sanity['edge_overlay_visual_check'] = True

    # ── Training state ───────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=tr['EARLY_STOPPING_PATIENCE'])
    if resume_ckpt is not None:
        early_stopping.prev      = resume_ckpt['early_stopping']['prev']
        early_stopping.counter   = resume_ckpt['early_stopping']['counter']
        early_stopping.triggered = resume_ckpt['early_stopping']['triggered']

    # checkpoint_index.json là nguồn sự thật cho best_miou/best_bfscore/
    # best_bareland — nạp lại nguyên trạng khi resume thay vì suy luận lại từ
    # CSV/checkpoint (đơn giản, không thể lệch pha).
    ckpt_index_path = os.path.join(out['WORK_DIR'], 'checkpoint_index.json')
    checkpoint_index = {}
    if resume_ckpt is not None and is_main() and os.path.exists(ckpt_index_path):
        with open(ckpt_index_path, encoding='utf-8') as f:
            checkpoint_index = json.load(f)
        log(f"Reloaded checkpoint_index.json ({len(checkpoint_index)} entries) for resume")

    best_miou     = checkpoint_index.get('best_miou', {}).get('metrics', {}).get('miou9', 0.0)
    best_bfscore  = checkpoint_index.get('best_bfscore', {}).get('metrics', {}).get('bf_score', 0.0) or 0.0
    best_bareland = checkpoint_index.get('best_bareland', {}).get('metrics', {}) \
        .get(f'iou_{_CLASS_NAMES[_BARELAND_IDX]}', 0.0) or 0.0
    history = []
    total_val_rounds = -(-tr['MAX_ITERS'] // tr['VAL_INTERVAL'])

    if resume_ckpt is not None and is_main():
        csv_path = os.path.join(out['WORK_DIR'], 'benchmark_results.csv')
        if os.path.exists(csv_path):
            _bool_cols = {'is_best_miou', 'is_best_bfscore', 'is_best_bareland', 'is_final'}
            _int_cols  = {'round', 'iter'}
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    for k, v in list(row.items()):
                        if k in _int_cols:
                            row[k] = int(v)
                        elif k in _bool_cols:
                            row[k] = (v == 'True')
                        else:
                            row[k] = float(v) if v not in (None, '') else float('nan')
                    history.append(row)
            log(f"Reconstructed {len(history)} history rows from {csv_path}")

    start_iter = resume_ckpt['iteration'] + 1 if resume_ckpt is not None else 1
    if args.dry_run and resume_ckpt is not None and start_iter > tr['MAX_ITERS']:
        log(f"Warning: --dry-run MAX_ITERS={tr['MAX_ITERS']} <= resumed "
            f"start_iter={start_iter}; loop will not execute.")

    lambda_log_path = os.path.join(out['WORK_DIR'], 'lambda_schedule_log.csv')
    lambda_log_write_header = is_main() and not os.path.exists(lambda_log_path)

    # ── Training loop ────────────────────────────────────────────────────────
    log(f"Training {tr['MAX_ITERS']} iterations | encoder={mdl['ENCODER']} | "
        f"loss=CombinedLoss+{alpha}*(lambda1(t)*BalancedBCEEdgeLoss+lambda2(t)*AffinityLoss) | "
        f"train_dir={ds['ROOT_DIR']}/{ds['TRAIN_IMG_DIR']}")

    last_l_region = last_l_bce = last_l_affinity = last_l_total = 0.0

    for iteration in range(start_iter, tr['MAX_ITERS'] + 1):
        model.train()
        cur_iter = iteration - 1  # 0-indexed, cùng quy ước với poly_lr(cur_iter=iteration-1, ...)

        lr = poly_lr(
            base_lr=opt['BASE_LR'], cur_iter=cur_iter, max_iters=tr['MAX_ITERS'],
            warmup_iters=tr.get('WARMUP_ITERS', 500),
        )
        set_lr(optimizer, lr)

        images, masks = next(train_iter)
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp_enabled):
            logits, edge_logits, fused_feature = model(images)
            l_total, parts = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight, cur_iter)

        if iteration == 1:
            # #1 shape/binary edge_gt + finite-loss checks — MỌI rank tự kiểm.
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
            log(f"Sanity (initial losses, cur_iter={cur_iter}): l_region={parts['l_region']:.4f} "
                f"l_bce={parts['l_bce']:.4f} l_affinity={parts['l_affinity']:.4f} l_total={parts['l_total']:.4f} "
                f"lambda1={parts['lambda1']:.4f} lambda2={parts['lambda2']:.4f}")
            for k in ('initial_l_region_finite', 'initial_l_bce_finite',
                      'initial_l_affinity_finite', 'initial_l_total_finite'):
                if not sanity[k]:
                    raise RuntimeError(f"Sanity check '{k}' thất bại (giá trị không hữu hạn ngay iteration 1).")

            # #9 (mục 5.2 spec) — đồng nhất công thức loss: l_total phải khớp
            # đúng l_region + alpha*(lambda1*l_bce + lambda2*l_affinity).
            expected = parts['l_region'] + parts['alpha'] * (
                parts['lambda1'] * parts['l_bce'] + parts['lambda2'] * parts['l_affinity'])
            ok9 = abs(parts['l_total'] - expected) < 1e-5
            sanity['loss_identity_check'] = ok9
            log(f"Sanity #9 (loss identity, l_total={parts['l_total']:.6f} vs expected={expected:.6f}): "
                f"{'PASS' if ok9 else 'FAIL'}")
            if not ok9:
                raise RuntimeError("Sanity check #9 (loss identity) thất bại — hệ số lambda1/lambda2 "
                                   "không tới được loss đúng công thức.")

            # #12 (mục 5.4 spec) — warmup thực sự vô hiệu hoá affinity: nếu
            # dynamic_weights=True và cur_iter=0 nằm trong warmup (warmup_frac>0),
            # alpha_affinity_effective phải =0 NHƯNG l_affinity vẫn hữu hạn
            # dương (đã tính, chỉ nhân trọng số 0 — không bỏ tính).
            if dynamic_weights and warmup_frac > 0:
                ok12 = (abs(parts['alpha_affinity_effective'] - 0.0) < 1e-12) and \
                       math.isfinite(parts['l_affinity']) and parts['l_affinity'] > 0.0
                sanity['warmup_disables_affinity_check'] = ok12
                log(f"Sanity #12 (warmup vo hieu hoa affinity, alpha_affinity_effective="
                    f"{parts['alpha_affinity_effective']:.6f}, l_affinity={parts['l_affinity']:.4f}): "
                    f"{'PASS' if ok12 else 'FAIL'}")
                if not ok12:
                    raise RuntimeError("Sanity check #12 (warmup disables affinity) thất bại.")
            else:
                sanity['warmup_disables_affinity_check'] = True
                log("Sanity #12: bo qua (dynamic_weights=False hoac warmup_frac=0 - khong warmup).")

        scaler.scale(l_total).backward()

        if iteration == 1:
            # #7 gradient khác 0 trên BoundaryHead (L_bce) VÀ trên decoder/frh
            # (qua l_region/l_bce/l_affinity) — kiểm NGAY SAU backward() thật.
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
            log(f"Sanity #8 (decoder/frh gradient nonzero, sum|grad|={grad_sum_affinity:.6g}): "
                f"{'PASS' if sanity['affinity_gradient_check'] else 'FAIL'}")
            if not sanity['affinity_gradient_check']:
                raise RuntimeError("Sanity check #8 (decoder/frh gradient) thất bại — "
                                   "không có gradient chảy tới decoder.")

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
                f"l_affinity={parts['l_affinity']:.4f} l_total={parts['l_total']:.4f}  "
                f"lambda1={parts['lambda1']:.4f} lambda2={parts['lambda2']:.4f}  "
                f"alpha_affinity_effective={parts['alpha_affinity_effective']:.4f}  lr={lr:.2e}")

        # ── lambda_schedule_log.csv — 1 hàng mỗi 200 iter (mục 4 spec) ───────
        if is_main() and (iteration % 200 == 0 or iteration == 1):
            with open(lambda_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                if lambda_log_write_header:
                    writer.writerow(['iter', 'lambda1', 'lambda2', 'alpha_bce_effective',
                                     'alpha_affinity_effective', 'l_region', 'l_bce',
                                     'l_affinity', 'l_total'])
                    lambda_log_write_header = False
                writer.writerow([iteration, parts['lambda1'], parts['lambda2'],
                                 parts['alpha_bce_effective'], parts['alpha_affinity_effective'],
                                 round(parts['l_region'], 6), round(parts['l_bce'], 6),
                                 round(parts['l_affinity'], 6), round(parts['l_total'], 6)])

        # ── Validation ───────────────────────────────────────────────────────
        if iteration % tr['VAL_INTERVAL'] == 0 or iteration == tr['MAX_ITERS']:
            val_round = iteration // tr['VAL_INTERVAL']
            val_result = validate(
                model, val_loader, loss_fn, pos_weight, tr['NUM_CLASSES'],
                device, amp_enabled=amp_enabled, cur_iter=cur_iter, boundary_distances=(1, 2, 4),
                connectivity=connectivity, dilation_radius=dilation_radius,
                ignore_index=ignore_index,
            )
            miou9 = val_result['mIoU']
            last_l_region, last_l_bce = val_result['l_region'], val_result['l_bce']
            last_l_affinity, last_l_total = val_result['l_affinity'], val_result['l_total']
            bnd = val_result['boundary']
            log(f"  [Val round {val_round}/{total_val_rounds}, iter {iteration}] mIoU-9={miou9:.4f}  "
                f"l_region={val_result['l_region']:.4f}  l_bce={val_result['l_bce']:.4f}  "
                f"l_affinity={val_result['l_affinity']:.4f}  l_total={val_result['l_total']:.4f}")
            log(f"    BFScore={bnd['bf_score']:.4f}  BIoU@1={bnd['boundary_iou_d1']:.4f}  "
                f"BIoU@2={bnd['boundary_iou_d2']:.4f}  BIoU@4={bnd['boundary_iou_d4']:.4f}  "
                f"ASD={bnd['asd']:.4f} (p2g={bnd['asd_pred_to_gt']:.4f} g2p={bnd['asd_gt_to_pred']:.4f})")

            stop_flag = False
            if is_main():
                per_class = val_result['per_class_iou']
                miou8 = float(np.mean(per_class[1:]))
                _abbr = ['Bg', 'Bare', 'Range', 'Dev', 'Road', 'Tree', 'Water', 'Agri', 'Bldg']
                pconst = val_result['loss_parts_const']

                row = {
                    'round': val_round,
                    'iter':  iteration,
                    'miou9': round(miou9, 4),
                    'miou8': round(miou8, 4),
                    **{f'iou_{n}': round(v, 4) for n, v in zip(_CLASS_NAMES, per_class)},
                    'bf_score':     _round4(bnd['bf_score']),
                    'bf_precision': _round4(bnd['bf_precision']),
                    'bf_recall':    _round4(bnd['bf_recall']),
                    'biou_d1': _round4(bnd['boundary_iou_d1']),
                    'biou_d2': _round4(bnd['boundary_iou_d2']),
                    'biou_d4': _round4(bnd['boundary_iou_d4']),
                }
                for d in (1, 2, 4):
                    for n, v in zip(_CLASS_NAMES, bnd[f'boundary_iou_d{d}_per_class']):
                        row[f'biou_d{d}_{n}'] = _round4(v)
                row.update({
                    'asd':             _round4(bnd['asd']),
                    'asd_pred_to_gt':  _round4(bnd['asd_pred_to_gt']),
                    'asd_gt_to_pred':  _round4(bnd['asd_gt_to_pred']),
                    'l_region':   round(val_result['l_region'], 4),
                    'l_bce':      round(val_result['l_bce'], 4),
                    'l_affinity': round(val_result['l_affinity'], 4),
                    'l_total':    round(val_result['l_total'], 4),
                    'lambda1': pconst['lambda1'],
                    'lambda2': pconst['lambda2'],
                    'alpha':   pconst['alpha'],
                    'alpha_bce_effective':      pconst['alpha_bce_effective'],
                    'alpha_affinity_effective': pconst['alpha_affinity_effective'],
                })

                is_best_miou     = miou9 > best_miou
                is_best_bfscore  = (bnd['bf_score'] == bnd['bf_score']) and bnd['bf_score'] > best_bfscore
                bareland_iou     = per_class[_BARELAND_IDX]
                is_best_bareland = bareland_iou > best_bareland

                stop_flag = early_stopping.step(miou9)
                is_final = stop_flag or (iteration == tr['MAX_ITERS'])

                row['is_best_miou']     = is_best_miou
                row['is_best_bfscore']  = is_best_bfscore
                row['is_best_bareland'] = is_best_bareland
                row['is_final']         = is_final
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
                                  iteration, device, amp_enabled=amp_enabled, miou=miou9)

                index_dirty = False
                if is_best_miou:
                    best_miou = miou9
                    _save_report_checkpoint(os.path.join(out['WORK_DIR'], _CKPT_FILES['best_miou']), model)
                    checkpoint_index['best_miou'] = {'file': _CKPT_FILES['best_miou'], 'iter': iteration,
                                                      'round': val_round, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  ✔ New best mIoU-9={best_miou:.4f} (round {val_round}/{total_val_rounds}, "
                        f"iter {iteration}) -> saved {_CKPT_FILES['best_miou']}")
                if is_best_bfscore:
                    best_bfscore = bnd['bf_score']
                    _save_report_checkpoint(os.path.join(out['WORK_DIR'], _CKPT_FILES['best_bfscore']), model)
                    checkpoint_index['best_bfscore'] = {'file': _CKPT_FILES['best_bfscore'], 'iter': iteration,
                                                         'round': val_round, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  ✔ New best BFScore={best_bfscore:.4f} (round {val_round}/{total_val_rounds}, "
                        f"iter {iteration}) -> saved {_CKPT_FILES['best_bfscore']}")
                if is_best_bareland:
                    best_bareland = bareland_iou
                    _save_report_checkpoint(os.path.join(out['WORK_DIR'], _CKPT_FILES['best_bareland']), model)
                    checkpoint_index['best_bareland'] = {'file': _CKPT_FILES['best_bareland'], 'iter': iteration,
                                                          'round': val_round, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  ✔ New best Bareland IoU={best_bareland:.4f} (round {val_round}/{total_val_rounds}, "
                        f"iter {iteration}) -> saved {_CKPT_FILES['best_bareland']}")
                if is_final:
                    final_file = f'final_iter{iteration}.pth'
                    _save_report_checkpoint(os.path.join(out['WORK_DIR'], final_file), model)
                    checkpoint_index['final'] = {'file': final_file, 'iter': iteration,
                                                  'round': val_round, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  ✔ Final checkpoint @ iter {iteration} -> saved {final_file}")

                if index_dirty:
                    with open(ckpt_index_path, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_index, f, indent=2, ensure_ascii=False)

                ckpt_path = os.path.join(out['WORK_DIR'], 'latest_checkpoint.pth')
                save_checkpoint(ckpt_path, model, optimizer, scaler, iteration,
                                early_stopping, best_miou,
                                edge_stats=edge_stats, edge_stats_source=edge_stats_source,
                                alpha=alpha, dynamic_weights=dynamic_weights,
                                schedule_name=schedule_name, schedule_kwargs=schedule_kwargs,
                                lambda1_static=lambda1_static, lambda2_static=lambda2_static)
                log(f"  Saved checkpoint (iter {iteration}) -> {ckpt_path}")

            should_stop = torch.tensor(int(stop_flag), device=device)
            dist.broadcast(should_stop, src=0)
            if should_stop.item():
                log(f"Early stopping at iteration {iteration}.")
                break

    # ── Export artefacts ─────────────────────────────────────────────────────
    if is_main():
        try:
            iters  = [h['iter']     for h in history]
            mious  = [h['miou9']    for h in history]
            losses = [h['l_total']  for h in history]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(iters, mious,  marker='o')
            ax1.set_title('Val mIoU-9'); ax1.set_xlabel('Iteration'); ax1.grid(True)
            ax2.plot(iters, losses, marker='o', color='orange')
            ax2.set_title('Val L_total'); ax2.set_xlabel('Iteration'); ax2.grid(True)
            fig.tight_layout()
            curve_path = os.path.join(out['WORK_DIR'], 'learning_curves.png')
            fig.savefig(curve_path, dpi=120)
            plt.close(fig)
            log(f"Saved learning curves to {curve_path}")

            best_miou_entry = checkpoint_index.get('best_miou')
            if best_miou_entry is not None:
                m = best_miou_entry['metrics']
                per_class_best = [m[f'iou_{n}'] for n in _CLASS_NAMES]
                colors = [
                    '#000000', '#800000', '#008000', '#808000', '#000080',
                    '#800080', '#008080', '#808080', '#400000',
                ]
                fig2, ax = plt.subplots(figsize=(11, 5))
                bars = ax.bar(_CLASS_NAMES, per_class_best, color=colors, edgecolor='white')
                for bar, val in zip(bars, per_class_best):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
                ax.set_ylim(0, 1.05)
                ax.set_ylabel('IoU')
                ax.set_title(f'Per-Class IoU at best_miou checkpoint  (mIoU-9 = {m["miou9"]:.4f}, '
                             f'round {best_miou_entry["round"]}/{total_val_rounds}, iter {best_miou_entry["iter"]})',
                             fontweight='bold')
                ax.tick_params(axis='x', rotation=30)
                ax.grid(axis='y', alpha=0.3)
                fig2.tight_layout()
                bar_path = os.path.join(out['WORK_DIR'], 'per_class_iou_best.png')
                fig2.savefig(bar_path, dpi=120)
                plt.close(fig2)
                log(f"Saved per-class IoU chart to {bar_path}")

            # ── Hình lịch trình cho bài báo (mục 7.4 spec) — từ
            #    lambda_schedule_log.csv: lambda1/lambda2 theo iter (trên),
            #    l_region/l_bce/l_affinity theo iter (dưới) ─────────────────
            if os.path.exists(lambda_log_path):
                with open(lambda_log_path, newline='') as f:
                    rows = list(csv.DictReader(f))
                if rows:
                    log_iters = [int(r['iter']) for r in rows]
                    log_lam1  = [float(r['lambda1']) for r in rows]
                    log_lam2  = [float(r['lambda2']) for r in rows]
                    log_lreg  = [float(r['l_region']) for r in rows]
                    log_lbce  = [float(r['l_bce']) for r in rows]
                    log_laff  = [float(r['l_affinity']) for r in rows]

                    fig3, (axa, axb) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
                    axa.plot(log_iters, log_lam1, label='lambda1', color='tab:blue')
                    axa.plot(log_iters, log_lam2, label='lambda2', color='tab:red')
                    axa.set_ylabel('lambda'); axa.set_ylim(-0.05, 1.15)
                    axa.legend(); axa.grid(True); axa.set_title('Run 4 — dynamic schedule (ramp_hold)')
                    axb.plot(log_iters, log_lreg, label='l_region', color='tab:green')
                    axb.plot(log_iters, log_lbce, label='l_bce', color='tab:orange')
                    axb.plot(log_iters, log_laff, label='l_affinity', color='tab:purple')
                    axb.set_xlabel('Iteration'); axb.set_ylabel('loss value')
                    axb.legend(); axb.grid(True)
                    fig3.tight_layout()
                    schedule_fig_path = os.path.join(out['WORK_DIR'], 'schedule_curves.png')
                    fig3.savefig(schedule_fig_path, dpi=120)
                    plt.close(fig3)
                    log(f"Saved schedule curves to {schedule_fig_path}")
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
        print(f'Saved timing -> {time_path}', flush=True)

        # final_summary.txt — cùng format baseline/Run 2/Run 3/3b (tương
        # thích ngược cho tooling/thói quen đọc cũ), nguồn số liệu =
        # best_miou entry trong checkpoint_index.
        summary_path = os.path.join(out['WORK_DIR'], 'final_summary.txt')
        best_miou_entry = checkpoint_index.get('best_miou')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('=== UNetFormer (ResNet-18) — Run 4 (dynamic ramp_hold) — final summary ===\n\n')
            f.write(f'alpha                     : {alpha}\n')
            f.write(f'dynamic_weights           : {dynamic_weights}\n')
            f.write(f'schedule                  : {schedule_name} ({schedule_kwargs})\n')
            f.write(f'Total iterations trained  : {history[-1]["iter"] if history else 0} '
                    f'(target MAX_ITERS = {tr["MAX_ITERS"]})\n')
            f.write(f'Validation interval       : every {tr["VAL_INTERVAL"]} iterations\n')
            f.write(f'Validation rounds run     : {len(history)} / {total_val_rounds}\n\n')
            if best_miou_entry is not None:
                m = best_miou_entry['metrics']
                f.write(f'Best mIoU-9                : {m["miou9"]:.4f}\n')
                f.write(f'  achieved at val round   : {best_miou_entry["round"]}/{total_val_rounds} '
                        f'(iteration {best_miou_entry["iter"]})\n\n')
                f.write('Per-class IoU at best_miou checkpoint:\n')
                for name in _CLASS_NAMES:
                    f.write(f'  {name:<12}: {m[f"iou_{name}"]:.4f}\n')
            else:
                f.write('  (no validation completed)\n')
            f.write(f'\nTotal training time        : {hours:02d}h {minutes:02d}m {seconds:02d}s\n')
            f.write(f'Run started                : {start_datetime}\n')
            f.write(f'Run ended                  : {end_datetime}\n')
        log(f'Saved final summary -> {summary_path}')

        boundary_metrics_note = (
            "day du moi vong validate (khong rut gon) tru khi 1 vong validate vuot ~5 phut "
            "tren Kaggle that - khi do rut gon con 4 vong cuoi (round cuoi), giu nguyen cach "
            "quyet dinh da dung o Run 3b (muc 4.3 docs/run3b_spec_lambda2_05.md)."
        )
        write_run4_summary(
            os.path.join(out['WORK_DIR'], 'summary.txt'),
            experiment_name=experiment_name, alpha=alpha,
            dynamic_weights=dynamic_weights, schedule_name=schedule_name,
            lambda1_const=lambda1_const, lambda2_end=lambda2_end,
            warmup_frac=warmup_frac, ramp_end_frac=ramp_end_frac,
            lambda1_static=lambda1_static, lambda2_static=lambda2_static,
            max_iters=schedule_max_iters,
            affinity_window_size=affinity_window_size, affinity_distance=affinity_distance,
            affinity_margin=affinity_margin,
            checkpoint_index=checkpoint_index, history=history, total_val_rounds=total_val_rounds,
            final_l_region=last_l_region, final_l_bce=last_l_bce,
            final_l_affinity=last_l_affinity, final_l_total=last_l_total,
            hours=hours, minutes=minutes, seconds=seconds,
            num_classes=tr['NUM_CLASSES'], ignore_index=ignore_index,
            connectivity=connectivity, dilation_radius=dilation_radius, seed=seed,
            edge_stats=edge_stats, edge_stats_source=edge_stats_source,
            pos_weight_max=pos_weight_max, sanity=sanity,
            boundary_metrics_note=boundary_metrics_note,
        )
        log(f"Saved summary report -> {os.path.join(out['WORK_DIR'], 'summary.txt')}")

    dist.destroy_process_group()
    log("Done.")
    if _log_file is not None:
        _log_file.close()


if __name__ == '__main__':
    main()
