"""
Training script — Run 7, "Affinity-only" (α_bce = 0, chỉ bật affinity)
(docs/spec-run7-affinity-only-ban-giao-claude-code.md): tắt hẳn nhánh
BCE-edge, chỉ bật affinity loss, liều đối xứng với Static (alpha=0.4):

    L_total = L_region + ALPHA * LAMBDA2_STATIC * L_Affinity

KHÔNG có số hạng BCE nào trong công thức — không phải chỉ nhân trọng số 0
(mục 1.1 spec). Vì AffinityLoss (src/losses/affinity.py) KHÔNG có tham số học
được, kiến trúc model ở đây là UNetFormer TRẦN (build_model(cfg), giống hệt
Run 1/baseline) — KHÔNG có BoundaryHead, KHÔNG có gì cần "vẫn forward cho DDP
an toàn" như Run 3/3b phải làm với nhánh BCE. Checkpoint sinh ra có
state_dict giống hệt cấu trúc checkpoint baseline (không tiền tố
`base.`/`boundary_head.`) — nên Tools/eval_boundary_metrics.py dùng thẳng
`--model-type baseline` để dump preds/per-image-stats, KHÔNG cần sửa gì.

File này ĐỘC LẬP HOÀN TOÀN với mọi script train_*.py trước đó — KHÔNG import,
KHÔNG sửa bất kỳ file nào của Run 1/2/3/3b/4/5, để các run đó tái lập lại
được y hệt bất kỳ lúc nào. Loss dùng
src/losses/total_loss_affinity_only.py::AffinityOnlyTotalLoss (file MỚI,
tách biệt total_loss.py/total_loss_weighted.py). Chỉ dùng chung (import,
không sửa): src/data, src/utils (metrics/callbacks/visualizer/
boundary_visualizer/boundary_metrics), src/models/unet_former_resnet18.py,
src/losses/affinity.py (file dùng chung sẵn có từ Run 3, không phải file
riêng của Run 3). KHÔNG import src/losses/boundary_bce.py — run này không có
nhánh BCE nào để cần tới nó.

Khác biệt cốt lõi so với train_static_boundary_weighted.py (Run 3b):
    - Model = build_model(cfg) trần, không wrapper +BoundaryHead. Forward:
      logits, fused_feature = model(images, return_fused_feature=True).
    - `--seed` BẮT BUỘC riêng cho script này (mục 1.3.2 spec): nếu
      cfg['TRAIN']['SEED'] vắng mặt VÀ --seed không được truyền ⇒ dừng ngay
      với lỗi rõ ràng, TRƯỚC khi khởi tạo torch.distributed (mỗi rank tự kiểm
      tra độc lập — torchrun chạy N process độc lập, không cần collective op
      nào để đồng bộ lỗi này, nên không có rủi ro hang).
    - OUTPUT.WORK_DIR của config BẮT BUỘC chứa placeholder '{SEED}' — thay
      bằng seed thật ngay sau khi seed được xác định; thiếu placeholder ⇒
      dừng ngay với lỗi rõ ràng (bảo vệ khỏi nhiều seed ghi đè lên nhau).
      --work-dir CLI override vẫn là ưu tiên tuyệt đối nếu truyền.
    - KHÔNG có pos_weight/EdgeStatsAccumulator/EDGE_STATS_FILE/BoundaryHead —
      các sanity check gắn với BCE của Run 3b (#1 edge_gt shape/binary, #2
      ignore-batch → l_bce=0, #6 overlay edge_gt, #7 gradient BoundaryHead)
      KHÔNG áp dụng ở đây (không có gì để kiểm). Sanity check giữ lại/thêm
      mới (mục 3 spec, ánh xạ vào ngữ cảnh affinity-only):
        * seed thật in ra dòng đầu log/summary/checkpoint_index.
        * hash danh sách file val in ra để đối chiếu thủ công với Run 3/Run 5
          (không có giá trị hash tham chiếu cứng trong repo này để assert
          tự động — nếu cần, thêm --val-hash-ref sau).
        * alpha_bce_effective == 0.0 mọi vòng (hằng số từ loss module).
        * alpha_affinity_effective == alpha*lambda2 mọi vòng.
        * đồng nhất thức l_total == l_region + alpha*lambda2*l_affinity,
          sai số < 1e-5.
        * l_bce ghi CSV là hằng số 0.0 tường minh — summary.txt ghi rõ đây là
          hành vi KỲ VỌNG (không có BoundaryHead trong kiến trúc), không phải
          lỗi.
        * pos_weight: không tính, không gate — ghi rõ lý do trong summary.txt.
        * assert model.module KHÔNG có submodule 'boundary_head' (xác nhận
          kiến trúc đúng — bảo vệ khỏi lẫn nhầm với script Run 3/3b).
        * gradient check: decoder/frh nhận gradient qua l_affinity.
    - `--dry-run` đặt MAX_ITERS=100 (đúng mục 3 spec — khác dry-run 5 iters
      của Run 3b, vì đây là script mới, tự chọn theo đúng spec Run 7).
    - Boundary metrics: tính ĐẦY ĐỦ mọi vòng validate (không rút gọn), giữ
      đúng cách làm đã kiểm chứng ở Run 3/3b — chỉ đo & log thời gian 1 vòng
      validate để người đọc summary.txt tự quyết có cần rút gọn cho lần chạy
      sau hay không (mục 2/3.9 spec), không thêm nhánh rút gọn chưa test.
    - Giữ nguyên style output của Run 3b: 4 checkpoint
      (best_miou.pth/best_bfscore.pth/best_bareland.pth/final_iter<N>.pth) +
      checkpoint_index.json; benchmark_results.csv 10 hàng đủ cột (per-class
      IoU, ASD 2 chiều, bf_precision/recall, Boundary IoU 3 ngưỡng per-class,
      lambda1/lambda2/alpha/alpha_bce_effective/alpha_affinity_effective);
      summary.txt với khối 4-checkpoint + mean±std 4 vòng cuối (ddof=1) +
      khối seed + khối liều.

Usage (Kaggle, 2x T4):
    torchrun --nproc_per_node=2 src/train_affinity_only.py \\
        --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \\
        --seed 19 --dry-run

    torchrun --nproc_per_node=2 src/train_affinity_only.py \\
        --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \\
        --seed 19

    torchrun --nproc_per_node=2 src/train_affinity_only.py \\
        --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \\
        --seed 19 --resume <work_dir>/latest_checkpoint.pth

    # Liều phụ (mục 1.2 spec) — CHỈ chạy sau khi đã có kết quả liều 0.4:
    torchrun --nproc_per_node=2 src/train_affinity_only.py \\
        --config configs/unet_former_resnet18_affinity_only/run7b_affinity_only_lambda02.yaml \\
        --seed 19 --dry-run

    # Seed thứ 2 (khuyến nghị 86, đúng vai trò Run 5 đã tạo cho Static):
    torchrun --nproc_per_node=2 src/train_affinity_only.py \\
        --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \\
        --seed 86 --dry-run
"""

import argparse
import csv
import hashlib
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
from src.utils.visualizer import save_visualization
from src.utils.boundary_visualizer import save_boundary_visualization
from src.losses.total_loss_affinity_only import AffinityOnlyTotalLoss

_CLASS_NAMES = ['Background', 'Bareland', 'Rangeland', 'Developed', 'Road',
                'Tree', 'Water', 'Agriculture', 'Building']
_BARELAND_IDX = _CLASS_NAMES.index('Bareland')
_TRAIN_CROP_SIZE = 512  # khớp src/data/transforms.py::get_train_transforms() mặc định
_SEED_TOKEN = '{SEED}'

# 4 checkpoint báo cáo — key -> tên file (mục 2 spec-run7).
_CKPT_FILES = {
    'best_miou':     'best_miou.pth',
    'best_bfscore':  'best_bfscore.pth',
    'best_bareland': 'best_bareland.pth',
}


# ── Helpers (duplicated on purpose từ các script train_*.py trước — mỗi
#    script trong project này tự chứa, không phụ thuộc lẫn nhau) ────────────

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
                     alpha: float = 0.4, lambda1_static: float = 1.0,
                     lambda2_static: float = 1.0, seed: int = None):
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
        'alpha':               alpha,
        'lambda1_static':      lambda1_static,
        'lambda2_static':      lambda2_static,
        'seed':                seed,
    }
    tmp_path = path + '.tmp'
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)


def _save_report_checkpoint(path: str, model):
    """1 trong 4 checkpoint báo cáo — raw state_dict, KHÔNG bọc thêm key
    'model'/'epoch', đúng format hiện dùng (tương thích ngược với mọi tool
    eval hiện có — vd Tools/eval_boundary_metrics.py --model-type baseline,
    vì kiến trúc Run 7 chính là UNetFormer trần, xem docstring đầu file)."""
    torch.save(model.module.state_dict(), path)


def _round4(v):
    """round() an toàn với NaN — trả '' (blank CSV cell) cho NaN, giữ nguyên
    convention đã dùng ở Run 3/3b."""
    return round(v, 4) if v == v else ''


# ── Validation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, loss_fn: AffinityOnlyTotalLoss,
             num_classes: int, device: torch.device, amp_enabled: bool,
             boundary_distances=(1, 2, 4), bf_tolerance: int = 2,
             connectivity: int = 4, dilation_radius: int = 0,
             ignore_index: int = 255) -> dict:
    """Tính ĐẦY ĐỦ boundary metrics (Boundary IoU/BF-Score/ASD, kể cả 2 chiều
    và per-class) mỗi vòng validate — không rút gọn, giữ đúng cách làm đã
    kiểm chứng ở Run 3/3b (xem docstring đầu file)."""
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)
    boundary_metrics = BoundaryMetrics(
        num_classes=num_classes, ignore_index=ignore_index,
        boundary_distances=boundary_distances, bf_tolerance=bf_tolerance,
        connectivity=connectivity, dilation_radius=dilation_radius,
    )
    total_l_region = total_l_affinity = total_l_total = 0.0
    n_batches = 0
    parts_const = None  # lambda1/lambda2/alpha/*_effective — hằng số (static), lấy từ batch đầu

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits, fused_feature = model(images, return_fused_feature=True)
            _l_total, parts = loss_fn(logits, fused_feature, masks)
        if parts_const is None:
            parts_const = {k: parts[k] for k in
                           ('lambda1', 'lambda2', 'alpha', 'alpha_bce_effective', 'alpha_affinity_effective')}

        total_l_region   += parts['l_region']
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
        [total_l_region, total_l_affinity, total_l_total, float(n_batches)],
        device=device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
    n = max(loss_t[3].item(), 1)

    result = metrics.compute()
    result['l_region']   = (loss_t[0] / n).item()
    result['l_bce']      = 0.0  # hằng số tường minh — run này không có BoundaryHead (xem docstring)
    result['l_affinity'] = (loss_t[1] / n).item()
    result['l_total']    = (loss_t[2] / n).item()
    result['val_loss']   = result['l_total']  # tương thích ngược với cột val_loss của Run 1/2/3/3b
    result['loss_parts_const'] = parts_const  # lambda1/lambda2/alpha/*_effective (hằng số, static)

    result['boundary'] = boundary_metrics.compute()
    return result


# ── Visualise a few samples (tái dùng nguyên save_visualization /
#    save_boundary_visualization — model không có edge_logits nên chỉ unpack
#    logits/fused_feature) ────────────────────────────────────────────────────

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
                logit = model(img)  # return_fused_feature=False mặc định -> chỉ logits
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


# ── summary.txt ──────────────────────────────────────────────────────────────

def write_run7_summary(path: str, *, experiment_name: str, alpha: float,
                        lambda1_static: float, lambda2_static: float,
                        affinity_window_size: int, affinity_distance: str,
                        affinity_margin: float,
                        checkpoint_index: dict, history: list, total_val_rounds: int,
                        final_l_region: float, final_l_affinity: float, final_l_total: float,
                        hours: int, minutes: int, seconds: int,
                        num_classes: int, ignore_index: int,
                        connectivity: int, dilation_radius: int, seed: int,
                        sanity: dict, boundary_metrics_note: str,
                        val_hash: str, val_round_seconds: float):
    alpha_bce_eff = 0.0
    alpha_affinity_eff = alpha * lambda2_static
    edge_width = 1 if dilation_radius == 0 else 2 * dilation_radius + 1

    def pf(ok: bool) -> str:
        return 'passed' if ok else 'FAILED'

    lines = []
    lines.append('=' * 64)
    lines.append(' AFFINITY-ONLY SUPERVISION STATISTICS (Run 7)')
    lines.append('=' * 64)
    lines.append('')
    lines.append(f'Experiment name            : {experiment_name}')
    lines.append('Dataset                    : OpenEarthMap (OEM)')
    lines.append('Task                       : 8-class land-cover semantic segmentation '
                 '(+ Background, 9 lop trong pipeline nay)')
    lines.append('Model                      : UNetFormer (TRAN - khong BoundaryHead, xem duoi)')
    lines.append('Backbone                   : ResNet-18')
    lines.append('Region loss                : CombinedLoss (CrossEntropy + Dice)')
    lines.append('Edge (BCE) loss            : KHONG DUNG - tat han, khong tinh vao tong '
                 '(khong phai chi nhan trong so 0)')
    lines.append('Affinity loss              : contrastive feature-distance gan bien '
                 f'(window={affinity_window_size}, distance={affinity_distance}, margin={affinity_margin})')
    lines.append('Total loss                 : L_total = L_region + alpha*lambda2*L_Affinity')
    lines.append(f'alpha                      : {alpha:.4f}')
    lines.append(f'lambda1 (static, KHONG dung): {lambda1_static:.4f}      <-- giu de log/CSV dong dang, khong co tac dung')
    lines.append(f'lambda2 (static)           : {lambda2_static:.4f}')
    lines.append(f'alpha_bce  (effective)     : {alpha_bce_eff:.4f}      <-- hang so, khong doi (khong co nhanh BCE)')
    lines.append(f'alpha_affinity (effective) : {alpha_affinity_eff:.4f}      <-- alpha * lambda2')
    lines.append('')
    lines.append('=' * 64)
    lines.append(' SEED (nap luc chay — muc 1.3 spec-run7-affinity-only)')
    lines.append('=' * 64)
    lines.append('')
    lines.append(f'Training seed (--seed)     : {seed}')
    lines.append(f'Val file-list hash (sha256): {val_hash}')
    lines.append('                              (doc tu thu muc co dinh, KHONG phu thuoc seed -- '
                 'phai khop hash cua Run 3/Run 5, doi chieu thu cong)')
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
    lines.append(f' MEAN +/- STD (ddof=1) -- {n_tail} VONG CUOI (round {history[-n_tail]["round"] if n_tail else "-"}-'
                 f'{history[-1]["round"] if history else "-"})')
    lines.append('=' * 64)
    lines.append('')
    if n_tail < 2:
        lines.append('(khong du >= 2 vong validate de tinh std voi ddof=1)')
    else:
        tail = history[-n_tail:]

        def mean_std(key):
            vals = [h[key] for h in tail if h.get(key, '') != '' and h[key] == h[key]]
            if len(vals) < 2:
                return float('nan'), float('nan')
            arr = np.array(vals, dtype=np.float64)
            return float(arr.mean()), float(arr.std(ddof=1))

        for label, key in (('mIoU-9', 'miou9'), ('mIoU-8', 'miou8'), ('BFScore', 'bf_score'),
                            ('BIoU@2', 'biou_d2'), ('BIoU@4', 'biou_d4'), ('ASD', 'asd')):
            mu, sd = mean_std(key)
            lines.append(f'{label:<12}: {mu:.4f} +/- {sd:.4f}' if mu == mu else f'{label:<12}: N/A')
        mu, sd = mean_std(f'iou_{_CLASS_NAMES[_BARELAND_IDX]}')
        lines.append(f'{"Bareland":<12}: {mu:.4f} +/- {sd:.4f}' if mu == mu else f'{"Bareland":<12}: N/A')
        lines.append('')
        lines.append('Per-class IoU (9 lop, mean +/- std, ddof=1):')
        for name in _CLASS_NAMES:
            mu, sd = mean_std(f'iou_{name}')
            lines.append(f'  {name:<12}: {mu:.4f} +/- {sd:.4f}' if mu == mu else f'  {name:<12}: N/A')
    lines.append('')
    lines.append(f'Boundary metrics coverage   : {boundary_metrics_note}')
    lines.append(f'Thoi gian 1 vong validate   : ~{val_round_seconds:.1f}s (do o vong dau tien)')
    lines.append('')
    lines.append(f'Final validation L_region   : {final_l_region:.4f}')
    lines.append(f'Final validation L_bce      : 0.0000  <-- hang so, KHONG co nhanh BCE (xem duoi)')
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
    lines.append('Ignore pixels in affinity   : excluded (ca pixel trung tam va hang xom, qua valid_mask)')
    lines.append('Ignore pixels in mIoU       : excluded (qua ignore_index trong SegmentationMetrics)')
    lines.append('mIoU reporting convention   : ghi ca mIoU-9 (dung chon best checkpoint) va mIoU-8 (chi bao cao)')
    lines.append('-' * 64)
    lines.append(' EDGE-TARGET DEFINITION (dung cho L_Affinity va BoundaryMetrics)')
    lines.append('-' * 64)
    lines.append('')
    lines.append('Edge source                 : semantic ground-truth mask (post-transform), '
                 'downsample NEAREST ve stride-4 cho L_Affinity')
    lines.append(f'Neighborhood connectivity   : {connectivity}-connected')
    lines.append('Edge criterion              : center valid pixel co it nhat 1 hang-xom valid mang nhan khac')
    lines.append(f'Edge dilation radius        : {dilation_radius} pixels')
    lines.append(f'Final edge-target width     : {edge_width} pixel(s)')
    lines.append(f'Crop size used for training : {_TRAIN_CROP_SIZE} x {_TRAIN_CROP_SIZE}')
    lines.append('-' * 64)
    lines.append(' BCE IMBALANCE CORRECTION (KHONG AP DUNG - run nay khong co nhanh BCE)')
    lines.append('-' * 64)
    lines.append('')
    lines.append('pos_weight                  : KHONG TINH, KHONG GATE -- BoundaryHead khong ton tai o '
                 'kien truc Run 7 (AffinityLoss khong co tham so hoc duoc, xem docstring dau file train script)')
    lines.append('-' * 64)
    lines.append(' AFFINITY LOSS CONFIGURATION')
    lines.append('-' * 64)
    lines.append('')
    lines.append('Feature source              : Fused Feature (decoder, C=64, stride 4, truoc upsample)')
    lines.append(f'Neighborhood window (KxK)   : {affinity_window_size} x {affinity_window_size}')
    lines.append(f'Distance metric             : {affinity_distance}')
    lines.append(f'Margin                      : {affinity_margin}')
    lines.append('Sampling region              : chi cac pixel trung tam trong vung gan bien (extract_edge_gt)')
    lines.append(f'lambda2 (L_Affinity weight) : {lambda2_static:.4f} (static)')
    lines.append('-' * 64)
    lines.append(' PRE-FLIGHT VALIDATION')
    lines.append('-' * 64)
    lines.append('')
    lines.append(f'No-BoundaryHead architecture check : {pf(sanity.get("no_boundary_head", False))}')
    lines.append(f'Affinity gradient check             : {pf(sanity.get("affinity_gradient_check", False))}')
    lines.append(f'Loss-formula identity check         : {pf(sanity.get("loss_identity_check", False))}')
    lines.append(f'alpha_bce_effective == 0 check      : {pf(sanity.get("alpha_bce_effective_zero", False))}')
    lines.append(f'alpha_affinity_effective check      : {pf(sanity.get("alpha_affinity_effective_check", False))}')
    lines.append(f'Initial L_region finite             : {pf(sanity.get("initial_l_region_finite", False))}')
    lines.append(f'Initial L_affinity finite           : {pf(sanity.get("initial_l_affinity_finite", False))}')
    lines.append(f'Initial L_total finite              : {pf(sanity.get("initial_l_total_finite", False))}')
    lines.append('')
    lines.append('l_bce trong CSV/log ghi hang so 0.0000 tuong minh -- day la hanh vi KY VONG '
                 '(khong co BoundaryHead trong kien truc Run 7), KHONG PHAI loi.')

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',  required=True, help='Path to YAML config')
    parser.add_argument('--dry-run', action='store_true',
                        help='Quick smoke-test: MAX_ITERS=100 (muc 3 spec-run7-affinity-only), '
                             'VAL_INTERVAL rut gon, subset du lieu nho.')
    parser.add_argument('--resume', default=None,
                        help='Path to latest_checkpoint.pth to resume training from')
    parser.add_argument('--data-root', default=None,
                        help='Override DATASET.ROOT_DIR/VAL_ROOT_DIR from the config')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Ghi de BOUNDARY_LOSS.ALPHA cua config de ablation nhanh — '
                             'khi truyen, WORK_DIR tu them hau to (SAU khi da thay the {SEED}).')
    parser.add_argument('--lambda2', type=float, default=None,
                        help='Ghi de BOUNDARY_LOSS.LAMBDA2_STATIC.')
    parser.add_argument('--seed', type=int, default=None,
                        help='BAT BUOC neu config khong co TRAIN.SEED (day la truong hop mac dinh '
                             'cua run7_affinity_only.yaml/run7b_..._lambda02.yaml — muc 1.3 spec). '
                             'CHI anh huong khoi tao trong so/thu tu lay batch/augmentation sampling '
                             '— KHONG dung de sinh lai splits (val set doc tu thu muc co dinh).')
    parser.add_argument('--work-dir', default=None,
                        help='Ghi de OUTPUT.WORK_DIR cua config truc tiep (uu tien tuyet doi — bo qua '
                             'ca co che thay the {SEED} lan hau to --alpha/--lambda2).')
    args = parser.parse_args()

    # ── Seed BAT BUOC — kiem tra TRUOC dist.init_process_group() (moi rank tu
    # kiem doc lap, khong can collective op nao, khong co rui ro hang) ────────
    cfg = load_config(args.config)
    tr  = cfg['TRAIN']
    out = cfg['OUTPUT']
    if tr.get('SEED') is None and args.seed is None:
        raise SystemExit(
            "SEED bat buoc cho run7_affinity_only (va run7b_...) — config nay KHONG co "
            "TRAIN.SEED co dinh. Truyen qua --seed, vi du: --seed 19  (xem muc 1.3 spec-"
            "run7-affinity-only-ban-giao-claude-code.md).")
    seed = args.seed if args.seed is not None else tr['SEED']
    tr['SEED'] = seed

    if args.work_dir is not None:
        out['WORK_DIR'] = args.work_dir
    else:
        if _SEED_TOKEN not in out['WORK_DIR']:
            raise SystemExit(
                f"OUTPUT.WORK_DIR cua config phai chua placeholder '{_SEED_TOKEN}' de nhieu seed "
                f"khong ghi de len nhau (muc 1.3.3 spec). WORK_DIR hien tai: {out['WORK_DIR']!r}. "
                f"Dung --work-dir de override truc tiep neu day la chu y.")
        out['WORK_DIR'] = out['WORK_DIR'].replace(_SEED_TOKEN, str(seed))

    ds  = cfg['DATASET']
    mdl = cfg['MODEL']
    opt = cfg['OPTIMIZER']
    bl  = cfg.setdefault('BOUNDARY_LOSS', {})

    alpha  = bl.get('ALPHA', 0.4)
    lambda1_static = bl.get('LAMBDA1_STATIC', 1.0)  # khong co tac dung trong cong thuc, chi de log/CSV
    lambda2_static = bl.get('LAMBDA2_STATIC', 1.0)
    suffix = ''
    if args.alpha is not None:
        alpha = args.alpha
        bl['ALPHA'] = alpha
        suffix += f'_alpha{alpha:.2f}'
    if args.lambda2 is not None:
        lambda2_static = args.lambda2
        bl['LAMBDA2_STATIC'] = lambda2_static
        suffix += f'_lambda2_{lambda2_static:.2f}'
    if args.work_dir is None and suffix:
        out['WORK_DIR'] = out['WORK_DIR'].rstrip('/') + suffix

    ignore_index      = bl.get('IGNORE_INDEX', 255)
    connectivity      = bl.get('CONNECTIVITY', 4)
    dilation_radius   = bl.get('DILATION_RADIUS', 0)
    affinity_window_size = bl.get('AFFINITY_WINDOW_K', 5)
    affinity_distance    = bl.get('AFFINITY_DISTANCE', 'cosine')
    affinity_margin      = bl.get('AFFINITY_MARGIN', 1.0)

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

    os.makedirs(out['WORK_DIR'], exist_ok=True)
    setup_log_file(os.path.join(out['WORK_DIR'], 'train_log.txt'))
    if not use_cuda:
        log("WARNING: CUDA not available — running on CPU with backend='gloo', AMP disabled. "
            "This is only intended for local dry-run/dev; real training must run on Kaggle GPUs.")

    experiment_name = os.path.basename(out['WORK_DIR'].rstrip('/'))
    log(f"Experiment: Run 7 (Affinity-only, seed={seed}) — {experiment_name}  "
        f"(alpha={alpha}, lambda2={lambda2_static}, alpha_bce_effective=0.0 [khong co nhanh BCE], "
        f"alpha_affinity_effective={alpha * lambda2_static}, "
        f"affinity_window={affinity_window_size}, affinity_distance={affinity_distance})")

    # Auto-resume
    if args.resume is None:
        auto_ckpt = os.path.join(out['WORK_DIR'], 'latest_checkpoint.pth')
        if os.path.exists(auto_ckpt):
            args.resume = auto_ckpt
            log(f"Auto-resume: found existing checkpoint at {auto_ckpt} — resuming from it.")
    resume_ckpt = load_checkpoint_file(args.resume) if args.resume else None

    set_seed(seed)
    log(f"Using seed={seed}  (BAT BUOC nap qua --seed hoac TRAIN.SEED, muc 1.3 spec-run7-affinity-only)")

    if args.dry_run:
        tr['MAX_ITERS']    = 100  # dung MAX_ITERS=100 (muc 3 spec-run7-affinity-only)
        tr['VAL_INTERVAL'] = min(tr.get('VAL_INTERVAL', 4000), 50)

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
    val_names_hash = hashlib.sha256(
        '\n'.join(os.path.basename(img_path) for img_path, _ in val_ds.samples).encode('utf-8')
    ).hexdigest()[:16]
    log(f"Val file list hash: {val_names_hash}  (doc tu thu muc co dinh, KHONG phu thuoc SEED — "
        f"PHAI khop hash cua Run 3/Run 5, doi chieu THU CONG — chua co gia tri tham chieu cung "
        f"trong repo nay de assert tu dong, muc 1.4/3.2 spec-run7-affinity-only)")

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

    # ── Model — UNetFormer TRAN, khong wrapper +BoundaryHead (xem docstring) ─
    model = build_model(cfg).to(device)
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model_state_dict'])
        log(f"Resumed model weights from {args.resume} (iteration {resume_ckpt['iteration']})")
    model = DDP(model, device_ids=[local_rank] if use_cuda else None,
                find_unused_parameters=False)

    assert not hasattr(model.module, 'boundary_head'), \
        "Sanity kien truc that bai — model Run 7 khong duoc co submodule 'boundary_head'."
    log("Sanity (kien truc): model KHONG co submodule 'boundary_head' — dung nhu ky vong cho Run 7.")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_cfg = cfg.get('LOSS', {})
    ce_w   = loss_cfg.get('CE_WEIGHT',   1.0)
    dice_w = loss_cfg.get('DICE_WEIGHT', 1.0)
    loss_fn = AffinityOnlyTotalLoss(
        num_classes=tr['NUM_CLASSES'], alpha=alpha, ignore_index=ignore_index,
        ce_weight=ce_w, dice_weight=dice_w,
        connectivity=connectivity, dilation_radius=dilation_radius,
        affinity_window_size=affinity_window_size, affinity_distance=affinity_distance,
        affinity_margin=affinity_margin,
        lambda1_static=lambda1_static, lambda2_static=lambda2_static,
    ).to(device)
    log(f"Loss = CombinedLoss(CE+Dice, ce_weight={ce_w}, dice_weight={dice_w}) "
        f"+ {alpha} * {lambda2_static} * AffinityLoss(window={affinity_window_size}, "
        f"distance={affinity_distance}, margin={affinity_margin})  "
        f"[alpha_bce_effective=0.0 (khong co nhanh BCE), "
        f"alpha_affinity_effective={alpha * lambda2_static}]")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opt['BASE_LR'], weight_decay=opt['WEIGHT_DECAY'],
    )
    scaler = GradScaler(enabled=amp_enabled)
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        scaler.load_state_dict(resume_ckpt['scaler_state_dict'])

    # ── Sanity checks ────────────────────────────────────────────────────────
    sanity = {'no_boundary_head': not hasattr(model.module, 'boundary_head')}

    # ── Training state ───────────────────────────────────────────────────────
    early_stopping = EarlyStopping(patience=tr['EARLY_STOPPING_PATIENCE'])
    if resume_ckpt is not None:
        early_stopping.prev      = resume_ckpt['early_stopping']['prev']
        early_stopping.counter   = resume_ckpt['early_stopping']['counter']
        early_stopping.triggered = resume_ckpt['early_stopping']['triggered']

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

    # ── Training loop ────────────────────────────────────────────────────────
    log(f"Training {tr['MAX_ITERS']} iterations | encoder={mdl['ENCODER']} | "
        f"loss=CombinedLoss+{alpha}*{lambda2_static}*AffinityLoss (khong co nhanh BCE) | "
        f"train_dir={ds['ROOT_DIR']}/{ds['TRAIN_IMG_DIR']}")

    last_l_region = last_l_affinity = last_l_total = 0.0
    val_round_seconds = 0.0

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
            logits, fused_feature = model(images, return_fused_feature=True)
            l_total, parts = loss_fn(logits, fused_feature, masks)

        if iteration == 1:
            sanity['initial_l_region_finite']   = math.isfinite(parts['l_region'])
            sanity['initial_l_affinity_finite'] = math.isfinite(parts['l_affinity'])
            sanity['initial_l_total_finite']    = math.isfinite(parts['l_total'])
            log(f"Sanity (initial losses): l_region={parts['l_region']:.4f} l_bce=0.0000 (khong co nhanh BCE) "
                f"l_affinity={parts['l_affinity']:.4f} l_total={parts['l_total']:.4f}")
            for k in ('initial_l_region_finite', 'initial_l_affinity_finite', 'initial_l_total_finite'):
                if not sanity[k]:
                    raise RuntimeError(f"Sanity check '{k}' thất bại (giá trị không hữu hạn ngay iteration 1).")

            # alpha_bce_effective phai luon = 0.0 (hang so, khong tinh vao tong — muc 3.3 spec).
            ok_bce0 = parts['alpha_bce_effective'] == 0.0
            sanity['alpha_bce_effective_zero'] = ok_bce0
            log(f"Sanity (alpha_bce_effective=0.0): {'PASS' if ok_bce0 else 'FAIL'}")
            if not ok_bce0:
                raise RuntimeError("Sanity check that bai — alpha_bce_effective phai la hang so 0.0.")

            # alpha_affinity_effective phai = alpha * lambda2 (muc 3.4 spec).
            expected_aae = alpha * lambda2_static
            ok_aae = abs(parts['alpha_affinity_effective'] - expected_aae) < 1e-9
            sanity['alpha_affinity_effective_check'] = ok_aae
            log(f"Sanity (alpha_affinity_effective={parts['alpha_affinity_effective']:.4f}, "
                f"expected alpha*lambda2={expected_aae:.4f}): {'PASS' if ok_aae else 'FAIL'}")
            if not ok_aae:
                raise RuntimeError("Sanity check that bai — alpha_affinity_effective khong khop alpha*lambda2.")

            # Dong nhat thuc: l_total == l_region + alpha*lambda2*l_affinity (muc 3.5 spec,
            # KHONG co so hang lambda1*l_bce vi khong co nhanh BCE).
            expected = parts['l_region'] + parts['alpha'] * parts['lambda2'] * parts['l_affinity']
            ok_identity = abs(parts['l_total'] - expected) < 1e-5
            sanity['loss_identity_check'] = ok_identity
            log(f"Sanity (loss identity, l_total={parts['l_total']:.6f} vs expected={expected:.6f}): "
                f"{'PASS' if ok_identity else 'FAIL'}")
            if not ok_identity:
                raise RuntimeError("Sanity check (loss identity) that bai.")

        scaler.scale(l_total).backward()

        if iteration == 1:
            grad_sum_affinity = sum(p.grad.abs().sum().item()
                                    for p in model.module.frh.parameters() if p.grad is not None)
            sanity['affinity_gradient_check'] = grad_sum_affinity > 0
            log(f"Sanity (decoder/frh gradient nonzero qua L_affinity, sum|grad|={grad_sum_affinity:.6g}): "
                f"{'PASS' if sanity['affinity_gradient_check'] else 'FAIL'}")
            if not sanity['affinity_gradient_check']:
                raise RuntimeError("Sanity check (affinity gradient) thất bại — "
                                   "không có gradient chảy tới decoder qua L_affinity.")

            n_pass = sum(1 for v in sanity.values() if v)
            log(f"{len(sanity)} sanity checks: {n_pass}/{len(sanity)} PASS — bắt đầu training thật.")

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt['GRAD_CLIP'])
        scaler.step(optimizer)
        scaler.update()

        last_l_region, last_l_affinity, last_l_total = parts['l_region'], parts['l_affinity'], parts['l_total']

        if iteration % 100 == 0:
            log(f"[{iteration:>6}/{tr['MAX_ITERS']}] l_region={parts['l_region']:.4f} l_bce=0.0000 "
                f"l_affinity={parts['l_affinity']:.4f} l_total={parts['l_total']:.4f}  "
                f"alpha_affinity_effective={parts['alpha_affinity_effective']:.4f}  lr={lr:.2e}")

        # ── Validation ───────────────────────────────────────────────────────
        if iteration % tr['VAL_INTERVAL'] == 0 or iteration == tr['MAX_ITERS']:
            val_round = iteration // tr['VAL_INTERVAL']
            val_t0 = time.time()
            val_result = validate(
                model, val_loader, loss_fn, tr['NUM_CLASSES'],
                device, amp_enabled=amp_enabled, boundary_distances=(1, 2, 4),
                connectivity=connectivity, dilation_radius=dilation_radius,
                ignore_index=ignore_index,
            )
            if val_round_seconds == 0.0:
                val_round_seconds = time.time() - val_t0
                log(f"Thoi gian vong validate dau tien: {val_round_seconds:.1f}s "
                    f"(dung de quyet dinh boundary metrics 4 vs 10 vong cho lan chay sau — muc 2/3.9 spec)")
            miou9 = val_result['mIoU']
            last_l_region, last_l_affinity, last_l_total = \
                val_result['l_region'], val_result['l_affinity'], val_result['l_total']
            bnd = val_result['boundary']
            log(f"  [Val round {val_round}/{total_val_rounds}, iter {iteration}] mIoU-9={miou9:.4f}  "
                f"l_region={val_result['l_region']:.4f}  l_bce=0.0000  "
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
                    'l_bce':      0.0,
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
                                                      'round': val_round, 'seed': seed, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  New best mIoU-9={best_miou:.4f} (round {val_round}/{total_val_rounds}, "
                        f"iter {iteration}) -> saved {_CKPT_FILES['best_miou']}")
                if is_best_bfscore:
                    best_bfscore = bnd['bf_score']
                    _save_report_checkpoint(os.path.join(out['WORK_DIR'], _CKPT_FILES['best_bfscore']), model)
                    checkpoint_index['best_bfscore'] = {'file': _CKPT_FILES['best_bfscore'], 'iter': iteration,
                                                         'round': val_round, 'seed': seed, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  New best BFScore={best_bfscore:.4f} (round {val_round}/{total_val_rounds}, "
                        f"iter {iteration}) -> saved {_CKPT_FILES['best_bfscore']}")
                if is_best_bareland:
                    best_bareland = bareland_iou
                    _save_report_checkpoint(os.path.join(out['WORK_DIR'], _CKPT_FILES['best_bareland']), model)
                    checkpoint_index['best_bareland'] = {'file': _CKPT_FILES['best_bareland'], 'iter': iteration,
                                                          'round': val_round, 'seed': seed, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  New best Bareland IoU={best_bareland:.4f} (round {val_round}/{total_val_rounds}, "
                        f"iter {iteration}) -> saved {_CKPT_FILES['best_bareland']}")
                if is_final:
                    final_file = f'final_iter{iteration}.pth'
                    _save_report_checkpoint(os.path.join(out['WORK_DIR'], final_file), model)
                    checkpoint_index['final'] = {'file': final_file, 'iter': iteration,
                                                  'round': val_round, 'seed': seed, 'metrics': dict(row)}
                    index_dirty = True
                    log(f"  Final checkpoint @ iter {iteration} -> saved {final_file}")

                if index_dirty:
                    checkpoint_index['seed'] = seed  # dong dau ro rang o cap manifest (muc 1.3.4 spec)
                    with open(ckpt_index_path, 'w', encoding='utf-8') as f:
                        json.dump(checkpoint_index, f, indent=2, ensure_ascii=False)

                ckpt_path = os.path.join(out['WORK_DIR'], 'latest_checkpoint.pth')
                save_checkpoint(ckpt_path, model, optimizer, scaler, iteration,
                                early_stopping, best_miou,
                                alpha=alpha, lambda1_static=lambda1_static,
                                lambda2_static=lambda2_static, seed=seed)
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
            f.write(f'Seed          : {seed}\n')
            f.write(f'Bắt đầu      : {start_datetime}\n')
            f.write(f'Kết thúc     : {end_datetime}\n')
            f.write(f'Tổng thời gian: {hours:02d}h {minutes:02d}m {seconds:02d}s\n')
        print(f'Saved timing → {time_path}', flush=True)

        summary_path = os.path.join(out['WORK_DIR'], 'final_summary.txt')
        best_miou_entry = checkpoint_index.get('best_miou')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('=== UNetFormer (ResNet-18) — Run 7 (Affinity-only) — final summary ===\n\n')
            f.write(f'seed                      : {seed}\n')
            f.write(f'alpha                     : {alpha}\n')
            f.write(f'lambda2                   : {lambda2_static}\n')
            f.write(f'alpha_bce_effective       : 0.0000 (khong co nhanh BCE)\n')
            f.write(f'alpha_affinity_effective  : {alpha * lambda2_static}\n')
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
        log(f'Saved final summary → {summary_path}')

        boundary_metrics_note = (
            "day du moi vong validate (khong rut gon) - giu nguyen cach lam da kiem "
            "chung o Run 3/3b."
        )
        write_run7_summary(
            os.path.join(out['WORK_DIR'], 'summary.txt'),
            experiment_name=experiment_name, alpha=alpha,
            lambda1_static=lambda1_static, lambda2_static=lambda2_static,
            affinity_window_size=affinity_window_size, affinity_distance=affinity_distance,
            affinity_margin=affinity_margin,
            checkpoint_index=checkpoint_index, history=history, total_val_rounds=total_val_rounds,
            final_l_region=last_l_region, final_l_affinity=last_l_affinity, final_l_total=last_l_total,
            hours=hours, minutes=minutes, seconds=seconds,
            num_classes=tr['NUM_CLASSES'], ignore_index=ignore_index,
            connectivity=connectivity, dilation_radius=dilation_radius, seed=seed,
            sanity=sanity, boundary_metrics_note=boundary_metrics_note,
            val_hash=val_names_hash, val_round_seconds=val_round_seconds,
        )
        log(f"Saved summary report → {os.path.join(out['WORK_DIR'], 'summary.txt')}")

    dist.destroy_process_group()
    log("Done.")
    if _log_file is not None:
        _log_file.close()


if __name__ == '__main__':
    main()
