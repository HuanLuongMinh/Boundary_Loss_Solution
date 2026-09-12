"""Tools/per_image_dump.py — Tinh "thong ke du" (sufficient statistics) per-image
cho Phan A cua docs/spec-per-image-bootstrap-eval-v2-ban-giao-claude-code.md
muc 2.2, va ham luu mask du doan ra PNG (muc 2.4).

File MOI, DOC LAP — KHONG sua src/utils/boundary_metrics.py hay
src/utils/metrics.py. Ham o day tinh lai CHINH XAC cung cong thuc/quy uoc voi
2 class do (cung goi extract_edge_gt tu src/losses/boundary_bce.py, cung
distance_transform_edt), nhung tra ve GIA TRI THO tung anh (numerator/
denominator rieng) thay vi cong don qua toan dataset — de giai doan 2
(Tools/bootstrap_boundary_ci.py) tu gop lai theo dung quy uoc (micro/macro)
ma khong can doan.

Duoc goi tu Tools/eval_boundary_metrics.py CHI KHI mot trong 2 co
--dump-preds/--dump-per-image-stats duoc truyen — duong tinh mac dinh (khong
co) khong goi toi module nay, xem gate 4.0 (muc 4.0 spec) va
Tools/test_eval_boundary_metrics_dump_gate.py.

Tu-test: `python Tools/test_per_image_dump.py` — kiem cac gia tri per-image
tra ve cong don DUNG lai chinh xac ket qua cua SegmentationMetrics/
BoundaryMetrics tren cung batch (ca 2 module do KHONG bi sua trong file nay).
"""

import os

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from src.losses.boundary_bce import extract_edge_gt


def compute_per_image_stats(logits: torch.Tensor, targets: torch.Tensor,
                             image_names, class_names,
                             ignore_index: int, connectivity: int, dilation_radius: int,
                             boundary_distances, bf_tolerance: int):
    """logits (B,C,H,W), targets (B,H,W) int64 — DUNG dinh dang voi
    SegmentationMetrics.update()/BoundaryMetrics.update(). image_names: list[str]
    do dai B, ten file goc (khong duoi), dung lam khoa ghep cap giua checkpoint.

    Tra ve (rows, preds_np):
      rows: list[dict], 1 dict/anh, DUNG cot theo dac ta muc 2.2 (khong tinh
        metric da chia — chi tu so/mau so tho + co trang thai rong).
      preds_np: (B,H,W) uint8 argmax prediction — dung de luu PNG (muc 2.4).
    """
    assert len(image_names) == logits.shape[0], (
        f"So ten anh ({len(image_names)}) khac batch size logits ({logits.shape[0]}).")

    preds = logits.argmax(dim=1)  # (B,H,W)
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    valid_np = targets_np != ignore_index
    num_classes = len(class_names)
    B = preds_np.shape[0]

    # Bien: dung CHUNG 1 dinh nghia voi BF-Score/ASD trong BoundaryMetrics —
    # goi lai dung extract_edge_gt(), khong dinh nghia lai.
    gt_edge, gt_valid = extract_edge_gt(targets, ignore_index, connectivity, dilation_radius)
    pred_edge, _ = extract_edge_gt(preds, ignore_index, connectivity, dilation_radius)
    pred_edge = pred_edge * gt_valid
    gt_edge_np = gt_edge.squeeze(1).cpu().numpy().astype(bool)      # (B,H,W)
    pred_edge_np = pred_edge.squeeze(1).cpu().numpy().astype(bool)  # (B,H,W)

    rows = []
    for b in range(B):
        t, p, v = targets_np[b], preds_np[b], valid_np[b]
        row = {'image': image_names[b], 'n_valid_px': int(v.sum())}

        # ---- Khoi IoU ngu nghia + co trang thai lop-co-mat (muc 2.2) ----
        for c, cname in enumerate(class_names):
            gt_c = (t == c) & v
            pred_c = (p == c) & v
            row[f'inter_{cname}'] = int(np.logical_and(gt_c, pred_c).sum())
            row[f'union_{cname}'] = int(np.logical_or(gt_c, pred_c).sum())
            row[f'class_present_{cname}'] = int(gt_c.any())

        # ---- Khoi Boundary IoU — per-class, per-distance (muc 2.2) ----
        # Cung thuat toan voi BoundaryMetrics.update(): distance_transform_edt
        # MOT LAN moi mask (khong lap lai cho tung d).
        for c, cname in enumerate(class_names):
            gt_c = (t == c) & v
            pred_c = (p == c) & v
            gt_dist = distance_transform_edt(gt_c) if gt_c.any() else None
            pred_dist = distance_transform_edt(pred_c) if pred_c.any() else None
            for d in boundary_distances:
                gt_band = (gt_c & (gt_dist <= d)) if gt_dist is not None else np.zeros_like(gt_c)
                pred_band = (pred_c & (pred_dist <= d)) if pred_dist is not None else np.zeros_like(pred_c)
                row[f'biou_inter_{cname}_d{d}'] = int(np.logical_and(gt_band, pred_band).sum())
                row[f'biou_union_{cname}_d{d}'] = int(np.logical_or(gt_band, pred_band).sum())

        # ---- Khoi BF-Score + ASD (muc 2.2) — cung 1 dinh nghia bien ----
        g, pr = gt_edge_np[b], pred_edge_np[b]
        n_gt, n_pred = int(g.sum()), int(pr.sum())
        row['has_gt_boundary'] = int(n_gt > 0)
        row['has_pred_boundary'] = int(n_pred > 0)

        bf_tp_pred = bf_tp_gt = 0
        asd_sum_p2g = asd_sum_g2p = 0.0
        asd_n_pred = asd_n_gt = 0
        if n_gt > 0:
            dist_to_gt = distance_transform_edt(~g)
            if n_pred > 0:
                bf_tp_pred = int((dist_to_gt[pr] <= bf_tolerance).sum())
                asd_sum_p2g = float(dist_to_gt[pr].sum())
                asd_n_pred = n_pred
        if n_pred > 0:
            dist_to_pred = distance_transform_edt(~pr)
            if n_gt > 0:
                bf_tp_gt = int((dist_to_pred[g] <= bf_tolerance).sum())
                asd_sum_g2p = float(dist_to_pred[g].sum())
                asd_n_gt = n_gt

        row['bf_tp_pred'] = bf_tp_pred
        row['bf_n_pred'] = n_pred
        row['bf_tp_gt'] = bf_tp_gt
        row['bf_n_gt'] = n_gt
        row['asd_sum_pred_to_gt'] = asd_sum_p2g
        row['asd_n_pred'] = asd_n_pred
        row['asd_sum_gt_to_pred'] = asd_sum_g2p
        row['asd_n_gt'] = asd_n_gt

        rows.append(row)

    return rows, preds_np


def per_image_stats_columns(class_names, boundary_distances=(1, 2, 4)):
    """Thu tu cot CO DINH cho CSV — dung chung giua writer va bat ky noi doc
    lai (khong bat buoc, pandas doc theo ten cot tu header, nhung giu thu tu
    nay giup file de doc bang mat khi mo thu cong). boundary_distances PHAI
    khop dung gia tri da truyen cho compute_per_image_stats() cung lan chay."""
    cols = ['image', 'n_valid_px']
    for cname in class_names:
        cols += [f'inter_{cname}', f'union_{cname}']
    for cname in class_names:
        for d in boundary_distances:
            cols += [f'biou_inter_{cname}_d{d}', f'biou_union_{cname}_d{d}']
    cols += ['bf_tp_pred', 'bf_n_pred', 'bf_tp_gt', 'bf_n_gt']
    cols += ['asd_sum_pred_to_gt', 'asd_n_pred', 'asd_sum_gt_to_pred', 'asd_n_gt']
    cols += ['has_gt_boundary', 'has_pred_boundary']
    for cname in class_names:
        cols.append(f'class_present_{cname}')
    return cols


def save_pred_masks(preds_np: np.ndarray, image_names, dump_dir: str):
    """Luu tung anh trong batch ra PNG uint8, gia tri 0..num_classes-1,
    KHONG palette (mode 'L'), cung ten file goc (muc 2.4)."""
    from PIL import Image
    os.makedirs(dump_dir, exist_ok=True)
    for name, arr in zip(image_names, preds_np):
        Image.fromarray(arr.astype(np.uint8), mode='L').save(os.path.join(dump_dir, name + '.png'))
