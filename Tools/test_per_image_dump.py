"""Tools/test_per_image_dump.py — Tester cho Tools/per_image_dump.py.

Chay bang du lieu SYNTHETIC (khong can checkpoint/dataset that) — xac nhan
cot loi cua Phan A: cong don cac cot tho (inter/union, biou_inter/union,
bf_*, asd_*) tra ve tu compute_per_image_stats() qua TOAN BO anh trong batch
phai TAI LAP CHINH XAC (khong sai so, cung phep tinh so nguyen/so thuc) ket
qua cua SegmentationMetrics.compute() va BoundaryMetrics.compute() tren CHINH
batch do — day la dieu kien tien quyet de Tools/bootstrap_boundary_ci.py tin
tuong duoc cac cot "thong ke du" nay dai dien dung cho duong tinh cu.

Chay: `python Tools/test_per_image_dump.py` tu repo root.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.utils.metrics import SegmentationMetrics
from src.utils.boundary_metrics import BoundaryMetrics
from Tools.per_image_dump import compute_per_image_stats, per_image_stats_columns, save_pred_masks


IGNORE = OpenEarthMapDataset.IGNORE_INDEX
CLASS_NAMES = OpenEarthMapDataset.CLASSES
NUM_CLASSES = len(CLASS_NAMES)
BOUNDARY_DISTANCES = (1, 2, 4)
BF_TOLERANCE = 2
CONNECTIVITY = 4
DILATION_RADIUS = 0


def make_synthetic_batch(seed, B=5, H=48, W=40, ignore_frac=0.05):
    """Batch gia lap: targets ngau nhien 0..8 (+ 1 vai vung ignore), logits =
    "gan dung" targets (tron mot phan de prediction # target o vai pixel, tao
    ca truong hop dung/sai/bien lech) — du de moi khoi thong ke (IoU, BIoU,
    BF, ASD, co rong) deu co gia tri khac 0/1 tam thuong."""
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, NUM_CLASSES, size=(B, H, W)).astype(np.int64)

    ignore_mask = rng.random((B, H, W)) < ignore_frac
    targets_with_ignore = targets.copy()
    targets_with_ignore[ignore_mask] = IGNORE

    # Prediction: bat dau tu targets, roi lam nhieu ~30% pixel (doi sang lop
    # khac ngau nhien) de tao loi thuc su (khong trung khop hoan toan).
    preds = targets.copy()
    flip_mask = rng.random((B, H, W)) < 0.3
    random_other = rng.integers(0, NUM_CLASSES, size=(B, H, W)).astype(np.int64)
    preds = np.where(flip_mask, random_other, preds)

    onehot = torch.nn.functional.one_hot(torch.from_numpy(preds), NUM_CLASSES)
    logits = (onehot.permute(0, 3, 1, 2).float() * 10.0 - 5.0)
    targets_t = torch.from_numpy(targets_with_ignore)
    return logits, targets_t


def aggregate_from_rows(rows, class_names, boundary_distances):
    """Cong don CA HAI quy uoc tu cac hang per-image — dung ham nay de doi
    chieu voi SegmentationMetrics/BoundaryMetrics (von dung quy uoc MICRO:
    cong tu so/mau so qua toan dataset roi moi chia — xem docstring cua 2
    class do)."""
    n = len(rows)

    # mIoU-9 kieu SegmentationMetrics: chi cac lop co (inter+union-inter... )
    # thuc chat can tp/fp/fn nhung inter/union da du: union = tp+fp+fn,
    # inter = tp. valid_classes = (tp+fp+fn) > 0 <=> union_sum > 0.
    per_class_iou = []
    valid_classes = []
    for cname in class_names:
        inter_sum = sum(r[f'inter_{cname}'] for r in rows)
        union_sum = sum(r[f'union_{cname}'] for r in rows)
        valid_classes.append(union_sum > 0)
        per_class_iou.append(inter_sum / (union_sum + 1e-10) if union_sum > 0 else 0.0)
    miou = float(np.mean([iou for iou, v in zip(per_class_iou, valid_classes) if v]))

    biou_by_d = {}
    for d in boundary_distances:
        per_class = []
        valid = []
        for cname in class_names:
            inter_sum = sum(r[f'biou_inter_{cname}_d{d}'] for r in rows)
            union_sum = sum(r[f'biou_union_{cname}_d{d}'] for r in rows)
            valid.append(union_sum > 0)
            per_class.append(inter_sum / union_sum if union_sum > 0 else float('nan'))
        vals = [v for v, ok in zip(per_class, valid) if ok]
        biou_by_d[d] = float(np.mean(vals)) if vals else float('nan')

    bf_tp_pred = sum(r['bf_tp_pred'] for r in rows)
    bf_n_pred = sum(r['bf_n_pred'] for r in rows)
    bf_tp_gt = sum(r['bf_tp_gt'] for r in rows)
    bf_n_gt = sum(r['bf_n_gt'] for r in rows)
    precision = bf_tp_pred / bf_n_pred if bf_n_pred > 0 else 0.0
    recall = bf_tp_gt / bf_n_gt if bf_n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    asd_sum_p2g = sum(r['asd_sum_pred_to_gt'] for r in rows)
    asd_n_pred = sum(r['asd_n_pred'] for r in rows)
    asd_sum_g2p = sum(r['asd_sum_gt_to_pred'] for r in rows)
    asd_n_gt = sum(r['asd_n_gt'] for r in rows)
    d_p2g = asd_sum_p2g / asd_n_pred if asd_n_pred > 0 else float('nan')
    d_g2p = asd_sum_g2p / asd_n_gt if asd_n_gt > 0 else float('nan')
    valid_d = [x for x in (d_p2g, d_g2p) if not np.isnan(x)]
    asd = float(np.mean(valid_d)) if valid_d else float('nan')

    assert n == len(rows)
    return {
        'mIoU': miou, 'per_class_iou': per_class_iou,
        'biou': biou_by_d,
        'bf_precision': precision, 'bf_recall': recall, 'bf_score': f1,
        'asd_pred_to_gt': d_p2g, 'asd_gt_to_pred': d_g2p, 'asd': asd,
    }


def run_case(seed):
    logits, targets = make_synthetic_batch(seed)
    image_names = [f'img_{seed}_{i}' for i in range(logits.shape[0])]

    seg_metrics = SegmentationMetrics(NUM_CLASSES, IGNORE)
    seg_metrics.update(logits, targets)
    seg_result = seg_metrics.compute()

    boundary_metrics = BoundaryMetrics(
        NUM_CLASSES, IGNORE, boundary_distances=BOUNDARY_DISTANCES,
        bf_tolerance=BF_TOLERANCE, connectivity=CONNECTIVITY, dilation_radius=DILATION_RADIUS)
    boundary_metrics.update(logits, targets)
    boundary_result = boundary_metrics.compute()

    rows, preds_np = compute_per_image_stats(
        logits, targets, image_names, CLASS_NAMES, IGNORE,
        CONNECTIVITY, DILATION_RADIUS, BOUNDARY_DISTANCES, BF_TOLERANCE)

    assert len(rows) == logits.shape[0]
    assert preds_np.shape == (logits.shape[0], logits.shape[2], logits.shape[3])
    assert [r['image'] for r in rows] == image_names, "thu tu/ten anh phai duoc giu nguyen"

    recomputed = aggregate_from_rows(rows, CLASS_NAMES, BOUNDARY_DISTANCES)

    assert abs(recomputed['mIoU'] - seg_result['mIoU']) < 1e-9, (
        f"mIoU tai lap tu per-image lech: {recomputed['mIoU']} vs {seg_result['mIoU']}")
    for i, cname in enumerate(CLASS_NAMES):
        got, exp = recomputed['per_class_iou'][i], seg_result['per_class_iou'][i]
        assert abs(got - exp) < 1e-9, f"per_class_iou[{cname}] lech: {got} vs {exp}"

    for d in BOUNDARY_DISTANCES:
        got = recomputed['biou'][d]
        exp = boundary_result[f'boundary_iou_d{d}']
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert abs(got - exp) < 1e-9, f"boundary_iou_d{d} lech: {got} vs {exp}"

    for key in ('bf_precision', 'bf_recall', 'bf_score'):
        assert abs(recomputed[key] - boundary_result[key]) < 1e-9, (
            f"{key} lech: {recomputed[key]} vs {boundary_result[key]}")

    for key in ('asd_pred_to_gt', 'asd_gt_to_pred', 'asd'):
        got, exp = recomputed[key], boundary_result[key]
        if np.isnan(exp):
            assert np.isnan(got)
        else:
            assert abs(got - exp) < 1e-9, f"{key} lech: {got} vs {exp}"

    # Cot co trang thai rong (muc 2.2) phai nhat quan voi tu so/mau so.
    for r in rows:
        for cname in CLASS_NAMES:
            if r[f'union_{cname}'] == 0:
                assert r[f'inter_{cname}'] == 0
                assert r[f'class_present_{cname}'] == 0
        if r['bf_n_gt'] == 0:
            assert r['has_gt_boundary'] == 0
        else:
            assert r['has_gt_boundary'] == 1
        if r['bf_n_pred'] == 0:
            assert r['has_pred_boundary'] == 0
        else:
            assert r['has_pred_boundary'] == 1

    return rows, preds_np, image_names


def test_reconstructs_aggregate_multiple_seeds():
    print("=== Test 1: tai lap chinh xac aggregate tren nhieu batch ngau nhien ===")
    for seed in range(8):
        run_case(seed)
    print(f"PASS ({8} batch synthetic, moi batch 5 anh 48x40, dung sai 1e-9)\n")


def test_all_ignore_image_no_crash():
    print("=== Test 2: 1 anh toan ignore trong batch — khong NaN/crash o cot tho ===")
    B, H, W = 3, 20, 16
    rng = np.random.default_rng(42)
    targets = rng.integers(0, NUM_CLASSES, size=(B, H, W)).astype(np.int64)
    targets[1] = IGNORE  # anh giua toan ignore
    preds = targets.copy()
    preds[0] = rng.integers(0, NUM_CLASSES, size=(H, W))

    onehot = torch.nn.functional.one_hot(torch.from_numpy(np.clip(preds, 0, NUM_CLASSES - 1)), NUM_CLASSES)
    logits = onehot.permute(0, 3, 1, 2).float() * 10.0 - 5.0
    targets_t = torch.from_numpy(targets)
    names = ['a', 'ignore_img', 'c']

    rows, preds_np = compute_per_image_stats(
        logits, targets_t, names, CLASS_NAMES, IGNORE,
        CONNECTIVITY, DILATION_RADIUS, BOUNDARY_DISTANCES, BF_TOLERANCE)

    ignore_row = rows[1]
    assert ignore_row['n_valid_px'] == 0
    assert ignore_row['bf_n_gt'] == 0
    assert ignore_row['has_gt_boundary'] == 0
    for cname in CLASS_NAMES:
        assert ignore_row[f'union_{cname}'] == 0
        assert ignore_row[f'inter_{cname}'] == 0
        assert ignore_row[f'class_present_{cname}'] == 0
    assert not any(np.isnan(v) for v in ignore_row.values() if isinstance(v, float))
    print("PASS (anh toan ignore cho toan 0, khong NaN)\n")


def test_column_schema_matches_spec_count():
    print("=== Test 3: so cot CSV dung dac ta muc 2.2 (9 lop, 3 nguong d) ===")
    cols = per_image_stats_columns(CLASS_NAMES, BOUNDARY_DISTANCES)
    expected_n = 2 + 9 * 2 + 9 * 3 * 2 + 4 + 4 + 2 + 9
    assert len(cols) == expected_n, f"so cot {len(cols)} != ky vong {expected_n}"
    assert len(cols) == len(set(cols)), "co cot trung ten"
    assert cols[0] == 'image' and cols[1] == 'n_valid_px'
    print(f"PASS ({len(cols)} cot, khong trung ten)\n")


def test_save_pred_masks_roundtrip(tmp_dir=None):
    print("=== Test 4: luu mask PNG roundtrip dung gia tri, khong palette ===")
    import shutil
    import tempfile
    from PIL import Image

    rows, preds_np, image_names = run_case(seed=7)
    tmp_dir = tempfile.mkdtemp(prefix='per_image_dump_test_')
    try:
        save_pred_masks(preds_np, image_names, tmp_dir)
        for name, arr in zip(image_names, preds_np):
            path = os.path.join(tmp_dir, name + '.png')
            assert os.path.exists(path)
            img = Image.open(path)
            assert img.mode == 'L', f"mode phai la 'L' (grayscale, khong palette), duoc {img.mode}"
            loaded = np.array(img)
            assert np.array_equal(loaded, arr.astype(np.uint8)), "gia tri PNG doc lai phai KHOP het prediction"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"PASS ({len(image_names)} mask PNG, mode 'L', gia tri khop 100%)\n")


if __name__ == '__main__':
    test_reconstructs_aggregate_multiple_seeds()
    test_all_ignore_image_no_crash()
    test_column_schema_matches_spec_count()
    test_save_pred_masks_roundtrip()
    print("Tat ca self-test Tools/per_image_dump.py PASS.")
