"""Tools/bootstrap_boundary_ci.py — Giai doan 2 (docs/spec-per-image-bootstrap-
eval-v2-ban-giao-claude-code.md Phan C): doc CSV thong ke du da dump boi
Tools/eval_boundary_metrics.py (--dump-per-image-stats), tai lap so gop, do
chinh sach anh rong bang du lieu, chay bootstrap ghep cap cho cac cap
checkpoint, xuat bootstrap_ci.csv/.md.

File MOI, DOC LAP voi Tools/eval_boundary_metrics.py — CHI DOC file CSV da
dump, KHONG chay inference, KHONG dung checkpoint/.pth nao. Chay trong vai
giay, lap lai bao nhieu lan cung duoc (muc 1.4 spec).

Quy uoc gop (micro/macro) — muc 1.1/4.1 spec:
  micro:  X = sum_i(num_i) / sum_i(den_i)                 (cong tu so/mau so
          THO qua toan dataset roi moi chia — DUNG cach SegmentationMetrics/
          BoundaryMetrics trong src/utils/*.py dang tinh, da doc code xac
          nhan — 2 accumulator do LA pixel-count accumulator, khong bao gio
          average-theo-anh).
  macro:  X = mean_i(num_i / den_i)                        (trung binh ti le
          tung anh — anh nao mau so=0 se can 1 trong 4 chinh sach thay the
          "empty_image_policy", muc 4.2).

Ca hai quy uoc deu duoc tinh o day; --checkpoint co --reference kem theo se
tu dong DO xem quy uoc/chinh sach nao tai lap dung so da cong bo (muc 4.2),
GHI RO trong output — khong doan.

Bootstrap ghep cap (muc 4.3) dung 1 phep nhan ma tran vector-hoa (khong vong
lap Python 10000 lan) de nhanh: voi B lan lap, dem so lan moi anh duoc chon
lai (counts, hinh (B, n_images)) tu idx = rng.integers(...), roi TICH VOI
ma tran thong ke tho (n_images, n_cot) — cho ca "micro" (tong tu so/mau so)
lan "macro" (trung binh ti le da ap chinh sach, quy ve trung binh co trong
so vi moi hang cua counts luon cong den dung n_images).

Tu-test: `python Tools/test_bootstrap_boundary_ci.py`.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset


CLASS_NAMES = list(OpenEarthMapDataset.CLASSES)
NUM_CLASSES = len(CLASS_NAMES)
RATIO01_KIND = 'ratio01'   # gia tri trong [0,1] — IoU/BIoU/precision/recall: chinh sach zero/one hop ly
DIST_KIND = 'dist'         # khoang cach (ASD) — chinh sach zero/diag hop ly, KHONG dung 'one'
DEFAULT_DIAG_VALUE = float(np.hypot(1024, 1024))  # get_val_transforms() resize mac dinh 1024x1024
EMPTY_POLICIES = ('skip', 'zero', 'one', 'diag')

# Muc 4.4 spec — 6 cap bat buoc. Tuple = (pair_id, label_MINUEND, label_TRU).
# delta = gia_tri(minuend) - gia_tri(tru). Nhan (ten hien thi) chi de doc,
# khong anh huong tinh toan.
#
# ⚠ CAP #6: bang 4.4 dat ten "Static s19@36k - Static s86@36k" (giong quy uoc
# "A - B" cua cac cap khac), NHUNG muc 7.5 CHOT huong tinh THUC SU dung de
# bao cao la NGUOC LAI: "+0.5343px theo chieu s86 - s19; giu dung dau nay,
# dung dao". Uu tien muc 7.5 (noi dung phan tich, khong phai ten bang) —
# minuend = s86, tru = s19. Neu doc lai spec va thay nguoc, DAY LA CHO CAN
# doi chieu dau tien.
DEFAULT_PAIRS = [
    ('pair1_static_s19_vs_bce04', 'run3_static_s19', 'run2b_bce04'),
    ('pair2_static_s19_vs_baseline', 'run3_static_s19', 'run1_baseline'),
    ('pair3_run4_vs_baseline', 'run4_dynamic', 'run1_baseline'),
    ('pair4_run4_vs_static_s19', 'run4_dynamic', 'run3_static_s19'),
    ('pair5_run3b_vs_bce04', 'run3b_aff02', 'run2b_bce04'),
    ('pair6_static_s86_vs_s19', 'run5_static_s86_best', 'run3_static_s19'),
]

# Muc 4.5 spec — cac dai luong PHAI tinh CI cho MOI cap.
HIGHLIGHT_CLASSES = ('Bareland', 'Water', 'Agriculture')


# ═══════════════════════════ Doc CSV / kiem tra schema ══════════════════════

def load_per_image_stats(csv_path, boundary_distances=(1, 2, 4)):
    df = pd.read_csv(csv_path)
    required = per_image_stats_columns(boundary_distances)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: thieu cot {missing} — file co dung dinh dang "
                          f"Tools/per_image_dump.py xuat ra khong?")
    return df


def per_image_stats_columns(boundary_distances=(1, 2, 4)):
    """Doc lap voi Tools/per_image_dump.py (khong import de giu 2 giai doan
    doc lap nhu spec yeu cau) — nhung PHAI tao ra dung cung danh sach cot."""
    cols = ['image', 'n_valid_px']
    for cname in CLASS_NAMES:
        cols += [f'inter_{cname}', f'union_{cname}']
    for cname in CLASS_NAMES:
        for d in boundary_distances:
            cols += [f'biou_inter_{cname}_d{d}', f'biou_union_{cname}_d{d}']
    cols += ['bf_tp_pred', 'bf_n_pred', 'bf_tp_gt', 'bf_n_gt']
    cols += ['asd_sum_pred_to_gt', 'asd_n_pred', 'asd_sum_gt_to_pred', 'asd_n_gt']
    cols += ['has_gt_boundary', 'has_pred_boundary']
    for cname in CLASS_NAMES:
        cols.append(f'class_present_{cname}')
    return cols


def assert_same_image_order(dfs_by_label):
    """Gate 5.7 — danh sach `image` cua MOI checkpoint phai GIONG HET nhau
    ca noi dung lan thu tu. Assert cung, khong tu sap xep ngam."""
    labels = list(dfs_by_label.keys())
    if len(labels) < 2:
        return
    ref_label = labels[0]
    ref_images = dfs_by_label[ref_label]['image'].tolist()
    for label in labels[1:]:
        images = dfs_by_label[label]['image'].tolist()
        if images != ref_images:
            if sorted(images) == sorted(ref_images):
                raise AssertionError(
                    f"GATE 5.7 FAIL: '{label}' co CUNG tap anh voi '{ref_label}' nhung KHAC THU TU — "
                    f"khong duoc tu sap xep ngam, kiem tra lai qua trinh dump (val split/DataLoader "
                    f"shuffle).")
            raise AssertionError(
                f"GATE 5.7 FAIL: '{label}' co danh sach anh KHAC '{ref_label}' — "
                f"kiem tra 2 checkpoint co dung chay tren cung 1 val split khong.")


# ═══════════════════════════ Gop theo quy uoc (point estimate) ═════════════

def _ratio_micro(df, num_col, den_col, zero_fallback):
    num = float(df[num_col].sum())
    den = float(df[den_col].sum())
    return (num / den) if den > 0 else zero_fallback


def _effective_ratio_per_image(df, num_col, den_col, kind, empty_policy, diag_value):
    """Vector 1 gia tri/anh — cho MACRO: ti le that neu mau so>0, hoac gia tri
    thay the theo empty_policy neu mau so=0 (muc 4.2). 'skip' -> NaN (loai
    khoi trung binh o buoc sau)."""
    nums = df[num_col].to_numpy(dtype=np.float64)
    dens = df[den_col].to_numpy(dtype=np.float64)
    vals = np.full(len(df), np.nan)
    valid = dens > 0
    vals[valid] = nums[valid] / dens[valid]
    invalid = ~valid
    if not invalid.any() or empty_policy == 'skip':
        return vals
    if empty_policy == 'zero':
        vals[invalid] = 0.0
    elif empty_policy == 'one':
        vals[invalid] = 1.0 if kind == RATIO01_KIND else 0.0
    elif empty_policy == 'diag':
        vals[invalid] = diag_value if kind == DIST_KIND else 0.0
    else:
        raise ValueError(f"empty_policy khong hop le: {empty_policy}")
    return vals


def aggregate_point_estimate(df, convention, empty_policy, boundary_distances=(1, 2, 4),
                              diag_value=DEFAULT_DIAG_VALUE):
    """Tinh TOAN BO dai luong muc 4.1/4.5 tren 1 checkpoint, 1 lan (khong
    bootstrap) — dung de doi chieu gate 5.1-5.5 va lam 'observed_delta'."""
    def agg(num_col, den_col, kind, zero_fallback=0.0):
        if convention == 'micro':
            return _ratio_micro(df, num_col, den_col, zero_fallback)
        vals = _effective_ratio_per_image(df, num_col, den_col, kind, empty_policy, diag_value)
        return float(np.nanmean(vals)) if not np.all(np.isnan(vals)) else float('nan')

    result = {}
    per_class_iou = {}
    class_valid = {}
    for c in CLASS_NAMES:
        class_valid[c] = bool((df[f'union_{c}'] > 0).any())
        per_class_iou[c] = agg(f'inter_{c}', f'union_{c}', RATIO01_KIND)
    result['per_class_iou'] = per_class_iou

    def masked_mean(values, classes_subset):
        vals = [values[c] for c in classes_subset if class_valid[c] and not np.isnan(values[c])]
        return float(np.mean(vals)) if vals else float('nan')

    result['mIoU_9'] = masked_mean(per_class_iou, CLASS_NAMES)
    result['mIoU_8'] = masked_mean(per_class_iou, CLASS_NAMES[1:])
    result['mIoU_no_bareland'] = masked_mean(per_class_iou, [c for c in CLASS_NAMES if c != 'Bareland'])
    for c in HIGHLIGHT_CLASSES:
        result[f'iou_{c}'] = per_class_iou[c]

    for d in boundary_distances:
        biou_per_class = {}
        biou_valid = {}
        for c in CLASS_NAMES:
            biou_valid[c] = bool((df[f'biou_union_{c}_d{d}'] > 0).any())
            biou_per_class[c] = agg(f'biou_inter_{c}_d{d}', f'biou_union_{c}_d{d}', RATIO01_KIND)
        vals8 = [biou_per_class[c] for c in CLASS_NAMES[1:]
                 if biou_valid[c] and not np.isnan(biou_per_class[c])]
        result[f'biou_8_d{d}'] = float(np.mean(vals8)) if vals8 else float('nan')
        result[f'biou_background_d{d}'] = biou_per_class[CLASS_NAMES[0]]
        result[f'biou_per_class_d{d}'] = biou_per_class

    precision = agg('bf_tp_pred', 'bf_n_pred', RATIO01_KIND, zero_fallback=0.0)
    recall = agg('bf_tp_gt', 'bf_n_gt', RATIO01_KIND, zero_fallback=0.0)
    result['bf_precision'] = precision
    result['bf_recall'] = recall
    result['bf_score'] = (2 * precision * recall / (precision + recall)
                           if (precision + recall) > 0 else 0.0)

    d_p2g = agg('asd_sum_pred_to_gt', 'asd_n_pred', DIST_KIND, zero_fallback=float('nan'))
    d_g2p = agg('asd_sum_gt_to_pred', 'asd_n_gt', DIST_KIND, zero_fallback=float('nan'))
    result['asd_pred_to_gt'] = d_p2g
    result['asd_gt_to_pred'] = d_g2p
    valid_d = [x for x in (d_p2g, d_g2p) if not np.isnan(x)]
    result['asd'] = float(np.mean(valid_d)) if valid_d else float('nan')

    return result


# ═══════════════════ Do chinh sach anh rong + quy uoc (muc 4.2) ════════════

def load_reference(path):
    """Chap nhan 2 dang: (a) JSON goc do Tools/eval_boundary_metrics.py
    --output sinh ra (co khoa 'segmentation'/'boundary'), (b) JSON "phang"
    tu viet tay voi dung ten khoa cua aggregate_point_estimate() (vd file
    dung de doi chieu truc tiep bang so muc 5.1-5.4 cua spec)."""
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    if 'segmentation' in raw and 'boundary' in raw:
        seg, bnd = raw['segmentation'], raw['boundary']
        per_class = seg.get('per_class_iou')
        flat = {
            'mIoU_9': seg.get('mIoU'),
            'bf_precision': bnd.get('bf_precision'), 'bf_recall': bnd.get('bf_recall'),
            'bf_score': bnd.get('bf_score'),
            'asd_pred_to_gt': bnd.get('asd_pred_to_gt'), 'asd_gt_to_pred': bnd.get('asd_gt_to_pred'),
            'asd': bnd.get('asd'),
        }
        if per_class:
            for c, v in zip(CLASS_NAMES, per_class):
                flat[f'iou_{c}'] = v
        for d in raw.get('boundary_distances', (1, 2, 4)):
            pc = bnd.get(f'boundary_iou_d{d}_per_class')
            if pc:
                flat[f'biou_background_d{d}'] = pc[0]
                vals8 = [v for v in pc[1:] if not (isinstance(v, float) and np.isnan(v))]
                flat[f'biou_8_d{d}'] = float(np.mean(vals8)) if vals8 else float('nan')
        return flat
    return raw


def detect_convention_and_policy(df, reference, boundary_distances=(1, 2, 4),
                                  tol=1e-6, diag_value=DEFAULT_DIAG_VALUE):
    """Muc 4.2 spec — thu 'micro' (khong can policy) roi 'macro' x 4 chinh
    sach, tra ve candidate DAU TIEN tai lap DUNG moi truong trong `reference`
    trong nguong `tol`. Tra ve (convention, policy, max_abs_diff, details) —
    None neu khong candidate nao khop."""
    candidates = [('micro', 'skip')] + [('macro', p) for p in EMPTY_POLICIES]
    best = None
    for convention, policy in candidates:
        computed = aggregate_point_estimate(df, convention, policy, boundary_distances, diag_value)
        diffs = {}
        ok = True
        for key, exp in reference.items():
            if exp is None or (isinstance(exp, float) and np.isnan(exp)):
                continue
            got = computed.get(key)
            if got is None or (isinstance(got, float) and np.isnan(got)):
                ok = False
                diffs[key] = float('nan')
                continue
            diff = abs(got - exp)
            diffs[key] = diff
            if diff >= tol:
                ok = False
        if ok and diffs:
            return convention, policy, max(diffs.values()) if diffs else 0.0, diffs
        if best is None or (diffs and max(diffs.values()) < best[2]):
            best = (convention, policy, max(diffs.values()) if diffs else float('inf'), diffs)
    return None if best is None else (*best[:3], best[3])


# ═══════════════════════════ Bootstrap ghep cap (muc 4.3) ═══════════════════

def build_raw_matrix(df, boundary_distances=(1, 2, 4)):
    """Tra ve (mat, col_index) — mat: (n_images, n_raw_cols) float64, moi cot
    la 1 trong cac cot tho (inter_c/union_c, biou_inter/union, bf_*, asd_*)
    can cho quy uoc MICRO. col_index: dict ten_cot -> chi so cot trong mat."""
    raw_cols = []
    for c in CLASS_NAMES:
        raw_cols += [f'inter_{c}', f'union_{c}']
    for c in CLASS_NAMES:
        for d in boundary_distances:
            raw_cols += [f'biou_inter_{c}_d{d}', f'biou_union_{c}_d{d}']
    raw_cols += ['bf_tp_pred', 'bf_n_pred', 'bf_tp_gt', 'bf_n_gt']
    raw_cols += ['asd_sum_pred_to_gt', 'asd_n_pred', 'asd_sum_gt_to_pred', 'asd_n_gt']
    mat = df[raw_cols].to_numpy(dtype=np.float64)
    col_index = {name: i for i, name in enumerate(raw_cols)}
    return mat, col_index


def build_effective_ratio_matrix(df, empty_policy, boundary_distances=(1, 2, 4),
                                  diag_value=DEFAULT_DIAG_VALUE):
    """Cho MACRO — tra ve (mat, col_index): mat (n_images, n_quantity) la ti
    le HIEU DUNG (da ap chinh sach) cho tung anh, tung dai luong co so (IoU
    tung lop, BIoU tung lop*d, bf precision/recall, asd 2 chieu). Bootstrap
    tren macro = trung binh co trong so cua CHINH cac cot nay (khong can gop
    lai tu so/mau so nua vi da quy ve ti le)."""
    quantities = []
    cols = {}
    for c in CLASS_NAMES:
        vals = _effective_ratio_per_image(df, f'inter_{c}', f'union_{c}', RATIO01_KIND,
                                           empty_policy, diag_value)
        cols[f'iou_{c}'] = vals
    for d in boundary_distances:
        for c in CLASS_NAMES:
            vals = _effective_ratio_per_image(
                df, f'biou_inter_{c}_d{d}', f'biou_union_{c}_d{d}', RATIO01_KIND,
                empty_policy, diag_value)
            cols[f'biou_{c}_d{d}'] = vals
    cols['bf_precision'] = _effective_ratio_per_image(df, 'bf_tp_pred', 'bf_n_pred', RATIO01_KIND,
                                                        empty_policy, diag_value)
    cols['bf_recall'] = _effective_ratio_per_image(df, 'bf_tp_gt', 'bf_n_gt', RATIO01_KIND,
                                                     empty_policy, diag_value)
    cols['asd_pred_to_gt'] = _effective_ratio_per_image(df, 'asd_sum_pred_to_gt', 'asd_n_pred', DIST_KIND,
                                                          empty_policy, diag_value)
    cols['asd_gt_to_pred'] = _effective_ratio_per_image(df, 'asd_sum_gt_to_pred', 'asd_n_gt', DIST_KIND,
                                                          empty_policy, diag_value)
    names = list(cols.keys())
    mat = np.stack([cols[n] for n in names], axis=1)
    # NaN ('skip' voi mau so=0 o TAT CA lan resample chua chac xay ra, nhung
    # tung anh rieng le co the la NaN) — thay NaN=0 va dem so "hop le" rieng
    # de bootstrap tinh trung binh co dieu kien dung (xem run_bootstrap()).
    valid_mask = ~np.isnan(mat)
    mat_filled = np.where(valid_mask, mat, 0.0)
    return mat_filled, valid_mask, {n: i for i, n in enumerate(names)}


def _resample_counts(idx, n_images):
    """idx: (B, n_images) chi so da resample (rng.integers). Tra ve counts
    (B, n_images) so lan moi anh goc duoc chon trong tung lan lap — dung de
    quy bootstrap ve 1 phep nhan ma tran (khong vong lap Python qua B)."""
    B = idx.shape[0]
    counts = np.zeros((B, n_images), dtype=np.float64)
    rows = np.repeat(np.arange(B), n_images)
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    return counts


def _micro_composites_from_sums(sums, col_index, boundary_distances):
    """sums: (B, n_raw_cols) tong tu so/mau so THO cho tung lan lap (da nhan
    ma tran voi counts) -> dict ten_dai_luong -> (B,) gia tri da chia, ap
    dung DUNG cong thuc/valid-class cua aggregate_point_estimate (macro=False)
    nhung vector hoa qua B lan lap."""
    def col(name):
        return sums[:, col_index[name]]

    B = sums.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = {}
        valid = {}
        for c in CLASS_NAMES:
            num, den = col(f'inter_{c}'), col(f'union_{c}')
            iou[c] = np.where(den > 0, num / den, np.nan)
            valid[c] = den > 0

        def masked_mean(values, classes_subset):
            stack = np.stack([values[c] for c in classes_subset], axis=1)          # (B, k)
            mask = np.stack([valid[c] for c in classes_subset], axis=1)            # (B, k)
            ma = np.ma.array(stack, mask=~mask)
            out = ma.mean(axis=1)
            return np.ma.filled(out, np.nan)

        out = {}
        out['mIoU_9'] = masked_mean(iou, CLASS_NAMES)
        out['mIoU_8'] = masked_mean(iou, CLASS_NAMES[1:])
        out['mIoU_no_bareland'] = masked_mean(iou, [c for c in CLASS_NAMES if c != 'Bareland'])
        for c in HIGHLIGHT_CLASSES:
            out[f'iou_{c}'] = iou[c]

        for d in boundary_distances:
            biou = {}
            biou_valid = {}
            for c in CLASS_NAMES:
                num, den = col(f'biou_inter_{c}_d{d}'), col(f'biou_union_{c}_d{d}')
                biou[c] = np.where(den > 0, num / den, np.nan)
                biou_valid[c] = den > 0
            stack = np.stack([biou[c] for c in CLASS_NAMES[1:]], axis=1)
            mask = np.stack([biou_valid[c] for c in CLASS_NAMES[1:]], axis=1)
            ma = np.ma.array(stack, mask=~mask)
            out[f'biou_8_d{d}'] = np.ma.filled(ma.mean(axis=1), np.nan)
            out[f'biou_background_d{d}'] = biou[CLASS_NAMES[0]]

        bf_n_pred, bf_n_gt = col('bf_n_pred'), col('bf_n_gt')
        precision = np.where(bf_n_pred > 0, col('bf_tp_pred') / bf_n_pred, 0.0)
        recall = np.where(bf_n_gt > 0, col('bf_tp_gt') / bf_n_gt, 0.0)
        out['bf_precision'] = precision
        out['bf_recall'] = recall
        pr_sum = precision + recall
        out['bf_score'] = np.where(pr_sum > 0, 2 * precision * recall / np.where(pr_sum > 0, pr_sum, 1), 0.0)

        asd_n_pred, asd_n_gt = col('asd_n_pred'), col('asd_n_gt')
        d_p2g = np.where(asd_n_pred > 0, col('asd_sum_pred_to_gt') / asd_n_pred, np.nan)
        d_g2p = np.where(asd_n_gt > 0, col('asd_sum_gt_to_pred') / asd_n_gt, np.nan)
        out['asd_pred_to_gt'] = d_p2g
        out['asd_gt_to_pred'] = d_g2p
        stack_asd = np.stack([d_p2g, d_g2p], axis=1)
        out['asd'] = np.nanmean(np.where(np.isnan(stack_asd), np.nan, stack_asd), axis=1)

    assert out['mIoU_9'].shape == (B,)
    return out


def _macro_composites_from_weighted(weighted, col_index, boundary_distances):
    """weighted: (B, n_quantity) = (counts @ mat_filled) / n_images — trung
    binh co trong so cua ti le hieu dung (da ap chinh sach) — day CHINH LA
    gia tri macro cho tung dai luong co so; ghep thanh cac dai luong composite
    (mIoU-9/8/no-Bareland, BIoU-8) bang trung binh don gian giua cac dai
    luong co so tuong ung (khong can valid-mask nua vi 'zero'/'one'/'diag' da
    dam bao moi anh co gia tri huu han — con 'skip' co the con NaN cho 1 lop
    NEU lop do vang mat trong toan bo lan resample, xu ly bang nanmean)."""
    def col(name):
        return weighted[:, col_index[name]]

    out = {}
    iou = {c: col(f'iou_{c}') for c in CLASS_NAMES}
    out['mIoU_9'] = np.nanmean(np.stack([iou[c] for c in CLASS_NAMES], axis=1), axis=1)
    out['mIoU_8'] = np.nanmean(np.stack([iou[c] for c in CLASS_NAMES[1:]], axis=1), axis=1)
    out['mIoU_no_bareland'] = np.nanmean(
        np.stack([iou[c] for c in CLASS_NAMES if c != 'Bareland'], axis=1), axis=1)
    for c in HIGHLIGHT_CLASSES:
        out[f'iou_{c}'] = iou[c]

    for d in boundary_distances:
        biou = {c: col(f'biou_{c}_d{d}') for c in CLASS_NAMES}
        out[f'biou_8_d{d}'] = np.nanmean(np.stack([biou[c] for c in CLASS_NAMES[1:]], axis=1), axis=1)
        out[f'biou_background_d{d}'] = biou[CLASS_NAMES[0]]

    precision, recall = col('bf_precision'), col('bf_recall')
    out['bf_precision'] = precision
    out['bf_recall'] = recall
    pr_sum = precision + recall
    out['bf_score'] = np.where(pr_sum > 0, 2 * precision * recall / np.where(pr_sum > 0, pr_sum, 1), 0.0)

    d_p2g, d_g2p = col('asd_pred_to_gt'), col('asd_gt_to_pred')
    out['asd_pred_to_gt'] = d_p2g
    out['asd_gt_to_pred'] = d_g2p
    out['asd'] = np.nanmean(np.stack([d_p2g, d_g2p], axis=1), axis=1)
    return out


def run_bootstrap(df_a, df_b, convention, empty_policy, n_boot, seed,
                   boundary_distances=(1, 2, 4), diag_value=DEFAULT_DIAG_VALUE):
    """Bootstrap GHEP CAP (muc 4.3) — CUNG idx cho ca df_a, df_b (gia dinh
    CUNG danh sach + thu tu anh, xem assert_same_image_order()). Tra ve
    (values_a, values_b): moi cai la dict ten_dai_luong -> mang (n_boot,)."""
    n_images = len(df_a)
    assert len(df_b) == n_images
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_images, size=(n_boot, n_images))
    counts = _resample_counts(idx, n_images)  # (B, n_images)

    if convention == 'micro':
        mat_a, col_index = build_raw_matrix(df_a, boundary_distances)
        mat_b, _ = build_raw_matrix(df_b, boundary_distances)
        sums_a = counts @ mat_a
        sums_b = counts @ mat_b
        values_a = _micro_composites_from_sums(sums_a, col_index, boundary_distances)
        values_b = _micro_composites_from_sums(sums_b, col_index, boundary_distances)
    else:
        mat_a, _valid_a, col_index = build_effective_ratio_matrix(
            df_a, empty_policy, boundary_distances, diag_value)
        mat_b, _valid_b, _ = build_effective_ratio_matrix(
            df_b, empty_policy, boundary_distances, diag_value)
        weighted_a = (counts @ mat_a) / n_images
        weighted_b = (counts @ mat_b) / n_images
        values_a = _macro_composites_from_weighted(weighted_a, col_index, boundary_distances)
        values_b = _macro_composites_from_weighted(weighted_b, col_index, boundary_distances)

    return values_a, values_b


QUANTITIES_FOR_CI = (
    ['asd_pred_to_gt', 'asd_gt_to_pred', 'asd', 'bf_precision', 'bf_recall', 'bf_score',
     'mIoU_9', 'mIoU_8', 'mIoU_no_bareland']
    + [f'biou_8_d{d}' for d in (1, 2, 4)]
    + [f'iou_{c}' for c in HIGHLIGHT_CLASSES]
)


def summarize_pair(values_a, values_b, observed_a, observed_b, quantities=QUANTITIES_FOR_CI):
    """Muc 4.3 spec — trung vi, CI 95% (percentile 2.5/97.5), P(Delta<0), cho
    tung dai luong. observed_a/observed_b: dict tu aggregate_point_estimate()
    (khong bootstrap) — dung lam observed_delta (khong phai trung binh cua
    cac lan bootstrap, dung diem uoc luong that)."""
    rows = []
    for q in quantities:
        delta = values_a[q] - values_b[q]
        delta_valid = delta[~np.isnan(delta)]
        n_nan = int(np.isnan(delta).sum())
        if len(delta_valid) == 0:
            rows.append({'quantity': q, 'median': float('nan'), 'ci_lo': float('nan'),
                         'ci_hi': float('nan'), 'p_delta_lt_0': float('nan'),
                         'observed_delta': float('nan'), 'n_boot_nan': n_nan})
            continue
        median = float(np.median(delta_valid))
        ci_lo, ci_hi = np.percentile(delta_valid, [2.5, 97.5])
        p_lt_0 = float(np.mean(delta_valid < 0))
        obs_delta = observed_a.get(q, float('nan')) - observed_b.get(q, float('nan'))
        rows.append({'quantity': q, 'median': median, 'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
                     'p_delta_lt_0': p_lt_0, 'observed_delta': obs_delta, 'n_boot_nan': n_nan})
    return rows


# ═══════════════════════════════ Output ═════════════════════════════════════

def write_bootstrap_ci_csv(all_rows, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(path, index=False)


def write_bootstrap_ci_md(all_rows, convention, empty_policy, n_boot, seed, path):
    lines = [
        '# Bootstrap CI ghep cap tren tap danh gia 384 anh',
        '',
        '> Nhan bat buoc (muc 7.7 spec): moi CI o day la "khoang tin cay bootstrap tren tap danh gia '
        '384 anh" — KHONG phai khoang tin cay tren lan huan luyen. No chi dong duoc cau hoi nhieu tap '
        'danh gia (chon 384 anh nao), KHONG tu dong dong cau hoi nhieu giua cac lan huan luyen/seed '
        '(xem muc 7.6 — doi chieu voi bien do lien-seed Run 5 truoc khi in bat ky CI nao vao bai).',
        '',
        f'Quy uoc gop: **{convention}**' + (f', chinh sach anh rong: **{empty_policy}**' if convention == 'macro' else ''),
        f'B (so lan bootstrap) = {n_boot}, seed = {seed} (seed cua script eval/bootstrap, '
        f'KHONG phai seed huan luyen — vd Run 5 dung seed huan luyen 86, hai khai niem khac nhau).',
        '',
        '| Cap | Dai luong | Median Delta | CI 95% | P(Delta<0) | Observed Delta |',
        '|---|---|---|---|---|---|',
    ]
    for r in all_rows:
        ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if not np.isnan(r['ci_lo']) else 'N/A'
        lines.append(
            f"| {r['pair_id']} ({r['label_minuend']} - {r['label_subtrahend']}) | {r['quantity']} | "
            f"{r['median']:.4f} | {ci} | {r['p_delta_lt_0']:.4f} | {r['observed_delta']:.4f} |")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ═══════════════════════════════════ CLI ════════════════════════════════════

def _parse_label_path(items, flag_name):
    out = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f"{flag_name} phai co dang LABEL=PATH, nhan: {item}")
        label, path = item.split('=', 1)
        out[label] = path
    return out


def _parse_pairs(items):
    pairs = []
    for i, item in enumerate(items):
        if '=' not in item:
            raise ValueError(f"--pair phai co dang MINUEND=TRU, nhan: {item}")
        a, b = item.split('=', 1)
        pairs.append((f'pair{i + 1}_{a}_vs_{b}', a, b))
    return pairs


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', action='append', default=[], required=True,
                     help='LABEL=duong_dan_per_image_stats.csv — lap lai cho moi checkpoint.')
    ap.add_argument('--reference', action='append', default=[],
                     help='(Tuy chon) LABEL=duong_dan.json — JSON goc cua eval_boundary_metrics.py '
                          '--output HOAC JSON phang tu viet — dung cho do chinh sach/quy uoc (muc 4.2) '
                          've gate 5.1-5.5. Bo qua checkpoint nao khong can doi chieu.')
    ap.add_argument('--pair', action='append', default=[],
                     help='MINUEND_LABEL=TRU_LABEL — lap lai de dinh nghia cac cap can bootstrap. '
                          'Bo qua ⇒ dung 6 cap mac dinh muc 4.4 (chi chay cap nao co du 2 nhan trong '
                          '--checkpoint).')
    ap.add_argument('--convention', choices=['auto', 'micro', 'macro'], default='auto',
                     help="'auto' (mac dinh): do tu --reference (muc 4.2), roi ve 'micro' neu khong co "
                          "reference nao (dung quy uoc THAT SU cua SegmentationMetrics/BoundaryMetrics).")
    ap.add_argument('--empty-policy', dest='empty_policy', choices=['auto'] + list(EMPTY_POLICIES),
                     default='auto', help="Chi dung khi --convention=macro (hoac auto do ra macro).")
    ap.add_argument('--gate-tol', dest='gate_tol', type=float, default=1e-6,
                     help='Nguong so sanh khi do quy uoc/chinh sach qua --reference (muc 4.2/5.1).')
    ap.add_argument('--n-boot', dest='n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=19, help='Seed CUA SCRIPT bootstrap (muc 4.3), '
                                                            'KHONG phai seed huan luyen.')
    ap.add_argument('--boundary-distances', default='1,2,4')
    ap.add_argument('--diag-value', dest='diag_value', type=float, default=DEFAULT_DIAG_VALUE,
                     help="Gia tri duong cheo anh dung cho chinh sach 'diag' (ASD) — mac dinh "
                          "sqrt(1024^2+1024^2), khop get_val_transforms() resize=1024 mac dinh cua "
                          "toan du an (moi anh val deu cung kich thuoc sau resize).")
    ap.add_argument('--output-dir', dest='output_dir', default='output/bootstrap')
    return ap.parse_args()


def main():
    args = parse_args()
    boundary_distances = tuple(int(x) for x in args.boundary_distances.split(','))

    checkpoint_paths = _parse_label_path(args.checkpoint, '--checkpoint')
    reference_paths = _parse_label_path(args.reference, '--reference')

    dfs = {label: load_per_image_stats(path, boundary_distances)
           for label, path in checkpoint_paths.items()}
    assert_same_image_order(dfs)
    print(f"Da doc {len(dfs)} checkpoint, {len(next(iter(dfs.values())))} anh/checkpoint, "
          f"danh sach anh khop nhau (gate 5.7 PASS).")

    references = {label: load_reference(path) for label, path in reference_paths.items()}

    # ── Muc 4.2: do quy uoc/chinh sach — CHUNG cho toan bo lan chay nay ──
    convention, empty_policy = args.convention, args.empty_policy
    if args.convention == 'auto':
        if references:
            first_ref_label = next(iter(references))
            detected = detect_convention_and_policy(
                dfs[first_ref_label], references[first_ref_label], boundary_distances, args.gate_tol,
                args.diag_value)
            if detected is None:
                raise RuntimeError(
                    "Khong tim duoc (convention, empty_policy) nao tai lap dung --reference trong nguong "
                    f"{args.gate_tol} — DUNG lai, kiem tra lai file dump/reference truoc khi bootstrap "
                    "(muc 4.2 spec: dung doan, dung bao cao).")
            convention, empty_policy, max_diff, diffs = detected
            print(f"Da DO quy uoc = '{convention}'" +
                  (f", empty_policy = '{empty_policy}'" if convention == 'macro' else '') +
                  f" (tu checkpoint '{first_ref_label}', lech toi da {max_diff:.2e}).")
        else:
            convention, empty_policy = 'micro', 'skip'
            print("Khong co --reference nao duoc truyen — mac dinh convention='micro' "
                  "(dung quy uoc thuc su cua SegmentationMetrics/BoundaryMetrics, xem docstring dau file).")
    elif convention == 'macro' and empty_policy == 'auto':
        raise RuntimeError("--convention macro bat buoc phai truyen --empty-policy cu the "
                            "(hoac dung --convention auto de tu do ca hai).")

    # Gate 5.1-5.5: bao cao lech cho MOI checkpoint co reference, khong gop chung.
    for label, ref in references.items():
        computed = aggregate_point_estimate(dfs[label], convention, empty_policy, boundary_distances,
                                             args.diag_value)
        print(f"\n[{label}] Doi chieu voi reference (quy uoc={convention}"
              f"{'/' + empty_policy if convention == 'macro' else ''}):")
        for key, exp in ref.items():
            if exp is None or (isinstance(exp, float) and np.isnan(exp)):
                continue
            got = computed.get(key)
            if got is None:
                print(f"  {key:24s} KHONG CO trong ket qua tinh duoc")
                continue
            diff = abs(got - exp)
            status = 'OK' if diff < args.gate_tol else 'LECH'
            print(f"  {key:24s} computed={got:.7f}  reference={exp:.7f}  lech={diff:.2e}  [{status}]")

    pairs = _parse_pairs(args.pair) if args.pair else DEFAULT_PAIRS
    available_pairs = [(pid, a, b) for pid, a, b in pairs if a in dfs and b in dfs]
    skipped = [(pid, a, b) for pid, a, b in pairs if (pid, a, b) not in available_pairs]
    if skipped:
        print(f"\nBo qua {len(skipped)} cap vi thieu checkpoint: "
              f"{[(pid, a, b) for pid, a, b in skipped]}")
    if not available_pairs:
        raise RuntimeError("Khong co cap nao du 2 checkpoint da truyen qua --checkpoint de bootstrap.")

    all_rows = []
    for pair_id, label_a, label_b in available_pairs:
        print(f"\n=== Bootstrap {pair_id}: {label_a} - {label_b} (B={args.n_boot}, seed={args.seed}) ===")
        values_a, values_b = run_bootstrap(
            dfs[label_a], dfs[label_b], convention, empty_policy, args.n_boot, args.seed,
            boundary_distances, args.diag_value)
        observed_a = aggregate_point_estimate(dfs[label_a], convention, empty_policy,
                                               boundary_distances, args.diag_value)
        observed_b = aggregate_point_estimate(dfs[label_b], convention, empty_policy,
                                               boundary_distances, args.diag_value)
        rows = summarize_pair(values_a, values_b, observed_a, observed_b)
        for r in rows:
            r['pair_id'] = pair_id
            r['label_minuend'] = label_a
            r['label_subtrahend'] = label_b
            all_rows.append(r)
            print(f"  {r['quantity']:16s} median={r['median']:+.4f}  "
                  f"CI95%=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                  f"P(Delta<0)={r['p_delta_lt_0']:.3f}  observed={r['observed_delta']:+.4f}")

    csv_path = os.path.join(args.output_dir, 'bootstrap_ci.csv')
    md_path = os.path.join(args.output_dir, 'bootstrap_ci.md')
    write_bootstrap_ci_csv(all_rows, csv_path)
    write_bootstrap_ci_md(all_rows, convention, empty_policy, args.n_boot, args.seed, md_path)
    print(f"\nDa ghi {csv_path} va {md_path}.")


if __name__ == '__main__':
    main()
