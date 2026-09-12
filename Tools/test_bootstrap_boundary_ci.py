"""Tools/test_bootstrap_boundary_ci.py — Tester cho Tools/bootstrap_boundary_ci.py.

Du lieu SYNTHETIC (khong can 7 checkpoint that tren Kaggle) — xac nhan:
  1. aggregate_point_estimate() quy uoc 'micro' tai lap DUNG SegmentationMetrics/
     BoundaryMetrics (bat cau qua Tools/per_image_dump.compute_per_image_stats,
     da tu-test rieng trong Tools/test_per_image_dump.py) — noi Phan A va
     Phan C khop nhau.
  2. Quy uoc 'macro' + 4 chinh sach anh rong (muc 4.2) khop TINH TAY tren 1
     bo du lieu nho, dung thiet ke de co ca truong hop mau so=0.
  3. detect_convention_and_policy() do DUNG lai combo da dung de tao reference
     (ca khi reference sinh tu micro lan tu macro/chinh sach cu the).
  4. Gate 5.7 (assert_same_image_order) bat loi khi danh sach anh khac ten
     hoac khac thu tu.
  5. Gate 5.6 (tinh tat dinh): chay run_bootstrap() 2 lan cung seed -> giong
     het; doi seed -> khac (xac nhan seed thuc su co tac dung).
  6. Bootstrap ghep cap dung "cung idx cho ca 2 model": cap voi 2 DataFrame
     GIONG HET nhau phai cho Delta = 0 TUYET DOI o MOI lan lap (khong chi
     trung binh ~0) — day la bang chung manh nhat rang paired resampling
     dung idx chung, khong resample doc lap.
  7. Ban vector-hoa (ma tran hoa qua np.add.at + matmul) khop CHINH XAC ban
     brute-force (vong lap Python, dung lai aggregate_point_estimate tren
     tung lan resample) tren B nho — xac nhan toi uu hoa hieu nang khong lam
     sai ket qua.

Chay: `python Tools/test_bootstrap_boundary_ci.py` tu repo root.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.utils.metrics import SegmentationMetrics
from src.utils.boundary_metrics import BoundaryMetrics
from Tools.per_image_dump import compute_per_image_stats
from Tools.bootstrap_boundary_ci import (
    aggregate_point_estimate, detect_convention_and_policy, assert_same_image_order,
    run_bootstrap, per_image_stats_columns, CLASS_NAMES, EMPTY_POLICIES,
)


IGNORE = OpenEarthMapDataset.IGNORE_INDEX
NUM_CLASSES = len(CLASS_NAMES)
BOUNDARY_DISTANCES = (1, 2, 4)


def make_synthetic_dataframe(seed, n_images=12, H=40, W=36, ignore_frac=0.05):
    """Nhieu 'anh' doc lap (tung anh 1 batch B=1) qua compute_per_image_stats
    da tu-test — dung lam nguon 'su that' de doi chieu voi SegmentationMetrics/
    BoundaryMetrics tren TOAN BO 'dataset' nay."""
    rng = np.random.default_rng(seed)
    all_logits, all_targets, all_rows = [], [], []
    seg_metrics = SegmentationMetrics(NUM_CLASSES, IGNORE)
    boundary_metrics = BoundaryMetrics(NUM_CLASSES, IGNORE, boundary_distances=BOUNDARY_DISTANCES,
                                        bf_tolerance=2, connectivity=4, dilation_radius=0)

    for i in range(n_images):
        targets = rng.integers(0, NUM_CLASSES, size=(1, H, W)).astype(np.int64)
        ignore_mask = rng.random((1, H, W)) < ignore_frac
        targets_ig = targets.copy()
        targets_ig[ignore_mask] = IGNORE

        preds = targets.copy()
        flip = rng.random((1, H, W)) < 0.35
        other = rng.integers(0, NUM_CLASSES, size=(1, H, W)).astype(np.int64)
        preds = np.where(flip, other, preds)

        onehot = torch.nn.functional.one_hot(torch.from_numpy(preds), NUM_CLASSES)
        logits = onehot.permute(0, 3, 1, 2).float() * 10.0 - 5.0
        targets_t = torch.from_numpy(targets_ig)

        seg_metrics.update(logits, targets_t)
        boundary_metrics.update(logits, targets_t)

        rows, _ = compute_per_image_stats(
            logits, targets_t, [f'img_{seed}_{i}'], CLASS_NAMES, IGNORE, 4, 0,
            BOUNDARY_DISTANCES, 2)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    return df, seg_metrics.compute(), boundary_metrics.compute()


def test_micro_matches_real_accumulators():
    print("=== Test 1: quy uoc 'micro' cua bootstrap script khop SegmentationMetrics/BoundaryMetrics ===")
    for seed in range(4):
        df, seg_result, boundary_result = make_synthetic_dataframe(seed)
        computed = aggregate_point_estimate(df, 'micro', 'skip', BOUNDARY_DISTANCES)

        assert abs(computed['mIoU_9'] - seg_result['mIoU']) < 1e-9, (
            f"[seed={seed}] mIoU_9 lech: {computed['mIoU_9']} vs {seg_result['mIoU']}")
        for c, exp in zip(CLASS_NAMES, seg_result['per_class_iou']):
            got = computed['per_class_iou'][c]
            assert abs(got - exp) < 1e-9, f"[seed={seed}] per_class_iou[{c}] lech: {got} vs {exp}"

        for d in BOUNDARY_DISTANCES:
            exp = boundary_result[f'boundary_iou_d{d}']
            exp_pc = boundary_result[f'boundary_iou_d{d}_per_class']
            valid8 = [v for v in exp_pc[1:] if not np.isnan(v)]
            exp_biou8 = float(np.mean(valid8)) if valid8 else float('nan')
            got_biou8 = computed[f'biou_8_d{d}']
            if np.isnan(exp_biou8):
                assert np.isnan(got_biou8)
            else:
                assert abs(got_biou8 - exp_biou8) < 1e-9, (
                    f"[seed={seed}] biou_8_d{d} lech: {got_biou8} vs {exp_biou8}")
            got_bg = computed[f'biou_background_d{d}']
            if np.isnan(exp_pc[0]):
                assert np.isnan(got_bg)
            else:
                assert abs(got_bg - exp_pc[0]) < 1e-9

        for key in ('bf_precision', 'bf_recall', 'bf_score'):
            assert abs(computed[key] - boundary_result[key]) < 1e-9, (
                f"[seed={seed}] {key} lech: {computed[key]} vs {boundary_result[key]}")
        for key in ('asd_pred_to_gt', 'asd_gt_to_pred', 'asd'):
            got, exp = computed[key], boundary_result[key]
            if np.isnan(exp):
                assert np.isnan(got)
            else:
                assert abs(got - exp) < 1e-9, f"[seed={seed}] {key} lech: {got} vs {exp}"
    print("PASS (4 dataset synthetic, moi cai 12 anh, dung sai 1e-9)\n")


def make_hand_crafted_df():
    """4 anh, 2 lop (dung 2 lop dau trong CLASS_NAMES: Background/Bareland) de
    tinh tay duoc — anh 3 KHONG co lop Bareland o ca pred lan gt (union=0,
    mau so=0) de kich hoat empty_image_policy."""
    rows = []
    base = {f'inter_{c}': 0 for c in CLASS_NAMES}
    base.update({f'union_{c}': 0 for c in CLASS_NAMES})
    for d in BOUNDARY_DISTANCES:
        base.update({f'biou_inter_{c}_d{d}': 0 for c in CLASS_NAMES})
        base.update({f'biou_union_{c}_d{d}': 0 for c in CLASS_NAMES})
    base.update({f'class_present_{c}': 0 for c in CLASS_NAMES})

    # Anh 0: Background inter=8,union=10 (IoU=0.8); Bareland inter=2,union=4 (IoU=0.5)
    r0 = dict(base); r0['image'] = 'i0'; r0['n_valid_px'] = 20
    r0['inter_Background'], r0['union_Background'] = 8, 10
    r0['inter_Bareland'], r0['union_Bareland'] = 2, 4
    r0['class_present_Background'] = 1
    r0['class_present_Bareland'] = 1
    r0['bf_tp_pred'], r0['bf_n_pred'] = 3, 5
    r0['bf_tp_gt'], r0['bf_n_gt'] = 4, 5
    r0['asd_sum_pred_to_gt'], r0['asd_n_pred'] = 10.0, 5
    r0['asd_sum_gt_to_pred'], r0['asd_n_gt'] = 15.0, 5
    r0['has_gt_boundary'], r0['has_pred_boundary'] = 1, 1

    # Anh 1: Background inter=6,union=6 (IoU=1.0); Bareland inter=0,union=2 (IoU=0)
    r1 = dict(base); r1['image'] = 'i1'; r1['n_valid_px'] = 15
    r1['inter_Background'], r1['union_Background'] = 6, 6
    r1['inter_Bareland'], r1['union_Bareland'] = 0, 2
    r1['class_present_Background'] = 1
    r1['class_present_Bareland'] = 1
    r1['bf_tp_pred'], r1['bf_n_pred'] = 2, 4
    r1['bf_tp_gt'], r1['bf_n_gt'] = 2, 6
    r1['asd_sum_pred_to_gt'], r1['asd_n_pred'] = 8.0, 4
    r1['asd_sum_gt_to_pred'], r1['asd_n_gt'] = 24.0, 6
    r1['has_gt_boundary'], r1['has_pred_boundary'] = 1, 1

    # Anh 2: Bareland KHONG xuat hien (union=0) — kich hoat empty policy cho IoU.
    # Boundary/ASD cung KHONG co bien (n_pred=n_gt=0) — kich hoat cho ASD.
    r2 = dict(base); r2['image'] = 'i2'; r2['n_valid_px'] = 12
    r2['inter_Background'], r2['union_Background'] = 5, 8
    r2['inter_Bareland'], r2['union_Bareland'] = 0, 0
    r2['class_present_Background'] = 1
    r2['class_present_Bareland'] = 0
    r2['bf_tp_pred'], r2['bf_n_pred'] = 0, 0
    r2['bf_tp_gt'], r2['bf_n_gt'] = 0, 0
    r2['asd_sum_pred_to_gt'], r2['asd_n_pred'] = 0.0, 0
    r2['asd_sum_gt_to_pred'], r2['asd_n_gt'] = 0.0, 0
    r2['has_gt_boundary'], r2['has_pred_boundary'] = 0, 0

    # Anh 3: Background inter=4,union=5; Bareland inter=1,union=1 (IoU=1.0)
    r3 = dict(base); r3['image'] = 'i3'; r3['n_valid_px'] = 10
    r3['inter_Background'], r3['union_Background'] = 4, 5
    r3['inter_Bareland'], r3['union_Bareland'] = 1, 1
    r3['class_present_Background'] = 1
    r3['class_present_Bareland'] = 1
    r3['bf_tp_pred'], r3['bf_n_pred'] = 4, 4
    r3['bf_tp_gt'], r3['bf_n_gt'] = 3, 3
    r3['asd_sum_pred_to_gt'], r3['asd_n_pred'] = 0.0, 4
    r3['asd_sum_gt_to_pred'], r3['asd_n_gt'] = 0.0, 3
    r3['has_gt_boundary'], r3['has_pred_boundary'] = 1, 1

    return pd.DataFrame([r0, r1, r2, r3])


def test_macro_empty_policies_hand_calc():
    print("=== Test 2: quy uoc 'macro' + 4 chinh sach anh rong khop tinh tay ===")
    df = make_hand_crafted_df()

    # IoU Bareland tung anh: i0=0.5, i1=0.0, i2=UNDEFINED(0/0), i3=1.0
    for policy, expected_i2, expected_mean in (
        ('skip', None, np.mean([0.5, 0.0, 1.0])),
        ('zero', 0.0, np.mean([0.5, 0.0, 0.0, 1.0])),
        ('one', 1.0, np.mean([0.5, 0.0, 1.0, 1.0])),
    ):
        result = aggregate_point_estimate(df, 'macro', policy, BOUNDARY_DISTANCES)
        got = result['iou_Bareland']
        assert abs(got - expected_mean) < 1e-9, (
            f"policy={policy}: iou_Bareland macro = {got}, ky vong {expected_mean}")
    print("PASS (IoU Bareland macro voi skip/zero/one khop cong thuc tay)\n")

    # ASD pred->gt tung anh: i0=10/5=2.0, i1=8/4=2.0, i2=UNDEFINED(0/0), i3=0/4=0.0
    diag_value = 111.0
    for policy, expected_mean in (
        ('skip', np.mean([2.0, 2.0, 0.0])),
        ('zero', np.mean([2.0, 2.0, 0.0, 0.0])),
        ('diag', np.mean([2.0, 2.0, diag_value, 0.0])),
    ):
        result = aggregate_point_estimate(df, 'macro', policy, BOUNDARY_DISTANCES, diag_value=diag_value)
        got = result['asd_pred_to_gt']
        assert abs(got - expected_mean) < 1e-9, (
            f"policy={policy}: asd_pred_to_gt macro = {got}, ky vong {expected_mean}")
    print("PASS (ASD pred->gt macro voi skip/zero/diag khop cong thuc tay, diag_value tuy chinh)\n")


def test_detect_convention_and_policy_roundtrip():
    print("=== Test 3: detect_convention_and_policy() do DUNG lai combo da tao reference ===")
    df = make_hand_crafted_df()

    ref_micro = aggregate_point_estimate(df, 'micro', 'skip', BOUNDARY_DISTANCES)
    ref_flat_micro = {'mIoU_9': ref_micro['mIoU_9'], 'bf_precision': ref_micro['bf_precision'],
                       'asd_pred_to_gt': ref_micro['asd_pred_to_gt']}
    detected = detect_convention_and_policy(df, ref_flat_micro, BOUNDARY_DISTANCES, tol=1e-9)
    assert detected is not None and detected[0] == 'micro', f"Ky vong do ra 'micro', duoc {detected}"
    print(f"PASS — reference sinh tu 'micro' duoc do dung: {detected[:2]}")

    for policy in ('zero', 'one'):
        ref_macro = aggregate_point_estimate(df, 'macro', policy, BOUNDARY_DISTANCES)
        ref_flat = {'iou_Bareland': ref_macro['iou_Bareland'], 'mIoU_9': ref_macro['mIoU_9']}
        detected = detect_convention_and_policy(df, ref_flat, BOUNDARY_DISTANCES, tol=1e-9)
        assert detected is not None, f"Khong do duoc gi ca cho policy={policy}"
        assert detected[:2] == ('macro', policy), (
            f"Ky vong do ra ('macro','{policy}'), duoc {detected[:2]}")
        print(f"PASS — reference sinh tu 'macro'/{policy} duoc do dung: {detected[:2]}")
    print()


def test_gate_5_7_image_order():
    print("=== Test 4 (Gate 5.7): assert_same_image_order bat loi khac ten/thu tu ===")
    df_a = pd.DataFrame({'image': ['a', 'b', 'c']})
    df_b_same = pd.DataFrame({'image': ['a', 'b', 'c']})
    assert_same_image_order({'A': df_a, 'B': df_b_same})  # khong duoc raise
    print("PASS (danh sach giong het nhau -> khong raise)")

    df_b_reordered = pd.DataFrame({'image': ['b', 'a', 'c']})
    try:
        assert_same_image_order({'A': df_a, 'B': df_b_reordered})
        raise AssertionError("Le ra phai raise khi thu tu khac nhau")
    except AssertionError as e:
        assert 'THU TU' in str(e) or 'thu tu' in str(e).lower() or 'KHAC' in str(e)
    print("PASS (thu tu khac nhau -> raise dung loai loi)")

    df_b_diff_content = pd.DataFrame({'image': ['a', 'b', 'd']})
    try:
        assert_same_image_order({'A': df_a, 'B': df_b_diff_content})
        raise AssertionError("Le ra phai raise khi noi dung khac nhau")
    except AssertionError:
        pass
    print("PASS (noi dung khac nhau -> raise)\n")


def test_gate_5_6_determinism_and_seed_effect():
    print("=== Test 5 (Gate 5.6): tinh tat dinh cua bootstrap, seed co tac dung ===")
    df_a, _, _ = make_synthetic_dataframe(seed=101, n_images=10)
    df_b, _, _ = make_synthetic_dataframe(seed=202, n_images=10)
    df_a['image'] = df_b['image'] = [f'img_{i}' for i in range(len(df_a))]

    va1, vb1 = run_bootstrap(df_a, df_b, 'micro', 'skip', n_boot=500, seed=19)
    va2, vb2 = run_bootstrap(df_a, df_b, 'micro', 'skip', n_boot=500, seed=19)
    for key in va1:
        assert np.array_equal(va1[key], va2[key], equal_nan=True), (
            f"GATE 5.6 FAIL: '{key}' khac nhau giua 2 lan chay cung seed")
    print("PASS (cung seed -> ket qua giong het byte-for-byte)")

    va3, vb3 = run_bootstrap(df_a, df_b, 'micro', 'skip', n_boot=500, seed=7)
    any_diff = any(not np.array_equal(va1[key], va3[key], equal_nan=True) for key in va1)
    assert any_diff, "Doi seed nhung ket qua khong doi — nghi ngo seed khong duoc dung that su"
    print("PASS (doi seed -> ket qua khac, xac nhan seed thuc su duoc dung)\n")


def test_paired_resampling_uses_same_idx():
    print("=== Test 6: cap voi 2 DataFrame GIONG HET nhau -> Delta = 0 TUYET DOI moi lan lap ===")
    df, _, _ = make_synthetic_dataframe(seed=55, n_images=15)
    df_copy = df.copy()

    for convention, policy in (('micro', 'skip'), ('macro', 'zero'), ('macro', 'diag')):
        values_a, values_b = run_bootstrap(df, df_copy, convention, policy, n_boot=300, seed=19)
        for key in values_a:
            delta = values_a[key] - values_b[key]
            finite = delta[~np.isnan(delta)]
            assert finite.size > 0, f"[{convention}/{policy}] '{key}': toan bo NaN, khong kiem duoc"
            assert np.allclose(finite, 0.0, atol=1e-10), (
                f"[{convention}/{policy}] '{key}': Delta khac 0 du 2 DataFrame giong het nhau — "
                f"paired resampling KHONG dung chung idx cho 2 model. max|delta|={np.abs(finite).max()}")
    print("PASS (micro + macro/zero + macro/diag: Delta=0 tuyet doi voi 2 DataFrame giong het, "
          "300 lan lap moi to hop) — xac nhan dung CHUNG idx cho ca 2 ve trong 1 cap.\n")


def test_vectorized_matches_brute_force():
    print("=== Test 7: ban vector-hoa (matmul) khop ban brute-force (vong lap Python) ===")
    df_a, _, _ = make_synthetic_dataframe(seed=3, n_images=9)
    df_b, _, _ = make_synthetic_dataframe(seed=4, n_images=9)
    df_a['image'] = df_b['image'] = [f'img_{i}' for i in range(len(df_a))]

    n_boot, seed = 25, 19
    n_images = len(df_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_images, size=(n_boot, n_images))  # PHAI khop dung logic run_bootstrap()

    for convention, policy in (('micro', 'skip'), ('macro', 'one')):
        values_a, values_b = run_bootstrap(df_a, df_b, convention, policy, n_boot, seed)

        for b in range(n_boot):
            sample_idx = idx[b]
            resampled_a = df_a.iloc[sample_idx].reset_index(drop=True)
            resampled_b = df_b.iloc[sample_idx].reset_index(drop=True)
            brute_a = aggregate_point_estimate(resampled_a, convention, policy)
            brute_b = aggregate_point_estimate(resampled_b, convention, policy)

            for key, brute_val_key in (
                ('mIoU_9', 'mIoU_9'), ('bf_precision', 'bf_precision'),
                ('asd_pred_to_gt', 'asd_pred_to_gt'), ('biou_8_d1', 'biou_8_d1'),
                ('iou_Water', 'iou_Water'),
            ):
                fast_val = values_a[key][b]
                slow_val = brute_a[brute_val_key]
                if np.isnan(slow_val):
                    assert np.isnan(fast_val), (
                        f"[{convention}/{policy}] rep={b} '{key}': fast={fast_val} nhung brute=NaN")
                else:
                    assert abs(fast_val - slow_val) < 1e-9, (
                        f"[{convention}/{policy}] rep={b} '{key}': fast={fast_val} != brute={slow_val}")
        print(f"PASS — [{convention}/{policy}] {n_boot} lan lap, ca 2 ve, 5 dai luong: "
              f"vector-hoa khop 100% brute-force.")
    print()


if __name__ == '__main__':
    test_micro_matches_real_accumulators()
    test_macro_empty_policies_hand_calc()
    test_detect_convention_and_policy_roundtrip()
    test_gate_5_7_image_order()
    test_gate_5_6_determinism_and_seed_effect()
    test_paired_resampling_uses_same_idx()
    test_vectorized_matches_brute_force()
    print("Tat ca self-test Tools/bootstrap_boundary_ci.py PASS.")
