"""Tools/test_bootstrap_boundary_ci_run7.py — Tester cho
Tools/bootstrap_boundary_ci_run7.py. Du lieu SYNTHETIC (khong can checkpoint
that/GPU) — xac nhan:

  1. scan_run7_manifest(): nhan dung thu muc run7/run7b hop le, suy ra dung
     seed/iter/run_name tu _meta.json + ten thu muc/checkpoint_path; BO QUA
     (khong raise) thu muc thieu file hoac khong suy ra duoc seed.
  2. run_bootstrap_contrast4() khop CHINH XAC ban brute-force dung lai HAM
     THAT aggregate_point_estimate() (import tu Tools/bootstrap_boundary_ci.py,
     khong viet lai cong thuc) tren tung lan resample, quy uoc 'micro' —
     xac nhan toi uu hoa vector hoa khong lam sai ket qua (giong ky thuat
     Test 7 cua Tools/test_bootstrap_boundary_ci.py cho cap doi).
  3. Tinh dinh: AffOnly==Baseline VA Static==BCE04 (4 DataFrame giong het
     nhau tung cap) -> contrast = 0 TUYET DOI o MOI lan lap bootstrap.
  4. write_interseed_amplitude(): sinh file khi >=2 seed cung run_family
     trong manifest; KHONG sinh file khi chi co 1 seed.

Chay: `python Tools/test_bootstrap_boundary_ci_run7.py` tu repo root.
"""

import json
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.bootstrap_boundary_ci import (
    CLASS_NAMES, aggregate_point_estimate, per_image_stats_columns,
)
from Tools.bootstrap_boundary_ci_run7 import (
    scan_run7_manifest, run_bootstrap_contrast4, summarize_contrast4,
    observed_contrast4, write_interseed_amplitude,
)

BOUNDARY_DISTANCES = (1, 2, 4)


def make_synthetic_stats_df(seed, n_images=10):
    """DataFrame voi dung schema per_image_stats_columns() — gia tri ngau
    nhien nhung hop le (inter <= union, n >= tp >= 0), du de bai tap toan
    tren cong thuc bootstrap (khong can qua SegmentationMetrics/
    BoundaryMetrics that, da duoc Tools/test_bootstrap_boundary_ci.py kiem
    chung rieng)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_images):
        row = {'image': f'img_{i}', 'n_valid_px': int(rng.integers(500, 2000))}
        for c in CLASS_NAMES:
            union = float(rng.integers(0, 300))
            inter = float(rng.integers(0, int(union) + 1)) if union > 0 else 0.0
            row[f'inter_{c}'] = inter
            row[f'union_{c}'] = union
            row[f'class_present_{c}'] = union > 0
        for c in CLASS_NAMES:
            for d in BOUNDARY_DISTANCES:
                bunion = float(rng.integers(0, 50))
                binter = float(rng.integers(0, int(bunion) + 1)) if bunion > 0 else 0.0
                row[f'biou_inter_{c}_d{d}'] = binter
                row[f'biou_union_{c}_d{d}'] = bunion
        n_pred = float(rng.integers(0, 40))
        n_gt = float(rng.integers(0, 40))
        row['bf_n_pred'] = n_pred
        row['bf_tp_pred'] = float(rng.integers(0, int(n_pred) + 1)) if n_pred > 0 else 0.0
        row['bf_n_gt'] = n_gt
        row['bf_tp_gt'] = float(rng.integers(0, int(n_gt) + 1)) if n_gt > 0 else 0.0
        asd_n_pred = float(rng.integers(1, 40))
        asd_n_gt = float(rng.integers(1, 40))
        row['asd_n_pred'] = asd_n_pred
        row['asd_sum_pred_to_gt'] = float(rng.uniform(0, 10)) * asd_n_pred
        row['asd_n_gt'] = asd_n_gt
        row['asd_sum_gt_to_pred'] = float(rng.uniform(0, 10)) * asd_n_gt
        row['has_gt_boundary'] = True
        row['has_pred_boundary'] = True
        rows.append(row)
    df = pd.DataFrame(rows)
    missing = [c for c in per_image_stats_columns(BOUNDARY_DISTANCES) if c not in df.columns]
    assert not missing, f"Thieu cot trong synthetic df: {missing}"
    return df


def test_manifest_scan():
    print("=== Test 1: scan_run7_manifest() ===")
    tmp = tempfile.mkdtemp(prefix='run7_manifest_test_')
    try:
        df = make_synthetic_stats_df(1, n_images=4)

        def make_entry(dirname, checkpoint_path, checkpoint_iter, run_name):
            d = os.path.join(tmp, dirname)
            os.makedirs(os.path.join(d, 'masks'), exist_ok=True)
            df.to_csv(os.path.join(d, 'per_image_stats.csv'), index=False)
            with open(os.path.join(d, 'per_image_stats_meta.json'), 'w', encoding='utf-8') as f:
                json.dump({'checkpoint_path': checkpoint_path, 'checkpoint_iter': checkpoint_iter,
                          'run_name': run_name}, f)

        make_entry('run7_affinity_only_seed19_best',
                   '/kaggle/working/.../run7_affinity_only_seed19/best_miou.pth', 36000,
                   'run7_affinity_only_seed19_best')
        make_entry('run7_affinity_only_seed19_final40000',
                   '/kaggle/working/.../run7_affinity_only_seed19/final_iter40000.pth', 40000,
                   'run7_affinity_only_seed19_final40000')
        # Thieu checkpoint_iter -> phai bi bo qua, khong raise.
        d_bad = os.path.join(tmp, 'run7_affinity_only_seed99_broken')
        os.makedirs(os.path.join(d_bad, 'masks'), exist_ok=True)
        df.to_csv(os.path.join(d_bad, 'per_image_stats.csv'), index=False)
        with open(os.path.join(d_bad, 'per_image_stats_meta.json'), 'w', encoding='utf-8') as f:
            json.dump({'checkpoint_path': None, 'checkpoint_iter': None, 'run_name': None}, f)

        manifest = scan_run7_manifest(tmp)
        assert len(manifest) == 2, f"Ky vong 2 entry hop le, nhan {len(manifest)}: {list(manifest)}"
        best = manifest['run7_affinity_only_seed19_best']
        assert best['seed'] == 19 and best['iter'] == 36000 and best['run_family'] == 'run7'
        final = manifest['run7_affinity_only_seed19_final40000']
        assert final['seed'] == 19 and final['iter'] == 40000
        print("PASS — manifest nhan dung 2 entry hop le, bo qua entry thieu checkpoint_iter.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_contrast4_matches_naive_reference():
    print("=== Test 2: run_bootstrap_contrast4() khop naive reference (dung ham that) ===")
    n_images = 8
    df_aff  = make_synthetic_stats_df(10, n_images)
    df_base = make_synthetic_stats_df(11, n_images)
    df_stat = make_synthetic_stats_df(12, n_images)
    df_bce  = make_synthetic_stats_df(13, n_images)

    B = 40
    seed = 19
    contrast = run_bootstrap_contrast4(df_aff, df_base, df_stat, df_bce,
                                        'micro', 'skip', B, seed, BOUNDARY_DISTANCES)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_images, size=(B, n_images))

    quantities = ['mIoU_9', 'bf_precision', 'asd_pred_to_gt']
    naive = {q: np.zeros(B) for q in quantities}
    for b in range(B):
        rows = idx[b]

        def resample(df):
            return df.iloc[rows].reset_index(drop=True)

        agg_aff  = aggregate_point_estimate(resample(df_aff),  'micro', 'skip', BOUNDARY_DISTANCES)
        agg_base = aggregate_point_estimate(resample(df_base), 'micro', 'skip', BOUNDARY_DISTANCES)
        agg_stat = aggregate_point_estimate(resample(df_stat), 'micro', 'skip', BOUNDARY_DISTANCES)
        agg_bce  = aggregate_point_estimate(resample(df_bce),  'micro', 'skip', BOUNDARY_DISTANCES)
        for q in quantities:
            naive[q][b] = (agg_aff[q] - agg_base[q]) - (agg_stat[q] - agg_bce[q])

    for q in quantities:
        max_diff = np.nanmax(np.abs(contrast[q] - naive[q]))
        assert max_diff < 1e-9, f"'{q}' lech {max_diff:.2e} giua ban vector hoa va naive reference"
    print(f"PASS — contrast4 vector hoa khop naive reference (dung aggregate_point_estimate that) "
          f"cho {quantities}.")


def test_contrast4_identical_pairs_gives_zero():
    print("=== Test 3: AffOnly==Baseline va Static==BCE04 -> contrast = 0 tuyet doi moi lan lap ===")
    n_images = 6
    df_x = make_synthetic_stats_df(20, n_images)
    df_y = make_synthetic_stats_df(21, n_images)
    contrast = run_bootstrap_contrast4(df_x, df_x.copy(), df_y, df_y.copy(),
                                        'micro', 'skip', 30, 19, BOUNDARY_DISTANCES)
    for q, vals in contrast.items():
        assert np.allclose(vals[~np.isnan(vals)], 0.0, atol=1e-12), \
            f"'{q}' phai = 0 tuyet doi khi AffOnly==Baseline va Static==BCE04, thay {vals}"
    print("PASS — contrast = 0 tuyet doi cho moi dai luong, moi lan lap bootstrap.")

    observed = observed_contrast4(df_x, df_x, df_y, df_y, 'micro', 'skip', BOUNDARY_DISTANCES)
    for q, v in observed.items():
        assert abs(v) < 1e-9, f"observed_contrast4['{q}']={v} phai ~0"
    rows = summarize_contrast4(contrast, observed)
    for r in rows:
        assert abs(r['median']) < 1e-9 and abs(r['observed_contrast']) < 1e-9
    print("PASS — observed_contrast4/summarize_contrast4 dong nhat voi contrast bootstrap.")


def test_interseed_amplitude():
    print("=== Test 4: write_interseed_amplitude() ===")
    tmp = tempfile.mkdtemp(prefix='run7_interseed_test_')
    try:
        df_s19 = make_synthetic_stats_df(30, 6)
        df_s86 = make_synthetic_stats_df(31, 6)
        csv_s19 = os.path.join(tmp, 'per_image_stats_s19.csv')
        csv_s86 = os.path.join(tmp, 'per_image_stats_s86.csv')
        df_s19.to_csv(csv_s19, index=False)
        df_s86.to_csv(csv_s86, index=False)

        manifest_2seed = {
            'run7_affinity_only_seed19_final40000': {
                'run_family': 'run7', 'run_name': 'run7_affinity_only_seed19_final40000',
                'iter': 40000, 'seed': 19, 'csv_path': csv_s19,
            },
            'run7_affinity_only_seed86_final40000': {
                'run_family': 'run7', 'run_name': 'run7_affinity_only_seed86_final40000',
                'iter': 40000, 'seed': 86, 'csv_path': csv_s86,
            },
        }
        out_path = os.path.join(tmp, 'run7_interseed_amplitude.md')
        write_interseed_amplitude(manifest_2seed, BOUNDARY_DISTANCES, 1024.0 * 1.41421356, out_path)
        assert os.path.exists(out_path), "Phai sinh file khi co >=2 seed cung run_family"
        content = open(out_path, encoding='utf-8').read()
        assert 'seed 19' in content and 'seed 86' in content
        print("PASS — sinh file bien do lien-seed khi co 2 seed.")

        manifest_1seed = {'a': manifest_2seed['run7_affinity_only_seed19_final40000']}
        out_path_1 = os.path.join(tmp, 'should_not_exist.md')
        write_interseed_amplitude(manifest_1seed, BOUNDARY_DISTANCES, 1024.0 * 1.41421356, out_path_1)
        assert not os.path.exists(out_path_1), "KHONG duoc sinh file khi chi co 1 seed"
        print("PASS — khong sinh file khi chi co 1 seed (dung tam bien do Static lam proxy).")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_manifest_scan()
    test_contrast4_matches_naive_reference()
    test_contrast4_identical_pairs_gives_zero()
    test_interseed_amplitude()
    print("\nALL TESTS PASSED (Tools/bootstrap_boundary_ci_run7.py)")
