"""Tools/bootstrap_boundary_ci_run7.py — mo rong bootstrap CI cho Run 7
"Affinity-only" (docs/spec-run7-affinity-only-ban-giao-claude-code.md muc 4),
HOAN TOAN TACH BIET voi Tools/bootstrap_boundary_ci.py va output/bootstrap/
da co — KHONG sua file goc, KHONG ghi de output cu, chay lai duoc bat ky luc
nao ma khong anh huong ket qua 6 cap bootstrap cu.

File nay CHI IMPORT (doc, khong sua) cac ham thuan tuy tu
Tools/bootstrap_boundary_ci.py: load_per_image_stats, assert_same_image_order,
aggregate_point_estimate, detect_convention_and_policy, build_raw_matrix,
build_effective_ratio_matrix, _resample_counts, _micro_composites_from_sums,
_macro_composites_from_weighted, run_bootstrap, summarize_pair,
write_bootstrap_ci_csv, QUANTITIES_FOR_CI, DEFAULT_DIAG_VALUE, EMPTY_POLICIES,
CLASS_NAMES, HIGHLIGHT_CLASSES. Vi cac ham do dong tren bien module-level
CLASS_NAMES/NUM_CLASSES (doc tu src.data.dataset.OpenEarthMapDataset), import
module nay se keo theo dependency rasterio — giong het Tools/bootstrap_
boundary_ci.py hien tai, khong phai gi moi.

3 viec chinh (mo rong bang cach CONG THEM, khong sua bang cu):

1. Manifest tu sinh (muc 4.1 spec) — quet `output/dump/` theo 2 pattern
   `run7_affinity_only_seed*_*` va `run7b_affinity_only_lambda02_seed*_*`
   (CHI 2 pattern nay — KHONG gop 7 checkpoint cu vao cung manifest, de
   khong tao phu thuoc cheo giua dump cu/moi theo yeu cau "doc lap hoan
   toan" cua nguoi dung). Ghi output/dump/checkpoints_manifest_run7.json.

   LUU Y quan trong (lech nho so voi van ban spec, bat buoc vi khong duoc
   sua Tools/eval_boundary_metrics.py): `<stem>_meta.json` hien tai KHONG co
   truong `seed` rieng (chi co checkpoint_path/checkpoint_iter/run_name).
   Seed duoc SUY RA bang regex `seed(\\d+)` tren `checkpoint_path` (uu tien)
   roi tren ten thu muc dump (fallback) — KHONG phai doc truc tiep 1 truong
   `seed` co san. Neu ca hai cach deu khong khop, checkpoint do bi BO QUA
   kem canh bao ro rang (khong doan lieu linh).

2. Cap #7/#8/#10 (pairwise, tai dung nguyen may moc run_bootstrap/
   summarize_pair) va #11 (lien-seed, dieu kien >=2 seed) — tinh cho MOI
   seed tim thay trong manifest (khong chi 1 seed "chinh"), dung nguyen tac
   "khong bo sot seed" (muc 6.3 spec).

3. Cap #9 — contrast cong tuyen 4 so hang, cong thuc dung muc 4.2 spec:

       contrast = (AffOnly - Baseline) - (Static - BCE04)

   CHUA co san trong Tools/bootstrap_boundary_ci.py (cong cu do chi lam duoc
   hieu so 2 checkpoint). Ham run_bootstrap_contrast4() o day tai dung
   build_raw_matrix/build_effective_ratio_matrix/_resample_counts/
   _micro_composites_from_sums/_macro_composites_from_weighted (import tu
   file goc) de vector hoa dung cach run_bootstrap() da lam — KHONG vong lap
   Python qua B=10000 lan.

Output: output/bootstrap_run7/bootstrap_ci_run7.csv + .md (cap 7-10) va
output/bootstrap_run7/run7_interseed_amplitude.md (cap 11, chi khi >=2 seed).
KHONG dung chung file voi output/bootstrap/bootstrap_ci.csv cu — day la lech
CO CHU Y so voi muc 4.4 cua spec goc (spec muon "them hang vao file da co"),
uu tien theo yeu cau tuong minh cua nguoi dung trong phien lam viec nay
(file cu/moi doc lap tuyet doi).

Usage:
    python Tools/bootstrap_boundary_ci_run7.py \\
        --checkpoint run1_baseline=output/dump/run1_baseline_iter40000/per_image_stats.csv \\
        --checkpoint run2b_bce04=output/dump/run2b_bce04_iter40000/per_image_stats.csv \\
        --checkpoint run3_static_s19=output/dump/run3_static_s19_iter36000/per_image_stats.csv \\
        --dump-dir output/dump --n-boot 10000 --seed 19 \\
        --output-dir output/bootstrap_run7

Tu-test: `python Tools/test_bootstrap_boundary_ci_run7.py`.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.bootstrap_boundary_ci import (  # noqa: E402 — doc, khong sua file goc
    CLASS_NAMES, HIGHLIGHT_CLASSES, DEFAULT_DIAG_VALUE, EMPTY_POLICIES, QUANTITIES_FOR_CI,
    load_per_image_stats, assert_same_image_order, aggregate_point_estimate,
    detect_convention_and_policy, load_reference,
    build_raw_matrix, build_effective_ratio_matrix, _resample_counts,
    _micro_composites_from_sums, _macro_composites_from_weighted,
    run_bootstrap, summarize_pair,
)

_MANIFEST_PATTERNS = {
    'run7':  'run7_affinity_only_seed*_*',
    'run7b': 'run7b_affinity_only_lambda02_seed*_*',
}
_SEED_RE = re.compile(r'seed(\d+)')
_ITER_LABEL_RE = re.compile(r'_(best|final\d+)$')


# ═══════════════════════════ Manifest tu sinh (muc 4.1) ═════════════════════

def _infer_seed(checkpoint_path: str, dump_dir_name: str, label: str) -> int:
    for source in (checkpoint_path or '', dump_dir_name):
        m = _SEED_RE.search(source)
        if m:
            return int(m.group(1))
    raise ValueError(
        f"[{label}] khong suy ra duoc seed tu checkpoint_path={checkpoint_path!r} hay ten thu muc "
        f"{dump_dir_name!r} — _meta.json khong co truong 'seed' rieng (Tools/eval_boundary_metrics.py "
        f"khong bi sua o day), can regex 'seed<N>' khop it nhat 1 trong 2 nguon tren.")


def scan_run7_manifest(dump_dir: str) -> dict:
    """Quet dump_dir theo 2 pattern run7/run7b, tra ve dict label -> entry
    (run_family, run_name, iter, seed, csv_path, meta_path, dump_dir). Bo qua
    (kem canh bao) checkpoint nao thieu file/khong suy ra duoc seed/iter."""
    manifest = {}
    warnings = []
    for run_family, pattern in _MANIFEST_PATTERNS.items():
        for entry_dir in sorted(glob.glob(os.path.join(dump_dir, pattern))):
            if not os.path.isdir(entry_dir):
                continue
            dir_name = os.path.basename(entry_dir.rstrip('/\\'))
            csv_path = os.path.join(entry_dir, 'per_image_stats.csv')
            meta_path = os.path.join(entry_dir, 'per_image_stats_meta.json')
            if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
                warnings.append(f"[{dir_name}] thieu per_image_stats.csv hoac _meta.json — bo qua.")
                continue
            mask_dirs = [d for d in glob.glob(os.path.join(entry_dir, '*')) if os.path.isdir(d)]
            if not mask_dirs:
                warnings.append(f"[{dir_name}] khong thay thu muc mask PNG nao — bo qua "
                                 f"(mong doi it nhat 1 thu muc con, vd 'masks/').")
                continue
            with open(meta_path, encoding='utf-8') as f:
                meta = json.load(f)
            iteration = meta.get('checkpoint_iter')
            if iteration is None:
                warnings.append(f"[{dir_name}] _meta.json khong co checkpoint_iter (--checkpoint-iter "
                                 f"khong duoc truyen luc dump) — bo qua, khong doan iteration.")
                continue
            label = meta.get('run_name') or dir_name
            try:
                seed = _infer_seed(meta.get('checkpoint_path'), dir_name, label)
            except ValueError as e:
                warnings.append(str(e) + " — bo qua.")
                continue
            manifest[label] = {
                'run_family': run_family, 'run_name': label, 'iter': int(iteration),
                'seed': seed, 'csv_path': csv_path, 'meta_path': meta_path, 'dump_dir': entry_dir,
            }
    for w in warnings:
        print(f"[manifest] CANH BAO: {w}")
    return manifest


def write_manifest(manifest: dict, out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Da ghi manifest ({len(manifest)} checkpoint) -> {out_path}")


# ═══════════════════ Cap #9 — contrast cong tuyen 4 so hang (muc 4.2) ═══════

def run_bootstrap_contrast4(df_affonly, df_baseline, df_static, df_bce04,
                             convention, empty_policy, n_boot, seed,
                             boundary_distances=(1, 2, 4), diag_value=DEFAULT_DIAG_VALUE):
    """contrast = (AffOnly - Baseline) - (Static - BCE04), CUNG bo chi so
    resample cho ca 4 checkpoint (muc 4.2 spec). Tra ve dict ten_dai_luong ->
    mang (n_boot,) gia tri contrast — vector hoa, khong vong lap Python qua B."""
    n_images = len(df_affonly)
    for name, df in (('baseline', df_baseline), ('static', df_static), ('bce04', df_bce04)):
        assert len(df) == n_images, f"So anh cua '{name}' ({len(df)}) khac AffOnly ({n_images})"

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_images, size=(n_boot, n_images))
    counts = _resample_counts(idx, n_images)

    def composites(df):
        if convention == 'micro':
            mat, col_index = build_raw_matrix(df, boundary_distances)
            sums = counts @ mat
            return _micro_composites_from_sums(sums, col_index, boundary_distances)
        mat, _valid, col_index = build_effective_ratio_matrix(df, empty_policy, boundary_distances, diag_value)
        weighted = (counts @ mat) / n_images
        return _macro_composites_from_weighted(weighted, col_index, boundary_distances)

    v_affonly  = composites(df_affonly)
    v_baseline = composites(df_baseline)
    v_static   = composites(df_static)
    v_bce04    = composites(df_bce04)

    contrast = {}
    for q in v_affonly:
        contrast[q] = (v_affonly[q] - v_baseline[q]) - (v_static[q] - v_bce04[q])
    return contrast


def summarize_contrast4(contrast_values, observed_contrast, quantities=QUANTITIES_FOR_CI):
    rows = []
    for q in quantities:
        vals = contrast_values[q]
        vals_valid = vals[~np.isnan(vals)]
        n_nan = int(np.isnan(vals).sum())
        if len(vals_valid) == 0:
            rows.append({'quantity': q, 'median': float('nan'), 'ci_lo': float('nan'),
                         'ci_hi': float('nan'), 'p_contrast_lt_0': float('nan'),
                         'observed_contrast': float('nan'), 'n_boot_nan': n_nan})
            continue
        median = float(np.median(vals_valid))
        ci_lo, ci_hi = np.percentile(vals_valid, [2.5, 97.5])
        p_lt_0 = float(np.mean(vals_valid < 0))
        rows.append({'quantity': q, 'median': median, 'ci_lo': float(ci_lo), 'ci_hi': float(ci_hi),
                     'p_contrast_lt_0': p_lt_0, 'observed_contrast': observed_contrast.get(q, float('nan')),
                     'n_boot_nan': n_nan})
    return rows


def observed_contrast4(df_affonly, df_baseline, df_static, df_bce04, convention, empty_policy,
                        boundary_distances=(1, 2, 4), diag_value=DEFAULT_DIAG_VALUE):
    obs = {df_name: aggregate_point_estimate(df, convention, empty_policy, boundary_distances, diag_value)
           for df_name, df in (('affonly', df_affonly), ('baseline', df_baseline),
                                ('static', df_static), ('bce04', df_bce04))}
    out = {}
    for q in QUANTITIES_FOR_CI:
        key = q if q in obs['affonly'] else None
        if key is None:
            continue
        out[q] = (obs['affonly'][q] - obs['baseline'][q]) - (obs['static'][q] - obs['bce04'][q])
    return out


# ═══════════════════════ Bang bien do lien-seed (muc 4.3) ═══════════════════

def write_interseed_amplitude(manifest: dict, boundary_distances, diag_value, path: str):
    """|AffOnly_seedA - AffOnly_seedB| tai iteration chung gan nhat co o ca
    hai (uu tien 40k neu ca hai seed deu co final_iter40000) — chi sinh khi
    >=2 seed cho CUNG 1 run_family (run7 hoac run7b) co trong manifest."""
    by_family = {}
    for label, entry in manifest.items():
        by_family.setdefault(entry['run_family'], {}).setdefault(entry['seed'], []).append(entry)

    lines = ['# Bien do lien-seed — Run 7 Affinity-only (tu sinh, muc 4.3 spec)', '']
    wrote_any = False
    for family, by_seed in by_family.items():
        seeds = sorted(by_seed)
        if len(seeds) < 2:
            continue
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                sA, sB = seeds[i], seeds[j]
                entries_a = {e['iter']: e for e in by_seed[sA]}
                entries_b = {e['iter']: e for e in by_seed[sB]}
                common_iters = sorted(set(entries_a) & set(entries_b))
                if not common_iters:
                    continue
                it = max(common_iters) if 40000 in common_iters else common_iters[0]
                ea, eb = entries_a[it], entries_b[it]
                df_a = load_per_image_stats(ea['csv_path'], boundary_distances)
                df_b = load_per_image_stats(eb['csv_path'], boundary_distances)
                assert_same_image_order({ea['run_name']: df_a, eb['run_name']: df_b})
                agg_a = aggregate_point_estimate(df_a, 'micro', 'skip', boundary_distances, diag_value)
                agg_b = aggregate_point_estimate(df_b, 'micro', 'skip', boundary_distances, diag_value)
                lines.append(f'## {family} — seed {sA} vs seed {sB} @ iter {it}')
                lines.append('')
                lines.append('| Dai luong | seed ' + str(sA) + ' | seed ' + str(sB) + ' | \\|A-B\\| |')
                lines.append('|---|---|---|---|')
                for q in QUANTITIES_FOR_CI:
                    va, vb = agg_a.get(q, float('nan')), agg_b.get(q, float('nan'))
                    amp = abs(va - vb) if (va == va and vb == vb) else float('nan')
                    lines.append(f'| {q} | {va:.4f} | {vb:.4f} | {amp:.4f} |')
                lines.append('')
                wrote_any = True

    if not wrote_any:
        print("[interseed] Chua co >=2 seed cho cung run_family trong manifest — khong sinh "
              f"{path} (dung tam bien do Static lam proxy, muc 4.3 spec).")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Da ghi bang bien do lien-seed -> {path}")


# ═══════════════════════════════════ CLI ════════════════════════════════════

def _parse_label_path(items, flag_name):
    out = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f"{flag_name} phai co dang LABEL=PATH, nhan: {item}")
        label, path = item.split('=', 1)
        out[label] = path
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', action='append', default=[],
                     help='LABEL=duong_dan_per_image_stats.csv cho cac checkpoint CU can lam moc '
                          '(run1_baseline/run2b_bce04/run3_static_s19) — lap lai. KHONG dung cho '
                          'checkpoint Run 7/7b (tu quet qua --dump-dir).')
    ap.add_argument('--dump-dir', dest='dump_dir', default='output/dump',
                     help='Thu muc chua dump cua Run 7/7b (quet manifest tu day, muc 4.1 spec).')
    ap.add_argument('--reference', action='append', default=[],
                     help='(Tuy chon) LABEL=duong_dan.json de tu DO quy uoc/chinh sach (muc 4.2), '
                          'giong het co che cua Tools/bootstrap_boundary_ci.py.')
    ap.add_argument('--convention', choices=['auto', 'micro', 'macro'], default='auto')
    ap.add_argument('--empty-policy', dest='empty_policy', choices=['auto'] + list(EMPTY_POLICIES),
                     default='auto')
    ap.add_argument('--gate-tol', dest='gate_tol', type=float, default=1e-6)
    ap.add_argument('--n-boot', dest='n_boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=19, help='Seed CUA SCRIPT bootstrap, KHONG phai seed '
                                                            'huan luyen.')
    ap.add_argument('--boundary-distances', default='1,2,4')
    ap.add_argument('--diag-value', dest='diag_value', type=float, default=DEFAULT_DIAG_VALUE)
    ap.add_argument('--output-dir', dest='output_dir', default='output/bootstrap_run7')
    return ap.parse_args()


def main():
    args = parse_args()
    boundary_distances = tuple(int(x) for x in args.boundary_distances.split(','))

    old_checkpoint_paths = _parse_label_path(args.checkpoint, '--checkpoint')
    reference_paths = _parse_label_path(args.reference, '--reference')

    manifest = scan_run7_manifest(args.dump_dir)
    write_manifest(manifest, os.path.join(args.dump_dir, 'checkpoints_manifest_run7.json'))
    if not manifest:
        raise RuntimeError(f"Khong tim thay checkpoint Run 7/7b nao duoi {args.dump_dir} — chay dump "
                            f"(Tools/eval_boundary_metrics.py --model-type baseline --dump-preds/"
                            f"--dump-per-image-stats) truoc.")

    dfs = {label: load_per_image_stats(path, boundary_distances)
           for label, path in old_checkpoint_paths.items()}
    for label, entry in manifest.items():
        dfs[label] = load_per_image_stats(entry['csv_path'], boundary_distances)
    assert_same_image_order(dfs)
    print(f"Da doc {len(dfs)} checkpoint ({len(old_checkpoint_paths)} cu + {len(manifest)} Run7/7b), "
          f"{len(next(iter(dfs.values())))} anh/checkpoint, danh sach anh khop nhau.")

    references = {label: load_reference(path) for label, path in reference_paths.items()}
    convention, empty_policy = args.convention, args.empty_policy
    if args.convention == 'auto':
        if references:
            first_ref_label = next(iter(references))
            detected = detect_convention_and_policy(
                dfs[first_ref_label], references[first_ref_label], boundary_distances, args.gate_tol,
                args.diag_value)
            if detected is None:
                raise RuntimeError("Khong DO duoc (convention, empty_policy) nao tai lap dung --reference.")
            convention, empty_policy, max_diff, _ = detected
            print(f"Da DO quy uoc = '{convention}'" +
                  (f", empty_policy = '{empty_policy}'" if convention == 'macro' else '') +
                  f" (tu '{first_ref_label}', lech toi da {max_diff:.2e}).")
        else:
            convention, empty_policy = 'micro', 'skip'
            print("Khong co --reference — mac dinh convention='micro' (quy uoc that cua "
                  "SegmentationMetrics/BoundaryMetrics).")
    elif convention == 'macro' and empty_policy == 'auto':
        raise RuntimeError("--convention macro bat buoc --empty-policy cu the.")

    all_rows = []

    def add_pair(pair_id, label_a, label_b):
        if label_a not in dfs or label_b not in dfs:
            print(f"Bo qua {pair_id}: thieu '{label_a}' hoac '{label_b}'.")
            return
        print(f"\n=== Bootstrap {pair_id}: {label_a} - {label_b} (B={args.n_boot}, seed={args.seed}) ===")
        values_a, values_b = run_bootstrap(dfs[label_a], dfs[label_b], convention, empty_policy,
                                            args.n_boot, args.seed, boundary_distances, args.diag_value)
        observed_a = aggregate_point_estimate(dfs[label_a], convention, empty_policy, boundary_distances, args.diag_value)
        observed_b = aggregate_point_estimate(dfs[label_b], convention, empty_policy, boundary_distances, args.diag_value)
        rows = summarize_pair(values_a, values_b, observed_a, observed_b)
        for r in rows:
            r['pair_id'] = pair_id
            r['label_minuend'] = label_a
            r['label_subtrahend'] = label_b
            all_rows.append(r)
            print(f"  {r['quantity']:16s} median={r['median']:+.4f}  CI95%=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                  f"P(Delta<0)={r['p_delta_lt_0']:.3f}  observed={r['observed_delta']:+.4f}")

    # Moi seed tim thay trong manifest, KHONG chi 1 seed "chinh" (muc 6.3 spec).
    run7_seeds = sorted({e['seed'] for e in manifest.values() if e['run_family'] == 'run7'})
    for s in run7_seeds:
        seed_entries = {e['iter']: label for label, e in manifest.items()
                         if e['run_family'] == 'run7' and e['seed'] == s}
        label_40k = seed_entries.get(40000)
        if label_40k is None:
            print(f"[seed {s}] khong co checkpoint @40000 trong manifest — bo qua cap 7/8/9 cho seed nay "
                  f"(can final_iter40000 de cung moc voi baseline/bce04, muc 4.2 spec).")
        else:
            add_pair(f'pair7_affonly_s{s}_vs_baseline', label_40k, 'run1_baseline')
            add_pair(f'pair8_affonly_s{s}_vs_bce04', label_40k, 'run2b_bce04')

            if all(l in dfs for l in (label_40k, 'run1_baseline', 'run3_static_s19', 'run2b_bce04')):
                print(f"\n=== Bootstrap pair9_additivity_contrast_s{s}: "
                      f"(AffOnly_s{s}@40k - Baseline@40k) - (Static_s19@36k - BCE04@40k) "
                      f"(B={args.n_boot}, seed={args.seed}) ===")
                contrast_vals = run_bootstrap_contrast4(
                    dfs[label_40k], dfs['run1_baseline'], dfs['run3_static_s19'], dfs['run2b_bce04'],
                    convention, empty_policy, args.n_boot, args.seed, boundary_distances, args.diag_value)
                obs_contrast = observed_contrast4(
                    dfs[label_40k], dfs['run1_baseline'], dfs['run3_static_s19'], dfs['run2b_bce04'],
                    convention, empty_policy, boundary_distances, args.diag_value)
                rows9 = summarize_contrast4(contrast_vals, obs_contrast)
                for r in rows9:
                    r['pair_id'] = f'pair9_additivity_contrast_s{s}'
                    r['label_minuend'] = f'({label_40k} - run1_baseline)'
                    r['label_subtrahend'] = '(run3_static_s19 - run2b_bce04)'
                    r['median'], r['ci_lo'], r['ci_hi'] = r['median'], r['ci_lo'], r['ci_hi']
                    r['p_delta_lt_0'] = r.pop('p_contrast_lt_0')
                    r['observed_delta'] = r.pop('observed_contrast')
                    all_rows.append(r)
                    print(f"  {r['quantity']:16s} median={r['median']:+.4f}  "
                          f"CI95%=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
                          f"P(contrast<0)={r['p_delta_lt_0']:.3f}  observed={r['observed_delta']:+.4f}")
            else:
                print(f"Bo qua pair9_additivity_contrast_s{s}: thieu 1 trong 4 checkpoint can thiet "
                      f"(run1_baseline/run2b_bce04/run3_static_s19 qua --checkpoint, hoac AffOnly@40k).")

        run7b_40k = next((label for label, e in manifest.items()
                          if e['run_family'] == 'run7b' and e['seed'] == s and e['iter'] == 40000), None)
        if run7b_40k and label_40k:
            add_pair(f'pair10_affonly02_vs_affonly04_s{s}', run7b_40k, label_40k)

    if len(run7_seeds) >= 2:
        for i in range(len(run7_seeds)):
            for j in range(i + 1, len(run7_seeds)):
                sA, sB = run7_seeds[i], run7_seeds[j]
                label_a = next((l for l, e in manifest.items()
                               if e['run_family'] == 'run7' and e['seed'] == sA and e['iter'] == 40000), None)
                label_b = next((l for l, e in manifest.items()
                               if e['run_family'] == 'run7' and e['seed'] == sB and e['iter'] == 40000), None)
                if label_a and label_b:
                    add_pair(f'pair11_affonly_s{sA}_vs_s{sB}', label_a, label_b)

    if not all_rows:
        raise RuntimeError("Khong tinh duoc cap nao — kiem tra lai --checkpoint/--dump-dir.")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'bootstrap_ci_run7.csv')
    md_path = os.path.join(args.output_dir, 'bootstrap_ci_run7.md')
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('# Bootstrap CI Run 7 (Affinity-only) — cap 7-10, tach biet output/bootstrap/ cu\n\n')
        f.write(f'Quy uoc gop: **{convention}**' +
                (f', chinh sach anh rong: **{empty_policy}**\n\n' if convention == 'macro' else '\n\n'))
        f.write(f'B={args.n_boot}, seed={args.seed}\n\n')
        f.write('| Cap | Dai luong | Median | CI 95% | P(<0) | Observed |\n|---|---|---|---|---|---|\n')
        for r in all_rows:
            ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]" if r['ci_lo'] == r['ci_lo'] else 'N/A'
            f.write(f"| {r['pair_id']} | {r['quantity']} | {r['median']:.4f} | {ci} | "
                    f"{r['p_delta_lt_0']:.4f} | {r['observed_delta']:.4f} |\n")
    print(f"\nDa ghi {csv_path} va {md_path}.")

    write_interseed_amplitude(manifest, boundary_distances, args.diag_value,
                              os.path.join(args.output_dir, 'run7_interseed_amplitude.md'))


if __name__ == '__main__':
    main()
