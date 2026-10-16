"""Tools/p0_rerun_eval.py — P0 cua docs/S4-spec-P0-R0-ban-giao-claude-code.md §2:
JSON eval hau ky cho run Static s19 chay lai o Phan D (snapshot iter_36000 / iter_40000).

Viec (§2.2):
  1. Kiem fingerprint 2 snapshot (P0-g2) — lech => DUNG, khong forward.
  2. Forward 384 anh val (model.eval(), fp32) bang CHINH Tools/eval_boundary_metrics.run_eval_and_dump:
     mask PNG (0..8, mode 'L' — cung dinh dang dump Buoc 1 / K2), per_image_stats.csv (+ _meta.json),
     eval.json (cung khoa voi `eval_boundary_metrics.py --output`).
  3. Cong P0-g1: mIoU-9 hau ky khop CSV val trong train (0.6404 @36k, 0.6548 @40k), dung sai 0.001.
  4. §2.4 (mo ta): so BF-P/R/F, BIoU@{1,2,4}, ASD hai chieu voi benchmark_results.csv hang iter 36000/40000.

Output (mac dinh output/refine/p0_rerun/):
  static_s19r_36k/ static_s19r_40k/   eval.json, per_image_stats.csv, per_image_stats_meta.json, masks/*.png
  p0_compare_csv.md, gates_p0.json, run_log.txt

  python Tools/p0_rerun_eval.py --snapshot-dir <.../phanD/snapshots> --data-root <OpenEarthMap> \\
      --bench-csv <.../phanD/logs/benchmark_results.csv>
"""

import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (  # noqa: E402
    CFG_STATIC, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, append_log, file_fingerprint, get_git_commit, read_json,
    write_json,
)
from Tools.refine_common import P0_CONFIG_MD5, P0_LABELS, P0_SNAPSHOTS, ckpt_info, file_md5  # noqa: E402

BOUNDARY_DISTANCES = (1, 2, 4)
BF_TOL = 2
G1_TOL = 1e-3
# (ten hien thi, cot benchmark_results.csv, ham lay gia tri hau ky tu (eval.json, aggregate per-image))
COMPARE = [
    ('mIoU-9', 'miou9', lambda e, a: e['segmentation']['mIoU']),
    ('mIoU-8', 'miou8', lambda e, a: a['mIoU_8']),
    ('BF-P', 'bf_precision', lambda e, a: e['boundary']['bf_precision']),
    ('BF-R', 'bf_recall', lambda e, a: e['boundary']['bf_recall']),
    ('BF-F', 'bf_score', lambda e, a: e['boundary']['bf_score']),
    ('BIoU@1', 'biou_d1', lambda e, a: e['boundary']['boundary_iou_d1']),
    ('BIoU@2', 'biou_d2', lambda e, a: e['boundary']['boundary_iou_d2']),
    ('BIoU@4', 'biou_d4', lambda e, a: e['boundary']['boundary_iou_d4']),
    ('ASD pred→gt', 'asd_pred_to_gt', lambda e, a: e['boundary']['asd_pred_to_gt']),
    ('ASD gt→pred', 'asd_gt_to_pred', lambda e, a: e['boundary']['asd_gt_to_pred']),
    ('ASD', 'asd', lambda e, a: e['boundary']['asd']),
]


def gate_fingerprints(snap_dir, labels):
    checks, ok = [], True
    for lb in labels:
        s = P0_SNAPSHOTS[lb]
        p = os.path.join(snap_dir, s['file'])
        got = file_fingerprint(p) if os.path.exists(p) else 'KHONG CO FILE'
        match = got == s['fingerprint']
        ok &= match
        checks.append({'label': lb, 'file': p, 'expected': s['fingerprint'], 'got': got, 'match': match})
    return {'gate': 'P0-g2', 'status': 'PASS' if ok else 'FAIL', 'threshold': 'khop tuyet doi', 'checks': checks,
            'detail': f"{sum(c['match'] for c in checks)}/{len(checks)} snapshot khop fingerprint"}


def eval_label(label, path, args, cfg, device, out_dir, n_images):
    import torch
    from torch.utils.data import DataLoader, Subset
    from Tools.eval_boundary_metrics import run_eval_and_dump
    from Tools.oracle_boundary_ceiling import build_val_dataset, image_ids_of
    from Tools.per_image_dump import per_image_stats_columns
    from Tools.refine_common import load_model

    info = ckpt_info(label)
    val_ds = build_val_dataset(cfg)
    names = image_ids_of(val_ds)
    if n_images < len(val_ds):
        val_ds = Subset(val_ds, list(range(n_images)))
        names = names[:n_images]
    loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=(device.type == 'cuda'))
    model = load_model(label, path, cfg, device)
    bcfg = cfg.get('BOUNDARY_LOSS', {})
    conn, dil = bcfg.get('CONNECTIVITY', 4), bcfg.get('DILATION_RADIUS', 0)
    ld = os.path.join(out_dir, label)
    os.makedirs(ld, exist_ok=True)
    seg, bnd, n, elapsed, rows = run_eval_and_dump(
        model, info['model_type'], loader, device, NUM_CLASSES, IGNORE_INDEX, conn, dil, BOUNDARY_DISTANCES, BF_TOL,
        names, dump_preds_dir=os.path.join(ld, 'masks'))
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    stats_csv = os.path.join(ld, 'per_image_stats.csv')
    with open(stats_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=per_image_stats_columns(CLASS_NAMES, BOUNDARY_DISTANCES))
        w.writeheader()
        w.writerows(rows)
    git = get_git_commit()
    write_json({'checkpoint_path': path, 'checkpoint_iter': info['iter'], 'run_name': label,
                'val_split_file': cfg['DATASET'].get('VAL_SPLIT_FILE'), 'n_images': n,
                'bf_tolerance_theta': BF_TOL, 'biou_dilations': list(BOUNDARY_DISTANCES), 'connectivity': conn,
                'ignore_index': IGNORE_INDEX, 'class_names': CLASS_NAMES,
                'asd_distance_transform': 'scipy.ndimage.distance_transform_edt',
                'empty_image_policy_observed': None, 'git_commit': git, 'eval_wall_time_s': elapsed},
               os.path.join(ld, 'per_image_stats_meta.json'))
    write_json({'checkpoint': path, 'config': args.config, 'model_type': info['model_type'], 'device': str(device),
                'n_images': n, 'elapsed_sec': elapsed, 'boundary_distances': list(BOUNDARY_DISTANCES),
                'bf_tolerance': BF_TOL, 'connectivity': conn, 'dilation_radius': dil,
                'segmentation': seg, 'boundary': bnd,
                # them cho P0 (khong co trong eval_boundary_metrics --output):
                'label': label, 'fingerprint': file_fingerprint(path), 'eval_mode': 'model.eval(), fp32',
                'git_commit': git, 'created_at': datetime.datetime.now().isoformat(timespec='seconds')},
               os.path.join(ld, 'eval.json'))
    print(f"[{label}] {n} anh, {elapsed / 60:.1f} phut — mIoU-9 {seg['mIoU']:.5f}, BF-F {bnd['bf_score']:.5f}, "
          f"ASD p→g {bnd['asd_pred_to_gt']:.4f} g→p {bnd['asd_gt_to_pred']:.4f}")


def read_bench(path):
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return {int(r['iter']): r for _, r in df.iterrows()}


def report(args, labels, n_images, g2):
    from Tools.bootstrap_boundary_ci import aggregate_point_estimate
    out = args.output_dir
    dryrun = n_images < 384
    bench = read_bench(args.bench_csv)
    evals, aggs = {}, {}
    for lb in labels:
        evals[lb] = read_json(os.path.join(out, lb, 'eval.json'))
        aggs[lb] = aggregate_point_estimate(pd.read_csv(os.path.join(out, lb, 'per_image_stats.csv')), 'micro', 'skip')

    # P0-g1
    checks = []
    for lb in labels:
        got = evals[lb]['segmentation']['mIoU']
        ref = P0_SNAPSHOTS[lb]['csv_miou9']
        checks.append({'label': lb, 'expected': ref, 'got': got, 'abs_diff': abs(got - ref),
                       'pass': None if dryrun else bool(abs(got - ref) <= G1_TOL)})
    g1 = {'gate': 'P0-g1', 'tol': G1_TOL, 'checks': checks,
          'status': 'SKIP_DRYRUN' if dryrun else ('PASS' if all(c['pass'] for c in checks) else 'FAIL'),
          'detail': '; '.join(f"{c['label']} {c['got']:.5f} vs {c['expected']:.4f} (|Δ| {c['abs_diff']:.5f})"
                              for c in checks)}
    # nhat quan noi bo (thong tin): aggregate micro tu per-image CSV == eval.json
    cons = []
    for lb in labels:
        e, a = evals[lb], aggs[lb]
        diffs = {'mIoU_9': abs(e['segmentation']['mIoU'] - a['mIoU_9']),
                 'bf_score': abs(e['boundary']['bf_score'] - a['bf_score']),
                 'asd_pred_to_gt': abs(e['boundary']['asd_pred_to_gt'] - a['asd_pred_to_gt']),
                 'asd_gt_to_pred': abs(e['boundary']['asd_gt_to_pred'] - a['asd_gt_to_pred'])}
        cons.append({'label': lb, 'max_abs_diff': max(diffs.values()), 'diffs': diffs})
    cmd5 = file_md5(args.config)
    gates = [g1, g2, {'gate': 'P0-info', 'status': 'INFO', 'blocking': False,
                      'config': args.config, 'config_md5': cmd5, 'config_md5_expected': P0_CONFIG_MD5,
                      'config_md5_match': cmd5 == P0_CONFIG_MD5, 'per_image_vs_eval_json': cons,
                      'detail': f"md5 config {'khop' if cmd5 == P0_CONFIG_MD5 else 'LECH'}; per-image micro vs "
                                f"eval.json lech toi da {max(c['max_abs_diff'] for c in cons):.2e}"}]
    overall = 'PASS' if all(g['status'] in ('PASS', 'SKIP_DRYRUN', 'INFO') for g in gates) else 'FAIL'
    write_json({'overall': overall, 'gates': gates, 'labels': labels, 'n_images': n_images, 'dryrun': dryrun,
                'bench_csv': args.bench_csv, 'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'git_commit': get_git_commit()}, os.path.join(out, 'gates_p0.json'))

    # §2.4
    L = ['# P0 — JSON hau ky run Static s19 chay lai (Phan D) vs CSV val trong train', '',
         f"Snapshot Phan D, val {n_images} anh, `model.eval()` fp32, `Tools/eval_boundary_metrics.run_eval_and_dump` "
         f"(BF θ = {BF_TOL} px, BIoU d ∈ {list(BOUNDARY_DISTANCES)} = trung binh 9 lop nhu BoundaryMetrics). "
         f"CSV: `{args.bench_csv}` (4 chu so). Tao {datetime.datetime.now().isoformat(timespec='seconds')} · "
         f"git {get_git_commit()[:10]}", '']
    if dryrun:
        L += [f'> **DRY-RUN ({n_images} anh)** — so lieu KHONG dung duoc.', '']
    L += ['## Cong', '', '| Cong | Trang thai | Chi tiet |', '|---|---|---|']
    for g in gates:
        L.append(f"| {g['gate']} | **{g['status']}** | {g['detail']} |")
    L.append('')
    L += ['## So voi CSV val trong train (§2.4, mo ta — khong cong)', '']
    if bench is None:
        L.append('_Khong co benchmark_results.csv (--bench-csv) — chi in gia tri hau ky._')
        L.append('')
    rows_csv = []
    for lb in labels:
        it = P0_SNAPSHOTS[lb]['iter']
        L += [f'### {lb} (iter {it})', '', '| Chi so | Hau ky | CSV train | |Δ| |', '|---|---|---|---|']
        r = bench.get(it) if bench is not None else None
        for name, col, fn in COMPARE:
            post = float(fn(evals[lb], aggs[lb]))
            ref = float(r[col]) if r is not None and col in r else float('nan')
            d = abs(post - ref)
            L.append(f"| {name} | {post:.5f} | {'—' if np.isnan(ref) else f'{ref:.4f}'} | "
                     f"{'—' if np.isnan(d) else f'{d:.5f}'} |")
            rows_csv.append({'label': lb, 'iter': it, 'metric': col, 'post_hoc': post, 'csv_train': ref, 'abs_diff': d})
        L.append('')
    pd.DataFrame(rows_csv).to_csv(os.path.join(out, 'p0_compare_csv.csv'), index=False)
    L += ['Ghi chu: CSV train lam tron 4 chu so ⇒ |Δ| ≤ 0.00005 la khop toi do phan giai cua CSV. '
          'BIoU o day la trung binh 9 lop (cung dinh nghia cot `biou_d*` cua CSV), khac `biou_8_d*` (8 lop) '
          'cua bootstrap R0.', '']
    with open(os.path.join(out, 'p0_compare_csv.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    for g in gates:
        print(f"  [{g['status']}] {g['gate']}: {g['detail']}")
    print(f"Da ghi gates_p0.json, p0_compare_csv.md trong {out} — cong: {overall}")
    return overall


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--snapshot-dir', required=True, help='thu muc chua iter_36000.pth, iter_40000.pth (Phan D)')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--config', default=CFG_STATIC)
    ap.add_argument('--bench-csv', default=None, help='Phan D logs/benchmark_results.csv (§2.4)')
    ap.add_argument('--output-dir', default='output/refine/p0_rerun')
    ap.add_argument('--labels', default=','.join(P0_LABELS))
    ap.add_argument('--max-images', type=int, default=None, help='dry-run: N anh dau (P0-g1 -> SKIP)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--device', default=None)
    ap.add_argument('--force', action='store_true', help='do lai ca nhan da co eval.json')
    args = ap.parse_args(argv)

    from Tools.oracle_boundary_ceiling import load_cfg
    from Tools.refine_common import resolve_device
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    labels = [x for x in args.labels.split(',') if x]
    append_log(os.path.join(out, 'run_log.txt'),
               f"[{datetime.datetime.now().isoformat(timespec='seconds')}] p0 labels={labels} "
               f"max_images={args.max_images} git={get_git_commit()} argv={' '.join(sys.argv)}")

    g2 = gate_fingerprints(args.snapshot_dir, labels)
    print(f"  [{g2['status']}] P0-g2: {g2['detail']}")
    for c in g2['checks']:
        print(f"      {c['label']}: {c['got']}  (ky vong {c['expected']})")
    if g2['status'] != 'PASS':
        write_json({'overall': 'FAIL', 'gates': [g2], 'labels': labels, 'stopped_at': 'P0-g2',
                    'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                    'git_commit': get_git_commit()}, os.path.join(out, 'gates_p0.json'))
        raise SystemExit("P0-g2 FAIL: fingerprint snapshot lech — DUNG (§2.2 buoc 1).")

    cfg = load_cfg(args.config, args.data_root)
    from Tools.oracle_boundary_ceiling import build_val_dataset
    n_val = len(build_val_dataset(cfg))
    n_images = n_val if args.max_images is None else min(args.max_images, n_val)
    device = resolve_device(args.device)
    for lb in labels:
        ej = os.path.join(out, lb, 'eval.json')
        if os.path.exists(ej) and not args.force and read_json(ej).get('n_images') == n_images:
            print(f"[{lb}] da co eval.json ({n_images} anh) — bo qua (--force de do lai).")
            continue
        t0 = time.time()
        eval_label(lb, os.path.join(args.snapshot_dir, P0_SNAPSHOTS[lb]['file']), args, cfg, device, out, n_images)
        append_log(os.path.join(out, 'run_log.txt'), f"  {lb}: xong {time.time() - t0:.0f}s device={device}")
    overall = report(args, labels, n_images, g2)
    if overall != 'PASS':
        sys.exit(1)


if __name__ == '__main__':
    main()
