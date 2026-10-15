"""Tools/hole_oracle_fill.py — K2-a cua docs/K2-ke-hoach-kiem-lo-gia-va-quyet-dinh-huong-giai-phap.md:
oracle lap lo CO CHON LOC (dung GT de chon lo -> TRAN, khong phai phuong phap).

CPU, khong can checkpoint. Doc mask du doan DA DUMP cua Buoc 1 (PNG 1024x1024, 0..8) + GT val
(get_val_transforms, Resize nearest -> 1024, cache .npy dung chung voi Tools/spurious_holes.py).

Dinh nghia lo (giong Tools/spurious_holes.py, KHONG doi):
  M_c = mask du doan lop c; lo = binary_fill_holes(M_c) - M_c, tach 4-lien thong (detect_holes_class).
  frac_gt_c = ti le pixel cua lo co GT = c (pixel ignore tinh la "khac c", nhu holes_for_image).
  KHONG ap A_min: lap moi lo co dien tich <= A_fill (giong B2 / fill_small_holes).

Bien the (moi A_fill in {64, 256}):
  F-all    moi lo                 tai lap B2 (fill_test.md) -> cong K-a1
  F-gia    frac_gt_c >= 0.5       ORACLE CHINH: tran cua viec chi xoa lo gia
  F-gia95  frac_gt_c >= 0.95      do nhay: chi lo ma nhan hoan toan dac
  F-that   frac_gt_c <  0.5       phan bu: cai gia cua viec xoa vat nho that
Thu tu lap y het fill_small_holes: lo lon truoc, lo nho sau (lo long nhau: lo trong thang).

Do: Tools/per_image_dump.compute_per_image_stats + Tools/bootstrap_boundary_ci (micro), nhu B2:
mIoU-9, mIoU-8, BF precision/recall/score (theta=2px), ASD pred->gt, gt->pred. Δ = sau − truoc;
bootstrap ghep cap per-image 10.000 lan, rng 19. (Boundary IoU khong can -> boundary_distances=().)

Cong:
  K-a1  F-all tai lap fill_test.md (Static s19, A_fill 64: Δprecision 0.01304, Δrecall −0.01633), sai so < 1e-4;
        neu co fill_test.csv cua Buoc 1 (--ref-fill-test) thi doi chieu MOI dong trung (nhan, A_fill, dai luong).
        Kem: mask F-all trung BIT voi spurious_holes.fill_small_holes (moi anh cua Static s19; 24 anh dau cua nhan khac).
  K-a2  Tap lo cua F-gia va F-that roi nhau, hop lai = F-all (dem lo khop tuyet doi); F-gia95 ⊆ F-gia.
  K-a0  (thong tin, khong chan) mIoU-9 "truoc" khop expected_miou9 cua Buoc 1 (sai so 1e-3).

Subcommand:
  measure  do (song song --workers tien trinh) -> per_image/*.csv, done_<nhan>.json, roi goi report.
  report   chi tong hop tu per_image/*.csv -> oracle_fill.{csv,md}, gates_oracle.json (+ §4.2).

Output (mac dinh output/hole_oracle/): oracle_fill.md, oracle_fill.csv, gates_oracle.json, per_image/, run_log.txt.
Huong dan Kaggle: docs/huong-dan-kaggle-K2.md
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (
    CHECKPOINTS, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, SEED, CFG_STATIC,
    parse_label_values, check_labels_known, get_git_commit, write_json, read_json, append_log, load_gt_cached,
)

K2_LABELS = ('static_s19_36k', 'static_s86_36k', 'bce04_40k', 'baseline_40k')
SEED_LABELS = ('static_s19_36k', 'static_s86_36k')      # §4.2: "ca hai seed Static"
DUMPNAME = {                                             # = scripts/run_grad_conflict_kaggle.sh
    'static_s19_36k': 'run3_static_s19_iter36000',
    'static_s86_36k': 'run5_static_s86_best_iter36000',
    'bce04_40k': 'run2b_bce04_iter40000',
    'baseline_40k': 'run1_baseline_iter40000',
}
A_FILLS = (64, 256)
SPURIOUS_FRAC = 0.5
SPURIOUS_FRAC_95 = 0.95
VARIANTS = ('F-all', 'F-gia', 'F-gia95', 'F-that')
VARIANT_DESC = {
    'F-all': 'moi lo (tai lap B2)',
    'F-gia': 'frac_gt_c >= 0.5 (oracle chinh)',
    'F-gia95': 'frac_gt_c >= 0.95 (do nhay)',
    'F-that': 'frac_gt_c < 0.5 (phan bu)',
}
QUANTITIES = ('mIoU_9', 'mIoU_8', 'bf_precision', 'bf_recall', 'bf_score', 'asd_pred_to_gt', 'asd_gt_to_pred')
BF_TOL = 2
NO_BIOU = ()                                             # khong tinh Boundary IoU (khong can cho K2)

KA1_LABEL, KA1_AFILL = 'static_s19_36k', 64
KA1_REF = {'bf_precision': 0.01304, 'bf_recall': -0.01633}   # fill_test.md cua Buoc 1 (lam tron 5 so)
KA1_TOL = 1e-4
KA0_TOL = 1e-3
BITCHECK_FIRST = 24           # nhan khac K-a1: kiem bit F-all tren 24 anh dau (tiet kiem ~20% thoi gian)
LEVER_PREC_MIN = 0.0014       # §4.2.1: bien do lien-seed precision cua Static
LEVER_RECALL_LO = -0.0094     # §4.2.2: phan cong huong recall Static − BCE


def _keep(variant, frac):
    if variant == 'F-all':
        return True
    if variant == 'F-gia':
        return frac >= SPURIOUS_FRAC
    if variant == 'F-gia95':
        return frac >= SPURIOUS_FRAC_95
    if variant == 'F-that':
        return frac < SPURIOUS_FRAC
    raise ValueError(variant)


# ═════════════════════════════ Lo + lap co chon loc ═══════════════════════════

def holes_with_frac(pred, gt):
    """Moi lo (moi lop, dien tich >= 1) theo DUNG thu tu fill_small_holes duyet:
    list dict {class_id, hole_id, area, sl, hmask, frac_gt_c}."""
    from Tools.spurious_holes import detect_holes_class
    out = []
    for c in range(NUM_CLASSES):
        holes, _ = detect_holes_class(pred == c, 1)
        for h in holes:
            g = gt[h['sl']][h['hmask']]
            out.append({'class_id': c, 'hole_id': h['hole_id'], 'area': h['area'], 'sl': h['sl'],
                        'hmask': h['hmask'], 'frac_gt_c': float((g == c).mean()) if g.size else 0.0})
    return out


def select_holes(holes, a_fill, variant):
    return [h for h in holes if h['area'] <= a_fill and _keep(variant, h['frac_gt_c'])]


def fill_holes(pred, selected):
    """Y het vong lap cua spurious_holes.fill_small_holes, nhung chi tren `selected`
    (sort on dinh theo -area: lo lon to truoc, lo nho to sau)."""
    out = pred.copy()
    jobs = sorted(((h['area'], h['class_id'], h) for h in selected), key=lambda t: -t[0])
    for _a, c, h in jobs:
        out[h['sl']][h['hmask']] = c
    return out


def hole_key(h):
    return (h['class_id'], h['hole_id'])


# ═════════════════════════════ Do 1 anh (worker) ══════════════════════════════

def image_stats(pred, gt, name):
    import torch
    from Tools.per_image_dump import compute_per_image_stats
    from Tools.oracle_boundary_ceiling import labels_to_pseudo_logits
    rows, _ = compute_per_image_stats(labels_to_pseudo_logits(pred, NUM_CLASSES), torch.from_numpy(gt)[None],
                                      [name], CLASS_NAMES, IGNORE_INDEX, 4, 0, NO_BIOU, BF_TOL)
    return rows[0]


def process_image(task):
    """task = (label, mask_dir, name, gt_path, a_fills, bitcheck). Tra ve dict:
    stats[(a_fill, variant) | 'before'] = hang per-image; counts = list hang dem lo; checks."""
    import torch
    from Tools.spurious_holes import load_pred_png, fill_small_holes
    torch.set_num_threads(1)
    label, mask_dir, name, gt_path, a_fills, bitcheck = task
    gt = np.load(gt_path).astype(np.int64)
    pred = load_pred_png(mask_dir, name)
    assert pred.shape == gt.shape, f"[{label}] {name}: mask {pred.shape} != GT {gt.shape}"
    holes = holes_with_frac(pred, gt)
    stats = {'before': image_stats(pred, gt, name)}
    counts, checks = [], {'fall_bit_mismatch': [], 'ka2_fail': []}
    cache = {}
    for a in a_fills:
        sel = {v: select_holes(holes, a, v) for v in VARIANTS}
        keys = {v: {hole_key(h) for h in sel[v]} for v in VARIANTS}
        ok = (not (keys['F-gia'] & keys['F-that']) and (keys['F-gia'] | keys['F-that']) == keys['F-all']
              and len(sel['F-gia']) + len(sel['F-that']) == len(sel['F-all']) and keys['F-gia95'] <= keys['F-gia'])
        if not ok:
            checks['ka2_fail'].append(a)
        for v in VARIANTS:
            ck = frozenset(keys[v])
            filled = fill_holes(pred, sel[v])
            if v == 'F-all' and bitcheck and not np.array_equal(filled, fill_small_holes(pred, a)):
                checks['fall_bit_mismatch'].append(a)
            if not ck:
                stats[(a, v)] = stats['before']                  # khong lap gi -> y het "truoc"
            elif ck in cache:
                stats[(a, v)] = cache[ck]
            else:
                stats[(a, v)] = cache[ck] = image_stats(filled, gt, name)
            counts.append({'image': name, 'a_fill': a, 'variant': v, 'n_holes': len(sel[v]),
                           'hole_px': int(sum(h['area'] for h in sel[v])),
                           'changed_px': int((filled != pred).sum())})
    return {'name': name, 'stats': stats, 'counts': counts, 'checks': checks, 'n_holes_all': len(holes),
            'bitchecked': bool(bitcheck)}


# ═════════════════════════════════ measure ════════════════════════════════════

def _per_image_path(out, label, key):
    tag = 'before' if key == 'before' else f"{key[1]}_a{key[0]}"
    return os.path.join(out, 'per_image', f'{label}__{tag}.csv')


def resolve_masks(args):
    masks = parse_label_values(args.masks, '--masks')
    if args.dump_root:
        for lb in K2_LABELS:
            d = os.path.join(args.dump_root, DUMPNAME[lb], 'masks')
            if lb not in masks and os.path.isdir(d):
                masks[lb] = d
    check_labels_known(masks, '--masks')
    if not masks:
        raise SystemExit("Khong co mask nao: truyen --dump-root <thu muc dump> hoac --masks LABEL=DIR.")
    return {lb: masks[lb] for lb in sorted(masks, key=lambda x: (K2_LABELS + (x,)).index(x))}


def cmd_measure(args):
    from multiprocessing import get_context
    from Tools.oracle_boundary_ceiling import load_cfg, build_val_dataset, image_ids_of
    from Tools.spurious_holes import _gate7_image_list

    masks = resolve_masks(args)
    a_fills = [int(x) for x in args.a_fill.split(',')]
    out = args.output_dir
    os.makedirs(os.path.join(out, 'per_image'), exist_ok=True)
    cfg = load_cfg(args.config, args.data_root)
    val_ds = build_val_dataset(cfg)
    image_ids = image_ids_of(val_ds)
    n = len(image_ids) if args.max_images is None else min(args.max_images, len(image_ids))
    append_log(os.path.join(out, 'run_log.txt'),
               f"[{datetime.datetime.now().isoformat(timespec='seconds')}] measure labels={list(masks)} "
               f"a_fills={a_fills} n={n} workers={args.workers} git={get_git_commit()} argv={' '.join(sys.argv)}")

    todo = []
    for label, mdir in masks.items():
        dp = os.path.join(out, f'done_{label}.json')
        if os.path.exists(dp) and not args.force:
            d = read_json(dp)
            if d.get('n_images') == n and d.get('a_fills') == a_fills and d.get('mask_dir') == mdir:
                print(f"[{label}] da do xong (done_{label}.json) — bo qua (--force de do lai).")
                continue
        if args.max_images is None:
            _gate7_image_list(label, mdir, image_ids, None)
            print(f"[{label}] mask dir: du {len(image_ids)} anh val — OK  ({mdir})")
        todo.append(label)

    if todo:
        t0 = time.time()
        gt_paths = []
        for i in range(n):
            name, _gt = load_gt_cached(val_ds, i, args.gt_cache)
            assert name == image_ids[i]
            gt_paths.append(os.path.join(args.gt_cache, f'{name}.npy'))
            print(f"  GT cache {i + 1}/{n} ({time.time() - t0:.0f}s)", end='\r')
        print()

    # 1 pool dung chung cho moi nhan (spawn + import torch ~10-40 s/lan)
    pool = get_context('spawn').Pool(args.workers) if todo and args.workers > 1 else None
    try:
        for label in todo:
            measure_label(label, masks[label], image_ids, gt_paths, n, a_fills, out, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if args.max_images is not None:
        args.dryrun = True
    cmd_report(args)


def measure_label(label, mask_dir, image_ids, gt_paths, n, a_fills, out, pool):
    t0 = time.time()
    # Kiem bit F-all == fill_small_holes: MOI anh cho nhan cua K-a1, BITCHECK_FIRST anh dau cho nhan khac
    tasks = [(label, mask_dir, image_ids[i], gt_paths[i], a_fills,
              label == KA1_LABEL or i < BITCHECK_FIRST) for i in range(n)]
    results = []
    if pool is not None:
        for k, r in enumerate(pool.imap(process_image, tasks, chunksize=1)):
            results.append(r)
            print(f"  [{label}] {k + 1}/{n} anh ({time.time() - t0:.0f}s)", end='\r')
    else:
        for k, t in enumerate(tasks):
            results.append(process_image(t))
            print(f"  [{label}] {k + 1}/{n} anh ({time.time() - t0:.0f}s)", end='\r')
    print()
    assert [r['name'] for r in results] == image_ids[:n]
    keys = ['before'] + [(a, v) for a in a_fills for v in VARIANTS]
    for key in keys:
        pd.DataFrame([r['stats'][key] for r in results]).to_csv(_per_image_path(out, label, key), index=False)
    pd.DataFrame([c for r in results for c in r['counts']]).to_csv(
        os.path.join(out, 'per_image', f'{label}__counts.csv'), index=False)
    checks = {
        'fall_bit_mismatch': sorted({(r['name'], a) for r in results for a in r['checks']['fall_bit_mismatch']}),
        'ka2_fail': sorted({(r['name'], a) for r in results for a in r['checks']['ka2_fail']}),
    }
    write_json({'label': label, 'mask_dir': mask_dir, 'n_images': n, 'a_fills': a_fills,
                'n_holes_all_sizes': int(sum(r['n_holes_all'] for r in results)),
                'n_images_bitchecked': int(sum(r['bitchecked'] for r in results)),
                'fall_bit_mismatch': [list(x) for x in checks['fall_bit_mismatch']],
                'ka2_fail': [list(x) for x in checks['ka2_fail']],
                'wall_time_s': time.time() - t0, 'workers': pool._processes if pool is not None else 1,
                'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'git_commit': get_git_commit()},
               os.path.join(out, f'done_{label}.json'))
    print(f"[{label}] xong sau {(time.time() - t0) / 60:.1f} phut — "
          f"F-all lech bit: {len(checks['fall_bit_mismatch'])}, K-a2 loi: {len(checks['ka2_fail'])}")


# ═════════════════════════════════ report ═════════════════════════════════════

def _fmt(x, nd=5):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x:+.{nd}f}' if nd >= 5 else f'{x:.{nd}f}'


def bootstrap_rows(out, label, a_fills, n_boot):
    from Tools.bootstrap_boundary_ci import aggregate_point_estimate, run_bootstrap, summarize_pair
    before = pd.read_csv(_per_image_path(out, label, 'before'))
    obs_b = aggregate_point_estimate(before, 'micro', 'skip', boundary_distances=NO_BIOU)
    counts = pd.read_csv(os.path.join(out, 'per_image', f'{label}__counts.csv'))
    rows = []
    for a in a_fills:
        for v in VARIANTS:
            after = pd.read_csv(_per_image_path(out, label, (a, v)))
            assert after['image'].tolist() == before['image'].tolist(), f"{label} {v} a{a}: lech thu tu anh"
            obs_a = aggregate_point_estimate(after, 'micro', 'skip', boundary_distances=NO_BIOU)
            va, vb = run_bootstrap(after, before, 'micro', 'skip', n_boot, SEED, boundary_distances=NO_BIOU)
            c = counts[(counts['a_fill'] == a) & (counts['variant'] == v)]
            for r in summarize_pair(va, vb, obs_a, obs_b, quantities=QUANTITIES):
                rows.append({'label': label, 'a_fill': a, 'variant': v, 'quantity': r['quantity'],
                             'before': obs_b[r['quantity']], 'after': obs_a[r['quantity']],
                             'delta': r['observed_delta'], 'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
                             'p_delta_lt_0': r['p_delta_lt_0'],
                             'ci_excludes_0': bool(r['ci_lo'] > 0 or r['ci_hi'] < 0),
                             'n_holes_filled': int(c['n_holes'].sum()), 'hole_px': int(c['hole_px'].sum()),
                             'changed_px': int(c['changed_px'].sum())})
    return rows


def _get(df, label, a, v, q, col='delta'):
    s = df[(df['label'] == label) & (df['a_fill'] == a) & (df['variant'] == v) & (df['quantity'] == q)]
    return float(s[col].iloc[0]) if len(s) else float('nan')


def find_ref_fill_test(arg):
    if arg:
        return arg if os.path.exists(arg) else None
    return None


def build_gates(df, out, labels, a_fills, ref_csv, dryrun):
    gates = []
    # K-a1
    if dryrun or KA1_LABEL not in labels or KA1_AFILL not in a_fills:
        gates.append({'gate': 'K-a1', 'status': 'SKIP_DRYRUN' if dryrun else 'SKIP',
                      'detail': 'dry-run (khong du 384 anh)' if dryrun else f'khong co {KA1_LABEL} / A_fill {KA1_AFILL}'})
    else:
        checks = []
        for q, ref in KA1_REF.items():
            got = _get(df, KA1_LABEL, KA1_AFILL, 'F-all', q)
            checks.append({'quantity': q, 'ref': ref, 'got': got, 'abs_err': abs(got - ref),
                           'pass': bool(abs(got - ref) < KA1_TOL), 'source': 'fill_test.md (K2 §1.4)'})
        if ref_csv:
            ref = pd.read_csv(ref_csv)
            for _, r in ref.iterrows():
                if r['label'] not in labels or int(r['a_fill']) not in a_fills or r['quantity'] not in QUANTITIES:
                    continue
                got = _get(df, r['label'], int(r['a_fill']), 'F-all', r['quantity'])
                err = abs(got - float(r['delta']))
                checks.append({'quantity': f"{r['label']}/a{int(r['a_fill'])}/{r['quantity']}", 'ref': float(r['delta']),
                               'got': got, 'abs_err': err, 'pass': bool(err < KA1_TOL), 'source': ref_csv})
        dn = {lb: read_json(os.path.join(out, f'done_{lb}.json')) for lb in labels}
        bit = {lb: len(dn[lb]['fall_bit_mismatch']) for lb in labels}
        nbit = {lb: dn[lb].get('n_images_bitchecked') for lb in labels}
        ok = all(c['pass'] for c in checks) and not any(bit.values())
        gates.append({'gate': 'K-a1', 'status': 'PASS' if ok else 'FAIL', 'tol': KA1_TOL, 'checks': checks,
                      'fall_bit_mismatch_images': bit, 'n_images_bitchecked': nbit,
                      'detail': f"{sum(c['pass'] for c in checks)}/{len(checks)} gia tri khop; "
                                f"F-all lech bit voi fill_small_holes: {sum(bit.values())} (anh, A_fill) "
                                f"tren {sum(v or 0 for v in nbit.values())} anh da kiem"})
    # K-a2
    det, ok = [], True
    for lb in labels:
        done = read_json(os.path.join(out, f'done_{lb}.json'))
        for a in a_fills:
            n = {v: _get(df, lb, a, v, 'mIoU_9', 'n_holes_filled') for v in VARIANTS}
            add = n['F-gia'] + n['F-that'] == n['F-all']
            sub = n['F-gia95'] <= n['F-gia']
            per_img = sum(1 for x in done['ka2_fail'] if x[1] == a)
            ok &= add and sub and per_img == 0
            det.append({'label': lb, 'a_fill': a, **{f'n_{v}': int(n[v]) for v in VARIANTS},
                        'gia_plus_that_eq_all': bool(add), 'gia95_subset_gia': bool(sub), 'images_failing': per_img})
    gates.append({'gate': 'K-a2', 'status': 'PASS' if ok else 'FAIL', 'checks': det,
                  'detail': 'F-gia ∩ F-that = ∅, F-gia ∪ F-that = F-all (dem lo + khoa (lop, hole_id) tung anh), '
                            'F-gia95 ⊆ F-gia'})
    # K-a0 (thong tin)
    det = []
    for lb in labels:
        exp = CHECKPOINTS[lb].get('expected_miou9')
        got = _get(df, lb, a_fills[0], 'F-all', 'mIoU_9', 'before')
        det.append({'label': lb, 'expected_miou9': exp, 'got': got,
                    'pass': None if exp is None or dryrun else bool(abs(got - exp) < KA0_TOL)})
    st = 'SKIP_DRYRUN' if dryrun else ('PASS' if all(d['pass'] in (True, None) for d in det) else 'WARN')
    gates.append({'gate': 'K-a0', 'status': st, 'blocking': False, 'tol': KA0_TOL, 'checks': det,
                  'detail': 'mIoU-9 "truoc" so voi Buoc 1 (CHECKPOINTS.expected_miou9) — chi thong tin'})
    return gates


def lever_decision(df, labels):
    """§4.2 — A_fill 64, CA HAI seed Static, bien the F-gia."""
    per = {}
    for lb in SEED_LABELS:
        if lb not in labels:
            per[lb] = None
            continue
        dp = _get(df, lb, 64, 'F-gia', 'bf_precision')
        p_lo, p_hi = _get(df, lb, 64, 'F-gia', 'bf_precision', 'ci_lo'), _get(df, lb, 64, 'F-gia', 'bf_precision', 'ci_hi')
        r_lo = _get(df, lb, 64, 'F-gia', 'bf_recall', 'ci_lo')
        c1 = bool((p_lo > 0 or p_hi < 0) and dp > LEVER_PREC_MIN)
        c2 = bool(r_lo > LEVER_RECALL_LO)
        dr_all = _get(df, lb, 64, 'F-all', 'bf_recall')
        dr_that = _get(df, lb, 64, 'F-that', 'bf_recall')
        per[lb] = {'d_precision': dp, 'precision_ci': [p_lo, p_hi], 'cond1_precision': c1,
                   'd_recall': _get(df, lb, 64, 'F-gia', 'bf_recall'), 'recall_ci_lo': r_lo, 'cond2_recall': c2,
                   'lever': c1 and c2,
                   'prediction_recall_loss_in_F_that': {
                       'd_recall_F_all': dr_all, 'd_recall_F_that': dr_that,
                       'share_F_that': dr_that / dr_all if dr_all else float('nan'),
                       'holds': bool(dr_all < 0 and dr_that / dr_all > 0.5)}}
    complete = all(per[lb] is not None for lb in SEED_LABELS)
    exists = complete and all(per[lb]['lever'] for lb in SEED_LABELS)
    return {'rule': f"A_fill 64, F-gia, ca 2 seed Static: (1) CI Δprecision loai 0 va Δ > {LEVER_PREC_MIN}; "
                    f"(2) can duoi CI Δrecall > {LEVER_RECALL_LO}",
            'per_seed': per, 'complete': complete,
            'lever_exists': exists if complete else None}


def write_md(df, gates, lever, labels, a_fills, n_boot, n_images, path):
    L = ['# K2-a — Oracle lap lo co chon loc', '',
         f"Lap lo co dien tich <= A_fill ∈ {a_fills} (khong ap A_min), chon lo theo `frac_gt_c` (DUNG GT -> **tran**, "
         f"khong phai phuong phap). Do bang `Tools/per_image_dump.compute_per_image_stats` + `Tools/bootstrap_boundary_ci` "
         f"(micro). Bootstrap ghep cap per-image {n_boot} lan, rng {SEED}; {n_images} anh. "
         f"precision/recall = BF precision/recall (theta = {BF_TOL} px). Δ = sau − truoc. "
         f"Tao {datetime.datetime.now().isoformat(timespec='seconds')} · git {get_git_commit()[:10]}", '']
    if n_images < 384:
        L += [f'> **DRY-RUN ({n_images} anh)** — chi kiem duong chay; so lieu va ket luan §4.2 duoi day KHONG dung duoc.', '']
    L += ['## Cong', '', '| Cong | Trang thai | Chi tiet |', '|---|---|---|']
    for g in gates:
        L.append(f"| {g['gate']} | **{g['status']}** | {g['detail']} |")
    L.append('')
    L += ['## §4.2 — Don bay lo gia (A_fill 64, F-gia, ca hai seed Static)', '', f"Luat: {lever['rule']}", '']
    if lever['complete']:
        L.append(f"**Don bay lo gia ton tai: {'CO' if lever['lever_exists'] else 'KHONG'}**")
    else:
        L.append('**Chua du hai seed Static -> chua ket luan §4.2**')
    L += ['', '| Seed | Δprecision | CI 95% | (1) | Δrecall | can duoi CI | (2) | Don bay |', '|---|---|---|---|---|---|---|---|']
    for lb, v in lever['per_seed'].items():
        if v is None:
            L.append(f'| {lb} | — | — | — | — | — | — | thieu |')
            continue
        L.append(f"| {lb} | {_fmt(v['d_precision'])} | [{_fmt(v['precision_ci'][0])}, {_fmt(v['precision_ci'][1])}] | "
                 f"{'dat' if v['cond1_precision'] else 'khong'} | {_fmt(v['d_recall'])} | {_fmt(v['recall_ci_lo'])} | "
                 f"{'dat' if v['cond2_recall'] else 'khong'} | **{'CO' if v['lever'] else 'KHONG'}** |")
    L += ['', 'Du doan ghi truoc (de kiem, KHONG dung quyet dinh): phan lon mat recall cua B2 nam o F-that.', '',
          '| Seed | Δrecall F-all | Δrecall F-that | Ti le F-that / F-all | Du doan dung |', '|---|---|---|---|---|']
    for lb, v in lever['per_seed'].items():
        if v is None:
            continue
        p = v['prediction_recall_loss_in_F_that']
        L.append(f"| {lb} | {_fmt(p['d_recall_F_all'])} | {_fmt(p['d_recall_F_that'])} | "
                 f"{_fmt(p['share_F_that'], 3)} | {'co' if p['holds'] else 'khong'} |")
    L.append('')
    short = {'mIoU_9': 'ΔmIoU-9', 'bf_precision': 'ΔBF-P', 'bf_recall': 'ΔBF-R', 'bf_score': 'ΔBF-F',
             'asd_pred_to_gt': 'ΔASD p→g', 'asd_gt_to_pred': 'ΔASD g→p'}
    for lb in labels:
        L.append(f'## {lb}')
        L.append('')
        for a in a_fills:
            L.append(f'**A_fill = {a}** — CI 95% trong ngoac; `*` = CI loai tru 0')
            L.append('')
            L.append('| Bien the | So lo lap | Pixel doi | ' + ' | '.join(short.values()) + ' |')
            L.append('|---|---|---|' + '---|' * len(short))
            for v in VARIANTS:
                cells = []
                for q in short:
                    s = df[(df['label'] == lb) & (df['a_fill'] == a) & (df['variant'] == v) & (df['quantity'] == q)].iloc[0]
                    star = '*' if s['ci_excludes_0'] else ''
                    cells.append(f"{_fmt(s['delta'])}{star} [{_fmt(s['ci_lo'])}, {_fmt(s['ci_hi'])}]")
                s0 = df[(df['label'] == lb) & (df['a_fill'] == a) & (df['variant'] == v)].iloc[0]
                L.append(f"| {v} — {VARIANT_DESC[v]} | {s0['n_holes_filled']} | {s0['changed_px']} | " + ' | '.join(cells) + ' |')
            L.append('')
        b = df[(df['label'] == lb)].iloc[0]
        L.append(f"Truoc khi lap: mIoU-9 {b['before']:.4f} (cac dai luong khac: `oracle_fill.csv`, cot `before`).")
        L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def cmd_report(args):
    out = args.output_dir
    labels = [lb for lb in K2_LABELS if os.path.exists(os.path.join(out, f'done_{lb}.json'))]
    labels += sorted(f[5:-5] for f in os.listdir(out)
                     if f.startswith('done_') and f.endswith('.json') and f[5:-5] not in labels)
    if not labels:
        raise SystemExit(f"Khong co done_<nhan>.json trong {out} — chay 'measure' truoc.")
    dones = {lb: read_json(os.path.join(out, f'done_{lb}.json')) for lb in labels}
    a_fills = dones[labels[0]]['a_fills']
    n_images = dones[labels[0]]['n_images']
    dryrun = getattr(args, 'dryrun', False) or n_images < 384
    rows = []
    for lb in labels:
        rows += bootstrap_rows(out, lb, a_fills, args.n_boot)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, 'oracle_fill.csv'), index=False)
    gates = build_gates(df, out, labels, a_fills, find_ref_fill_test(args.ref_fill_test), dryrun)
    lever = lever_decision(df, labels)
    blocking = [g for g in gates if g.get('blocking', True)]
    overall = 'PASS' if all(g['status'] in ('PASS', 'SKIP', 'SKIP_DRYRUN') for g in blocking) else 'FAIL'
    write_json({'overall': overall, 'gates': gates, 'lever_42': lever, 'labels': labels, 'a_fills': a_fills,
                'n_images': n_images, 'n_boot': args.n_boot, 'dryrun': dryrun,
                'created_at': datetime.datetime.now().isoformat(timespec='seconds'), 'git_commit': get_git_commit()},
               os.path.join(out, 'gates_oracle.json'))
    write_md(df, gates, lever, labels, a_fills, args.n_boot, n_images, os.path.join(out, 'oracle_fill.md'))
    for g in gates:
        print(f"  [{g['status']}] {g['gate']}: {g['detail']}")
    print(f"  §4.2 don bay lo gia: {lever['lever_exists']}")
    print(f"Da ghi oracle_fill.csv/.md, gates_oracle.json trong {out} — cong: {overall}")
    if overall != 'PASS':
        sys.exit(1)


# ═════════════════════════════════ CLI ════════════════════════════════════════

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--output-dir', default='output/hole_oracle')
        p.add_argument('--n-boot', type=int, default=10000)
        p.add_argument('--ref-fill-test', default=None,
                       help='(tuy chon) fill_test.csv cua Buoc 1 -> K-a1 doi chieu moi dong trung')

    p = sub.add_parser('measure', help='Do 4 bien the x A_fill cho tung nhan, roi report')
    common(p)
    p.add_argument('--dump-root', default=None, help='thu muc dump/ (chua run3_static_s19_iter36000/masks, ...)')
    p.add_argument('--masks', action='append', default=[], metavar='LABEL=DIR', help='ghi de / them thu muc mask')
    p.add_argument('--config', default=CFG_STATIC, help='YAML de doc DATASET (val split dung chung)')
    p.add_argument('--data-root', default=None)
    p.add_argument('--gt-cache', default='output/_cache/gt1024')
    p.add_argument('--a-fill', default=','.join(str(a) for a in A_FILLS))
    p.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument('--max-images', type=int, default=None, help='dry-run: chi N anh dau (cong K-a1 -> SKIP)')
    p.add_argument('--force', action='store_true')

    p = sub.add_parser('report', help='Chi tong hop tu per_image/*.csv')
    common(p)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    {'measure': cmd_measure, 'report': cmd_report}[args.cmd](args)


if __name__ == '__main__':
    main()
