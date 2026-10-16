"""Tools/test_refine.py — test CPU (du lieu tong hop) cho P0/R0 (docs/S4-spec-P0-R0-ban-giao-claude-code.md):
Tools/refine_common.py, Tools/p0_rerun_eval.py, Tools/r0_pcrf.py.

    python Tools/test_refine.py                # test CRF that bi SKIP neu chua cai pydensecrf
    python Tools/test_refine.py --require-crf  # (Kaggle) thieu pydensecrf -> FAIL
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools import r0_pcrf as R  # noqa: E402
from Tools import p0_rerun_eval as P  # noqa: E402
from Tools.grad_conflict_common import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES  # noqa: E402
from Tools.refine_common import bounded_imap, per_image_stats  # noqa: E402

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def synth(seed, size=96):
    """gt nhieu hinh chu nhat; pred = gt dich + nhieu dom nho."""
    rng = np.random.default_rng(seed)
    gt = np.full((size, size), 2, dtype=np.int64)
    for _ in range(12):
        c = int(rng.integers(0, NUM_CLASSES))
        y, x = rng.integers(0, size - 20, 2)
        h, w = rng.integers(4, 30, 2)
        gt[y:y + h, x:x + w] = c
    pred = np.roll(gt, int(rng.integers(-2, 3)), axis=1).copy()
    for _ in range(15):
        y, x = rng.integers(0, size - 4, 2)
        pred[y:y + 3, x:x + 3] = int(rng.integers(0, NUM_CLASSES))
    gt[0:2, :] = IGNORE_INDEX
    return pred.astype(np.uint8), gt


def synth_prob(pred, seed, sharp=3.0):
    rng = np.random.default_rng(seed)
    logit = rng.normal(0, 1, (NUM_CLASSES,) + pred.shape).astype(np.float32)
    np.put_along_axis(logit, pred[None].astype(np.int64), sharp + 2.0, axis=0)
    e = np.exp(logit - logit.max(0, keepdims=True))
    return (e / e.sum(0, keepdims=True)).astype(np.float32)


def test_grid():
    print('1. Luoi chot truoc (§3.4)')
    check('27 to hop', len(R.GRID) == 27)
    order = [(g['bi_xy_std'], g['bi_rgb_std'], g['bi_w']) for g in R.GRID]
    check('thu tu tang dan (bi_xy, bi_rgb, bi_w)', order == sorted(order))
    check('gia tri dung spec', sorted({o[0] for o in order}) == [40, 80, 121] and sorted({o[1] for o in order}) == [3, 5, 10]
          and sorted({o[2] for o in order}) == [3, 4, 5])
    check('co dinh pos 3/3, 10 vong', R.FIXED == {'pos_xy_std': 3, 'pos_w': 3, 'n_iter': 10})
    check('G3 = (80, 5, 4)', R.GRID[R.G3_COMBO] == {'bi_xy_std': 80, 'bi_rgb_std': 5, 'bi_w': 4})


def test_select_tie():
    print('2. Chon to hop: min, hoa -> to hop dau theo thu tu luoi, bo NaN')
    rows = [{'combo': -1, 'asd_mean2': 0.1}] + [{'combo': k, 'asd_mean2': 5.0 - (k in (3, 7))} for k in range(27)]
    rows[1]['asd_mean2'] = float('nan')                  # combo 0 = NaN -> bo qua
    b = R.select_best(rows)
    check('hoa 3 & 7 -> 3; bo hang truoc CRF + NaN', b['combo'] == 3, f"chon {b['combo']}")


def test_unary():
    print('3. Unary (§3.2)')
    p = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)          # C=2, H=1, W=2
    u = R.unary_from_prob(p, 'softmax')
    check('kep p >= 1e-8', np.isclose(u.max(), -np.log(1e-8)) and u.min() == 0 and u.flags['C_CONTIGUOUS']
          and u.dtype == np.float32 and u.shape == (2, 2))
    q = synth_prob(np.zeros((4, 4), np.uint8), 0)
    ug = R.unary_from_prob(q, 'gt_prob').reshape(q.shape)
    check('gt_prob 0.7', np.allclose(ug[0], -np.log(0.7)) and np.allclose(ug[1:], -np.log(0.3 / (NUM_CLASSES - 1))))


def stats_df(preds, gts, bd):
    return pd.DataFrame([per_image_stats(p, g, f'im{i}', bd) for i, (p, g) in enumerate(zip(preds, gts))])


def test_bootstrap_extra():
    print('4. Bootstrap mo rong (BIoU 9 lop / tung lop) dung CUNG resample voi run_bootstrap')
    from Tools.bootstrap_boundary_ci import build_raw_matrix, run_bootstrap
    from src.utils.boundary_metrics import BoundaryMetrics
    from Tools.oracle_boundary_ceiling import labels_to_pseudo_logits
    pairs = [synth(s) for s in range(10)]
    after = [np.roll(p, 1, axis=0) for p, _ in pairs]
    da = stats_df(after, [g for _, g in pairs], R.BD)
    db = stats_df([p for p, _ in pairs], [g for _, g in pairs], R.BD)
    va, vb = run_bootstrap(da, db, 'micro', 'skip', 300, 19, boundary_distances=R.BD)
    ma, ci = build_raw_matrix(da, R.BD)
    cnt = R.boot_counts(len(da), 300, 19)
    v = R.biou_class_values(cnt @ ma, ci, 2)
    mine8 = R._nanmean_cols(v, CLASS_NAMES[1:])
    check('biou_8_d2 bootstrap trung run_bootstrap', np.allclose(mine8, va['biou_8_d2'], equal_nan=True))
    bm = BoundaryMetrics(NUM_CLASSES, IGNORE_INDEX, boundary_distances=R.BD, bf_tolerance=2, connectivity=4,
                         dilation_radius=0)
    for p, (_q, g) in zip(after, pairs):
        bm.update(labels_to_pseudo_logits(p, NUM_CLASSES), torch.from_numpy(g)[None])
    ref = bm.compute()
    ex = R.extra_quantities(ma.sum(0), ci)
    check('biou_9_d* diem == BoundaryMetrics (cot biou_d* CSV train)',
          all(abs(float(ex[f'biou_9_d{d}']) - ref[f'boundary_iou_d{d}']) < 1e-12 for d in R.BD))
    with tempfile.TemporaryDirectory() as tmp:
        da.to_csv(os.path.join(tmp, 'x__after.csv'), index=False)
        db.to_csv(os.path.join(tmp, 'x__before.csv'), index=False)
        rows, extra, sc = R.bootstrap_label(tmp, 'x', 300)
    df = pd.DataFrame(rows)
    check('du dai luong', set(df['quantity']) == set(R.QUANTITIES), f'{len(df)} hang')
    check('tu kiem biou_8_d2', sc < 1e-12)
    grp = pd.DataFrame(extra)
    st = float(grp[grp['quantity'] == 'biou_d2_grp_stuff']['after'].iloc[0])
    per = [float(grp[grp['quantity'] == f'biou_d2_{c}']['after'].iloc[0]) for c in R.STUFF]
    check('nhom stuff = trung binh 4 lop', abs(st - np.nanmean(per)) < 1e-12)


def _rule_df(vals):
    rows = []
    for lb, (p2g, p2g_hi, bi, bi_lo, g2p, rlo) in vals.items():
        rows += [dict(label=lb, quantity='asd_pred_to_gt', delta=p2g, ci_lo=p2g - 1, ci_hi=p2g_hi),
                 dict(label=lb, quantity='biou_8_d2', delta=bi, ci_lo=bi_lo, ci_hi=bi + 1),
                 dict(label=lb, quantity='biou_9_d2', delta=bi, ci_lo=bi_lo, ci_hi=bi + 1),
                 dict(label=lb, quantity='asd_gt_to_pred', delta=g2p, ci_lo=0, ci_hi=0),
                 dict(label=lb, quantity='bf_recall', delta=0.0, ci_lo=rlo, ci_hi=0)]
    return pd.DataFrame(rows)


def test_rule():
    print('5. Luat §3.6')
    good = (-0.3, -0.1, 0.01, 0.002, 0.2, -0.005)
    cases = {
        'R-co': {'static_s19_36k': good, 'static_s86_36k': good},
        'R-danh-doi': {'static_s19_36k': good, 'static_s86_36k': (-0.3, -0.1, 0.01, 0.002, 0.6, -0.005)},
        'R-danh-doi ': {'static_s19_36k': good, 'static_s86_36k': (-0.3, -0.1, 0.01, 0.002, 0.2, -0.0095)},
        'R-khong': {'static_s19_36k': (-0.3, 0.05, 0.01, 0.002, 0.2, -0.005),
                    'static_s86_36k': (-0.3, -0.1, 0.01, -0.001, 0.2, -0.005)},
        'KHONG-PHAN-LOAI': {'static_s19_36k': good, 'static_s86_36k': (0.1, 0.2, 0.01, 0.002, 0.2, -0.005)},
    }
    for want, v in cases.items():
        got = R.rule_36(_rule_df(v), list(v))['code']
        check(f'{want.strip()}', got == want.strip(), f'nhan {got}')
    edge = R.rule_36(_rule_df({'static_s19_36k': (-0.3, -0.1, 0.01, 0.002, 0.534, -0.0093),
                                'static_s86_36k': good}), list(DECISION()))['code']
    check('(iii) = 0.534 dung bien van dat; (iv) -0.0093 > -0.0094', edge == 'R-co', edge)


def DECISION():
    return R.DECISION_LABELS


def test_small_and_bands():
    print('6. Mo ta §3.8 (thanh phan nho, dai bien)')
    before = np.zeros((60, 60), np.uint8)
    before[5:10, 5:10] = 3           # 25 px -> bi xoa
    before[20:25, 20:25] = 3         # 25 px -> giu 1 px
    before[30:60, 30:60] = 5         # 900 px -> khong tinh (> 700)
    after = before.copy()
    after[5:10, 5:10] = 0
    after[20:25, 20:24] = 0
    sm = {r['class']: r for r in R.small_components(before, after, np.ones_like(before, bool))}
    check('Developed: 2 nho, 1 bi xoa', sm['Developed']['n_small'] == 2 and sm['Developed']['n_deleted'] == 1
          and sm['Developed']['px_deleted'] == 25)
    check('Tree 900 px khong tinh', sm['Tree']['n_small'] == 0)
    gt = np.zeros((40, 40), np.int64)
    gt[:, 20:] = 1
    d = R.gt_edge_dist(gt)
    check('khoang cach bien GT: cot 19/20 = 0', d[5, 19] == 0 and d[5, 20] == 0 and d[5, 10] == 9)


def test_bounded_imap():
    print('7. bounded_imap giu thu tu')
    check('tuan tu', list(bounded_imap(None, abs, [-3, 1, -2], 2)) == [3, 1, 2])


def test_fingerprint_gate():
    print('8. P0-g2 fingerprint')
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'iter_36000.pth'), 'wb') as f:
            f.write(b'x' * 1000)
        g = P.gate_fingerprints(tmp, ['static_s19r_36k', 'static_s19r_40k'])
        check('file sai / thieu -> FAIL', g['status'] == 'FAIL' and g['checks'][1]['got'] == 'KHONG CO FILE')


def test_aggregate_grid():
    print('9. Gop luoi (micro, ASD hai chieu trung binh)')
    from Tools.bootstrap_boundary_ci import aggregate_point_estimate
    pairs = [synth(s, 64) for s in range(3)]
    rows = []
    for k in [-1] + list(range(27)):
        for i, (p, g) in enumerate(pairs):
            q = p if k < 0 else np.roll(p, k % 3, axis=0)
            rows.append(dict(per_image_stats(q, g, f'im{i}', ()), combo=k))
    df = pd.DataFrame(rows)
    agg = R.aggregate_grid(df)
    a = aggregate_point_estimate(df[df['combo'] == 4], 'micro', 'skip', boundary_distances=())
    r4 = [r for r in agg if r['combo'] == 4][0]
    check('asd_mean2 = (p→g + g→p)/2 micro', abs(r4['asd_mean2'] - (a['asd_pred_to_gt'] + a['asd_gt_to_pred']) / 2) < 1e-12)
    check('28 hang (truoc + 27)', len(agg) == 28 and all(r['n_images'] == 3 for r in agg))


def test_fingerprint_parts():
    print('10. Dau van tay CRF (R0-g5): unary tu GT, hash')
    gt = np.array([[0, 8], [IGNORE_INDEX, 3]], dtype=np.int64)
    p = R.gt_unary_prob(gt)
    check('0.7 o nhan GT, 0.3/8 lop khac', np.isclose(p[0, 0, 0], 0.7) and np.isclose(p[8, 0, 1], 0.7)
          and np.isclose(p[1, 0, 0], 0.3 / 8) and np.allclose(p.sum(0), 1))
    check('pixel ignore = deu 1/9', np.allclose(p[:, 1, 0], 1 / 9))
    m = np.arange(12, dtype=np.uint8).reshape(3, 4)
    check('sha256 tat dinh, phan biet shape', R.mask_sha256(m) == R.mask_sha256(m.copy())
          and R.mask_sha256(m) != R.mask_sha256(m.reshape(4, 3)))
    check('tham so fingerprint = GRID[13] + co dinh', R.FP_COMBO == R.G3_COMBO and R.FP_N == 3)


def test_crf(require):
    print('11. CRF that (pydensecrf)')
    try:
        import pydensecrf.densecrf  # noqa: F401
    except Exception:
        check('pydensecrf co mat', not require, 'SKIP (chua cai)' if not require else 'THIEU')
        return
    pred, _gt = synth(3, 128)
    prob = synth_prob(pred, 3, sharp=0.5)
    rng = np.random.default_rng(0)
    rgb = (rng.integers(0, 255, (NUM_CLASSES, 3))[np.clip(pred, 0, 8)] + rng.integers(0, 8, pred.shape + (3,))).astype(np.uint8)
    p0 = {**R.FIXED, **R.GRID[0], 'bi_w': 0, 'pos_w': 0}
    out = R.crf_infer(prob, rgb, p0)
    agree = float((out == prob.argmax(0)).mean())
    check('chi unary == argmax (R0-g2)', agree >= R.G2_MIN, f'{agree:.6f}')
    p = {**R.FIXED, **R.GRID[R.G3_COMBO]}
    a, b = R.crf_infer(prob, rgb, p), R.crf_infer(prob, rgb, p)
    check('tat dinh (R0-g3)', np.array_equal(a, b))
    check('co pairwise -> nhan thay doi', int((a != prob.argmax(0)).sum()) > 0, f'{int((a != prob.argmax(0)).sum())} px')
    info = R.crf_backend_info()
    check('ghi duoc phien ban', 'version' in info, str(info))


if __name__ == '__main__':
    require = '--require-crf' in sys.argv
    for fn in (test_grid, test_select_tie, test_unary, test_bootstrap_extra, test_rule, test_small_and_bands,
               test_bounded_imap, test_fingerprint_gate, test_aggregate_grid, test_fingerprint_parts):
        fn()
    test_crf(require)
    n_fail = sum(1 for _n, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    sys.exit(1 if n_fail else 0)
