"""Tools/test_hole_oracle.py — test CPU (du lieu tong hop) cho K2:
Tools/hole_oracle_fill.py (K2-a) va Tools/hole_audit_sample.py (K2-b).

    python Tools/test_hole_oracle.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools import hole_oracle_fill as HO
from Tools import hole_audit_sample as HA
from Tools.grad_conflict_common import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES
from Tools.per_image_dump import compute_per_image_stats
from Tools.oracle_boundary_ceiling import labels_to_pseudo_logits
from Tools.spurious_holes import fill_small_holes

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")


def synth_pair(seed, size=192):
    """pred co nhieu lo (ke ca lo long nhau, lo cham bien), gt = pred voi mot nua so lo bi 'dac' lai."""
    rng = np.random.default_rng(seed)
    pred = np.full((size, size), 2, dtype=np.int64)
    gt = pred.copy()
    for _ in range(40):
        c = int(rng.integers(0, NUM_CLASSES))
        y, x = rng.integers(0, size - 30, 2)
        h, w = rng.integers(2, 26, 2)
        pred[y:y + h, x:x + w] = c
        if rng.random() < 0.5:
            gt[y:y + h, x:x + w] = c
    # lo long nhau: vanh 7 quanh dao 5 co lo 2
    pred[40:100, 120:180] = 7
    pred[55:85, 135:165] = 5
    pred[65:75, 145:155] = 7
    gt[40:100, 120:180] = 7
    gt[0:3, 0:size] = IGNORE_INDEX
    return pred, gt


def stats(pred, gt, bd):
    rows, _ = compute_per_image_stats(labels_to_pseudo_logits(pred, NUM_CLASSES), torch.from_numpy(gt)[None],
                                      ['x'], CLASS_NAMES, IGNORE_INDEX, 4, 0, bd, 2)
    return rows[0]


def test_fill_equivalence():
    print('1. F-all trung bit fill_small_holes; K-a2 cong tinh')
    ok_bit, ok_add, n_tot = True, True, 0
    for s in range(12):
        pred, gt = synth_pair(s)
        holes = HO.holes_with_frac(pred, gt)
        for a in (16, 64, 256):
            sel = {v: HO.select_holes(holes, a, v) for v in HO.VARIANTS}
            ok_bit &= np.array_equal(HO.fill_holes(pred, sel['F-all']), fill_small_holes(pred, a))
            k = {v: {HO.hole_key(h) for h in sel[v]} for v in HO.VARIANTS}
            ok_add &= (not (k['F-gia'] & k['F-that'])) and (k['F-gia'] | k['F-that']) == k['F-all'] \
                and k['F-gia95'] <= k['F-gia']
            n_tot += len(sel['F-all'])
    check('F-all == fill_small_holes (12 anh x 3 A_fill)', ok_bit)
    check('F-gia ⊔ F-that = F-all, F-gia95 ⊆ F-gia', ok_add, f'{n_tot} lo')
    check('co lo de test', n_tot > 50, f'{n_tot}')


def test_selective_fill_uses_gt():
    print('2. Chon lo theo frac_gt_c')
    pred = np.full((64, 64), 2, dtype=np.int64)
    pred[10:20, 10:20] = 5                     # lo 100 px, GT = 2 -> gia
    pred[40:50, 40:50] = 5                     # lo 100 px, GT = 5 -> that
    gt = np.full((64, 64), 2, dtype=np.int64)
    gt[40:50, 40:50] = 5
    holes = [h for h in HO.holes_with_frac(pred, gt) if h['class_id'] == 2]
    gia = HO.fill_holes(pred, HO.select_holes(holes, 256, 'F-gia'))
    that = HO.fill_holes(pred, HO.select_holes(holes, 256, 'F-that'))
    check('F-gia lap lo gia, giu lo that', (gia[15, 15] == 2) and (gia[45, 45] == 5))
    check('F-that lap lo that, giu lo gia', (that[15, 15] == 5) and (that[45, 45] == 2))
    check('A_fill 64 khong lap lo 100 px', np.array_equal(HO.fill_holes(pred, HO.select_holes(holes, 64, 'F-all')), pred))


def test_no_biou_same_stats():
    print('3. boundary_distances=() khong doi mIoU/BF/ASD (cot tho)')
    pred, gt = synth_pair(3)
    full, fast = stats(pred, gt, (1, 2, 4)), stats(pred, gt, ())
    same = all(full[k] == v for k, v in fast.items())
    check('moi cot cua ban nhanh == ban day du', same)
    check('ban nhanh khong co cot biou', not any(k.startswith('biou') for k in fast))


def test_bootstrap_equivalence():
    print('4. Bootstrap NO_BIOU == bootstrap (1,2,4) tren 7 dai luong K2')
    from Tools.bootstrap_boundary_ci import aggregate_point_estimate, run_bootstrap, summarize_pair
    rows_b, rows_a = [], []
    for s in range(10):
        pred, gt = synth_pair(100 + s, 96)
        rows_b.append(stats(pred, gt, (1, 2, 4)))
        rows_a.append(stats(fill_small_holes(pred, 64), gt, (1, 2, 4)))
    b, a = pd.DataFrame(rows_b), pd.DataFrame(rows_a)
    out = []
    for bd in ((1, 2, 4), ()):
        ob, oa = aggregate_point_estimate(b, 'micro', 'skip', boundary_distances=bd), \
            aggregate_point_estimate(a, 'micro', 'skip', boundary_distances=bd)
        va, vb = run_bootstrap(a, b, 'micro', 'skip', 500, 19, boundary_distances=bd)
        out.append({r['quantity']: (r['observed_delta'], r['ci_lo'], r['ci_hi'])
                    for r in summarize_pair(va, vb, oa, ob, quantities=HO.QUANTITIES)})
    same = all(np.allclose(out[0][q], out[1][q], equal_nan=True, rtol=0, atol=1e-12) for q in HO.QUANTITIES)
    check('Δ va CI trung', same)


def test_process_image_e2e():
    print('5. process_image + report e2e (2 nhan gia, 3 anh)')
    with tempfile.TemporaryDirectory() as tmp:
        names = ['img_a', 'img_b', 'img_c']
        mdirs = {lb: os.path.join(tmp, lb) for lb in ('static_s19_36k', 'static_s86_36k')}
        gts = []
        for k, n in enumerate(names):
            for j, d in enumerate(mdirs.values()):
                os.makedirs(d, exist_ok=True)
                pred, gt = synth_pair(10 * k + j, 128)
                Image.fromarray(pred.astype(np.uint8), mode='L').save(os.path.join(d, n + '.png'))
            _p, gt = synth_pair(10 * k, 128)
            gp = os.path.join(tmp, n + '.npy')
            np.save(gp, gt.astype(np.uint8))
            gts.append(gp)
        out = os.path.join(tmp, 'out')
        os.makedirs(os.path.join(out, 'per_image'))
        for lb, d in mdirs.items():
            res = [HO.process_image((lb, d, n, g, [64, 256], True)) for n, g in zip(names, gts)]
            for key in ['before'] + [(a, v) for a in (64, 256) for v in HO.VARIANTS]:
                pd.DataFrame([r['stats'][key] for r in res]).to_csv(HO._per_image_path(out, lb, key), index=False)
            pd.DataFrame([c for r in res for c in r['counts']]).to_csv(
                os.path.join(out, 'per_image', f'{lb}__counts.csv'), index=False)
            HO.write_json({'label': lb, 'mask_dir': d, 'n_images': 3, 'a_fills': [64, 256],
                           'fall_bit_mismatch': [], 'ka2_fail': [[r['name'], a] for r in res for a in r['checks']['ka2_fail']]},
                          os.path.join(out, f'done_{lb}.json'))
            p0 = HO.process_image((lb, d, names[0], gts[0], [64], True))
            pred0 = np.array(Image.open(os.path.join(d, names[0] + '.png'))).astype(np.int64)
            gt0 = np.load(gts[0]).astype(np.int64)
            direct = stats(HO.fill_holes(pred0, HO.select_holes(HO.holes_with_frac(pred0, gt0), 64, 'F-gia')), gt0, ())
            got = {k: v for k, v in p0['stats'][(64, 'F-gia')].items() if k != 'image'}
            check(f'[{lb}] stats F-gia a64 == tinh truc tiep', got == {k: v for k, v in direct.items() if k != 'image'})
        args = type('A', (), {'output_dir': out, 'n_boot': 200, 'ref_fill_test': None, 'dryrun': True})()
        try:
            HO.cmd_report(args)
            code = 0
        except SystemExit as e:
            code = e.code
        g = HO.read_json(os.path.join(out, 'gates_oracle.json'))
        st = {x['gate']: x['status'] for x in g['gates']}
        check('report chay, K-a2 PASS, K-a1 SKIP_DRYRUN', code in (0, None) and st['K-a2'] == 'PASS'
              and st['K-a1'] == 'SKIP_DRYRUN', str(st))
        df = pd.read_csv(os.path.join(out, 'oracle_fill.csv'))
        check('oracle_fill.csv du 2 nhan x 2 A_fill x 4 bien the x 7 dai luong', len(df) == 2 * 2 * 4 * 7, str(len(df)))
        check('oracle_fill.md co bang §4.2', '§4.2' in open(os.path.join(out, 'oracle_fill.md'), encoding='utf-8').read())


def test_lever_rule():
    print('6. Luat §4.2')
    rows = []
    for lb, dp, plo, rlo in (('static_s19_36k', 0.003, 0.001, -0.005), ('static_s86_36k', 0.002, 0.0005, -0.009)):
        for v in HO.VARIANTS:
            rows.append({'label': lb, 'a_fill': 64, 'variant': v, 'quantity': 'bf_precision', 'delta': dp,
                         'ci_lo': plo, 'ci_hi': dp * 2})
            rows.append({'label': lb, 'a_fill': 64, 'variant': v, 'quantity': 'bf_recall', 'delta': -0.002,
                         'ci_lo': rlo, 'ci_hi': 0.0})
    df = pd.DataFrame(rows)
    lv = HO.lever_decision(df, list(HO.SEED_LABELS))
    check('ca 2 seed dat -> don bay CO', lv['lever_exists'] is True)
    df.loc[(df['label'] == 'static_s86_36k') & (df['quantity'] == 'bf_precision'), 'delta'] = 0.0010
    check('Δprecision s86 <= 0.0014 -> KHONG', HO.lever_decision(df, list(HO.SEED_LABELS))['lever_exists'] is False)
    df.loc[(df['label'] == 'static_s86_36k') & (df['quantity'] == 'bf_precision'), 'delta'] = 0.002
    df.loc[(df['label'] == 'static_s19_36k') & (df['quantity'] == 'bf_recall'), 'ci_lo'] = -0.0095
    check('can duoi Δrecall < −0.0094 -> KHONG', HO.lever_decision(df, list(HO.SEED_LABELS))['lever_exists'] is False)
    check('thieu 1 seed -> chua ket luan', HO.lever_decision(df, ['static_s19_36k'])['lever_exists'] is None)


def fake_population(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    sizes = {'64-149': (64, 149), '150-699': (150, 699), '>=700': (700, 3000)}
    for g, members in HA.GROUPS:
        for b, (lo, hi) in sizes.items():
            n = 3 if (g == 'Khac' and b == '>=700') else 25
            for i in range(n):
                cname = members[i % min(len(members), 3)]
                rows.append({'image_id': f'im{rng.integers(0, 50):03d}', 'class': cname,
                             'class_id': CLASS_NAMES.index(cname), 'hole_id': int(rng.integers(1, 10 ** 6)),
                             'area': int(rng.integers(lo, hi + 1)), 'frac_gt_c': 0.9, 'bbox': '0:0:10:10',
                             'class_group': g, 'size_bin': HA.size_bin(int(rng.integers(lo, hi + 1)))})
    pop = pd.DataFrame(rows)
    pop['size_bin'] = [HA.size_bin(a) for a in pop['area']]
    return pop.sort_values(['image_id', 'class_id', 'hole_id'], kind='stable').reset_index(drop=True)


def test_sampling():
    print('7. Chon mau K2-b')
    pop = fake_population()
    s1, s2 = HA.draw_sample(pop), HA.draw_sample(pop)
    check('tat dinh (rng 19)', s1.equals(s2))
    main = s1[s1['duplicate_of'] == '']
    dup = s1[s1['duplicate_of'] != '']
    exp_main = sum(min(HA.PER_CELL, n) for n in HA.population_table(pop)['N_pop'])
    check('so muc chinh = sum min(7, N_o)', len(main) == exp_main, f'{len(main)} vs {exp_main}')
    check('20 lap an', len(dup) == HA.N_DUP)
    check('item_id duy nhat, H001..', s1['item_id'].is_unique and s1['item_id'].iloc[0] == 'H001'
          and sorted(s1['item_id']) == [f'H{i:03d}' for i in range(1, len(s1) + 1)])
    key = ['image_id', 'class', 'hole_id']
    ok = all(main.set_index('item_id').loc[r['duplicate_of'], key].tolist() == [r[k] for k in key]
             for _, r in dup.iterrows())
    check('duplicate_of tro dung lo goc', ok)
    check('muc chinh khong trung lo', not main.duplicated(key).any())
    cnt = main.groupby('stratum').size()
    check('o 3 lo -> lay het 3', cnt.get('Khac|>=700', 0) == 3)
    pos_dup = [s1.index[s1['item_id'] == i][0] for i in dup['item_id']]
    check('lap an bi xao (khong don cuoi)', min(pos_dup) < len(s1) - HA.N_DUP)


def test_size_bins():
    print('8. Nhom kich thuoc / nhom lop')
    check('149 -> 64-149, 150 -> 150-699, 699 -> 150-699, 700 -> >=700',
          [HA.size_bin(a) for a in (64, 149, 150, 699, 700)] == ['64-149', '64-149', '150-699', '150-699', '>=700'])
    check('Road/Bareland/Water -> Khac', {HA.class_group(c) for c in ('Road', 'Bareland', 'Water')} == {'Khac'})
    check('tong ky vong = 8836', sum(HA.EXPECTED_POP.values()) == 8836)


def test_crop():
    print('9. Crop')
    t, l, s = HA.crop_window((500, 500, 510, 520), (1024, 1024))
    check('canh >= 128', s == 128 and t <= 500 and t + s >= 510 and l <= 500 and l + s >= 520)
    t, l, s = HA.crop_window((0, 1000, 100, 1024), (1024, 1024))
    check('x3 bbox, dich vao trong anh', s == 300 and t == 0 and l + s == 1024)
    rgb = np.full((256, 256, 3), 90, dtype=np.uint8)
    hm = np.zeros((256, 256), dtype=bool)
    hm[120:136, 120:140] = True
    im = np.array(HA.render_crop(rgb, hm, (120, 120, 136, 140), 'H001 | lop c = Tree'))
    check('kich thuoc 2 o x 400 + tieu de', im.shape == (400 + 34, 2 * 400 + 12, 3), str(im.shape))
    red = (im[..., 0] == 255) & (im[..., 1] == 0) & (im[..., 2] == 0)
    check('vien do chi o o phai', red[:, 412:].sum() > 100 and red[34:, :400].sum() == 0)
    check('vien khong de len trong lo', not red[34 + 200, 412 + 200])


def test_hole_mask_roundtrip():
    print('10. Lay lai mask lo tu (class, hole_id)')
    pred, gt = synth_pair(5, 160)
    ok = True
    from Tools.spurious_holes import holes_for_image
    hs = [h for h in holes_for_image(pred, gt, 64) if h['area'] >= 64]
    for h in hs:
        m = HA.hole_mask_full(pred, h['class_id'], h['hole_id'])
        ok &= int(m.sum()) == h['area']
    check('dien tich khop', ok and len(hs) > 0, f'{len(hs)} lo')


def test_rating_template_not_overwritten():
    print('11. File cham khong bi ghi de')
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, 'r.csv')
        s = pd.DataFrame({'item_id': ['H001', 'H002']})
        HA.write_rating_template(s, p)
        df = pd.read_csv(p, keep_default_na=False)
        check('cot dung', list(df.columns) == HA.RATING_COLS and len(df) == 2)
        df.loc[0, 'nhan'] = 'A'
        df.to_csv(p, index=False)
        HA.write_rating_template(s, p)
        check('giu nhan da cham', pd.read_csv(p, keep_default_na=False).loc[0, 'nhan'] == 'A')


if __name__ == '__main__':
    for fn in (test_fill_equivalence, test_selective_fill_uses_gt, test_no_biou_same_stats, test_bootstrap_equivalence,
               test_process_image_e2e, test_lever_rule, test_sampling, test_size_bins, test_crop,
               test_hole_mask_roundtrip, test_rating_template_not_overwritten):
        fn()
    n_fail = sum(1 for _n, ok in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} PASS")
    sys.exit(1 if n_fail else 0)
