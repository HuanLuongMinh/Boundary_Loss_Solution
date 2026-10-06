"""Tools/test_spurious_holes.py — Tester cho Tools/spurious_holes.py (Phan B).

Du lieu SYNTHETIC, khong can checkpoint/dataset that. Kiem:
  1. Cong 6 (muc 4): dia dac / vanh khuyen / 2 dia roi / lo 63 px o A_min=64.
  2. Lo gia vs lo that (>= 50% pixel GT = c), thanh phan GT chua lo.
  3. Vanh khuyen: nguong 10% dien tich sau khi lap.
  4. Lo gia moi: IoU < 0.1 voi lo gia cua Baseline.
  5. fill_small_holes(): lap dung lo <= A_fill, lo long nhau.
  6. compare: bootstrap + verdict bang 5.4 tren CSV per-image tu dung
     (kich ban "do affinity" va "khong do loss"), cong tan suat 3.3.

Chay: `python Tools/test_spurious_holes.py` tu repo root.
"""

import os
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd

try:
    import rasterio  # noqa: F401
except ImportError:   # vd may Windows chan DLL rasterio — test nay khong doc .tif
    sys.modules['rasterio'] = types.ModuleType('rasterio')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.spurious_holes import (
    run_selftest, detect_holes_class, holes_for_image, max_iou_vs_reference, per_image_rows,
    fill_small_holes, cmd_compare, parse_args, A_MIN_MAIN,
)
from Tools.grad_conflict_common import CHECKPOINTS, CLASS_NAMES

WATER = CLASS_NAMES.index('Water')
BLD = CLASS_NAMES.index('Building')


def test_gate6():
    print("=== Test 1: cong 6 ===")
    res = run_selftest()
    for name, ok, detail in res:
        assert ok, f"{name}: {detail}"
    print("PASS\n")


def _square_with_hole(size=128, outer=(20, 100), inner=(50, 70)):
    m = np.zeros((size, size), dtype=bool)
    m[outer[0]:outer[1], outer[0]:outer[1]] = True
    m[inner[0]:inner[1], inner[0]:inner[1]] = False
    return m


def test_spurious_and_gt_component():
    print("=== Test 2: lo gia vs lo that ===")
    ring = _square_with_hole()
    pred = np.zeros((128, 128), dtype=np.int64)
    pred[ring] = WATER
    gt_solid = np.zeros_like(pred)
    gt_solid[20:100, 20:100] = WATER              # GT dac -> lo la lo gia
    holes = holes_for_image(pred, gt_solid, 16)
    hw = [h for h in holes if h['class_id'] == WATER]
    assert len(hw) == 1 and hw[0]['area'] == 400 and hw[0]['is_spurious'], hw
    assert hw[0]['gt_comp_id'] > 0 and hw[0]['gt_comp_area'] == 80 * 80

    gt_true = gt_solid.copy()
    gt_true[50:70, 50:70] = BLD                    # GT cung co lo -> lo that
    hw = [h for h in holes_for_image(pred, gt_true, 16) if h['class_id'] == WATER]
    assert len(hw) == 1 and not hw[0]['is_spurious'] and hw[0]['frac_gt_c'] == 0.0
    print("PASS\n")


def test_annulus_threshold():
    print("=== Test 3: vanh khuyen (10%) ===")
    # filled = 80*80 = 6400; lo 20x20 = 400 (6.25%) -> khong vanh khuyen; lo 30x30 = 900 (14%) -> co
    for inner, expect in (((50, 70), 0), ((45, 75), 1)):
        pred = np.zeros((128, 128), dtype=np.int64)
        pred[_square_with_hole(inner=inner)] = WATER
        gt = np.zeros_like(pred)
        gt[20:100, 20:100] = WATER
        holes = holes_for_image(pred, gt, 16)
        new = {a: max_iou_vs_reference(holes, [], gt.shape, a) for a in (64,)}
        rows = per_image_rows('x', holes, new, (64,))
        r = [r for r in rows if r['class'] == 'all'][0]
        assert r['n_vanh_khuyen'] == expect, (inner, r)
        assert r['n_lo_gia'] == 1 and r['n_lo_gia_moi'] == 1
    print("PASS\n")


def test_new_vs_baseline():
    print("=== Test 4: lo gia moi (IoU < 0.1) ===")
    gt = np.zeros((128, 128), dtype=np.int64)
    gt[20:100, 20:100] = WATER
    p_base = np.zeros_like(gt)
    p_base[_square_with_hole(inner=(50, 70))] = WATER
    p_same = p_base.copy()
    p_far = np.zeros_like(gt)
    p_far[_square_with_hole(inner=(25, 35))] = WATER     # lo khac cho, khong chong
    base = holes_for_image(p_base, gt, 16)
    for pred, expect_new in ((p_same, 0), (p_far, 1)):
        holes = holes_for_image(pred, gt, 16)
        iou = max_iou_vs_reference(holes, base, gt.shape, 64)
        rows = per_image_rows('x', holes, {64: iou}, (64,))
        r = [r for r in rows if r['class'] == 'all'][0]
        assert r['n_lo_gia_moi'] == expect_new, (expect_new, r, iou)
    print("PASS\n")


def test_fill_small_holes():
    print("=== Test 5: fill_small_holes ===")
    pred = np.zeros((128, 128), dtype=np.int64)
    pred[_square_with_hole(inner=(50, 70))] = WATER     # lo 400 px, ben trong = 0 (Background)
    out64 = fill_small_holes(pred, 64)
    assert (out64 == pred).all(), "lo 400 px khong duoc lap o A_fill=64"
    out1024 = fill_small_holes(pred, 1024)
    assert (out1024[50:70, 50:70] == WATER).all()
    # Lo long nhau: Water vanh ngoai, Building vanh trong, dao Tree o giua (100 px)
    nest = np.zeros((128, 128), dtype=np.int64)
    nest[10:110, 10:110] = WATER
    nest[30:90, 30:90] = BLD
    nest[55:65, 55:65] = CLASS_NAMES.index('Tree')
    out = fill_small_holes(nest, 100)
    assert (out[55:65, 55:65] == BLD).all(), "lo trong cung (100 px) phai lap bang lop bao quanh truc tiep"
    assert (out[30:55, 30:90] == BLD).all(), "lo lon hon A_fill khong duoc lap"
    print("PASS\n")


def _write_fake_per_image(od, n=60, effect_labels=(), effect=1.0, seed=0):
    rng = np.random.default_rng(seed)
    for lb in CHECKPOINTS:
        rows = []
        for i in range(n):
            for a in (16, 64, 256):
                for cls in ['all'] + CLASS_NAMES:
                    lam = 0.5 + (effect if (lb in effect_labels and cls == 'all') else 0.0)
                    k = int(rng.poisson(lam))
                    new = 0 if lb == 'baseline_40k' else int(rng.poisson(lam * 0.6))
                    rows.append({'image_id': f'img_{i:03d}', 'a_min': a, 'class': cls, 'n_lo_gia': k,
                                 'dien_tich_lo_gia': 100 * k, 'n_vanh_khuyen': k // 2, 'n_lo_gia_moi': new,
                                 'n_holes_total': k + 1})
        pd.DataFrame(rows).to_csv(os.path.join(od, f'per_image_{lb}.csv'), index=False)


def test_compare_verdicts():
    print("=== Test 6: compare + bang 5.4 ===")
    tmp = tempfile.mkdtemp()
    try:
        _write_fake_per_image(tmp, effect_labels=('affonly_s19_40k', 'affonly_s86_40k', 'static_s19_36k',
                                                  'static_s86_36k', 'static_s86_40k'), effect=3.0)
        cmd_compare(parse_args(['compare', '--output-dir', tmp, '--n-boot', '2000']))
        s = pd.read_json(os.path.join(tmp, 'pairs_bootstrap.json'), typ='series')
        assert s['verdict_54']['key'] == 'affinity', s['verdict_54']
        assert s['frequency_gate']['rare'] is False
        assert s['run_part_b2'] is True and s['run_part_c'] is True
        assert os.path.exists(os.path.join(tmp, 'pairs_bootstrap.md'))
        res = pd.read_csv(os.path.join(tmp, 'pairs_bootstrap.csv'))
        assert set(res['pair_id']) >= {'H1', 'H2', "H2'", 'H3', "H3'", 'H4', 'H5', "H5'", 'N1', 'N2'}

        shutil.rmtree(tmp)
        os.makedirs(tmp)
        _write_fake_per_image(tmp, effect_labels=(), seed=1)
        cmd_compare(parse_args(['compare', '--output-dir', tmp, '--n-boot', '2000']))
        s = pd.read_json(os.path.join(tmp, 'pairs_bootstrap.json'), typ='series')
        assert s['verdict_54']['key'] in ('none', 'unmatched'), s['verdict_54']
        print(f"  kich ban null -> {s['verdict_54']['key']}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


class _FakeValDS:
    """Thay OpenEarthMapDataset (khong doc .tif): 2 anh 128x128, GT Water dac."""

    def __init__(self):
        import torch
        self.samples = [(f'/v/img_{k}.tif', f'/l/img_{k}.tif') for k in range(2)]
        gt = np.zeros((128, 128), dtype=np.int64)
        gt[20:100, 20:100] = WATER
        self._items = [(torch.zeros(3, 128, 128), torch.from_numpy(gt.copy())) for _ in range(2)]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, i):
        return self._items[i]


def test_count_end_to_end():
    print("=== Test 7: count end-to-end (dataset gia lap) ===")
    from PIL import Image
    import Tools.oracle_boundary_ceiling as obc
    from Tools.spurious_holes import cmd_count
    orig = (obc.load_cfg, obc.build_val_dataset)
    obc.load_cfg = lambda path, data_root: {}
    obc.build_val_dataset = lambda cfg: _FakeValDS()
    tmp = tempfile.mkdtemp()
    try:
        dirs = {}
        for lb, inner in (('baseline_40k', None), ('static_s19_36k', (50, 70))):
            d = os.path.join(tmp, lb, 'masks')
            os.makedirs(d)
            pred = np.zeros((128, 128), dtype=np.uint8)
            if inner is None:
                pred[20:100, 20:100] = WATER
            else:
                pred[_square_with_hole(inner=inner)] = WATER
            for k in range(2):
                Image.fromarray(pred, mode='L').save(os.path.join(d, f'img_{k}.png'))
            dirs[lb] = d
        od = os.path.join(tmp, 'out')
        args = parse_args(['count', '--output-dir', od, '--gt-cache', '',
                           '--masks', f"baseline_40k={dirs['baseline_40k']}",
                           '--masks', f"static_s19_36k={dirs['static_s19_36k']}"])
        cmd_count(args)
        df = pd.read_csv(os.path.join(od, 'per_image_static_s19_36k.csv'))
        r = df[(df['a_min'] == A_MIN_MAIN) & (df['class'] == 'all')]
        assert r['n_lo_gia'].tolist() == [1, 1] and r['n_lo_gia_moi'].tolist() == [1, 1], r
        assert r['dien_tich_lo_gia'].tolist() == [400, 400]
        b = pd.read_csv(os.path.join(od, 'per_image_baseline_40k.csv'))
        assert b['n_lo_gia'].sum() == 0
        h = pd.read_csv(os.path.join(od, 'holes_static_s19_36k.csv'))
        sp = h[(h['a_min'] == 64) & (h['is_spurious'] == 1)]
        assert len(sp) == 2 and (sp['gt_comp_id'] > 0).all() and (sp['class'] == 'Water').all(), sp
        cmd_count(args)   # lan 2: phai bo qua (resume)
    finally:
        obc.load_cfg, obc.build_val_dataset = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def test_link_part_c():
    print("=== Test 8: Phan C (link_holes_conflict) ===")
    import json
    from Tools.link_holes_conflict import main as link_main
    tmp = tempfile.mkdtemp()
    try:
        gd, hd = os.path.join(tmp, 'g'), os.path.join(tmp, 'h')
        os.makedirs(gd)
        os.makedirs(hd)
        rng = np.random.default_rng(0)
        comp, holes = [], []
        for i in range(80):
            has = i < 40
            for pn in ('bce_aff', 'region_aff'):
                d = (-0.5 if has and pn == 'bce_aff' else 0.2) + 0.05 * rng.standard_normal()
                comp.append({'image_id': f'img_{i // 4}', 'pair': pn, 'comp_id': i, 'class': 'Water', 'area': 900,
                             'sum_dot': d, 'sum_n1': 1.0, 'sum_n2': 1.0, 'n_pos': 50, 'n_support': 50})
            if has:
                holes.append({'image_id': f'img_{i // 4}', 'class': 'Water', 'a_min': 64, 'hole_id': 1, 'area': 80,
                              'frac_gt_c': 1.0, 'is_spurious': 1, 'is_new': 1, 'max_iou_baseline': 0.0,
                              'comp_id': 1, 'comp_filled_area': 900, 'gt_comp_id': i, 'gt_comp_area': 900,
                              'bbox': '0:0:1:1'})
        pd.DataFrame(comp).to_csv(os.path.join(gd, 'per_component_small_static_s19_36k.csv'), index=False)
        pd.DataFrame(holes).to_csv(os.path.join(hd, 'holes_static_s19_36k.csv'), index=False)
        with open(os.path.join(hd, 'pairs_bootstrap.json'), 'w') as f:
            json.dump({'run_part_c': True, 'verdict_54': {'text': 'x'}}, f)
        with open(os.path.join(hd, 'pairs_bootstrap.md'), 'w') as f:
            f.write('# B\n')
        link_main(['--grad-dir', gd, '--holes-dir', hd, '--n-boot', '1000'])
        link_main(['--grad-dir', gd, '--holes-dir', hd, '--n-boot', '1000'])   # chay lai: thay the muc C
        res = json.load(open(os.path.join(hd, 'part_c_link.json')))
        assert res['supports_common_root'] and res['verdict_by_pair']['bce_aff'] == 'ben co lo gia AM HON ro', res
        assert res['verdict_by_pair']['region_aff'] == 'khong khac biet ro', res
        md = open(os.path.join(hd, 'pairs_bootstrap.md'), encoding='utf-8').read()
        assert md.count('PHAN C BEGIN') == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def test_fill_test_end_to_end():
    print("=== Test 9: fill-test (Phan B2) end-to-end ===")
    from PIL import Image
    import Tools.oracle_boundary_ceiling as obc
    from Tools.spurious_holes import cmd_fill_test
    orig = (obc.load_cfg, obc.build_val_dataset)
    obc.load_cfg = lambda path, data_root: {}
    obc.build_val_dataset = lambda cfg: _FakeValDS()
    tmp = tempfile.mkdtemp()
    try:
        d = os.path.join(tmp, 'masks')
        os.makedirs(d)
        pred = np.zeros((128, 128), dtype=np.uint8)
        pred[_square_with_hole(inner=(50, 70))] = WATER          # lo 400 px, GT dac
        for k in range(2):
            Image.fromarray(pred, mode='L').save(os.path.join(d, f'img_{k}.png'))
        cmd_fill_test(parse_args(['fill-test', '--output-dir', tmp, '--gt-cache', '', '--a-fill', '64,1024',
                                  '--n-boot', '200', '--masks', f'static_s19_36k={d}']))
        df = pd.read_csv(os.path.join(tmp, 'fill_test.csv'))
        m = df[df['quantity'] == 'mIoU_9'].set_index('a_fill')
        assert m.loc[64, 'delta'] == 0.0, m
        assert m.loc[1024, 'delta'] > 0.0, m
        assert os.path.exists(os.path.join(tmp, 'fill_test.md'))
    finally:
        obc.load_cfg, obc.build_val_dataset = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


if __name__ == '__main__':
    test_gate6()
    test_count_end_to_end()
    test_link_part_c()
    test_fill_test_end_to_end()
    test_spurious_and_gt_component()
    test_annulus_threshold()
    test_new_vs_baseline()
    test_fill_small_holes()
    test_compare_verdicts()
    print("Tat ca self-test Tools/spurious_holes.py PASS.")
