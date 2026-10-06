"""Tools/test_grad_conflict_probe.py — Tester cho Tools/grad_conflict_probe.py (Phan A)
va Tools/grad_conflict_common.py.

Mac dinh (nhanh, < 1 phut, CPU): test don vi tren du lieu tu dung —
  1. gt_strata_maps(): tang khoang cach / lop / kich thuoc + ha mau nearest.
  2. feature_pair_stats(): tong dung, null hoan vi giu nguyen tang (g2 hang theo
     tang -> null == quan sat), dem ho tro / cos<0.
  3. batch_partition(): tat dinh, phu du, khong trung.
  4. decide_branch(): bang 5.2 tren summary tu dung — kich ban G, L, C, RA, N.

`--e2e` (cham, ~15-20 phut CPU): chay THAT gates -> measure -> summarize tren
dataset gia lap trong bo nho (3 anh) va checkpoint khoi tao ngau nhien (Static
+ Baseline). Kiem: 6 cong PASS, du file output, bao cao co nhanh 5.2.

Chay: `python Tools/test_grad_conflict_probe.py [--e2e]` tu repo root.
"""

import os
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd
import torch

try:
    import rasterio  # noqa: F401
except ImportError:   # vd may Windows chan DLL rasterio — test nay khong doc .tif
    sys.modules['rasterio'] = types.ModuleType('rasterio')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (CHECKPOINTS, CLASS_NAMES, AXES, DIST_STRATA, SIZE_STRATA,
                                        gt_strata_maps, gt_components)
from Tools import grad_conflict_probe as gcp


def test_gt_strata_maps():
    print("=== Test 1: gt_strata_maps ===")
    gt = np.zeros((256, 256), dtype=np.int64)
    gt[:, 128:] = 2                      # bien doc tai cot 127/128
    gt[10:20, 10:20] = 8                 # Building nho (100 px)
    st = gt_strata_maps(gt, (64, 64))
    assert st['dist'].shape == (64, 64)
    # nearest: cot stride-4 j <- cot full-res 4j. Bien GT o cot 127 va 128.
    assert st['dist'][40, 32] == 0          # cot 128: tren bien
    assert st['dist'][40, 31] == 0          # cot 124: cach 3 px -> [0,4]
    assert st['dist'][40, 30] == 1          # cot 120: cach 7 px -> (4,8]
    assert st['dist'][40, 20] == 4          # cot 80: cach 47 px -> >32
    assert st['class'][40, 40] == 2 and st['class'][3, 3] == 8
    assert st['size'][3, 3] == 0            # Building 100 px -> nho
    assert st['size'][40, 40] == 1          # 128x256 = 32768 px -> vua
    _cm, table = gt_components(gt)
    areas = sorted(a for _c, a in table.values())
    assert areas == [100, 128 * 256 - 100, 128 * 256], areas
    print("PASS\n")


def test_feature_pair_stats():
    print("=== Test 2: feature_pair_stats ===")
    torch.manual_seed(0)
    gt = np.zeros((64, 64), dtype=np.int64)
    gt[:, 32:] = 1
    st = gcp.ImageStrata(gt, (16, 16), torch.device('cpu'), 0, n_perm=5, want_components=True)
    g1 = torch.randn(4, 16, 16)
    # g2 hang so trong moi lop -> hoan vi trong tang 'class' khong doi tong dot
    cls = torch.from_numpy(st.idx['class'].numpy().reshape(16, 16))
    g2 = torch.where(cls[None] == 0, torch.tensor([1., 2, 3, 4])[:, None, None],
                     torch.tensor([-1., 0, 1, 0])[:, None, None])
    g2[:, 0, 0] = 0.0                    # 1 vi tri khong ho tro
    fs = gcp.feature_pair_stats(g1, g2, st, 5)
    dot = (g1 * g2).sum(0).double()
    a = fs['strata']['all'][0]
    assert torch.allclose(a[0], dot.sum()) and a[3] == 256 and a[4] == 255
    assert int(a[5]) == int(((dot < 0) & ((g2 ** 2).sum(0) > 1e-12)).sum())
    c = fs['strata']['class']
    assert torch.allclose(c[:2, 0].sum(), dot.sum()) and c[2:].abs().sum() == 0
    # null 'class': g2[:,0,0] = 0 pha vo tinh hang so -> chi kiem n1/n2 bat bien va so chieu
    assert fs['null']['class'].shape == (5, len(AXES['class']))
    # tinh hang so that su: bo diem 0 di
    g2b = torch.where(cls[None] == 0, torch.tensor([1., 2, 3, 4])[:, None, None],
                      torch.tensor([-1., 0, 1, 0])[:, None, None])
    fsb = gcp.feature_pair_stats(g1, g2b, st, 5)
    assert torch.allclose(fsb['null']['class'][:, :2], fsb['strata']['class'][:2, 0].expand(5, 2))
    assert not torch.allclose(fsb['null']['all'][:, 0], fsb['strata']['all'][0, 0].expand(5))
    print("PASS\n")


def test_batch_partition():
    print("=== Test 3: batch_partition ===")
    b1 = gcp.batch_partition(384, 8)
    b2 = gcp.batch_partition(384, 8)
    assert b1 == b2 and len(b1) == 48 and all(len(b) == 8 for b in b1)
    flat = sorted(i for b in b1 for i in b)
    assert flat == list(range(384))
    assert b1[0] == np.random.default_rng(19).permutation(384)[:8].tolist()
    print("PASS\n")


def _fake_summary(label, confirmed):
    """confirmed: set (level, pair, axis, stratum) se 'xac nhan'."""
    rows = []
    pairs = [p for p in ('bce_aff', 'region_aff', 'region_bce')]
    for pn in pairs:
        keys = [('param', pn, 'all', 'all')] + [('feat', pn, ax, s) for ax, ss in AXES.items() for s in ss]
        for k in keys:
            ok = k in confirmed
            rows.append({'label': label, 'level': k[0], 'pair': pn, 'axis': k[2], 'stratum': k[3],
                         'value': -0.3 if ok else 0.01, 'ci_lo': -0.4, 'ci_hi': -0.2 if ok else 0.05,
                         'null_lo': -0.05, 'null_hi': 0.05, 'frac_neg': 0.5, 'mag_ratio': 1.0,
                         'support_cov': 0.5, 'flag_insufficient': False, 'confirmed_single_ckpt': ok})
    return pd.DataFrame(rows)


def test_decide_branch():
    print("=== Test 4: decide_branch (bang 5.2) ===")
    cases = {
        'G': {('param', 'bce_aff', 'all', 'all')},
        'L': {('feat', 'bce_aff', 'dist', 'd0-4')},
        'C': {('feat', 'bce_aff', 'class', 'Water'), ('feat', 'bce_aff', 'class', 'Agriculture')},
        'RA': {('feat', 'region_aff', 'dist', 'd16-32')},
        'N': set(),
    }
    for expect, conf in cases.items():
        dfs = {lb: _fake_summary(lb, conf) for lb in ('static_s19_36k', 'static_s86_36k')}
        d = gcp.decide_branch(dfs)
        assert d['branch'] == expect, (expect, d)
    # chi 1 seed xac nhan -> khong tinh
    dfs = {'static_s19_36k': _fake_summary('static_s19_36k', cases['G']),
           'static_s86_36k': _fake_summary('static_s86_36k', set())}
    assert gcp.decide_branch(dfs)['branch'] == 'N'
    # G + RA cung dat tho -> chon G, RA thanh bien the ablation
    both = cases['G'] | {('param', 'region_aff', 'all', 'all')}
    dfs = {lb: _fake_summary(lb, both) for lb in ('static_s19_36k', 'static_s86_36k')}
    d = gcp.decide_branch(dfs)
    assert d['branch'] == 'G' and 'RA' in d['ablation_variants'], d
    print("PASS\n")


# ───────────────────────────── e2e (tuy chon) ─────────────────────────────────

class _FakeValDS:
    def __init__(self, n=3, H=256):
        from src.data.transforms import get_val_transforms
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[:H, :H]
        self.items, self.samples = [], []
        for k in range(n):
            m = np.zeros((H, H), np.uint8)
            m[:, H // 2:] = 2
            for c in (1, 6, 7, 8):
                cy, cx = rng.integers(30, 220, 2)
                r = rng.integers(10, 40)
                m[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = c
            img = np.stack([(m * 25 + rng.integers(0, 40, (H, H))).astype(np.uint8)] * 3, -1)
            self.items.append((img, m.astype(np.int64)))
            self.samples.append((f'/x/img_{k}.tif', f'/y/img_{k}.tif'))
        self.t = get_val_transforms()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, m = self.items[i]
        o = self.t(image=img, mask=m)
        return o['image'].float(), o['mask'].long()


def test_e2e():
    print("=== Test 5 (e2e): gates -> measure -> summarize ===")
    import yaml
    import Tools.oracle_boundary_ceiling as obc
    from Tools.eval_boundary_metrics import build_eval_model
    from Tools.per_image_dump import compute_per_image_stats, save_pred_masks
    from Tools import select_figure_images as sfi
    orig = (obc.load_cfg, obc.build_val_dataset)
    obc.load_cfg = lambda path, data_root: yaml.safe_load(open(path, encoding='utf-8'))
    obc.build_val_dataset = lambda cfg: _FakeValDS()
    tmp = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        ck = {}
        for lb in ('static_s19_36k', 'static_s86_36k', 'baseline_40k'):
            info = CHECKPOINTS[lb]
            cfg = yaml.safe_load(open(info['config'], encoding='utf-8'))
            m = build_eval_model(cfg, info['model_type'])
            if hasattr(m, 'boundary_head'):
                with torch.no_grad():
                    m.boundary_head.conv_out.weight.add_(0.05 * torch.randn_like(m.boundary_head.conv_out.weight))
                    for n_, b in m.boundary_head.named_buffers():
                        if n_.endswith('running_mean'):
                            b.sub_(0.1)
            ck[lb] = os.path.join(tmp, f'{lb}.pth')
            torch.save(m.state_dict(), ck[lb])
            dd = os.path.join(tmp, f'dump_{lb}')
            os.makedirs(os.path.join(dd, 'masks'))
            m.eval()
            ds, rows = _FakeValDS(), []
            with torch.no_grad():
                for i in range(len(ds)):
                    x, y = ds[i]
                    out = m(x[None])
                    logits = out[0] if isinstance(out, tuple) else out
                    r, p = compute_per_image_stats(logits, y[None], [f'img_{i}'], CLASS_NAMES, 255, 4, 0, (1, 2, 4), 2)
                    rows += r
                    save_pred_masks(p, [f'img_{i}'], os.path.join(dd, 'masks'))
            pd.DataFrame(rows).to_csv(os.path.join(dd, 'per_image_stats.csv'), index=False)

        od = os.path.join(tmp, 'out')
        sys.argv = ['x', '--output', os.path.join(od, 'figure_images.json'), '--gt-cache', '']
        sfi.main()
        common = []
        for lb in ck:
            common += ['--ckpt', f'{lb}={ck[lb]}', '--dump-dir', f"{lb}={os.path.join(tmp, f'dump_{lb}')}"]
        common += ['--pos-weight', 'static_s19_36k=11.0', '--pos-weight', 'static_s86_36k=11.0',
                   '--output-dir', od, '--max-images', '2', '--n-perm', '3', '--num-workers', '0', '--gt-cache', '']
        gcp.main(['gates'] + common)
        for lb in ck:
            g = pd.read_json(os.path.join(od, f'gates_{lb}.json'), typ='series')
            assert g['overall'] == 'PASS', g['gates']
        gcp.main(['measure'] + common + ['--batch-size', '2', '--scratch-dir', os.path.join(tmp, 'scr')])
        gcp.main(['summarize', '--output-dir', od, '--holes-dir', os.path.join(tmp, 'holes'), '--n-boot', '500'])
        for f in ('summary_grad_conflict.md', 'summary_grad_conflict.json', 'per_image_strata_static_s19_36k.csv',
                  'per_batch_param_static_s19_36k.csv', 'null_param_static_s19_36k.csv',
                  'null_feat_static_s19_36k.csv', 'per_component_small_static_s19_36k.csv',
                  'summary_baseline_40k.csv', 'run_log.txt'):
            assert os.path.exists(os.path.join(od, f)), f
        s = pd.read_csv(os.path.join(od, 'summary_static_s19_36k.csv'))
        assert s[(s['level'] == 'param') & (s['pair'] == 'bce_aff')]['value'].notna().all()
        assert os.listdir(os.path.join(od, 'figures'))
    finally:
        obc.load_cfg, obc.build_val_dataset = orig
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


if __name__ == '__main__':
    test_gt_strata_maps()
    test_feature_pair_stats()
    test_batch_partition()
    test_decide_branch()
    if '--e2e' in sys.argv:
        test_e2e()
    else:
        print("(bo qua e2e — them --e2e de chay gates/measure/summarize that tren du lieu gia lap)")
    print("Tat ca self-test Tools/grad_conflict_probe.py PASS.")
