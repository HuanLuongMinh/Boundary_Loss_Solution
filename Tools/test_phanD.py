"""Tools/test_phanD.py — Tester cho Phan D (docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md):
src/utils/phanD_snapshots.py, Tools/phanD_probe.py, Tools/phanD_traj.py, Tools/phanD_dryrun.py.

Mac dinh (nhanh, CPU, < 1 phut):
  1. Lich snapshot: 35 moc (co 36000), lich gian D5 (co 36000).
  2. probe_ids: tat dinh, dung quy tac default_rng(19).choice, khong trung.
  3. Thong ke Gram: cos_all / cos batch / bootstrap / null nua-nua tu Gram == tinh truc tiep tren vector.
  4. Cong D3 (gate_d3) tren memmap gia lap.
  5. Cua so §6.2: >= 3 snapshot lien tiep, bo t=0, chuoi bi ngat.
  6. Bang 6.3: D-G, D-RA, D-L, D-dai-dang (ghi de), D-N, khong ma nao.
  7. SnapshotSaver + verify_snapshot_load (D2) + prune_after (resume) + LossLogger.
  8. Cong so loss D0/D1 (phanD_dryrun compare): trung bit, nhieu, lech; pos_weight.
  9. phanD_traj.main tren Gram + feat gia lap -> du file output + verdict.

`--e2e` (cham hon, CPU): measure_snapshot THAT (Probe + model Static khoi tao ngau nhien) tren dataset
gia lap 3 anh, roi D3 + phanD_traj.

Chay: `python Tools/test_phanD.py [--e2e]` tu repo root.
"""

import json
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
except ImportError:
    sys.modules['rasterio'] = types.ModuleType('rasterio')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import phanD_snapshots as snp
from Tools import phanD_probe as pp
from Tools import phanD_traj as tj
from Tools import phanD_dryrun as dr
from Tools.grad_conflict_common import ALL_PAIRS, pair_name, AXES

TERMS = ['region', 'bce', 'aff']


def _gram(V):
    """V (n, nt, P) -> Gram (n*nt, n*nt), hang i*nt + j."""
    n, nt, P = V.shape
    X = V.reshape(n * nt, P).astype(np.float64)
    return X @ X.T


def test_schedule():
    print("=== Test 1: lich snapshot ===")
    d = snp.parse_schedule('default')
    assert len(d) == 35 and d[0] == 0 and d[-1] == 40000 and 36000 in d, d
    assert d[:5] == [0, 250, 500, 750, 1000] and 1500 in d and 8000 in d and 9000 not in d and 10000 in d
    s = snp.parse_schedule('sparse')
    assert 36000 in s and 40000 in s and 1500 not in s and 2000 in s and 12000 in s and 10000 not in s, s
    assert snp.parse_schedule('0,500,250') == [0, 250, 500]
    assert snp.parse_schedule(None) is None
    print("PASS\n")


def test_probe_ids():
    print("=== Test 2: probe_ids ===")
    names = [f'img_{i:04d}' for i in range(3000)]
    a = pp.make_probe_ids(names)
    b = pp.make_probe_ids(names)
    assert a == b and len(a) == 32 and len(set(a)) == 32
    idx = np.random.default_rng(19).choice(3000, 32, replace=False)
    assert a == [names[i] for i in idx]
    print("PASS\n")


def test_gram_stats():
    print("=== Test 3: thong ke tu Gram == tinh truc tiep ===")
    rng = np.random.default_rng(0)
    n, P = 32, 40
    V = rng.normal(size=(n, 3, P))
    V[:, 2] -= 0.6 * V[:, 1]                    # aff nguoc huong bce mot phan
    G = _gram(V)
    batches = [list(b) for b in np.random.default_rng(19).permutation(n).reshape(4, 8)]
    W = tj.boot_weights(n, 200)
    U, Vh = tj.half_splits(n, 100)
    for k1, k2 in ALL_PAIRS:
        a, b = V[:, TERMS.index(k1)], V[:, TERMS.index(k2)]
        s = tj.param_stats(G, TERMS, batches, k1, k2, W, U, Vh)
        A, B = a.sum(0), b.sum(0)
        assert abs(s['cos_all'] - A @ B / np.linalg.norm(A) / np.linalg.norm(B)) < 1e-12
        assert abs(s['ratio'] - np.linalg.norm(B) / np.linalg.norm(A)) < 1e-12
        for bi, mem in enumerate(batches):
            x, y = a[mem].sum(0), b[mem].sum(0)
            assert abs(s['cos_batch'][bi] - x @ y / np.linalg.norm(x) / np.linalg.norm(y)) < 1e-12
        # 1 hang bootstrap va 1 lan chia nua — tinh lai truc tiep
        G11, G22, G12 = tj.gram_pair_blocks(G, TERMS, k1, k2)
        x, y = W[3] @ a, W[3] @ b
        assert abs(tj.weighted_cos(W[3:4], G11, G22, G12)[0] - x @ y / np.linalg.norm(x) / np.linalg.norm(y)) < 1e-12
        x, y = U[5] @ a, Vh[5] @ b
        assert abs(tj.weighted_cos(U[5:6], G11, G22, G12, Vh[5:6])[0]
                   - x @ y / np.linalg.norm(x) / np.linalg.norm(y)) < 1e-12
        assert s['ci_lo'] <= s['cos_all'] <= s['ci_hi'] or k1 == 'region'   # CI thuong chua gia tri quan sat
    assert ((U + Vh) == 1).all() and (U.sum(1) == 16).all()          # 2 nua roi nhau, phu du
    s = tj.param_stats(G, TERMS, batches, 'bce', 'aff', W, U, Vh)
    assert s['cos_all'] < 0
    print("PASS\n")


def test_gate_d3():
    print("=== Test 4: cong D3 ===")
    tmp = tempfile.mkdtemp()
    try:
        rng = np.random.default_rng(1)
        n, P = 6, 1000
        mm_path = os.path.join(tmp, 'a.f32')
        mm = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(n, 3, P))
        mm[:] = rng.normal(size=(n, 3, P)).astype(np.float32)
        mm.flush()
        from Tools.grad_conflict_probe import gram_from_memmap
        G = gram_from_memmap(mm, torch.device('cpu'), chunk=128)        # "tang dan" theo khoi nho
        V = np.asarray(mm, dtype=np.float64)
        pairs, vals = [], []
        for k1, k2 in ALL_PAIRS:
            A, B = V[:, TERMS.index(k1)].sum(0), V[:, TERMS.index(k2)].sum(0)
            pairs.append(pair_name((k1, k2)))
            vals.append([A @ B, A @ A, B @ B])
        del mm
        r = pp.gate_d3(G, TERMS, pairs, vals, mm_path, (n, 3, P))
        assert r['status'] == 'PASS', r
        vals[0][0] *= 1.01                                           # lam hong tong truc tiep -> FAIL
        r = pp.gate_d3(G, TERMS, pairs, vals, mm_path, (n, 3, P))
        assert r['status'] == 'FAIL'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def _cand(flags):
    """flags: {(level, pair, axis, stratum): {t: bool}} -> DataFrame cho find_windows."""
    rows = []
    for (lv, pn, ax, st), d in flags.items():
        for t, c in d.items():
            rows.append({'t': t, 'level': lv, 'pair': pn, 'axis': ax, 'stratum': st, 'conflict': c})
    return pd.DataFrame(rows)


TS = snp.parse_schedule('default')


def _series(on):
    return {t: (t in on) for t in TS}


def test_windows():
    print("=== Test 5: cua so §6.2 ===")
    P = ('param', 'bce_aff', 'all', 'all')
    # t=0 + 250 + 500: chi 2 snapshot hop le -> KHONG la cua so
    w = tj.find_windows(_cand({P: _series({0, 250, 500})}))
    assert len(w) == 0, w
    # 250, 500, 750 -> cua so 3
    w = tj.find_windows(_cand({P: _series({250, 500, 750})}))
    assert len(w) == 1 and w.iloc[0]['t_start'] == 250 and w.iloc[0]['t_end'] == 750
    # chuoi bi ngat: 250, 500, (750 khong), 1000, 1500, 2000 -> 1 cua so (1000-2000)
    w = tj.find_windows(_cand({P: _series({250, 500, 1000, 1500, 2000})}))
    assert len(w) == 1 and w.iloc[0]['t_start'] == 1000 and w.iloc[0]['n_snapshots'] == 3
    # 8000 -> 10000 -> 12000 lien tiep theo lich (khong co 9000)
    w = tj.find_windows(_cand({P: _series({8000, 10000, 12000})}))
    assert len(w) == 1
    print("PASS\n")


def test_decide():
    print("=== Test 6: bang 6.3 ===")
    def W(*rows):
        return pd.DataFrame([dict(level=a, pair=b, axis=c, stratum=d, t_start=s, t_end=e, n_snapshots=3,
                                  snapshots='') for a, b, c, d, s, e in rows],
                            columns=['level', 'pair', 'axis', 'stratum', 't_start', 't_end', 'n_snapshots',
                                     'snapshots'])
    assert tj.decide(W())['verdict'] == 'D-N'
    assert tj.decide(W(('param', 'bce_aff', 'all', 'all', 250, 1000)))['verdict'] == 'D-G'
    d = tj.decide(W(('param', 'region_aff', 'all', 'all', 250, 1000), ('feat', 'bce_aff', 'dist', 'd0-4', 250, 750)))
    assert d['verdict'] == 'D-RA'
    assert tj.decide(W(('feat', 'bce_aff', 'size', 'small', 1000, 2000)))['verdict'] == 'D-L'
    assert tj.decide(W(('feat', 'bce_aff', 'class', 'Road', 1000, 2000)))['verdict'] is None   # chi truc lop
    d = tj.decide(W(('param', 'bce_aff', 'all', 'all', 250, 1000), ('param', 'region_bce', 'all', 'all', 32000, 40000)))
    assert d['verdict'] == 'D-dai-dang' and d['literal_verdict'] == 'D-G'
    d = tj.decide(W(('param', 'bce_aff', 'all', 'all', 30000, 36000)))   # keo toi 36000 -> khong phai D-G
    assert d['verdict'] == 'D-dai-dang' and not d['literal']['D-G']
    print("PASS\n")


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x))


def test_snapshots():
    print("=== Test 7: SnapshotSaver / D2 / prune / LossLogger ===")
    tmp = tempfile.mkdtemp()
    try:
        torch.manual_seed(0)
        m = _Tiny()
        m(torch.randn(2, 3, 8, 8))                       # cap nhat BN running stats
        rng_before = torch.get_rng_state().clone()
        sv = snp.SnapshotSaver([0, 250, 500], os.path.join(tmp, 'snap'), 19, 'abc', True)
        assert sv.maybe_save(m, 100) is None              # khong thuoc lich
        p = sv.maybe_save(m, 500)
        assert torch.equal(torch.get_rng_state(), rng_before), 'luu snapshot khong duoc cham RNG'
        sd, meta = snp.load_snapshot_state(p)
        assert meta == {'iter': 500, 'seed': 19, 'config_md5': 'abc'}
        assert snp.verify_snapshot_load(p, _Tiny, m)['status'] == 'PASS'
        with torch.no_grad():
            m.conv.weight[0, 0, 0, 0] += 1e-7
        r = snp.verify_snapshot_load(p, _Tiny, m)
        assert r['status'] == 'FAIL' and r['mismatched'] == ['conv.weight']
        # file da co thi khong ghi de (resume)
        mt = os.path.getmtime(p)
        sv.maybe_save(m, 500)
        assert os.path.getmtime(p) == mt
        sv.maybe_save(m, 250)
        assert sv.prune_after(250) == [500] and not os.path.exists(p)
        assert not snp.SnapshotSaver([0], os.path.join(tmp, 'x'), 19, 'a', False).maybe_save(m, 0)  # rank != 0
        lg = snp.LossLogger(os.path.join(tmp, 'loss.csv'), True)
        for it in (1, 2):
            lg.log(it, 1e-4, {'l_region': 1.0 / 3, 'l_bce': 0.5, 'l_affinity': 0.25, 'l_total': 2.0})
        lg.close()
        A, _ = dr.read_loss_log(os.path.join(tmp, 'loss.csv'))
        assert A[1]['l_region'] == 1.0 / 3                 # repr -> doc lai trung bit
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def _write_loss(path, vals, pw=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lg = snp.LossLogger(path, True)
    for it, v in enumerate(vals, 1):
        lg.log(it, 1e-4, {'l_region': v, 'l_bce': v / 2, 'l_affinity': v / 4, 'l_total': v * 2})
    lg.close()
    if pw is not None:
        with open(os.path.join(os.path.dirname(path), 'train_log.txt'), 'w') as f:
            f.write(f"[00:00:00] --stop-at-iter: dung sau iter {len(vals)} (pos_weight={pw:.6f}).\n")


def _compare(tmp, a, b, c, gate='D1', selection=False, pw=True):
    out = os.path.join(tmp, f'{gate}.json')
    argv = ['compare', '--gate', gate, '--a', a, '--b', b, '--c', c, '--out', out]
    if selection:
        argv.append('--selection')
    if pw:
        argv += ['--pos-weight-log', os.path.join(os.path.dirname(c), 'train_log.txt')]
    try:
        dr.main(argv)
        code = 0
    except SystemExit as e:
        code = e.code
    return code, json.load(open(out))


def test_compare():
    print("=== Test 8: cong so loss D0/D1 ===")
    tmp = tempfile.mkdtemp()
    try:
        base = np.linspace(2.0, 1.0, 50)
        f = {k: os.path.join(tmp, k, 'loss.csv') for k in ('a', 'b', 'same', 'small', 'big', 'badpw')}
        _write_loss(f['a'], base, 11.0014)
        _write_loss(f['b'], base + 1e-4, 11.0014)
        _write_loss(f['same'], base, 11.0014)
        _write_loss(f['small'], base - 5e-5, 11.0014)
        _write_loss(f['big'], base + 1e-2, 11.0014)
        _write_loss(f['badpw'], base, 10.5)
        assert _compare(tmp, f['a'], f['b'], f['small'])[0] == 0
        assert _compare(tmp, f['a'], f['b'], f['big'])[0] == 2
        # A == B (trung bit) -> yeu cau C == A
        assert _compare(tmp, f['a'], f['same'], f['same'])[0] == 0
        assert _compare(tmp, f['a'], f['same'], f['small'])[0] == 2
        # D0 la cong chon: FAIL -> ma 3 (khong chan); pos_weight sai -> ma 2 (chan)
        code, r = _compare(tmp, f['a'], f['b'], f['big'], gate='D0', selection=True)
        assert code == 3 and r['blocking'] is False
        code, r = _compare(tmp, f['a'], f['b'], f['badpw'], gate='D0', selection=True)
        assert code == 2 and r['blocking'] is True and r['value']['pos_weight_ok'] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def test_d0_static():
    print("=== Test 8b: D0 phan tinh (5 muc) + bat duoc dot bien ===")
    from Tools.phanD_d0_static import run_static_checks, WEIGHTED
    r = run_static_checks()
    assert r['status'] == 'PASS', [(i['key'], i['status']) for i in r['items']]
    assert [i['key'] for i in r['items']] == ['lr', 'init_rng', 'sampler', 'posweight_lambda', 'validation']
    src = open(WEIGHTED, encoding='utf-8').read()
    mutations = {   # muc mong doi KHAC: (chuoi goc, chuoi dot bien)
        'lr': ("power: float = 0.9", "power: float = 0.8"),
        'init_rng': ("    model = StaticBoundaryUNetFormer(cfg).to(device)",
                     "    torch.rand(1)\n    model = StaticBoundaryUNetFormer(cfg).to(device)"),
        'sampler': ("sampler=train_sampler, num_workers=tr['NUM_WORKERS'],",
                    "sampler=train_sampler, num_workers=tr['NUM_WORKERS'], worker_init_fn=None,"),
        'posweight_lambda': ("for _ in range(edge_stats_nbatch):", "for _ in range(edge_stats_nbatch + 1):"),
        'validation': ("is_best_miou     = miou9 > best_miou", "is_best_miou     = miou9 >= best_miou"),
    }
    tmp = tempfile.mkdtemp()
    try:
        for key, (old, new) in mutations.items():
            assert src.count(old) >= 1, (key, old)
            p = os.path.join(tmp, f'w_{key}.py')
            open(p, 'w', encoding='utf-8').write(src.replace(old, new, 1))
            rr = run_static_checks(weighted=p)
            st = {i['key']: i['status'] for i in rr['items']}
            assert rr['status'] == 'FAIL' and st[key] == 'KHAC', (key, st)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def test_d0_iter1_and_static():
    print("=== Test 8c: D0 dieu kien loss iter 1 trung bit + phan tinh ===")
    tmp = tempfile.mkdtemp()
    try:
        base = np.linspace(2.0, 1.0, 50)
        f = {k: os.path.join(tmp, k, 'loss.csv') for k in ('a', 'b', 'c_ok', 'c_it1')}
        _write_loss(f['a'], base, 11.0014)
        _write_loss(f['b'], base + 1e-4, 11.0014)
        _write_loss(f['c_ok'], np.r_[base[:1], base[1:] + 5e-5], 11.0014)      # iter 1 trung, sau do trong nhieu
        _write_loss(f['c_it1'], np.r_[base[:1] + 1e-7, base[1:]], 11.0014)    # iter 1 lech 1e-7
        st_pass = os.path.join(tmp, 'st_pass.json')
        st_fail = os.path.join(tmp, 'st_fail.json')
        json.dump({'status': 'PASS', 'items': [{'key': 'lr', 'status': 'GIONG'}]}, open(st_pass, 'w'))
        json.dump({'status': 'FAIL', 'items': [{'key': 'lr', 'status': 'KHAC'}]}, open(st_fail, 'w'))

        def run(c, st):
            out = os.path.join(tmp, 'D0.json')
            argv = ['compare', '--gate', 'D0', '--selection', '--iter1-exact', '--static-json', st,
                    '--a', f['a'], '--b', f['b'], '--c', c, '--out', out,
                    '--pos-weight-log', os.path.join(os.path.dirname(f['a']), 'train_log.txt')]
            try:
                dr.main(argv)
                code = 0
            except SystemExit as e:
                code = e.code
            return code, json.load(open(out))

        code, r = run(f['c_ok'], st_pass)
        assert code == 0 and r['value']['iter1']['A_equals_C'], r
        code, r = run(f['c_it1'], st_pass)            # trong nhieu 500 iter nhung iter 1 khong trung -> FAIL
        assert code == 3 and not r['value']['iter1']['A_equals_C'] and r['blocking'] is False
        code, r = run(f['c_ok'], st_fail)             # phan tinh FAIL -> D0 FAIL (chuyen script goc)
        assert code == 3 and r['value']['static']['status'] == 'FAIL'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def _fake_measure_outputs(od, ts, conflict_ts=()):
    """Gram + feat gia lap: (BCE, Aff) nguoc huong o cac t trong conflict_ts."""
    for sub in ('gram', 'feat'):
        os.makedirs(os.path.join(od, sub), exist_ok=True)
    rng = np.random.default_rng(3)
    n, P = 32, 30
    images = [f'img_{i}' for i in range(n)]
    for t in ts:
        V = rng.normal(size=(n, 3, P)) * 0.3
        common = rng.normal(size=P)
        V[:, 0] += common                                     # region
        V[:, 1] += 0.5 * common                               # bce cung huong region
        V[:, 2] += (-1.0 if t in conflict_ts else 0.5) * V[:, 1].mean(0) * 2
        np.savez_compressed(os.path.join(od, 'gram', f'iter_{t}.npz'), G=_gram(V), terms=np.array(TERMS),
                            image_ids=np.array(images), iter=t,
                            batches=np.random.default_rng(19).permutation(n).reshape(4, 8))
        srows, nrows = [], []
        for pn in [pair_name(p) for p in ALL_PAIRS]:
            for ax, names in AXES.items():
                for s in names:
                    for im in images:
                        d = rng.normal() + 1.0
                        srows.append({'image_id': im, 'pair': pn, 'axis': ax, 'stratum': s, 'sum_dot': d,
                                      'sum_n1': 4.0, 'sum_n2': 4.0, 'n_pos': 100, 'n_support': 50, 'n_neg': 10})
                        for r in range(3):
                            nrows.append((pn, ax, s, r, im, rng.normal() * 0.1))
        pd.DataFrame(srows).to_csv(os.path.join(od, 'feat', f'iter_{t}_strata.csv'), index=False)
        pd.DataFrame(nrows, columns=['pair', 'axis', 'stratum', 'perm_id', 'image_id', 'sum_dot_perm']) \
            .to_csv(os.path.join(od, 'feat', f'iter_{t}_null.csv'), index=False)


def test_traj_main():
    print("=== Test 9: phanD_traj.main tren du lieu gia lap ===")
    tmp = tempfile.mkdtemp()
    try:
        ts = [0, 250, 500, 750, 1000, 1500, 32000, 34000, 36000, 38000, 40000]
        _fake_measure_outputs(tmp, ts, conflict_ts={250, 500, 750})
        gates = {'meta': {'config_md5': 'x', 'train_script': 'src/train_static_boundary_weighted.py'},
                 'post': {'interpretable': False, 'sanity5': {'status': 'FAIL'}}}
        json.dump(gates, open(os.path.join(tmp, 'g.json'), 'w'))
        dec = tj.main(['--output-dir', tmp, '--gates-json', os.path.join(tmp, 'g.json'),
                       '--n-boot', '300', '--n-null', '100'])
        for fn in ('traj_param.csv', 'traj_feat_strata.csv', 'windows.csv', 'summary_phanD.md',
                   'summary_phanD.json', 'figures/traj_cos_param.png', 'figures/traj_ratio.png'):
            assert os.path.exists(os.path.join(tmp, fn)), fn
        tp = pd.read_csv(os.path.join(tmp, 'traj_param.csv'))
        assert set(tp['t']) == set(ts) and len(tp) == 3 * len(ts)
        ba = tp[tp['pair'] == 'bce_aff'].set_index('t')
        assert (ba.loc[[250, 500, 750], 'cos_all'] < 0).all() and (ba.loc[[1000, 36000], 'cos_all'] > 0).all()
        assert dec['verdict'] == 'D-G', dec
        md = open(os.path.join(tmp, 'summary_phanD.md'), encoding='utf-8').read()
        assert md.startswith('**Config:**') and 'KHONG duoc dien giai' in md and 'D-G' in md
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


# ───────────────────────────── e2e (tuy chon) ─────────────────────────────────

def test_e2e():
    print("=== Test 10 (e2e): measure_snapshot that + D3 + phanD_traj ===")
    import argparse
    import yaml
    from Tools.test_grad_conflict_probe import _FakeValDS
    from Tools.eval_boundary_metrics import build_eval_model
    from Tools.grad_conflict_common import CFG_STATIC
    tmp = tempfile.mkdtemp()
    try:
        cfg = yaml.safe_load(open(CFG_STATIC, encoding='utf-8'))
        torch.manual_seed(0)
        m = build_eval_model(cfg, 'static_boundary')
        snap_dir = os.path.join(tmp, 'snap')
        sv = snp.SnapshotSaver([0, 500], snap_dir, 19, 'md5', True)
        sv.maybe_save(m, 0)
        with torch.no_grad():
            for p in m.parameters():
                p.add_(0.01 * torch.randn_like(p))
        sv.maybe_save(m, 500)
        ds = _FakeValDS(n=3)
        ids = [f'img_{k}' for k in range(3)]
        od = os.path.join(tmp, 'out')
        args = argparse.Namespace(output_dir=od, scratch_dir=os.path.join(tmp, 'scr'), gt_cache='', n_perm=3,
                                  num_workers=0)
        dev = torch.device('cpu')
        for t, path in snp_list(snap_dir):
            res = pp.measure_snapshot(t, path, args, cfg, dev, 11.0014, ds, ids, keep_memmap=(t == 500))
            if t == 500:
                z = np.load(os.path.join(od, 'gram', 'iter_500.npz'))
                assert z['G'].shape == (9, 9)
                r = pp.gate_d3(z['G'], z['terms'], z['direct_pairs'], z['direct_vals'], res['memmap_path'],
                               res['memmap_shape'])
                assert r['status'] == 'PASS', r
        dec = tj.main(['--output-dir', od, '--n-boot', '200', '--n-null', '50'])
        assert dec['verdict'] in ('D-N', 'D-G', 'D-RA', 'D-L', 'D-dai-dang', None)
        tf = pd.read_csv(os.path.join(od, 'traj_feat_strata.csv'))
        assert set(tf['t']) == {0, 500}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS\n")


def snp_list(d):
    return pp.list_snapshots(d)


if __name__ == '__main__':
    test_schedule()
    test_probe_ids()
    test_gram_stats()
    test_gate_d3()
    test_windows()
    test_decide()
    test_snapshots()
    test_compare()
    test_d0_static()
    test_d0_iter1_and_static()
    test_traj_main()
    if '--e2e' in sys.argv:
        test_e2e()
    else:
        print("(bo qua e2e — them --e2e de chay measure_snapshot that tren du lieu gia lap)")
    print("Tat ca self-test Tools/test_phanD.py PASS.")
