"""Tools/phanD_probe.py — Phan D (docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md §2): do offline
xung dot gradient tren cac snapshot `iter_<t>.pth` cua run Static s19 moi, tren 32 anh train co dinh.

Duoc goi qua subcommand cua Tools/grad_conflict_probe.py (chi THEM, khong doi che do cu):

  phanD-probe-ids   §2.1: sinh probe_ids.txt (32 anh train, default_rng(19).choice tren danh sach train
                    da sap xep theo ten). Truyen --check <file> de doi chieu voi ban da commit.
  phanD-measure     §2.2/2.3: lap qua moi snapshot trong --snapshot-dir (chia shard theo GPU), moi snapshot:
                      gram/iter_<t>.npz          Gram (32*3)x(32*3) cua a_k,i = d(w_k L_k,i)/d theta_shared
                      feat/iter_<t>_strata.csv   tong theo tang (muc dac trung, giong Phan A)
                      feat/iter_<t>_null.csv     null hoan vi trong tang
                      feat/iter_<t>_loss.csv     loss tung anh
                      done/iter_<t>.json         dau van tay snapshot + thoi gian (resume: co thi bo qua)
  phanD-gates       Cong D3 (Gram dung), D4 (cong 2/3/5 S1 tren anh train), D6 (cong 1 S1, val 384 anh).

Logic do GIU NGUYEN cua Phan A: Probe.image_grads (fp32, model.eval(), 1 anh/backward, seed 19+idx,
trong so hieu dung w_region=1, w_bce=0.4, w_aff=0.4, hook F tren frh, shared_params), ImageStrata,
feature_pair_stats, gram_from_memmap, run_gates_for.
"""

import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (
    AXES, SEED, CFG_STATIC, PHAND_POS_WEIGHT, pair_name, get_git_commit, file_fingerprint, write_json,
    read_json, append_log, load_gt_cached,
)

LABEL = 'phanD_s19'
N_PROBE = 32
PROBE_BATCH = 8
D3_TOL = 1e-6


# ═════════════════════════════════ du lieu ════════════════════════════════════

def train_names(cfg):
    ds = cfg['DATASET']
    d = os.path.join(ds['ROOT_DIR'], ds['TRAIN_IMG_DIR'])
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.lower().endswith('.tif'))


def make_probe_ids(names, n=N_PROBE, seed=SEED):
    """§2.1: default_rng(19).choice(N_train, 32, replace=False) tren danh sach da sap xep theo ten.
    Giu THU TU cua choice (vi tri trong file = image_idx dung cho seed/hoan vi)."""
    idx = np.random.default_rng(seed).choice(len(names), n, replace=False)
    return [names[i] for i in idx]


def read_image_list(path):
    with open(path, encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_probe_dataset(cfg, image_list, split):
    """Anh train (hoac val) theo danh sach cho truoc; tien xu ly = get_val_transforms()
    (resize 1024, chuan hoa ImageNet, ToTensor) — §2.1."""
    from src.data.dataset import OpenEarthMapDataset
    from src.data.transforms import get_val_transforms
    ds = cfg['DATASET']
    if split == 'train':
        root, img, msk = ds['ROOT_DIR'], ds['TRAIN_IMG_DIR'], ds['TRAIN_MASK_DIR']
    else:
        root, img, msk = ds.get('VAL_ROOT_DIR', ds['ROOT_DIR']), ds['VAL_IMG_DIR'], ds['VAL_MASK_DIR']
    return OpenEarthMapDataset(root_dir=root, img_dir=img, mask_dir=msk, split_file=image_list,
                               transform=get_val_transforms())


def list_snapshots(snapshot_dir):
    out = []
    for f in os.listdir(snapshot_dir):
        if f.startswith('iter_') and f.endswith('.pth'):
            try:
                out.append((int(f[5:-4]), os.path.join(snapshot_dir, f)))
            except ValueError:
                pass
    return sorted(out)


# ═════════════════════════════════ do 1 snapshot ══════════════════════════════

def measure_snapshot(t, path, args, cfg, device, pos_weight, ds, image_ids, keep_memmap=False):
    """Do 1 snapshot. Tra ve dict thong tin (+ 'memmap_path' neu keep_memmap)."""
    from Tools.grad_conflict_probe import Probe, ImageStrata, feature_pair_stats, gram_from_memmap, batch_partition
    od = args.output_dir
    for sub in ('gram', 'feat', 'done'):
        os.makedirs(os.path.join(od, sub), exist_ok=True)
    probe = Probe(LABEL, path, cfg, device, pos_weight)
    n = len(image_ids)
    terms = list(probe.terms)
    nt, P = len(terms), probe.n_shared
    os.makedirs(args.scratch_dir, exist_ok=True)
    mm_path = os.path.join(args.scratch_dir, f'phanD_a_iter{t}_{os.getpid()}.f32')
    mm = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(n, nt, P))
    # Tong truc tiep (float64, GPU) — doi chieu cos_all tu Gram (cong D3), ghi kem moi snapshot.
    direct = {k: torch.zeros(P, dtype=torch.float64, device=device) for k in terms}

    strata_rows, null_rows, loss_rows = [], [], []
    t0 = time.time()
    for i in range(n):
        name, gt = load_gt_cached(ds, i, args.gt_cache or None)
        assert name == image_ids[i], (name, image_ids[i])
        x, m = ds[i]
        x, m = x[None].to(device), m[None].to(device)
        L, gF, gT = probe.image_grads(x, m, i, need_theta=True)
        for j, k in enumerate(terms):
            mm[i, j, :] = gT[k].cpu().numpy()
            direct[k] += gT[k].double()
        del gT
        loss_rows.append({'image_id': name, 'image_idx': i, **{f'L_{k}': v for k, v in L.items()}})
        st = ImageStrata(np.asarray(gt), gF['region'].shape[-2:], device, i, args.n_perm, False)
        for p in probe.pairs:
            pn = pair_name(p)
            fs = feature_pair_stats(gF[p[0]], gF[p[1]], st, args.n_perm)
            for ax, names in AXES.items():
                a = fs['strata'][ax].cpu().numpy()
                nl = fs['null'][ax].cpu().numpy()
                for s, sname in enumerate(names):
                    strata_rows.append({'image_id': name, 'pair': pn, 'axis': ax, 'stratum': sname,
                                        'sum_dot': a[s, 0], 'sum_n1': a[s, 1], 'sum_n2': a[s, 2],
                                        'n_pos': int(a[s, 3]), 'n_support': int(a[s, 4]), 'n_neg': int(a[s, 5])})
                    if a[s, 3] > 0:
                        for r in range(args.n_perm):
                            null_rows.append((pn, ax, sname, r, name, nl[r, s]))
        del gF
        el = time.time() - t0
        print(f"  [iter {t}] {i + 1}/{n} anh ({el:.0f}s, {el / (i + 1):.2f}s/anh)", end='\r')
    print()
    mm.flush()
    wall_grad = time.time() - t0

    t1 = time.time()
    G = gram_from_memmap(mm, device)          # (n*nt, n*nt), hang = i*nt + j
    wall_gram = time.time() - t1
    direct_stats = {}
    for p in probe.pairs:
        a, b = direct[p[0]], direct[p[1]]
        direct_stats[pair_name(p)] = [float(a @ b), float(a @ a), float(b @ b)]
    del direct
    np.savez_compressed(os.path.join(od, 'gram', f'iter_{t}.npz'), G=G, terms=np.array(terms),
                        image_ids=np.array(image_ids), iter=t,
                        batches=np.array(batch_partition(n, PROBE_BATCH)),
                        direct_pairs=np.array(list(direct_stats)),
                        direct_vals=np.array(list(direct_stats.values()), dtype=np.float64))
    pd.DataFrame(strata_rows).to_csv(os.path.join(od, 'feat', f'iter_{t}_strata.csv'), index=False)
    pd.DataFrame(null_rows, columns=['pair', 'axis', 'stratum', 'perm_id', 'image_id', 'sum_dot_perm']) \
        .to_csv(os.path.join(od, 'feat', f'iter_{t}_null.csv'), index=False)
    pd.DataFrame(loss_rows).to_csv(os.path.join(od, 'feat', f'iter_{t}_loss.csv'), index=False)
    del mm
    res = {'iter': t, 'n_images': n, 'terms': terms, 'pairs': [pair_name(p) for p in probe.pairs],
           'weights': probe.weights, 'n_shared_params': P, 'wall_time_grad_s': wall_grad,
           'wall_time_gram_s': wall_gram, 'wall_time_s': time.time() - t0}
    if keep_memmap:
        res['memmap_path'] = mm_path
        res['memmap_shape'] = [n, nt, P]
    else:
        try:
            os.remove(mm_path)
        except OSError:
            pass
    return res


def _setup(args):
    from Tools.grad_conflict_probe import resolve_device_det
    from Tools.oracle_boundary_ceiling import load_cfg
    cfg = load_cfg(args.config, args.data_root)
    device = resolve_device_det(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    return cfg, device


def _probe_dataset(args, cfg):
    from Tools.oracle_boundary_ceiling import image_ids_of
    ds = build_probe_dataset(cfg, args.image_list, args.split)
    ids = image_ids_of(ds)
    if ids != read_image_list(args.image_list):
        raise SystemExit(f"Dataset khong khop {args.image_list} (thu tu/ten anh).")
    return ds, ids


# ═════════════════════════════════ subcommand ═════════════════════════════════

def cmd_probe_ids(args):
    from Tools.oracle_boundary_ceiling import load_cfg
    cfg = load_cfg(args.config, args.data_root)
    names = train_names(cfg)
    ids = make_probe_ids(names)
    if args.check and os.path.exists(args.check):
        old = read_image_list(args.check)
        if old != ids:
            raise SystemExit(f"[LOI] probe_ids sinh lai KHAC {args.check} — khong duoc doi tap do (S2 §8).")
        print(f"  probe_ids khop ban da luu: {args.check}")
    if os.path.exists(args.output) and read_image_list(args.output) != ids:
        raise SystemExit(f"[LOI] {args.output} da ton tai va KHAC ban sinh lai — khong ghi de.")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ids) + '\n')
    write_json({'n_train': len(names), 'n_probe': len(ids), 'rule': 'np.random.default_rng(19).choice(N_train, 32, '
                'replace=False) tren danh sach train sap xep theo ten', 'probe_ids': ids,
                'created_at': datetime.datetime.now().isoformat(timespec='seconds')},
               os.path.splitext(args.output)[0] + '_meta.json')
    print(f"  {len(ids)} anh / N_train={len(names)} -> {args.output}")


def cmd_measure(args):
    cfg, device = _setup(args)
    ds, ids = _probe_dataset(args, cfg)
    snaps = list_snapshots(args.snapshot_dir)
    if not snaps:
        raise SystemExit(f"Khong co iter_<t>.pth trong {args.snapshot_dir}")
    k, nshard = (int(x) for x in args.shard.split('/'))
    mine = [s for j, s in enumerate(snaps) if j % nshard == k]
    log_path = os.path.join(args.output_dir, 'run_log.txt')
    print(f"Phan D measure: {len(mine)}/{len(snaps)} snapshot (shard {args.shard}) tren {device}")
    for t, path in mine:
        done_path = os.path.join(args.output_dir, 'done', f'iter_{t}.json')
        fp = file_fingerprint(path)
        if os.path.exists(done_path) and not args.force:
            d = read_json(done_path)
            if d.get('snapshot_fingerprint') == fp and d.get('n_perm') == args.n_perm:
                print(f"[iter {t}] da do xong — bo qua")
                continue
        res = measure_snapshot(t, path, args, cfg, device, args.pos_weight, ds, ids)
        done = {'label': LABEL, 'snapshot': path, 'snapshot_fingerprint': fp, 'pos_weight': args.pos_weight,
                'image_list': args.image_list, 'split': args.split, 'n_perm': args.n_perm, 'device': str(device),
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'mode': 'model.eval() — BN running stats, dropout/drop-path tat; fp32; 1 anh/backward',
                'torch': torch.__version__, 'git_commit': get_git_commit(),
                'created_at': datetime.datetime.now().isoformat(timespec='seconds'), **res}
        write_json(done, done_path)
        append_log(log_path, f"[{done['created_at']}] measure iter {t}: {res['wall_time_s']:.0f}s gpu={done['gpu']}")
        print(f"[iter {t}] xong trong {res['wall_time_s']:.0f}s")


def gate_d3(G, terms, direct_pairs, direct_vals, mm_path, shape):
    """D3: (a) cos_all tu Gram == cos tren tong gradient truc tiep; (b) Gram tang dan (theo khoi tham so,
    tu memmap) == Gram day du (1 khoi, nap toan bo vao RAM)."""
    from Tools.grad_conflict_probe import gram_from_memmap
    from Tools.phanD_traj import gram_pair_blocks
    errs_a = {}
    for pn, (dot, n1, n2) in zip(direct_pairs, direct_vals):
        k1, k2 = str(pn).split('_')
        G11, G22, G12 = gram_pair_blocks(G, list(terms), k1, k2)
        one = np.ones(G11.shape[0])
        cos_g = (one @ G12 @ one) / np.sqrt((one @ G11 @ one) * (one @ G22 @ one))
        cos_d = dot / np.sqrt(n1 * n2)
        errs_a[str(pn)] = abs(float(cos_g - cos_d))
    mm = np.memmap(mm_path, dtype=np.float32, mode='c', shape=tuple(shape))
    G_full = gram_from_memmap(mm, torch.device('cpu'), chunk=int(shape[2]))
    del mm
    scale = float(np.abs(G_full).max()) or 1.0
    err_b = float(np.abs(G - G_full).max()) / scale
    ok = max(errs_a.values()) < D3_TOL and err_b < D3_TOL
    return {'gate': 'D3', 'name': 'Gram dung', 'status': 'PASS' if ok else 'FAIL', 'threshold': D3_TOL,
            'value': {'abs_err_cos_all_gram_vs_direct': errs_a, 'max_rel_err_gram_incremental_vs_full': err_b},
            'note': 'Gram tang dan = gram_from_memmap theo khoi 2^19 tham so; day du = 1 khoi tren CPU float64'}


def cmd_gates(args):
    from Tools.grad_conflict_probe import run_gates_for
    cfg, device = _setup(args)
    out = {}
    if args.gate == 'd3':
        ds, ids = _probe_dataset(args, cfg)
        t = int(os.path.basename(args.snapshot)[5:-4])
        res = measure_snapshot(t, args.snapshot, args, cfg, device, args.pos_weight, ds, ids, keep_memmap=True)
        z = np.load(os.path.join(args.output_dir, 'gram', f'iter_{t}.npz'))
        try:
            out = gate_d3(z['G'], z['terms'], z['direct_pairs'], z['direct_vals'], res['memmap_path'],
                          res['memmap_shape'])
        finally:
            os.remove(res['memmap_path'])
        out['value']['s_per_snapshot'] = res['wall_time_s']
        out['value']['s_per_image'] = res['wall_time_grad_s'] / max(res['n_images'], 1)
        out['value']['n_shared_params'] = res['n_shared_params']
    elif args.gate == 'd4':
        ds, ids = _probe_dataset(args, cfg)
        args.max_images = 4
        r = run_gates_for(LABEL, args, args.snapshot, cfg, device, args.pos_weight, None, None,
                          dataset=ds, gate_ids={2, 3, 5})
        out = {'gate': 'D4', 'name': 'cong 2/3/5 cua S1 tren 4 anh do (train)', 'status': r['overall'],
               'value': r['gates'], 'threshold': '< 1e-5; < 1e-5; < 1e-6 (nhu S1)'}
    elif args.gate == 'd6':
        args.max_images = None
        r = run_gates_for(LABEL, args, args.snapshot, cfg, device, args.pos_weight, None, args.expected_miou,
                          dataset=None, gate_ids={1})
        out = {'gate': 'D6', 'name': 'nap dung snapshot 36k (mIoU-9 val 384 anh vs log train)',
               'status': r['overall'], 'value': r['gates'], 'threshold': 0.001}
    out.update({'snapshot': args.snapshot, 'snapshot_fingerprint': file_fingerprint(args.snapshot),
                'device': str(device), 'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'git_commit': get_git_commit()})
    write_json(out, args.json_out)
    print(f"  cong {out['gate']}: {out['status']} -> {args.json_out}")
    if out['status'] != 'PASS' and not args.no_fail_exit:
        raise SystemExit(2)


# ═════════════════════════════════ CLI ════════════════════════════════════════

def _common(p):
    p.add_argument('--config', default=CFG_STATIC)
    p.add_argument('--data-root', default=None)
    p.add_argument('--output-dir', default='output/phanD')
    p.add_argument('--device', default=None)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--pos-weight', type=float, default=PHAND_POS_WEIGHT)
    p.add_argument('--image-list', default='output/phanD/probe_ids.txt')
    p.add_argument('--split', choices=['train', 'val'], default='train')
    p.add_argument('--gt-cache', default='output/_cache/gt1024_train', help="cache GT .npy ('' = tat)")
    p.add_argument('--n-perm', type=int, default=20)
    p.add_argument('--scratch-dir', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'phanD_scratch')
                   if os.name != 'nt' else os.path.join(os.environ.get('TEMP', '.'), 'phanD_scratch'))


def add_subparsers(sub):
    p = sub.add_parser('phanD-probe-ids', help='Phan D §2.1: sinh probe_ids.txt')
    p.add_argument('--config', default=CFG_STATIC)
    p.add_argument('--data-root', default=None)
    p.add_argument('--output', default='output/phanD/probe_ids.txt')
    p.add_argument('--check', default=None, help='File probe_ids da commit — phai trung')

    p = sub.add_parser('phanD-measure', help='Phan D §2: do moi snapshot trong --snapshot-dir')
    _common(p)
    p.add_argument('--snapshot-dir', required=True)
    p.add_argument('--shard', default='0/1', help='k/n: chi do snapshot thu j voi j %% n == k (chia GPU)')
    p.add_argument('--force', action='store_true')

    p = sub.add_parser('phanD-gates', help='Phan D: cong D3 / D4 / D6')
    _common(p)
    p.add_argument('--gate', choices=['d3', 'd4', 'd6'], required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--expected-miou', type=float, default=None, help='D6: mIoU-9 vong 36000 trong log train')
    p.add_argument('--json-out', required=True)
    p.add_argument('--no-fail-exit', action='store_true', help='Ghi FAIL nhung thoat 0 (dung cho D6 sau train)')


def dispatch(args):
    return {'phanD-probe-ids': cmd_probe_ids, 'phanD-measure': cmd_measure, 'phanD-gates': cmd_gates}[args.cmd](args)
