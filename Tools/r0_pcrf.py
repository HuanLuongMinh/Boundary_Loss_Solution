"""Tools/r0_pcrf.py — R0 cua docs/S4-spec-P0-R0-ban-giao-claude-code.md §3: P-CRF (DenseCRF, Kraehenbuehl &
Koltun 2011, thu vien pydensecrf) — baseline KHONG hoc, thuc hien nguyen van K2 §5.1.

Stage (subcommand `run --stages ...`, mac dinh theo thu tu duoi; moi stage co file done_* de resume):
  g1      R0-g1: forward val (model.eval(), fp32), mIoU-9 "truoc CRF" khop bang checkpoint (dung sai 0.001)
          — TRUOC khi chay CRF. FAIL -> DUNG.
  g23     R0-g2: CRF chi unary (bi_w = pos_w = 0) == argmax softmax >= 99.99% pixel tren 4 anh;
          R0-g3: chay lai 2 anh cung tham so -> trung tuyet doi. FAIL -> DUNG.
  grid    §3.4: 27 to hop x 32 anh train cua probe_ids.txt (Phan D), moi checkpoint -> grid/<nhan>.csv
  select  §3.4: tieu chi = min (ASD pred→gt + ASD gt→pred)/2, micro tren 32 anh; hoa -> to hop dau theo thu tu
          luoi -> crf_grid_train.csv, crf_selected.json; R0-g4 (khong ro val). FAIL -> DUNG.
  val     §3.5: 384 anh val, to hop da chon -> per-image truoc/sau, mo ta §3.8, mask sau CRF, hinh
  report  bootstrap ghep cap (micro, 10.000 lan, rng 19), luat §3.6, mo ta §3.8 -> crf_val.csv/.md,
          per_class.csv, changed_pixels.csv, small_regions.csv, figures/, gates_r0.json

Dau vao CRF (§3.2): softmax fp32 1024x1024 cua checkpoint; unary = -log(max(p, 1e-8)); anh RGB goc uint8
1024x1024 CHUA chuan hoa cho kernel song phuong. Du phong `--input-mode gt_prob` (chi khi softmax khong kha
thi): unary tu nhan argmax voi gt_prob = 0.7. Che do ghi trong MOI output.
Cố dinh: pos_xy_std 3, pos_w 3, 10 vong lap. Luoi: bi_xy_std {40,80,121} x bi_rgb_std {3,5,10} x bi_w {3,4,5}.

KHONG tu viet CRF thay the: thieu pydensecrf -> dung (crf_backend_info() raise).
Huong dan Kaggle: docs/huong-dan-kaggle-P0-R0.md
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (  # noqa: E402
    CFG_STATIC, CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, SEED, STRUCT_8, append_log, get_git_commit, read_json,
    write_json, load_gt_cached,
)
from Tools.refine_common import (  # noqa: E402
    DECISION_LABELS, R0_LABELS, bounded_imap, ckpt_info, iter_softmax, load_model, load_rgb_1024, make_pool,
    per_image_stats, resolve_ckpt_path, save_png,
)

# ─────────────────────────────── Hang so spec ─────────────────────────────────

GRID = [dict(bi_xy_std=x, bi_rgb_std=r, bi_w=w) for x in (40, 80, 121) for r in (3, 5, 10) for w in (3, 4, 5)]
FIXED = dict(pos_xy_std=3, pos_w=3, n_iter=10)
P_MIN = 1e-8
GT_PROB = 0.7
N_BOOT = 10000
BD = (1, 2, 4)
NO_BD = ()
N_PROBE = 32
N_VAL = 384

G1_TOL = 1e-3
G2_LABEL, G2_N, G2_MIN = 'static_s19_36k', 4, 0.9999
G3_N, G3_COMBO = 2, 13                      # GRID[13] = (80, 5, 4) — giua luoi
EXPECTED_MIOU9_EXTRA = {'static_s19r_36k': 0.6404}   # §2.3 (CSV val Phan D) — nhan ngoai CHECKPOINTS

RULE_G2P_MAX = 0.534                        # §3.6 (iii): bien do lien-seed Static
RULE_RECALL_LO = -0.0094                    # §3.6 (iv)
RULE_BIOU_Q = 'biou_8_d2'                   # §3.6 (ii): BIoU@d2 theo quy uoc bootstrap_boundary_ci (8 lop)
B2_REF = {'bf_precision': 0.0130, 'bf_recall': -0.0163, 'asd_gt_to_pred': 0.693}   # §3.8.4 (Static s19@36k)

STUFF = ('Rangeland', 'Agriculture', 'Bareland', 'Water')
THING = ('Building', 'Road')
SEPARATE = ('Developed', 'Tree')
SMALL_MAX = 700                             # §3.8.3
BANDS = (('le2', 0, 2), ('2to8', 2, 8), ('gt8', 8, np.inf))   # §3.8.2: (lo, hi] px toi bien GT

QUANTITIES = ('mIoU_9', 'mIoU_8', 'bf_precision', 'bf_recall', 'bf_score', 'biou_8_d1', 'biou_8_d2', 'biou_8_d4',
              'biou_9_d1', 'biou_9_d2', 'biou_9_d4', 'asd_pred_to_gt', 'asd_gt_to_pred', 'asd')
QSHORT = {'mIoU_9': 'mIoU-9', 'mIoU_8': 'mIoU-8', 'bf_precision': 'BF-P', 'bf_recall': 'BF-R', 'bf_score': 'BF-F',
          'biou_8_d1': 'BIoU8@1', 'biou_8_d2': 'BIoU8@2', 'biou_8_d4': 'BIoU8@4', 'biou_9_d1': 'BIoU9@1',
          'biou_9_d2': 'BIoU9@2', 'biou_9_d4': 'BIoU9@4', 'asd_pred_to_gt': 'ASD p→g', 'asd_gt_to_pred': 'ASD g→p',
          'asd': 'ASD'}

K2_52_NOTE = ('Theo K2 §5.2 (ban sua 2), moi ma cua R0 la **mo ta**: R0 la baseline khong hoc, '
              '**khong** la cong dung huong tinh chinh hau ky.')


def combo_name(p):
    return f"xy{p['bi_xy_std']}_rgb{p['bi_rgb_std']}_w{p['bi_w']}"


def expected_miou9(label):
    return EXPECTED_MIOU9_EXTRA.get(label, ckpt_info(label)['expected_miou9'])


# ═════════════════════════════════ CRF ════════════════════════════════════════

def crf_backend_info():
    """Phien ban/commit pydensecrf. Khong import duoc -> SystemExit (§3.3: KHONG tu viet CRF thay the)."""
    try:
        import pydensecrf
        import pydensecrf.densecrf  # noqa: F401
    except Exception as e:  # pragma: no cover - phu thuoc moi truong
        raise SystemExit(f"[R0] Khong import duoc pydensecrf ({type(e).__name__}: {e}) — DUNG, khong tu viet CRF "
                         f"thay the (§3.3). Cai: pip install git+https://github.com/lucasb-eyer/pydensecrf.git")
    info = {'module_file': getattr(pydensecrf, '__file__', None)}
    try:
        import importlib.metadata as md
        dist = md.distribution('pydensecrf')
        info['version'] = dist.version
        du = dist.read_text('direct_url.json')
        if du:
            import json
            j = json.loads(du)
            info['source_url'] = j.get('url')
            info['commit'] = (j.get('vcs_info') or {}).get('commit_id')
    except Exception as e:  # pragma: no cover
        info['metadata_error'] = str(e)
    return info


def unary_from_prob(prob, mode):
    """prob (C,H,W) fp32 -> unary (C,H*W) fp32 C-contiguous.
    softmax : -log(max(p, 1e-8))                                   (§3.2 che do chinh)
    gt_prob : nhan = argmax(p); -log(0.7) cho nhan, -log(0.3/(C-1)) cho lop khac (= pydensecrf.utils.unary_from_labels,
              zero_unsure=False) (§3.2 du phong)."""
    C = prob.shape[0]
    if mode == 'softmax':
        u = -np.log(np.maximum(prob, P_MIN))
    elif mode == 'gt_prob':
        lab = prob.argmax(0)
        u = np.full(prob.shape, -np.log((1.0 - GT_PROB) / (C - 1)), dtype=np.float32)
        np.put_along_axis(u, lab[None], -np.log(GT_PROB), axis=0)
    else:
        raise ValueError(mode)
    return np.ascontiguousarray(u.reshape(C, -1).astype(np.float32))


def crf_infer(prob, rgb, params, mode='softmax', unary=None):
    """DenseCRF2D (pydensecrf): Gaussian (pos_xy_std, pos_w) + bilateral (bi_xy_std, bi_rgb_std, bi_w), Potts,
    n_iter vong mean-field. Tra ve nhan uint8 (H,W) = argmax Q."""
    import pydensecrf.densecrf as dcrf
    C, H, W = prob.shape
    d = dcrf.DenseCRF2D(W, H, C)
    d.setUnaryEnergy(unary_from_prob(prob, mode) if unary is None else unary)
    d.addPairwiseGaussian(sxy=float(params['pos_xy_std']), compat=float(params['pos_w']))
    d.addPairwiseBilateral(sxy=float(params['bi_xy_std']), srgb=float(params['bi_rgb_std']),
                           rgbim=np.ascontiguousarray(rgb, dtype=np.uint8), compat=float(params['bi_w']))
    Q = np.asarray(d.inference(int(params['n_iter'])), dtype=np.float32).reshape(C, H, W)
    return Q.argmax(0).astype(np.uint8)


# ═════════════════════════════ Worker (tien trinh con) ════════════════════════

def grid_worker(task):
    """task = (name, prob, img_path, gt, combos, mode) -> before row + 1 row/to hop (chi mIoU/BF/ASD)."""
    name, prob, img_path, gt, combos, mode = task
    rgb = load_rgb_1024(img_path, gt.shape[0])
    unary = unary_from_prob(prob, mode)
    before = prob.argmax(0).astype(np.uint8)
    rows = [dict(per_image_stats(before, gt, name, NO_BD), combo=-1)]
    t_crf = 0.0
    for k, p in combos:
        t = time.time()
        pred = crf_infer(prob, rgb, {**FIXED, **p}, mode, unary=unary)
        t_crf += time.time() - t
        rows.append(dict(per_image_stats(pred, gt, name, NO_BD), combo=k))
    return {'name': name, 'rows': rows, 't_crf': t_crf, 'n_crf': len(combos)}


def gt_edge_dist(gt):
    from scipy.ndimage import distance_transform_edt
    from Tools.grad_conflict_common import gt_edge_np
    edge = gt_edge_np(gt)
    return distance_transform_edt(~edge) if edge.any() else np.full(gt.shape, np.inf)


def small_components(before, after, valid):
    """§3.8.3: thanh phan du doan (8-lien thong, theo lop) dien tich <= 700 px; 'bi xoa' = khong con pixel nao
    cua lop do sau CRF. Tra ve list dict/lop."""
    from scipy.ndimage import label as cc_label
    out = []
    for c in range(NUM_CLASSES):
        lab, n = cc_label(before == c, structure=STRUCT_8)
        if n == 0:
            out.append({'class': CLASS_NAMES[c], 'n_small': 0, 'n_deleted': 0, 'px_small': 0, 'px_deleted': 0})
            continue
        areas = np.bincount(lab.ravel(), minlength=n + 1)
        kept = np.bincount(lab[after == c].ravel(), minlength=n + 1)
        small = np.zeros(n + 1, dtype=bool)
        small[1:] = areas[1:] <= SMALL_MAX
        deleted = small & (kept == 0)
        out.append({'class': CLASS_NAMES[c], 'n_small': int(small.sum()), 'n_deleted': int(deleted.sum()),
                    'px_small': int(areas[small].sum()), 'px_deleted': int(areas[deleted].sum())})
    return out


def val_worker(task):
    """task = (name, prob, img_path, gt, params, mode) -> per-image truoc/sau (BIoU 1,2,4) + mo ta §3.8 + mask."""
    name, prob, img_path, gt, params, mode = task
    rgb = load_rgb_1024(img_path, gt.shape[0])
    before = prob.argmax(0).astype(np.uint8)
    t = time.time()
    after = crf_infer(prob, rgb, {**FIXED, **params}, mode)
    t_crf = time.time() - t
    valid = gt != IGNORE_INDEX
    changed = (after != before) & valid
    dist = gt_edge_dist(gt)
    ch = {'image': name, 'n_valid_px': int(valid.sum()), 'n_changed': int(changed.sum())}
    for tag, lo, hi in BANDS:
        band = valid & (dist > lo) & (dist <= hi) if lo > 0 else valid & (dist <= hi)
        ch[f'n_band_{tag}'] = int(band.sum())
        ch[f'n_changed_{tag}'] = int((changed & band).sum())
    small = [dict(image=name, **r) for r in small_components(before, after, valid)]
    return {'name': name, 'before': per_image_stats(before, gt, name, BD), 'after': per_image_stats(after, gt, name, BD),
            'changed': ch, 'small': small, 'after_mask': after, 'before_mask': before, 't_crf': t_crf}


# ═════════════════════════════ Du lieu / model ════════════════════════════════

class Ctx:
    """Moi thu dung chung giua cac stage."""

    def __init__(self, args):
        from Tools.oracle_boundary_ceiling import build_val_dataset, image_ids_of, load_cfg
        from Tools.refine_common import resolve_device
        self.args = args
        self.out = args.output_dir
        self.labels = [x for x in args.labels.split(',') if x]
        for lb in self.labels:
            ckpt_info(lb)
        self.cfg_data = load_cfg(CFG_STATIC, args.data_root)
        self.val_ds = build_val_dataset(self.cfg_data)
        self.val_ids = image_ids_of(self.val_ds)
        self.n_val = len(self.val_ids) if args.max_val is None else min(args.max_val, len(self.val_ids))
        self.device = resolve_device(args.device)
        self.mode = args.input_mode
        self.dryrun = args.max_val is not None or args.max_probe is not None
        self._probe = None

    def cfg_of(self, label):
        from Tools.oracle_boundary_ceiling import load_cfg
        return load_cfg(ckpt_info(label)['config'], self.args.data_root)

    def ckpt_path(self, label):
        p = resolve_ckpt_path(label, self.args.ckpt_dir, self.args.snapshot_dir)
        if not os.path.exists(p):
            raise SystemExit(f"[{label}] khong thay checkpoint: {p}")
        return p

    def model(self, label):
        return load_model(label, self.ckpt_path(label), self.cfg_of(label), self.device)

    def probe(self):
        """(dataset train 32 anh theo probe_ids.txt, list ten) — dung thu tu file."""
        if self._probe is None:
            from Tools.phanD_probe import build_probe_dataset, read_image_list
            from Tools.oracle_boundary_ceiling import image_ids_of
            ids = read_image_list(self.args.probe_ids)
            ds = build_probe_dataset(self.cfg_data, self.args.probe_ids, 'train')
            if image_ids_of(ds) != ids:
                raise SystemExit(f"Dataset train khong khop {self.args.probe_ids}")
            n = len(ids) if self.args.max_probe is None else min(self.args.max_probe, len(ids))
            self._probe = (ds, ids, n)
        return self._probe

    def log(self, text):
        append_log(os.path.join(self.out, 'run_log.txt'),
                   f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {text}")


def free(model, device):
    import gc
    import torch
    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ═════════════════════════════════ Stage g1 ═══════════════════════════════════

def stage_g1(ctx):
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.utils.metrics import SegmentationMetrics
    from Tools.refine_common import model_logits
    gd = os.path.join(ctx.out, 'gates')
    os.makedirs(gd, exist_ok=True)
    ds = ctx.val_ds if ctx.n_val == len(ctx.val_ds) else Subset(ctx.val_ds, list(range(ctx.n_val)))
    checks = []
    for lb in ctx.labels:
        t0 = time.time()
        model = ctx.model(lb)
        sm = SegmentationMetrics(NUM_CLASSES, IGNORE_INDEX)
        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=ctx.device.type == 'cuda')
        with torch.no_grad():
            for images, masks in loader:
                logits = model_logits(model, ckpt_info(lb)['model_type'], images.to(ctx.device).float()).float()
                sm.update(logits, masks.to(ctx.device))
        free(model, ctx.device)
        got = sm.compute()['mIoU']
        exp = expected_miou9(lb)
        ok = None if ctx.dryrun or exp is None else bool(abs(got - exp) <= G1_TOL)
        checks.append({'label': lb, 'expected_miou9': exp, 'got': got,
                       'abs_diff': None if exp is None else abs(got - exp), 'pass': ok,
                       'checkpoint': ctx.ckpt_path(lb), 'n_images': ctx.n_val, 'seconds': time.time() - t0})
        print(f"  [{lb}] mIoU-9 truoc CRF {got:.5f} (ky vong {exp}) — {time.time() - t0:.0f}s")
    status = 'SKIP_DRYRUN' if ctx.dryrun else ('PASS' if all(c['pass'] for c in checks) else 'FAIL')
    g = {'gate': 'R0-g1', 'status': status, 'tol': G1_TOL, 'checks': checks,
         'detail': '; '.join(f"{c['label']} {c['got']:.4f}" + (f" vs {c['expected_miou9']:.4f}" if c['expected_miou9']
                                                                 else '') for c in checks)}
    write_json(g, os.path.join(gd, 'R0-g1.json'))
    ctx.log(f"g1 {status}: {g['detail']}")
    if status == 'FAIL':
        raise SystemExit("R0-g1 FAIL: mIoU-9 truoc CRF lech bang checkpoint — DUNG truoc khi chay CRF (§3.1).")


# ═════════════════════════════════ Stage g23 ══════════════════════════════════

def stage_g23(ctx):
    gd = os.path.join(ctx.out, 'gates')
    os.makedirs(gd, exist_ok=True)
    lb = G2_LABEL if G2_LABEL in ctx.labels else ctx.labels[0]
    ds, ids, _n = ctx.probe()
    n = max(G2_N, G3_N)
    model = ctx.model(lb)
    items = list(iter_softmax(model, ckpt_info(lb)['model_type'], ds, range(n), ctx.device))
    free(model, ctx.device)
    # R0-g2
    unary_only = {**FIXED, **GRID[0], 'bi_w': 0, 'pos_w': 0}
    per, same, tot = [], 0, 0
    for name, prob, gt, path in items[:G2_N]:
        rgb = load_rgb_1024(path, gt.shape[0])
        pred = crf_infer(prob, rgb, unary_only, ctx.mode)
        ref = prob.argmax(0)
        s = int((pred == ref).sum())
        per.append({'image': name, 'agree_frac': s / pred.size})
        same += s
        tot += pred.size
    frac = same / tot
    g2 = {'gate': 'R0-g2', 'status': 'PASS' if frac >= G2_MIN else 'FAIL', 'threshold': G2_MIN, 'label': lb,
          'params': unary_only, 'agree_frac_pooled': frac, 'per_image': per,
          'detail': f"CRF chi unary vs argmax softmax: trung {frac:.6f} pixel tren {len(per)} anh train ({lb})"}
    # R0-g3
    p3 = {**FIXED, **GRID[G3_COMBO]}
    per3, ok3 = [], True
    for name, prob, gt, path in items[:G3_N]:
        rgb = load_rgb_1024(path, gt.shape[0])
        a = crf_infer(prob, rgb, p3, ctx.mode)
        b = crf_infer(prob, rgb, p3, ctx.mode)
        eq = bool(np.array_equal(a, b))
        ok3 &= eq
        per3.append({'image': name, 'identical': eq, 'n_diff_px': int((a != b).sum())})
    g3 = {'gate': 'R0-g3', 'status': 'PASS' if ok3 else 'FAIL', 'threshold': 'trung tuyet doi', 'label': lb,
          'params': p3, 'per_image': per3,
          'detail': f"chay lai {len(per3)} anh, {combo_name(GRID[G3_COMBO])}: "
                    f"{'trung tuyet doi' if ok3 else 'KHAC'}"}
    for g in (g2, g3):
        write_json(g, os.path.join(gd, f"{g['gate']}.json"))
        print(f"  [{g['status']}] {g['gate']}: {g['detail']}")
        ctx.log(f"{g['gate']} {g['status']}: {g['detail']}")
    if g2['status'] != 'PASS' or g3['status'] != 'PASS':
        raise SystemExit("R0-g2/g3 FAIL — DUNG (cai dat CRF khong dung nhu ky vong).")


# ═════════════════════════════════ Stage grid ═════════════════════════════════

def stage_grid(ctx, pool):
    gdir = os.path.join(ctx.out, 'grid')
    os.makedirs(gdir, exist_ok=True)
    ds, ids, n = ctx.probe()
    combos = list(enumerate(GRID))
    for lb in ctx.labels:
        dp = os.path.join(gdir, f'done_{lb}.json')
        if os.path.exists(dp) and not ctx.args.force:
            d = read_json(dp)
            if d.get('n_images') == n and d.get('input_mode') == ctx.mode:
                print(f"  [{lb}] luoi da xong — bo qua")
                continue
        t0 = time.time()
        model = ctx.model(lb)
        mt = ckpt_info(lb)['model_type']
        tasks = ((name, prob, path, gt, combos, ctx.mode)
                 for name, prob, gt, path in iter_softmax(model, mt, ds, range(n), ctx.device))
        rows, names, t_crf, n_crf = [], [], 0.0, 0
        for k, r in enumerate(bounded_imap(pool, grid_worker, tasks, 2 * max(1, ctx.args.workers))):
            rows += r['rows']
            names.append(r['name'])
            t_crf += r['t_crf']
            n_crf += r['n_crf']
            print(f"  [{lb}] luoi {k + 1}/{n} anh ({time.time() - t0:.0f}s, CRF {t_crf / max(n_crf, 1):.2f}s/lan)",
                  end='\r')
        print()
        free(model, ctx.device)
        assert names == ids[:n], f"[{lb}] lech thu tu anh probe"
        pd.DataFrame(rows).to_csv(os.path.join(gdir, f'{lb}.csv'), index=False)
        write_json({'label': lb, 'n_images': n, 'images': names, 'input_mode': ctx.mode, 'n_combos': len(GRID),
                    'crf_s_mean': t_crf / max(n_crf, 1), 'wall_time_s': time.time() - t0,
                    'workers': ctx.args.workers, 'device': str(ctx.device), 'git_commit': get_git_commit(),
                    'created_at': datetime.datetime.now().isoformat(timespec='seconds')}, dp)
        ctx.log(f"grid {lb}: {n} anh x {len(GRID)} to hop, {(time.time() - t0) / 60:.1f} ph, "
                f"CRF {t_crf / max(n_crf, 1):.2f}s/lan")


# ═════════════════════════════════ Stage select ═══════════════════════════════

def aggregate_grid(df):
    from Tools.bootstrap_boundary_ci import aggregate_point_estimate
    rows = []
    for k in [-1] + list(range(len(GRID))):
        sub = df[df['combo'] == k]
        a = aggregate_point_estimate(sub, 'micro', 'skip', boundary_distances=NO_BD)
        p = GRID[k] if k >= 0 else {'bi_xy_std': None, 'bi_rgb_std': None, 'bi_w': None}
        rows.append({'combo': k, **p, 'asd_pred_to_gt': a['asd_pred_to_gt'], 'asd_gt_to_pred': a['asd_gt_to_pred'],
                     'asd_mean2': (a['asd_pred_to_gt'] + a['asd_gt_to_pred']) / 2, 'bf_precision': a['bf_precision'],
                     'bf_recall': a['bf_recall'], 'bf_score': a['bf_score'], 'mIoU_9': a['mIoU_9'],
                     'n_images': len(sub)})
    return rows


def select_best(grid_rows):
    """Min asd_mean2; hoa -> to hop xuat hien truoc theo thu tu luoi (bi_xy_std, bi_rgb_std, bi_w tang dan)."""
    best = None
    for r in grid_rows:
        if r['combo'] < 0 or not np.isfinite(r['asd_mean2']):
            continue
        if best is None or r['asd_mean2'] < best['asd_mean2']:
            best = r
    return best


def stage_select(ctx):
    from Tools.phanD_probe import make_probe_ids, read_image_list, train_names
    gdir = os.path.join(ctx.out, 'grid')
    _ds, probe_ids, n = ctx.probe()
    all_rows, selected, used = [], {}, {}
    for lb in ctx.labels:
        df = pd.read_csv(os.path.join(gdir, f'{lb}.csv'))
        used[lb] = list(dict.fromkeys(df['image'].tolist()))
        rows = aggregate_grid(df)
        best = select_best(rows)
        if best is None:
            raise SystemExit(f"[{lb}] moi to hop co ASD hai chieu NaN tren tap chon (khong co bien du doan) — "
                             f"khong ap duoc tieu chi §3.4; kiem checkpoint/du lieu.")
        for r in rows:
            all_rows.append({'label': lb, **r, 'selected': bool(r['combo'] == best['combo'])})
        selected[lb] = {'combo': best['combo'], 'name': combo_name(GRID[best['combo']]),
                        'params': {**GRID[best['combo']], **FIXED}, 'asd_mean2_train': best['asd_mean2'],
                        'asd_mean2_train_before': rows[0]['asd_mean2'], 'n_images': n}
        print(f"  [{lb}] chon {selected[lb]['name']}: ASD2 {best['asd_mean2']:.4f} (truoc CRF {rows[0]['asd_mean2']:.4f})")
    pd.DataFrame(all_rows).to_csv(os.path.join(ctx.out, 'crf_grid_train.csv'), index=False)

    # R0-g4: tap chon = dung 32 id cua probe_ids.txt, khong id val nao
    val_all = set(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ctx.val_ds.root_dir, ctx.val_ds.img_dir))
                  if f.lower().endswith('.tif'))
    regen = make_probe_ids(train_names(ctx.cfg_data))
    file_ids = read_image_list(ctx.args.probe_ids)
    checks = []
    for lb, u in used.items():
        checks.append({'label': lb, 'n_used': len(u), 'n_unique': len(set(u)),
                       'equals_probe_ids': set(u) == set(file_ids) and len(u) == len(file_ids) == N_PROBE,
                       'subset_of_probe_ids': set(u) <= set(file_ids), 'n_val_ids_used': len(set(u) & val_all),
                       'val_ids_used': sorted(set(u) & val_all)})
    leak = sum(c['n_val_ids_used'] for c in checks)
    regen_ok = regen == file_ids
    if ctx.dryrun:
        status = 'SKIP_DRYRUN' if leak == 0 and all(c['subset_of_probe_ids'] for c in checks) and regen_ok else 'FAIL'
    else:
        status = 'PASS' if leak == 0 and all(c['equals_probe_ids'] for c in checks) and regen_ok else 'FAIL'
    g4 = {'gate': 'R0-g4', 'status': status, 'checks': checks, 'probe_ids_file': ctx.args.probe_ids,
          'probe_ids_regenerated_match': regen_ok, 'n_probe_ids': len(file_ids),
          'detail': f"anh dung de chon: {', '.join(str(c['n_used']) for c in checks)} / {len(file_ids)} id probe; "
                    f"id val bi dung: {leak}; probe_ids sinh lai (rng 19) {'khop' if regen_ok else 'LECH'}"}
    write_json(g4, os.path.join(ctx.out, 'gates', 'R0-g4.json'))
    write_json({'rule': 'min (ASD pred→gt + ASD gt→pred)/2, micro tren tap chon; hoa -> to hop dau theo thu tu luoi '
                        '(bi_xy_std, bi_rgb_std, bi_w tang dan)',
                'grid': GRID, 'fixed': FIXED, 'input_mode': ctx.mode, 'selection_images': file_ids[:n],
                'selection_images_used': used, 'selected': selected, 'dryrun': ctx.dryrun,
                'git_commit': get_git_commit(), 'created_at': datetime.datetime.now().isoformat(timespec='seconds')},
               os.path.join(ctx.out, 'crf_selected.json'))
    picks = ', '.join(lb + '=' + s['name'] for lb, s in selected.items())
    ctx.log(f"select: {picks}; anh chon = {file_ids[:n]}")
    ctx.log(f"R0-g4 {status}: {g4['detail']}")
    print(f"  [{status}] R0-g4: {g4['detail']}")
    if status == 'FAIL':
        raise SystemExit("R0-g4 FAIL (ro val / sai tap chon) — DUNG.")


# ═════════════════════════════════ Stage val ══════════════════════════════════

def figure_ids(ctx):
    p = ctx.args.figure_images
    if p and os.path.exists(p):
        return [x['image_id'] for x in read_json(p)['images']]
    return []


def stage_val(ctx, pool):
    sel = read_json(os.path.join(ctx.out, 'crf_selected.json'))['selected']
    vdir = os.path.join(ctx.out, 'val')
    os.makedirs(vdir, exist_ok=True)
    figs = set(figure_ids(ctx))
    for lb in ctx.labels:
        params = {k: sel[lb]['params'][k] for k in ('bi_xy_std', 'bi_rgb_std', 'bi_w')}
        dp = os.path.join(vdir, f'done_{lb}.json')
        if os.path.exists(dp) and not ctx.args.force:
            d = read_json(dp)
            if d.get('n_images') == ctx.n_val and d.get('params') == params and d.get('input_mode') == ctx.mode:
                print(f"  [{lb}] val da xong — bo qua")
                continue
        t0 = time.time()
        model = ctx.model(lb)
        mt = ckpt_info(lb)['model_type']
        tasks = ((name, prob, path, gt, params, ctx.mode)
                 for name, prob, gt, path in iter_softmax(model, mt, ctx.val_ds, range(ctx.n_val), ctx.device))
        before, after, changed, small, names, t_crf = [], [], [], [], [], 0.0
        mdir = os.path.join(ctx.out, 'masks', lb)
        for k, r in enumerate(bounded_imap(pool, val_worker, tasks, 2 * max(1, ctx.args.workers))):
            before.append(r['before'])
            after.append(r['after'])
            changed.append(r['changed'])
            small += r['small']
            names.append(r['name'])
            t_crf += r['t_crf']
            save_png(r['after_mask'], os.path.join(mdir, r['name'] + '.png'))
            if r['name'] in figs and lb in DECISION_LABELS:
                save_png(r['before_mask'], os.path.join(ctx.out, 'figures', '_masks', lb, r['name'] + '_before.png'))
            print(f"  [{lb}] val {k + 1}/{ctx.n_val} anh ({time.time() - t0:.0f}s)", end='\r')
        print()
        free(model, ctx.device)
        assert names == ctx.val_ids[:ctx.n_val], f"[{lb}] lech thu tu anh val"
        from Tools.per_image_dump import per_image_stats_columns
        cols = per_image_stats_columns(CLASS_NAMES, BD)
        pd.DataFrame(before)[cols].to_csv(os.path.join(vdir, f'{lb}__before.csv'), index=False)
        pd.DataFrame(after)[cols].to_csv(os.path.join(vdir, f'{lb}__after.csv'), index=False)
        pd.DataFrame(changed).to_csv(os.path.join(vdir, f'{lb}__changed.csv'), index=False)
        pd.DataFrame(small).to_csv(os.path.join(vdir, f'{lb}__small.csv'), index=False)
        write_json({'label': lb, 'n_images': ctx.n_val, 'params': params, 'input_mode': ctx.mode,
                    'crf_s_mean': t_crf / max(ctx.n_val, 1), 'wall_time_s': time.time() - t0,
                    'checkpoint': ctx.ckpt_path(lb), 'device': str(ctx.device), 'git_commit': get_git_commit(),
                    'created_at': datetime.datetime.now().isoformat(timespec='seconds')}, dp)
        ctx.log(f"val {lb}: {ctx.n_val} anh, {combo_name(params)}, {(time.time() - t0) / 60:.1f} ph")


# ═════════════════════════════════ Report ═════════════════════════════════════

def boot_counts(n, n_boot=N_BOOT, seed=SEED):
    """CUNG phep resample voi bootstrap_boundary_ci.run_bootstrap (cung rng, cung thu tu goi)."""
    from Tools.bootstrap_boundary_ci import _resample_counts
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    return _resample_counts(idx, n)


def biou_class_values(sums, col_index, d):
    """sums (B, n_cot) hoac (n_cot,) -> dict lop -> BIoU@d (NaN neu union = 0)."""
    out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for c in CLASS_NAMES:
            num = sums[..., col_index[f'biou_inter_{c}_d{d}']]
            den = sums[..., col_index[f'biou_union_{c}_d{d}']]
            out[c] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    return out


def _nanmean_cols(vals, classes):
    import warnings
    st = np.stack([np.asarray(vals[c], dtype=np.float64) for c in classes], axis=-1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(st, axis=-1)


def extra_quantities(sums, col_index):
    """BIoU 9 lop (= BoundaryMetrics / cot biou_d* cua CSV train), BIoU@2 tung lop va theo nhom (§3.8.1)."""
    out = {}
    for d in BD:
        v = biou_class_values(sums, col_index, d)
        out[f'biou_9_d{d}'] = _nanmean_cols(v, CLASS_NAMES)
        if d == 2:
            for c in CLASS_NAMES:
                out[f'biou_d2_{c}'] = v[c]
            out['biou_d2_grp_stuff'] = _nanmean_cols(v, STUFF)
            out['biou_d2_grp_thing'] = _nanmean_cols(v, THING)
    return out


def ci_row(delta_boot, obs):
    dv = delta_boot[~np.isnan(delta_boot)]
    if len(dv) == 0:
        return dict(delta=obs, ci_lo=np.nan, ci_hi=np.nan, p_delta_lt_0=np.nan, ci_excludes_0=False)
    lo, hi = np.percentile(dv, [2.5, 97.5])
    return dict(delta=obs, ci_lo=float(lo), ci_hi=float(hi), p_delta_lt_0=float(np.mean(dv < 0)),
                ci_excludes_0=bool(lo > 0 or hi < 0))


def bootstrap_label(vdir, lb, n_boot):
    from Tools.bootstrap_boundary_ci import aggregate_point_estimate, build_raw_matrix, run_bootstrap, summarize_pair
    b = pd.read_csv(os.path.join(vdir, f'{lb}__before.csv'))
    a = pd.read_csv(os.path.join(vdir, f'{lb}__after.csv'))
    assert a['image'].tolist() == b['image'].tolist(), f"{lb}: lech thu tu anh"
    obs_b = aggregate_point_estimate(b, 'micro', 'skip', boundary_distances=BD)
    obs_a = aggregate_point_estimate(a, 'micro', 'skip', boundary_distances=BD)
    va, vb = run_bootstrap(a, b, 'micro', 'skip', n_boot, SEED, boundary_distances=BD)
    main = {r['quantity']: r for r in summarize_pair(va, vb, obs_a, obs_b,
                                                     quantities=[q for q in QUANTITIES if not q.startswith('biou_9')])}
    # BIoU 9 lop + tung lop / nhom — cung resample (boot_counts == idx cua run_bootstrap)
    mat_a, ci = build_raw_matrix(a, BD)
    mat_b, _ = build_raw_matrix(b, BD)
    cnt = boot_counts(len(a), n_boot, SEED)
    ex_a, ex_b = extra_quantities(cnt @ mat_a, ci), extra_quantities(cnt @ mat_b, ci)
    pt_a, pt_b = extra_quantities(mat_a.sum(0), ci), extra_quantities(mat_b.sum(0), ci)
    rows, extra = [], []
    for q in QUANTITIES:
        if q in main:
            r = main[q]
            rows.append(dict(label=lb, quantity=q, before=obs_b[q], after=obs_a[q],
                             **ci_row(va[q] - vb[q], r['observed_delta'])))
        else:
            rows.append(dict(label=lb, quantity=q, before=float(pt_b[q]), after=float(pt_a[q]),
                             **ci_row(ex_a[q] - ex_b[q], float(pt_a[q] - pt_b[q]))))
    for q in [k for k in pt_a if k.startswith('biou_d2_')]:
        extra.append(dict(label=lb, quantity=q, before=float(pt_b[q]), after=float(pt_a[q]),
                          **ci_row(ex_a[q] - ex_b[q], float(pt_a[q] - pt_b[q]))))
    # tu kiem: biou_8_d2 tu tong cua chinh minh == aggregate_point_estimate
    self_chk = abs(float(_nanmean_cols(biou_class_values(mat_a.sum(0), ci, 2), CLASS_NAMES[1:])) - obs_a['biou_8_d2'])
    return rows, extra, self_chk


def seed_conditions(df, lb):
    def g(q, col='delta'):
        s = df[(df['label'] == lb) & (df['quantity'] == q)]
        return float(s[col].iloc[0])
    p2g, p2g_hi = g('asd_pred_to_gt'), g('asd_pred_to_gt', 'ci_hi')
    bi, bi_lo = g(RULE_BIOU_Q), g(RULE_BIOU_Q, 'ci_lo')
    bi9, bi9_lo = g('biou_9_d2'), g('biou_9_d2', 'ci_lo')
    g2p = g('asd_gt_to_pred')
    rec_lo = g('bf_recall', 'ci_lo')
    c = {'i': bool(p2g < 0 and p2g_hi < 0), 'ii': bool(bi > 0 and bi_lo > 0), 'iii': bool(g2p <= RULE_G2P_MAX),
         'iv': bool(rec_lo > RULE_RECALL_LO), 'ii_biou9': bool(bi9 > 0 and bi9_lo > 0)}
    return {'d_asd_pred_to_gt': p2g, 'ci': [g('asd_pred_to_gt', 'ci_lo'), p2g_hi], 'd_biou_d2': bi,
            'biou_ci': [bi_lo, g(RULE_BIOU_Q, 'ci_hi')], 'd_biou9_d2': bi9, 'd_asd_gt_to_pred': g2p,
            'd_bf_recall': g('bf_recall'), 'bf_recall_ci_lo': rec_lo, 'cond': c,
            'seed_code': ('R-co' if all(c[k] for k in ('i', 'ii', 'iii', 'iv')) else
                          'R-danh-doi' if c['i'] and c['ii'] else 'R-khong')}


def rule_36(df, labels, ii_key='ii'):
    """§3.6, CA HAI seed Static. Doc: R-khong = o MOI seed, khong dat (i) hoac khong dat (ii)."""
    per = {lb: seed_conditions(df, lb) for lb in DECISION_LABELS if lb in labels}
    if len(per) < len(DECISION_LABELS):
        return {'code': None, 'per_seed': per, 'complete': False}
    i_ii = {lb: v['cond']['i'] and v['cond'][ii_key] for lb, v in per.items()}
    iii_iv = {lb: v['cond']['iii'] and v['cond']['iv'] for lb, v in per.items()}
    if all(i_ii.values()) and all(iii_iv.values()):
        code = 'R-co'
    elif not any(i_ii.values()):
        code = 'R-khong'
    elif all(i_ii.values()):
        code = 'R-danh-doi'
    else:
        code = 'KHONG-PHAN-LOAI'
    return {'code': code, 'per_seed': per, 'complete': True, 'biou_key': RULE_BIOU_Q if ii_key == 'ii' else 'biou_9_d2'}


CODE_TEXT = {'R-co': 'R-có', 'R-khong': 'R-không', 'R-danh-doi': 'R-đánh-đổi',
             'KHONG-PHAN-LOAI': 'KHÔNG PHÂN LOẠI ĐƯỢC (một seed đạt (i)–(ii), seed kia không — §3.6 không phủ)',
             None: 'chưa đủ hai seed'}


def _f(x, nd=4, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '—'
    return f'{x:+.{nd}f}' if sign else f'{x:.{nd}f}'


def changed_summary(vdir, labels):
    rows = []
    for lb in labels:
        c = pd.read_csv(os.path.join(vdir, f'{lb}__changed.csv'))
        s = c.drop(columns=['image']).sum()
        r = {'label': lb, 'n_valid_px': int(s['n_valid_px']), 'n_changed': int(s['n_changed']),
             'frac_changed': s['n_changed'] / s['n_valid_px']}
        for tag, _lo, _hi in BANDS:
            r[f'share_changed_in_{tag}'] = s[f'n_changed_{tag}'] / s['n_changed'] if s['n_changed'] else np.nan
            r[f'frac_band_{tag}_changed'] = s[f'n_changed_{tag}'] / s[f'n_band_{tag}'] if s[f'n_band_{tag}'] else np.nan
            r[f'n_changed_{tag}'] = int(s[f'n_changed_{tag}'])
            r[f'n_band_{tag}'] = int(s[f'n_band_{tag}'])
        rows.append(r)
    return pd.DataFrame(rows)


def small_summary(vdir, labels):
    rows = []
    for lb in labels:
        s = pd.read_csv(os.path.join(vdir, f'{lb}__small.csv'))
        for cls, g in list(s.groupby('class', sort=False)) + [('ALL', s)]:
            n, k = int(g['n_small'].sum()), int(g['n_deleted'].sum())
            rows.append({'label': lb, 'class': cls, 'n_small_le700': n, 'n_deleted': k,
                         'frac_deleted': k / n if n else np.nan, 'px_small': int(g['px_small'].sum()),
                         'px_deleted': int(g['px_deleted'].sum())})
    return pd.DataFrame(rows)


def b2_position(df, lb):
    def g(q):
        return float(df[(df['label'] == lb) & (df['quantity'] == q)]['delta'].iloc[0])
    dp, dr, dg = g('bf_precision'), g('bf_recall'), g('asd_gt_to_pred')
    if dp > 0 and dr < 0:
        where = 'cung huong B2 (precision ↑, recall ↓) — truc danh doi'
    elif dp > 0 and dr >= 0:
        where = 'ca precision va recall khong giam — NGOAI truc danh doi (tot hon B2)'
    elif dp <= 0 and dr < 0:
        where = 'ca precision va recall khong tang — kem hon B2'
    else:
        where = 'nguoc B2 (precision ↓, recall ↑)'
    ratio = dp / -dr if dr < 0 else np.nan
    b2_ratio = B2_REF['bf_precision'] / -B2_REF['bf_recall']
    return {'label': lb, 'd_bf_precision': dp, 'd_bf_recall': dr, 'd_asd_gt_to_pred': dg, 'where': where,
            'precision_gain_per_recall_loss': ratio, 'b2_precision_gain_per_recall_loss': b2_ratio}


def render_figures(ctx):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from src.utils.visualizer import mask_to_rgb
    fdir = os.path.join(ctx.out, 'figures')
    sel = read_json(os.path.join(ctx.out, 'crf_selected.json'))['selected']
    made = []
    idx = {n: i for i, n in enumerate(ctx.val_ids)}
    for lb in DECISION_LABELS:
        if lb not in ctx.labels:
            continue
        for name in figure_ids(ctx):
            pb = os.path.join(fdir, '_masks', lb, name + '_before.png')
            pa = os.path.join(ctx.out, 'masks', lb, name + '.png')
            if not (os.path.exists(pb) and os.path.exists(pa)) or name not in idx:
                continue
            _n, gt = load_gt_cached(ctx.val_ds, idx[name], None)
            before, after = np.array(Image.open(pb)), np.array(Image.open(pa))
            fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
            for a, m, t in zip(ax, (before, after, gt), ('Truoc CRF', f"Sau CRF ({sel[lb]['name']})", 'GT')):
                a.imshow(mask_to_rgb(np.asarray(m).astype(np.int64)))
                a.set_title(t)
                a.axis('off')
            fig.suptitle(f'{lb} — {name} (doi nhan: {int((before != after).sum())} px)')
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            p = os.path.join(fdir, f'{lb}__{name}.png')
            fig.savefig(p, dpi=80)
            plt.close(fig)
            made.append(os.path.relpath(p, ctx.out))
    return made


def stage_report(ctx):
    out, vdir = ctx.out, os.path.join(ctx.out, 'val')
    labels = [lb for lb in ctx.labels if os.path.exists(os.path.join(vdir, f'done_{lb}.json'))]
    if not labels:
        raise SystemExit("Chua co ket qua val nao — chay stage val truoc.")
    dones = {lb: read_json(os.path.join(vdir, f'done_{lb}.json')) for lb in labels}
    n_images = dones[labels[0]]['n_images']
    dryrun = ctx.dryrun or n_images < N_VAL
    rows, extra, self_chk = [], [], {}
    for lb in labels:
        r, e, s = bootstrap_label(vdir, lb, ctx.args.n_boot)
        rows += r
        extra += e
        self_chk[lb] = s
        print(f"  [{lb}] bootstrap xong")
    df = pd.DataFrame(rows)
    sel = read_json(os.path.join(out, 'crf_selected.json'))
    for lb in labels:
        df.loc[df['label'] == lb, 'params'] = sel['selected'][lb]['name']
    df['input_mode'] = ctx.mode
    df.to_csv(os.path.join(out, 'crf_val.csv'), index=False)
    pc = pd.DataFrame(extra)
    pc.to_csv(os.path.join(out, 'per_class.csv'), index=False)
    ch = changed_summary(vdir, labels)
    ch.to_csv(os.path.join(out, 'changed_pixels.csv'), index=False)
    sm = small_summary(vdir, labels)
    sm.to_csv(os.path.join(out, 'small_regions.csv'), index=False)
    rule = rule_36(df, labels)
    rule9 = rule_36(df, labels, 'ii_biou9')
    b2 = [b2_position(df, lb) for lb in labels]
    figs = render_figures(ctx)

    # gates
    gd = os.path.join(out, 'gates')
    gates = [read_json(os.path.join(gd, f'{g}.json')) if os.path.exists(os.path.join(gd, f'{g}.json'))
             else {'gate': g, 'status': 'MISSING', 'detail': 'chua chay'} for g in ('R0-g1', 'R0-g2', 'R0-g3', 'R0-g4')]
    if not dryrun:
        g1 = gates[0]
        for c in g1.get('checks', []):     # nhat quan: mIoU-9 'truoc' cua stage val == g1
            got = float(df[(df['label'] == c['label']) & (df['quantity'] == 'mIoU_9')]['before'].iloc[0])
            c['val_stage_before_miou9'] = got
            c['val_stage_consistent'] = bool(abs(got - c['got']) < 1e-9)
    crf = crf_backend_info()
    overall = 'PASS' if all(g['status'] in ('PASS', 'SKIP_DRYRUN') for g in gates) else 'FAIL'
    eta = None
    gdone = [read_json(os.path.join(out, 'grid', f'done_{lb}.json')) for lb in labels
             if os.path.exists(os.path.join(out, 'grid', f'done_{lb}.json'))]
    if dryrun and gdone:
        t = float(np.mean([d['crf_s_mean'] for d in gdone]))
        w = max(1, ctx.args.workers)
        eta = {'crf_s_per_call': t, 'workers': w,
               'grid_min': len(R0_LABELS) * N_PROBE * len(GRID) * t * 1.15 / w / 60,
               'val_min': len(R0_LABELS) * N_VAL * (t + 3.0) / w / 60,
               'note': 'uoc tinh tu dry-run: +15% thong ke cho luoi, +3 s/anh thong ke BIoU + mo ta cho val'}
    write_json({'overall': overall, 'gates': gates, 'rule_36': rule, 'rule_36_sensitivity_biou9': rule9,
                'rule_note': K2_52_NOTE, 'input_mode': ctx.mode, 'crf_backend': crf, 'fixed': FIXED,
                'labels': labels, 'n_images': n_images, 'n_boot': ctx.args.n_boot, 'dryrun': dryrun,
                'self_check_biou8_d2_abs_diff': self_chk, 'eta_full_run': eta,
                'git_commit': get_git_commit(), 'created_at': datetime.datetime.now().isoformat(timespec='seconds')},
               os.path.join(out, 'gates_r0.json'))
    write_md(ctx, df, pc, ch, sm, rule, rule9, b2, gates, sel, figs, crf, n_images, dryrun)
    ctx.log(f"report: cong {overall}, ma §3.6 = {rule['code']} (mo ta), mode={ctx.mode}, "
            f"pydensecrf={crf.get('version')}@{crf.get('commit')}, git={get_git_commit()}")
    for g in gates:
        print(f"  [{g['status']}] {g['gate']}: {g.get('detail', '')}")
    print(f"  §3.6 (mo ta): {CODE_TEXT[rule['code']]}" + ('   (DRY-RUN — khong dung duoc)' if dryrun else ''))
    if eta:
        print(f"  Uoc tinh chay that: luoi ~{eta['grid_min']:.0f} ph, val ~{eta['val_min']:.0f} ph "
              f"(CRF {eta['crf_s_per_call']:.2f} s/lan, {eta['workers']} tien trinh)")
    print(f"Da ghi crf_val.csv/.md, per_class.csv, changed_pixels.csv, small_regions.csv, gates_r0.json — cong: {overall}")
    return overall


def write_md(ctx, df, pc, ch, sm, rule, rule9, b2, gates, sel, figs, crf, n_images, dryrun):
    L = ['# R0 — P-CRF (DenseCRF hau ky, baseline khong hoc)', '',
         f"> {K2_52_NOTE}", '',
         f"- Che do dau vao CRF: **{ctx.mode}** "
         f"({'softmax fp32 1024², unary = −log(max(p, 1e-8))' if ctx.mode == 'softmax' else 'nhan argmax, gt_prob = 0.7'})",
         f"- pydensecrf: version {crf.get('version')}, commit {crf.get('commit')}, nguon {crf.get('source_url')}",
         f"- Co dinh: pos_xy_std {FIXED['pos_xy_std']}, pos_w {FIXED['pos_w']}, {FIXED['n_iter']} vong lap; "
         f"luoi 27 to hop chon tren 32 anh train `probe_ids.txt` (tieu chi: min ASD hai chieu trung binh, micro)",
         f"- Do: val {n_images} anh, `compute_per_image_stats` + `bootstrap_boundary_ci` (micro), bootstrap ghep cap "
         f"{ctx.args.n_boot} lan, rng {SEED}. Δ = sau CRF − truoc. `*` = CI 95% loai 0.",
         f"- Tao {datetime.datetime.now().isoformat(timespec='seconds')} · git {get_git_commit()[:10]}", '']
    if dryrun:
        L += [f'> **DRY-RUN ({n_images} anh val)** — chi kiem duong chay; so lieu va ma §3.6 KHONG dung duoc.', '']
    L += ['## Cong', '', '| Cong | Trang thai | Chi tiet |', '|---|---|---|']
    for g in gates:
        L.append(f"| {g['gate']} | **{g['status']}** | {g.get('detail', '')} |")
    L += ['', '## To hop chon (tren train)', '', '| Checkpoint | To hop | ASD2 train sau CRF | ASD2 train truoc |',
          '|---|---|---|---|']
    for lb, s in sel['selected'].items():
        L.append(f"| {lb} | `{s['name']}` | {s['asd_mean2_train']:.4f} | {s['asd_mean2_train_before']:.4f} |")
    L += ['', '## §3.6 — Luat doc (hai checkpoint quyet dinh) — MO TA', '',
          f"**Ma: {CODE_TEXT[rule['code']]}**", '',
          f"Dieu kien: (i) Δ ASD pred→gt < 0, CI loai 0; (ii) Δ BIoU@d2 > 0, CI loai 0 (`{RULE_BIOU_Q}`, quy uoc "
          f"8 lop cua `bootstrap_boundary_ci`); (iii) Δ ASD gt→pred ≤ {RULE_G2P_MAX} px; (iv) can duoi CI Δ BF-recall "
          f"> {RULE_RECALL_LO}. Doc R-không = o moi seed, truot (i) hoac (ii).", '',
          '| Seed | Δ p→g [CI] | (i) | Δ BIoU8@2 [CI] | (ii) | Δ g→p | (iii) | Δ BF-R (can duoi CI) | (iv) | Ma seed |',
          '|---|---|---|---|---|---|---|---|---|---|']
    yn = {True: 'dat', False: 'khong'}
    for lb, v in rule['per_seed'].items():
        c = v['cond']
        L.append(f"| {lb} | {_f(v['d_asd_pred_to_gt'])} [{_f(v['ci'][0])}, {_f(v['ci'][1])}] | {yn[c['i']]} | "
                 f"{_f(v['d_biou_d2'], 5)} [{_f(v['biou_ci'][0], 5)}, {_f(v['biou_ci'][1], 5)}] | {yn[c['ii']]} | "
                 f"{_f(v['d_asd_gt_to_pred'])} | {yn[c['iii']]} | {_f(v['d_bf_recall'])} ({_f(v['bf_recall_ci_lo'])}) | "
                 f"{yn[c['iv']]} | {CODE_TEXT[v['seed_code']]} |")
    L += ['', f"Do nhay (khong dung quyet dinh): (ii) voi BIoU 9 lop (`biou_9_d2`, dinh nghia cot `biou_d2` cua CSV "
              f"train) -> ma {CODE_TEXT[rule9['code']]}.", '',
          '## Moi chi so §1 — ca 5 checkpoint', '']
    qs = ['mIoU_9', 'mIoU_8', 'bf_precision', 'bf_recall', 'bf_score', 'biou_8_d1', 'biou_8_d2', 'biou_8_d4',
          'asd_pred_to_gt', 'asd_gt_to_pred']
    L += ['| Checkpoint | ' + ' | '.join(QSHORT[q] for q in qs) + ' |', '|---|' + '---|' * len(qs)]
    for lb in df['label'].unique():
        cells = []
        for q in qs:
            s = df[(df['label'] == lb) & (df['quantity'] == q)].iloc[0]
            cells.append(f"{s['before']:.4f}→{s['after']:.4f}<br>{_f(s['delta'])}{'*' if s['ci_excludes_0'] else ''} "
                         f"[{_f(s['ci_lo'])}, {_f(s['ci_hi'])}]")
        L.append(f"| {lb} | " + ' | '.join(cells) + ' |')
    L += ['', 'BIoU 9 lop (`biou_9_d*`) va ASD tong: `crf_val.csv`.', '',
          '## §3.8.1 — Δ BIoU@2 theo lop va nhom', '',
          f"Nhom = trung binh BIoU@2 cac lop trong nhom: stuff = {', '.join(STUFF)}; thing = {', '.join(THING)}; "
          f"{', '.join(SEPARATE)} bao cao rieng.", '']
    show = [f'biou_d2_{c}' for c in CLASS_NAMES] + ['biou_d2_grp_stuff', 'biou_d2_grp_thing']
    names = {f'biou_d2_{c}': c for c in CLASS_NAMES}
    names.update({'biou_d2_grp_stuff': '**stuff**', 'biou_d2_grp_thing': '**thing**'})
    labs = list(pc['label'].unique())
    L += ['| Lop / nhom | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    for q in show:
        cells = []
        for lb in labs:
            s = pc[(pc['label'] == lb) & (pc['quantity'] == q)].iloc[0]
            cells.append(f"{_f(s['delta'], 5)}{'*' if s['ci_excludes_0'] else ''}")
        L.append(f"| {names[q]} | " + ' | '.join(cells) + ' |')
    L += ['', '## §3.8.2 — Pixel doi nhan sau CRF', '',
          '| Checkpoint | Ti le pixel doi | trong ≤2 px bien GT | 2–8 px | >8 px |', '|---|---|---|---|---|']
    for _, r in ch.iterrows():
        L.append(f"| {r['label']} | {r['frac_changed']:.4%} | {r['share_changed_in_le2']:.1%} | "
                 f"{r['share_changed_in_2to8']:.1%} | {r['share_changed_in_gt8']:.1%} |")
    L += ['', 'Ti le trong dai = phan cua pixel doi nhan nam trong dai (cong = 100%). Khoang cach toi bien GT '
              '4-lien thong (`gt_edge_np`).', '',
          '## §3.8.3 — Thanh phan du doan ≤ 700 px bi xoa sau CRF', '',
          '| Checkpoint | So thanh phan ≤ 700 px | Bi xoa | Ti le |', '|---|---|---|---|']
    for _, r in sm[sm['class'] == 'ALL'].iterrows():
        L.append(f"| {r['label']} | {r['n_small_le700']} | {r['n_deleted']} | {_f(r['frac_deleted'], 4, False)} |")
    L += ['', 'Thanh phan 8-lien thong theo lop cua du doan truoc CRF; "bi xoa" = khong con pixel nao cua lop do '
              'trong thanh phan sau CRF. Theo lop: `small_regions.csv`.', '',
          '## §3.8.4 — Vi tri tren truc danh doi precision ↔ recall (so B2)', '',
          f"B2 (lap mu lo ≤ 64 px, Static s19@36k): ΔBF-P {B2_REF['bf_precision']:+.4f}, ΔBF-R "
          f"{B2_REF['bf_recall']:+.4f}, Δ g→p {B2_REF['asd_gt_to_pred']:+.3f} px "
          f"(ΔP/−ΔR = {B2_REF['bf_precision'] / -B2_REF['bf_recall']:.2f}).", '',
          '| Checkpoint | ΔBF-P | ΔBF-R | Δ g→p | ΔP/−ΔR | Vi tri |', '|---|---|---|---|---|---|']
    for v in b2:
        L.append(f"| {v['label']} | {_f(v['d_bf_precision'])} | {_f(v['d_bf_recall'])} | {_f(v['d_asd_gt_to_pred'], 3)} | "
                 f"{_f(v['precision_gain_per_recall_loss'], 2, False)} | {v['where']} |")
    L += ['', '## §3.8.5 — Luoi tren tap chon (ASD hai chieu trung binh, train)', '',
          'Day du: `crf_grid_train.csv` (27 to hop × 5 checkpoint + hang `combo = -1` = truoc CRF).', '']
    g = pd.read_csv(os.path.join(ctx.out, 'crf_grid_train.csv'))
    labs = list(g['label'].unique())
    L += ['| To hop | ' + ' | '.join(labs) + ' |', '|---|' + '---|' * len(labs)]
    for k in [-1] + list(range(len(GRID))):
        cells = []
        for lb in labs:
            s = g[(g['label'] == lb) & (g['combo'] == k)].iloc[0]
            cells.append(f"{'**' if s['selected'] else ''}{s['asd_mean2']:.4f}{'**' if s['selected'] else ''}")
        L.append(f"| {'truoc CRF' if k < 0 else combo_name(GRID[k])} | " + ' | '.join(cells) + ' |')
    L += ['', '## Hinh', '']
    L += [f'- `{p}`' for p in figs] or ['- (khong co — thieu figure_images.json hoac anh khong nam trong tap val da do)']
    with open(os.path.join(ctx.out, 'crf_val.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


# ═════════════════════════════════ CLI ════════════════════════════════════════

STAGES = ('g1', 'g23', 'grid', 'select', 'val', 'report')


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=('run', 'report', 'crf-info'))
    ap.add_argument('--stages', default=' '.join(STAGES))
    ap.add_argument('--ckpt-dir', default=None, help='dataset checkpoint (best_model_Static_S19.pth, ...)')
    ap.add_argument('--snapshot-dir', default=None, help='snapshots/ cua Phan D (iter_36000.pth)')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--probe-ids', default=None, help='probe_ids.txt cua Phan D (32 anh train)')
    ap.add_argument('--figure-images', default=None, help='figure_images.json (Phan D / Buoc 1)')
    ap.add_argument('--output-dir', default='output/refine/r0_crf')
    ap.add_argument('--labels', default=','.join(R0_LABELS))
    ap.add_argument('--input-mode', default='softmax', choices=('softmax', 'gt_prob'))
    ap.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument('--device', default=None)
    ap.add_argument('--n-boot', type=int, default=N_BOOT)
    ap.add_argument('--max-val', type=int, default=None, help='dry-run: N anh val dau')
    ap.add_argument('--max-probe', type=int, default=None, help='dry-run: N anh probe dau')
    ap.add_argument('--force', action='store_true')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.cmd == 'crf-info':
        print(crf_backend_info())
        return
    os.makedirs(args.output_dir, exist_ok=True)
    crf = crf_backend_info()                      # thieu pydensecrf -> dung ngay (§3.3)
    ctx = Ctx(args)
    stages = STAGES if args.cmd == 'run' and not args.stages else tuple(args.stages.split())
    if args.cmd == 'report':
        stages = ('report',)
    ctx.log(f"r0 {args.cmd} stages={stages} labels={ctx.labels} mode={ctx.mode} max_val={args.max_val} "
            f"max_probe={args.max_probe} workers={args.workers} device={ctx.device} pydensecrf={crf} "
            f"git={get_git_commit()} argv={' '.join(sys.argv)}")
    if args.probe_ids is None and any(s in stages for s in ('g23', 'grid', 'select')):
        raise SystemExit("can --probe-ids (probe_ids.txt cua Phan D)")
    pool = make_pool(args.workers) if any(s in stages for s in ('grid', 'val')) else None
    try:
        for s in stages:
            print(f"\n----- R0 stage {s} -----")
            if s == 'g1':
                stage_g1(ctx)
            elif s == 'g23':
                stage_g23(ctx)
            elif s == 'grid':
                stage_grid(ctx, pool)
            elif s == 'select':
                stage_select(ctx)
            elif s == 'val':
                stage_val(ctx, pool)
            elif s == 'report':
                if stage_report(ctx) != 'PASS':
                    sys.exit(1)
            else:
                raise SystemExit(f"stage khong biet: {s}")
    finally:
        if pool is not None:
            pool.close()
            pool.join()


if __name__ == '__main__':
    main()
