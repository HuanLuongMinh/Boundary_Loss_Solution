"""Tools/grad_conflict_probe.py — Phan A cua
docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md: do xung dot gradient giua
L_region, L_BCE, L_aff tren Fused Feature F (64 kenh, stride 4) va tren tham so
dung chung, tai cac checkpoint DA CO. KHONG train, KHONG sua src/losses/.

Subcommand (chay 1 lan het, hoac tach tung checkpoint / tung phien Kaggle):

  gates      Cong 1,2,3,4,5,7 (muc 4) cho tung checkpoint -> gates_<ckpt>.json.
  measure    Do tren 384 anh val -> per_image_strata_<ckpt>.csv,
             per_batch_param_<ckpt>.csv, null_param_<ckpt>.csv,
             null_feat_<ckpt>.csv, per_component_small_<ckpt>.csv (Static),
             figures/ (Static s19@36k), done_<ckpt>.json.
             Tu choi chay neu gates_<ckpt>.json chua PASS (tru --skip-gate-check).
  summarize  Gop MOI checkpoint da do trong --output-dir -> summary_<ckpt>.csv,
             summary_grad_conflict.md (tieu chi 5.1, nhanh 5.2, mo ta 5.3).

Checkpoint truyen luc chay: --ckpt LABEL=duong_dan.pth (lap lai). LABEL phai
thuoc bang CHECKPOINTS (Tools/grad_conflict_common.py) — bang do co dinh
config/model-type/iter/seed/trong so/cap hop le cua tung run.

Quy uoc tinh (muc 2.1): fp32 (khong autocast), model.eval() (BN running stats,
dropout/drop-path tat), batch = 1 anh / lan backward, torch.manual_seed(19+idx)
truoc moi anh. F lay bang forward hook tren `frh` (FeatureRefinementHead).
shared_params = moi tham so TRUOC F (bo classifier.* va boundary_head.*).
L_region tinh tren tung anh (Dice gop trong 1 anh) — khac Dice gop batch 4 luc
train; cosine/ti le do lon theo batch o muc 2.2 la TONG gradient theo anh.

Xem docs/huong-dan-chay-grad-conflict-va-lo-gia.md cho lenh day du.
"""

import argparse
import datetime
import math
import os
import sys
import time

# Cong 5 (tat dinh): phai dat TRUOC khi CUDA khoi tao.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (
    CHECKPOINTS, MISSING_RUNS, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, SEED, AXES, ALL_PAIRS,
    DECISION_CKPTS, FIGURE_CKPT, DIST_STRATA, SIZE_SMALL_MAX,
    pair_name, terms_of, term_weights, gt_strata_maps, parse_label_values, check_labels_known,
    get_git_commit, file_fingerprint, write_json, read_json, append_log, load_gt_cached,
)

TAU_PARAM = 0.05
TAU_FEAT = 0.10
MAG_LO, MAG_HI = 0.1, 10.0
SUPPORT_EPS = 1e-12
SUPPORT_MIN_COV = 0.01
GATE_MIOU_TOL = 1e-3
GATE_LOSS_TOL = 1e-5
GATE_LIN_TOL = 1e-5
GATE_DET_TOL = 1e-6
N_GATE_LOSS_IMAGES = 8
N_GATE_LIN_IMAGES = 4
N_GATE_DET_IMAGES = 4
STATIC_LABELS = [lb for lb, v in CHECKPOINTS.items() if v['model_type'] == 'static_boundary']


# ═══════════════════════════════ Thiet lap ════════════════════════════════════

def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class Probe:
    """Model + loss + hook cho 1 checkpoint."""

    def __init__(self, label, ckpt_path, cfg, device, pos_weight, dtype=torch.float32):
        from Tools.eval_boundary_metrics import build_eval_model
        from src.losses.total_loss import StaticBoundaryTotalLoss

        self.label = label
        self.info = CHECKPOINTS[label]
        self.cfg = cfg
        self.device = device
        self.model = build_eval_model(cfg, self.info['model_type']).to(device)
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and 'model_state_dict' in sd and not any(k.startswith('base.') for k in sd):
            sd = sd['model_state_dict']
        self.model.load_state_dict(sd)   # strict
        self.state_keys = list(sd.keys())
        self.dtype = dtype
        self.model.to(dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.base = self.model.base if hasattr(self.model, 'base') else self.model

        bl = cfg.get('BOUNDARY_LOSS', {}) or {}
        lc = cfg.get('LOSS', {}) or {}
        self.loss_kw = dict(
            num_classes=cfg['TRAIN']['NUM_CLASSES'], ignore_index=IGNORE_INDEX,
            ce_weight=lc.get('CE_WEIGHT', 1.0), dice_weight=lc.get('DICE_WEIGHT', 1.0),
            connectivity=bl.get('CONNECTIVITY', 4), dilation_radius=bl.get('DILATION_RADIUS', 0),
            affinity_window_size=bl.get('AFFINITY_WINDOW_K', 5),
            affinity_distance=bl.get('AFFINITY_DISTANCE', 'cosine'),
            affinity_margin=bl.get('AFFINITY_MARGIN', 1.0))
        # Chi dung cac submodule (region_loss / bce_edge / affinity) de lay so hang RIENG o dang tensor.
        self.loss_mod = StaticBoundaryTotalLoss(alpha=0.4, **self.loss_kw).to(device)
        self.terms = terms_of(label)
        self.weights = term_weights(label)
        self.pairs = [tuple(p) for p in self.info['pairs']]
        self.pos_weight = pos_weight
        if self.info['has_bce'] and pos_weight is None:
            raise ValueError(f"[{label}] can --pos-weight {label}=<gia tri trong log | auto>")

        self.shared = [(n, p) for n, p in self.model.named_parameters()
                       if not (n.startswith('classifier.') or n.startswith('base.classifier.')
                               or n.startswith('boundary_head.'))]
        self.shared_params = [p for _n, p in self.shared]
        self.n_shared = sum(p.numel() for p in self.shared_params)
        self._F = None
        self.base.frh.register_forward_hook(self._hook)

    def _hook(self, _m, _i, out):
        self._F = out

    def forward(self, x):
        self._F = None
        out = self.model(x.to(self.dtype))
        if self.info['model_type'] == 'static_boundary':
            logits, edge, _f = out
        elif self.info['model_type'] == 'bce_edge':
            logits, edge = out
        else:
            logits, edge = out, None
        assert self._F is not None, 'forward hook tren frh khong bat duoc F'
        return logits, edge, self._F

    def losses(self, logits, edge, F, m):
        L = {'region': self.loss_mod.region_loss(logits, m),
             'aff': self.loss_mod.affinity(F, m)}
        if 'bce' in self.terms:
            L['bce'] = self.loss_mod.bce_edge(edge, m, self.pos_weight)
        return L

    def image_grads(self, x, m, image_idx, need_theta=True):
        """Tra ve (L dict float, gF dict (64,h,w), gT dict flat (P,) hoac None)."""
        torch.manual_seed(SEED + image_idx)
        logits, edge, F = self.forward(x)
        L = self.losses(logits, edge, F, m)
        inputs = [F] + (self.shared_params if need_theta else [])
        gF, gT = {}, {}
        for k in self.terms:
            g = torch.autograd.grad(self.weights[k] * L[k], inputs, retain_graph=True, allow_unused=True)
            gF[k] = (g[0] if g[0] is not None else torch.zeros_like(F))[0].detach()
            if need_theta:
                gT[k] = flatten_grads(g[1:], self.shared_params)
        return {k: float(v.detach()) for k, v in L.items()}, gF, (gT if need_theta else None)


def flatten_grads(grads, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for g, p in zip(grads, params)]).detach()


def resolve_pos_weights(labels, raw, cfg_by_label, output_dir):
    out, src = {}, {}
    for lb in labels:
        if not CHECKPOINTS[lb]['has_bce']:
            continue
        v = raw.get(lb)
        if v is None:
            raise SystemExit(f"Thieu --pos-weight {lb}=<gia tri trong log train | auto> (checkpoint co BCE).")
        if v.lower() == 'auto':
            out[lb], src[lb] = estimate_pos_weight(cfg_by_label[lb], output_dir)
        else:
            out[lb], src[lb] = float(v), 'log train (nguoi dung cung cap)'
    return out, src


def estimate_pos_weight(cfg, output_dir):
    """Uoc luong lai pos_weight nhu luc train (EdgeStatsAccumulator, 100 batch/rank x 2 rank x 4 crop,
    get_train_transforms, seed 19). XAP XI — khong trung tuyet doi gia tri luc train."""
    cache = os.path.join(output_dir, 'pos_weight_auto.json')
    if os.path.exists(cache):
        d = read_json(cache)
        return d['pos_weight'], d['source']
    import random
    from torch.utils.data import DataLoader
    from src.data.dataset import OpenEarthMapDataset
    from src.data.transforms import get_train_transforms
    from src.losses.boundary_bce import EdgeStatsAccumulator
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    ds = cfg['DATASET']
    train_ds = OpenEarthMapDataset(root_dir=ds['ROOT_DIR'], img_dir=ds['TRAIN_IMG_DIR'],
                                   mask_dir=ds['TRAIN_MASK_DIR'], transform=get_train_transforms())
    g = torch.Generator().manual_seed(SEED)
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, generator=g, num_workers=0, drop_last=True)
    bl = cfg.get('BOUNDARY_LOSS', {}) or {}
    acc = EdgeStatsAccumulator(IGNORE_INDEX, bl.get('CONNECTIVITY', 4), bl.get('DILATION_RADIUS', 0))
    for i, (_x, m) in enumerate(loader):
        if i >= 200:
            break
        acc.update(m)
    res = acc.result(bl.get('POS_WEIGHT_MAX', 20.0))
    source = 'auto: uoc luong lai 200 batch x 4 crop (train transforms, seed 19) — XAP XI'
    write_json({'pos_weight': res['pos_weight'], 'source': source, 'stats': res}, cache)
    return res['pos_weight'], source


def load_cfg_for(label, config_override, data_root):
    from Tools.oracle_boundary_ceiling import load_cfg
    return load_cfg(config_override or CHECKPOINTS[label]['config'], data_root)


def dump_paths(dump_dir):
    """Chap nhan <dump>/masks/*.png + <dump>/per_image_stats.csv, hoac thu muc chua PNG truc tiep."""
    if dump_dir is None:
        return None, None
    masks = os.path.join(dump_dir, 'masks')
    if not os.path.isdir(masks):
        masks = dump_dir
    csv = os.path.join(dump_dir, 'per_image_stats.csv')
    return masks, (csv if os.path.exists(csv) else None)


def miou_from_confusion(cm):
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    den = tp + fp + fn
    iou = tp / (den + 1e-10)
    valid = den > 0
    miou9 = float(iou[valid].mean())
    miou8 = float(iou[1:][valid[1:]].mean())
    return miou9, miou8, iou.tolist()


def miou_from_dump_csv(csv_path, image_subset=None):
    df = pd.read_csv(csv_path)
    if image_subset is not None:
        df = df.set_index('image').loc[image_subset].reset_index()
    ious, valid = [], []
    for c in CLASS_NAMES:
        u = float(df[f'union_{c}'].sum())
        ious.append(float(df[f'inter_{c}'].sum()) / u if u > 0 else 0.0)
        valid.append(u > 0)
    ious, valid = np.array(ious), np.array(valid)
    return float(ious[valid].mean()), float(ious[1:][valid[1:]].mean())


# ═══════════════════════ Thong ke muc dac trung (2.3) ════════════════════════

class ImageStrata:
    """Tang cua 1 anh tai stride 4 (tensor tren device) + hoan vi null (20 lan, rng 19)."""

    def __init__(self, gt, out_hw, device, image_idx, n_perm, want_components):
        maps = gt_strata_maps(gt, out_hw)
        self.idx = {ax: torch.from_numpy(maps[ax].reshape(-1)).to(device) for ax in AXES}
        self.comp = torch.from_numpy(maps['comp'].reshape(-1)).to(device) if want_components else None
        self.comp_table = maps['comp_table']
        self.perms = {}
        gen = torch.Generator().manual_seed(SEED * 1_000_003 + image_idx)
        for ax in AXES:
            idx = self.idx[ax]
            vpos = torch.nonzero(idx >= 0).squeeze(1)
            keys = idx[vpos].double()
            order = vpos[torch.argsort(keys, stable=True)]
            perms = []
            for _r in range(n_perm):
                rnd = torch.rand(len(vpos), generator=gen, dtype=torch.float64).to(device)
                perm = vpos[torch.argsort(keys + rnd * 0.5)]
                perms.append(perm)
            self.perms[ax] = (order, perms)


def feature_pair_stats(g1, g2, strata, n_perm):
    """g1, g2 (64,h,w). Tra ve dict:
      'strata': {ax: tensor (S,6) [sum_dot, sum_n1, sum_n2, n_pos, n_support, n_neg]} (float64)
      'null':   {ax: tensor (n_perm, S) sum_dot_perm}
      'maps':   (dot, n1, n2) flat (de ve hinh / thanh phan)
    """
    a = g1.reshape(g1.shape[0], -1).double()
    b = g2.reshape(g2.shape[0], -1).double()
    dot = (a * b).sum(0)
    n1 = (a * a).sum(0)
    n2 = (b * b).sum(0)
    sup = (n1 > SUPPORT_EPS) & (n2 > SUPPORT_EPS)
    neg = sup & (dot < 0)
    ones = torch.ones_like(dot)
    feats = torch.stack([dot, n1, n2, ones, sup.double(), neg.double()], 1)   # (N,6)
    out = {'strata': {}, 'null': {}, 'maps': (dot, n1, n2)}
    for ax, names in AXES.items():
        idx = strata.idx[ax]
        valid = idx >= 0
        S = len(names)
        acc = torch.zeros(S, 6, dtype=torch.float64, device=dot.device)
        acc.index_add_(0, idx[valid], feats[valid])
        out['strata'][ax] = acc
        order, perms = strata.perms[ax]
        sidx = idx[order]
        nulls = torch.zeros(n_perm, S, dtype=torch.float64, device=dot.device)
        a_o = a[:, order]
        for r, perm in enumerate(perms[:n_perm]):
            dp = (a_o * b[:, perm]).sum(0)
            nulls[r].index_add_(0, sidx, dp)
        out['null'][ax] = nulls
    return out


def component_stats(maps, strata):
    """Thong ke theo thanh phan GT NHO (area full-res < 4096). Tra ve list row dict (khong co pair)."""
    dot, n1, n2 = maps
    comp = strata.comp
    if comp is None or not strata.comp_table:
        return []
    M = max(strata.comp_table) + 1
    sup = ((n1 > SUPPORT_EPS) & (n2 > SUPPORT_EPS)).double()
    feats = torch.stack([dot, n1, n2, torch.ones_like(dot), sup], 1)
    acc = torch.zeros(M, 5, dtype=torch.float64, device=dot.device)
    valid = comp > 0
    acc.index_add_(0, comp[valid], feats[valid])
    acc = acc.cpu().numpy()
    rows = []
    for cid, (c, area) in strata.comp_table.items():
        if area >= SIZE_SMALL_MAX:
            continue
        rows.append({'comp_id': cid, 'class': CLASS_NAMES[c], 'area': area,
                     'sum_dot': acc[cid, 0], 'sum_n1': acc[cid, 1], 'sum_n2': acc[cid, 2],
                     'n_pos': int(acc[cid, 3]), 'n_support': int(acc[cid, 4])})
    return rows


# ════════════════════════════════ gates ═══════════════════════════════════════

def gate(gid, name, status, value=None, threshold=None, note=''):
    return {'gate': gid, 'name': name, 'status': status, 'value': value, 'threshold': threshold, 'note': note}


def run_gates_for(label, args, ckpt_path, cfg, device, pos_weight, dump_dir, expected_override):
    from torch.utils.data import DataLoader
    from Tools.oracle_boundary_ceiling import build_val_dataset, image_ids_of
    from src.utils.losses import CombinedLoss
    from src.losses.total_loss import StaticBoundaryTotalLoss
    from src.losses.total_loss_affinity_only import AffinityOnlyTotalLoss
    from src.losses.boundary_bce import BoundaryHead
    from PIL import Image

    info = CHECKPOINTS[label]
    probe = Probe(label, ckpt_path, cfg, device, pos_weight)
    val_ds = build_val_dataset(cfg)
    image_ids = image_ids_of(val_ds)
    n = len(image_ids) if args.max_images is None else min(args.max_images, len(image_ids))
    dry = args.max_images is not None
    masks_dir, dump_csv = dump_paths(dump_dir)
    gates = []

    # ── Cong 7 (danh sach anh) + Cong 1 (mIoU) — chung 1 luot forward ──
    list_ok, list_note = None, ''
    if masks_dir is None:
        list_ok, list_note = False, 'khong truyen --dump-dir — khong kiem duoc danh sach anh voi dump'
    else:
        pngs = sorted(os.path.splitext(f)[0] for f in os.listdir(masks_dir) if f.lower().endswith('.png'))
        list_ok = pngs == sorted(image_ids)
        list_note = f'{len(pngs)} PNG trong {masks_dir}; val {len(image_ids)} anh'
        if dump_csv:
            csv_imgs = pd.read_csv(dump_csv, usecols=['image'])['image'].astype(str).tolist()
            same_order = csv_imgs == image_ids
            list_ok = list_ok and same_order
            list_note += f"; per_image_stats.csv {'khop' if same_order else 'KHAC'} danh sach+thu tu"
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    agree, total_px = 0, 0
    loader = DataLoader(torch.utils.data.Subset(val_ds, list(range(n))), batch_size=1, shuffle=False,
                        num_workers=args.num_workers)
    t0 = time.time()
    with torch.no_grad():
        for i, (x, m) in enumerate(loader):
            logits, _e, _F = probe.forward(x.to(device))
            pred = logits.argmax(1)[0].cpu().numpy()
            gt = m[0].numpy()
            v = gt != IGNORE_INDEX
            cm += np.bincount(NUM_CLASSES * gt[v].astype(np.int64) + pred[v],
                              minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
            if masks_dir is not None and list_ok is not False:
                pth = os.path.join(masks_dir, image_ids[i] + '.png')
                if os.path.exists(pth):
                    dm = np.array(Image.open(pth))
                    agree += int((dm == pred).sum())
                    total_px += pred.size
            print(f"  [{label}] cong 1: {i + 1}/{n} ({time.time() - t0:.0f}s)", end='\r')
    print()
    miou9, miou8, per_class = miou_from_confusion(cm)
    expected, exp_src = info['expected_miou9'], 'hang so spec (muc 4 cong 1)'
    if expected_override is not None:
        expected, exp_src = expected_override, '--expected-miou'
    if dry:
        if dump_csv:
            expected, _e8 = miou_from_dump_csv(dump_csv, image_ids[:n])
            exp_src = f'dump per_image_stats.csv tren {n} anh dau (dry-run)'
        else:
            expected, exp_src = None, 'dry-run, khong co dump csv'
    elif expected is None and dump_csv:
        expected, _e8 = miou_from_dump_csv(dump_csv)
        exp_src = f'dump per_image_stats.csv ({dump_csv}) — spec khong co hang so'
    if expected is None:
        st = 'SKIP_DRYRUN' if dry else 'FAIL'
        gates.append(gate(1, 'nap dung checkpoint (mIoU-9)', st, {'miou9': miou9, 'miou8': miou8}, GATE_MIOU_TOL,
                          'khong co tham chieu — truyen --dump-dir (co per_image_stats.csv) hoac --expected-miou'))
    else:
        diff = abs(miou9 - expected)
        gates.append(gate(1, 'nap dung checkpoint (mIoU-9)', 'PASS' if diff < GATE_MIOU_TOL else 'FAIL',
                          {'miou9': miou9, 'miou8_bo_background': miou8, 'expected_miou9': expected,
                           'abs_diff': diff, 'expected_source': exp_src,
                           'per_class_iou': dict(zip(CLASS_NAMES, per_class))},
                          GATE_MIOU_TOL, 'neu expected trung mIoU-8 -> nham quy uoc 8/9 lop'))
    print(f"  [{label}] mIoU-9 = {miou9:.7f}   mIoU-8 (bo Background) = {miou8:.7f}   "
          f"expected mIoU-9 = {expected} ({exp_src})")

    pix = (agree / total_px) if total_px else None
    gates.append(gate(7, 'danh sach anh khop dump', 'PASS' if list_ok else 'FAIL',
                      {'pixel_agreement_with_dumped_masks': pix}, 'khop tuyet doi (noi dung + thu tu)',
                      list_note + ('' if pix is None or pix >= 0.99 else
                                   f' | CANH BAO: argmax chi khop {pix:.4f} pixel voi mask dump')))

    # ── Cong 2: dong nhat thuc loss tren 8 anh dau ──
    errs = []
    for i in range(min(N_GATE_LOSS_IMAGES, n)):
        x, m = val_ds[i]
        x, m = x[None].to(device), m[None].to(device)
        with torch.no_grad():
            logits, edge, F = probe.forward(x)
            L = probe.losses(logits, edge, F, m)
            tt = info['trained_total']
            if tt == 'static':
                ref = StaticBoundaryTotalLoss(alpha=0.4, **probe.loss_kw).to(device)
                l_total, _ = ref(logits, edge, F, m, pos_weight)
                mine = L['region'] + 0.4 * (1.0 * L['bce'] + 1.0 * L['aff'])
            elif tt == 'affonly':
                bl = cfg['BOUNDARY_LOSS']
                ref = AffinityOnlyTotalLoss(alpha=bl['ALPHA'], lambda1_static=bl.get('LAMBDA1_STATIC', 1.0),
                                            lambda2_static=bl.get('LAMBDA2_STATIC', 1.0), **probe.loss_kw).to(device)
                l_total, _ = ref(logits, F, m)
                mine = L['region'] + bl['ALPHA'] * bl.get('LAMBDA2_STATIC', 1.0) * L['aff']
            elif tt == 'bce':
                from src.losses.boundary_bce import BalancedBCEEdgeLoss
                seg = CombinedLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).to(device)
                edge_l = BalancedBCEEdgeLoss(IGNORE_INDEX, probe.loss_kw['connectivity'],
                                             probe.loss_kw['dilation_radius']).to(device)
                l_total = seg(logits, m) + info['w_bce'] * edge_l(edge, m, pos_weight)
                mine = L['region'] + info['w_bce'] * L['bce']
            else:
                l_total = CombinedLoss(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX).to(device)(logits, m)
                mine = L['region']
            errs.append(abs(float(l_total) - float(mine)))
    e = max(errs) if errs else float('nan')
    gates.append(gate(2, 'dong nhat thuc loss (8 anh dau)', 'PASS' if e < GATE_LOSS_TOL else 'FAIL',
                      {'max_abs_err': e, 'trained_total': info['trained_total'],
                       'weights': probe.weights, 'pos_weight': pos_weight}, GATE_LOSS_TOL,
                      'L_total THAT luc train (module repo) vs tong so hang rieng; so hang probe khong nam trong L_total'))

    # ── Cong 3: tuyen tinh gradient ──
    # Kiem o float64 (ban sao model): muc dich la xac nhan viec TACH so hang dung (khong thieu/trung so
    # hang), khong phai do sai so lam tron. O fp32, tong ~11.6M phan tu theo thu tu backward khac nhau cho
    # sai so tuong doi ~1e-5 — van ghi lai de tham khao, KHONG dung de quyet dinh.
    def lin_errors(pr):
        worst = {'F': 0.0, 'theta': 0.0}
        for i in range(min(N_GATE_LIN_IMAGES, n)):
            x, m = val_ds[i]
            x, m = x[None].to(device), m[None].to(device)
            _L, gF, gT = pr.image_grads(x, m, i, need_theta=True)
            torch.manual_seed(SEED + i)
            logits, edge, F = pr.forward(x)
            L = pr.losses(logits, edge, F, m)
            tot = sum(pr.weights[k] * L[k] for k in pr.terms)
            g = torch.autograd.grad(tot, [F] + pr.shared_params, allow_unused=True)
            gF_tot = g[0][0]
            gT_tot = flatten_grads(g[1:], pr.shared_params)
            sF = sum(gF[k] for k in pr.terms)
            sT = sum(gT[k] for k in pr.terms)
            worst['F'] = max(worst['F'], float((sF - gF_tot).norm() / gF_tot.norm().clamp_min(1e-30)))
            worst['theta'] = max(worst['theta'], float((sT - gT_tot).norm() / gT_tot.norm().clamp_min(1e-30)))
            del gT, gT_tot, sT, g
        return worst

    worst32 = lin_errors(probe)
    probe64 = Probe(label, ckpt_path, cfg, device, pos_weight, dtype=torch.float64)
    worst = lin_errors(probe64)
    del probe64
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    ok = worst['F'] < GATE_LIN_TOL and worst['theta'] < GATE_LIN_TOL
    gates.append(gate(3, 'tuyen tinh gradient (F va shared_params)', 'PASS' if ok else 'FAIL',
                      {'max_rel_err_F_float64': worst['F'], 'max_rel_err_theta_float64': worst['theta'],
                       'max_rel_err_F_float32_thamkhao': worst32['F'],
                       'max_rel_err_theta_float32_thamkhao': worst32['theta'],
                       'n_images': min(N_GATE_LIN_IMAGES, n), 'n_shared_params': probe.n_shared}, GATE_LIN_TOL,
                      'quyet dinh tren float64 (kiem tach so hang); so float32 chi tham khao'))

    # ── Cong 4: dung pham vi cap ──
    notes, ok = [], True
    has_bh_keys = any(k.startswith('boundary_head.') for k in probe.state_keys)
    if not info['has_bce']:
        if any('bce' in p for p in probe.pairs):
            ok = False
            notes.append('co cap chua bce cho checkpoint khong co BCE')
        if has_bh_keys:
            ok = False
            notes.append('state_dict co boundary_head.* nhung bang checkpoint noi khong co BCE')
        notes.append('khong co cap nao chua L_BCE (dung)')
    else:
        if not has_bh_keys:
            ok = False
            notes.append('thieu boundary_head.* trong state_dict')
        bh = probe.model.boundary_head
        torch.manual_seed(SEED)
        fresh = BoundaryHead(in_channels=bh.conv_out.in_channels).to(device)
        with torch.no_grad():
            w_diff = float((bh.conv_out.weight - fresh.conv_out.weight).abs().max())
        rm = [b for n_, b in bh.named_buffers() if n_.endswith('running_mean')]
        rv = [b for n_, b in bh.named_buffers() if n_.endswith('running_var')]
        bn_moved = any(float(t.abs().max()) > 1e-6 for t in rm) or any(float((t - 1).abs().max()) > 1e-6 for t in rv)
        if not (w_diff > 1e-3 and bn_moved):
            ok = False
        notes.append(f'BoundaryHead: |conv_out - khoi tao moi seed19|max = {w_diff:.4g}; BN running stats '
                     f"{'da doi' if bn_moved else 'CON MAC DINH'}")
    gates.append(gate(4, 'dung pham vi cap', 'PASS' if ok else 'FAIL',
                      {'pairs': [pair_name(p) for p in probe.pairs]}, None, '; '.join(notes)))

    # ── Cong 5: tat dinh ──
    max_rel = 0.0
    for i in range(min(N_GATE_DET_IMAGES, n)):
        name, gt = load_gt_cached(val_ds, i, args.gt_cache or None)
        x, m = val_ds[i]
        x, m = x[None].to(device), m[None].to(device)
        runs = []
        for _rep in range(2):
            _L, gF, gT = probe.image_grads(x, m, i, need_theta=True)
            st = ImageStrata(np.asarray(gt), gF['region'].shape[-2:], device, i, args.n_perm, True)
            vals = []
            for p in probe.pairs:
                fs = feature_pair_stats(gF[p[0]], gF[p[1]], st, args.n_perm)
                vals += [t.flatten().cpu() for t in fs['strata'].values()]
                vals += [t.flatten().cpu() for t in fs['null'].values()]
                vals.append(torch.tensor([float(gT[p[0]] @ gT[p[1]])], dtype=torch.float64))
            vals += [gT[k].double().cpu() for k in probe.terms]
            runs.append(vals)
            del gT
        for a, b in zip(*runs):
            scale = float(a.abs().max()) if a.numel() else 0.0
            if scale > 0:
                max_rel = max(max_rel, float((a - b).abs().max()) / scale)
    gates.append(gate(5, 'tat dinh (2 lan, 4 anh)', 'PASS' if max_rel <= GATE_DET_TOL else 'FAIL',
                      {'max_rel_diff': max_rel}, GATE_DET_TOL,
                      'cudnn.deterministic + use_deterministic_algorithms(warn_only); KHONG noi nguong neu FAIL'))

    statuses = [g['status'] for g in gates]
    overall = 'PASS' if all(s in ('PASS', 'SKIP_DRYRUN') for s in statuses) else 'FAIL'
    res = {'label': label, 'checkpoint_path': ckpt_path, 'checkpoint_fingerprint': file_fingerprint(ckpt_path),
           'config': CHECKPOINTS[label]['config'], 'iter': info['iter'], 'seed': info['seed'],
           'pos_weight': pos_weight, 'dry_run': dry, 'n_images': n, 'device': str(device),
           'overall': overall, 'gates': sorted(gates, key=lambda g: g['gate']),
           'note': 'cong 6 (unit test lo) nam o output/spurious_holes/gates_holes.json',
           'created_at': datetime.datetime.now().isoformat(timespec='seconds'), 'git_commit': get_git_commit()}
    return res


def cmd_gates(args):
    ckpts, extra = parse_common(args)
    device = resolve_device_det(args.device)
    for label, path in ckpts.items():
        out = os.path.join(args.output_dir, f'gates_{label}.json')
        if os.path.exists(out) and not args.force:
            g = read_json(out)
            if g.get('overall') == 'PASS' and g.get('checkpoint_fingerprint') == file_fingerprint(path) \
                    and g.get('dry_run') == (args.max_images is not None):
                print(f"[{label}] gates da PASS truoc do ({out}) — bo qua (--force de chay lai).")
                continue
        t0 = time.time()
        res = run_gates_for(label, args, path, extra['cfg'][label], device, extra['pos_weight'].get(label),
                            extra['dump'].get(label), extra['expected'].get(label))
        res['wall_time_s'] = time.time() - t0
        write_json(res, out)
        for g in res['gates']:
            print(f"  [{label}] cong {g['gate']} {g['name']}: {g['status']}  {g['note']}")
        print(f"[{label}] TONG: {res['overall']} -> {out}")
        append_log(os.path.join(args.output_dir, 'run_log.txt'),
                   f"[{res['created_at']}] gates {label} -> {res['overall']} ({res['wall_time_s']:.0f}s) "
                   f"pos_weight={res['pos_weight']} ({extra['pos_src'].get(label, '-')})")
        if res['overall'] != 'PASS' and not args.continue_on_fail:
            raise SystemExit(f"[{label}] CONG KIEM FAIL — DUNG (muc 4). Xem {out}.")


# ═══════════════════════════════ measure ══════════════════════════════════════

def batch_partition(n, batch_size):
    perm = np.random.default_rng(SEED).permutation(n)
    return [perm[i:i + batch_size].tolist() for i in range(0, n, batch_size)]


def measure_one(label, args, ckpt_path, cfg, device, pos_weight, figure_ids):
    from Tools.oracle_boundary_ceiling import build_val_dataset, image_ids_of
    probe = Probe(label, ckpt_path, cfg, device, pos_weight)
    val_ds = build_val_dataset(cfg)
    image_ids = image_ids_of(val_ds)
    n = len(image_ids) if args.max_images is None else min(args.max_images, len(image_ids))
    batches = batch_partition(n, args.batch_size)
    want_comp = label in STATIC_LABELS
    nt = len(probe.terms)
    P = probe.n_shared
    os.makedirs(args.scratch_dir, exist_ok=True)
    mm_path = os.path.join(args.scratch_dir, f'gtheta_{label}.f32')
    mm = np.memmap(mm_path, dtype=np.float32, mode='w+', shape=(len(batches), nt, P))
    fig_dir = os.path.join(args.output_dir, 'figures')

    strata_rows, null_rows, comp_rows = [], [], []
    loss_rows = []
    t0 = time.time()
    done = 0
    for b, members in enumerate(batches):
        acc = {k: torch.zeros(P, dtype=torch.float32, device=device) for k in probe.terms}
        for i in members:
            name, gt = load_gt_cached(val_ds, i, args.gt_cache or None)
            assert name == image_ids[i]
            x, m = val_ds[i]
            x, m = x[None].to(device), m[None].to(device)
            L, gF, gT = probe.image_grads(x, m, i, need_theta=True)
            for k in probe.terms:
                acc[k] += gT[k]
            del gT
            loss_rows.append({'image_id': name, 'image_idx': i, 'batch_id': b, **{f'L_{k}': v for k, v in L.items()}})
            st = ImageStrata(np.asarray(gt), gF['region'].shape[-2:], device, i, args.n_perm, want_comp)
            fig_maps = {}
            for p in probe.pairs:
                pn = pair_name(p)
                fs = feature_pair_stats(gF[p[0]], gF[p[1]], st, args.n_perm)
                for ax, names in AXES.items():
                    a = fs['strata'][ax].cpu().numpy()
                    nl = fs['null'][ax].cpu().numpy()
                    for s, sname in enumerate(names):
                        strata_rows.append({'image_id': name, 'image_idx': i, 'pair': pn, 'axis': ax,
                                            'stratum': sname, 'sum_dot': a[s, 0], 'sum_n1': a[s, 1],
                                            'sum_n2': a[s, 2], 'n_pos': int(a[s, 3]), 'n_support': int(a[s, 4]),
                                            'n_neg': int(a[s, 5])})
                        if a[s, 3] > 0:
                            for r in range(args.n_perm):
                                null_rows.append((pn, ax, sname, r, name, nl[r, s]))
                if want_comp:
                    for row in component_stats(fs['maps'], st):
                        comp_rows.append({'image_id': name, 'pair': pn, **row})
                if label == FIGURE_CKPT and name in figure_ids:
                    dot, n1, n2 = fs['maps']
                    cos = torch.where((n1 > SUPPORT_EPS) & (n2 > SUPPORT_EPS),
                                      dot / torch.sqrt(n1 * n2).clamp_min(1e-300),
                                      torch.full_like(dot, float('nan')))
                    fig_maps[pn] = cos.reshape(gF['region'].shape[-2:]).cpu().numpy()
            if fig_maps:
                render_figure(val_ds, i, name, np.asarray(gt), fig_maps, fig_dir, label)
            del gF
            done += 1
            el = time.time() - t0
            print(f"  [{label}] {done}/{n} anh ({el:.0f}s, {el / done:.2f}s/anh)", end='\r')
        for j, k in enumerate(probe.terms):
            mm[b, j, :] = acc[k].cpu().numpy()
        del acc
    print()
    mm.flush()
    wall_grad = time.time() - t0

    # ── Muc tham so: ma tran Gram (b, k) x (b', k') ──
    gram = gram_from_memmap(mm, device)
    del mm
    if not args.keep_scratch:
        try:
            os.remove(mm_path)
        except OSError:
            pass
    tidx = {k: j for j, k in enumerate(probe.terms)}
    nb = len(batches)
    param_rows, null_param = [], []

    def gv(bb, k1, bb2, k2):
        return gram[bb * nt + tidx[k1], bb2 * nt + tidx[k2]]

    for p in probe.pairs:
        pn = pair_name(p)
        for bb in range(nb):
            d = gv(bb, p[0], bb, p[1])
            n1 = math.sqrt(max(gv(bb, p[0], bb, p[0]), 0.0))
            n2 = math.sqrt(max(gv(bb, p[1], bb, p[1]), 0.0))
            cos = d / (n1 * n2) if n1 > 0 and n2 > 0 else float('nan')
            param_rows.append({'batch_id': bb, 'image_ids': '|'.join(image_ids[i] for i in batches[bb]),
                               'n_images': len(batches[bb]), 'pair': pn, 'dot': d, 'norm1': n1, 'norm2': n2,
                               'cos': cos, 'mag_ratio': (n2 / n1) if n1 > 0 else float('nan')})
            for bb2 in range(nb):
                if bb2 == bb:
                    continue
                d2 = gv(bb, p[0], bb2, p[1])
                m1 = math.sqrt(max(gv(bb, p[0], bb, p[0]), 0.0))
                m2 = math.sqrt(max(gv(bb2, p[1], bb2, p[1]), 0.0))
                null_param.append({'pair': pn, 'b': bb, 'b_prime': bb2,
                                   'cos': d2 / (m1 * m2) if m1 > 0 and m2 > 0 else float('nan')})

    od = args.output_dir
    pd.DataFrame(strata_rows).sort_values(['image_idx', 'pair', 'axis'], kind='stable') \
        .drop(columns=['image_idx']).to_csv(os.path.join(od, f'per_image_strata_{label}.csv'), index=False)
    pd.DataFrame(param_rows).to_csv(os.path.join(od, f'per_batch_param_{label}.csv'), index=False)
    pd.DataFrame(null_param, columns=['pair', 'b', 'b_prime', 'cos']).to_csv(os.path.join(od, f'null_param_{label}.csv'), index=False)
    pd.DataFrame(null_rows, columns=['pair', 'axis', 'stratum', 'perm_id', 'image_id', 'sum_dot_perm']) \
        .to_csv(os.path.join(od, f'null_feat_{label}.csv'), index=False)
    pd.DataFrame(loss_rows).sort_values('image_idx').to_csv(os.path.join(od, f'per_image_loss_{label}.csv'),
                                                            index=False)
    if want_comp:
        pd.DataFrame(comp_rows).to_csv(os.path.join(od, f'per_component_small_{label}.csv'), index=False)
    return {'n_images': n, 'n_batches': nb, 'batch_size': args.batch_size, 'n_shared_params': P,
            'terms': list(probe.terms), 'pairs': [pair_name(p) for p in probe.pairs], 'weights': probe.weights,
            'wall_time_grad_s': wall_grad, 'wall_time_total_s': time.time() - t0}


def gram_from_memmap(mm, device, chunk=1 << 19):
    nb, nt, P = mm.shape
    V = mm.reshape(nb * nt, P)
    dev = device if device.type == 'cuda' else torch.device('cpu')
    G = torch.zeros(nb * nt, nb * nt, dtype=torch.float64, device=dev)
    for s in range(0, P, chunk):
        blk = torch.from_numpy(np.ascontiguousarray(V[:, s:s + chunk])).to(dev).double()
        G += blk @ blk.T
    return G.cpu().numpy()


def render_figure(val_ds, i, name, gt, maps, fig_dir, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.utils.visualizer import mask_to_rgb, denormalize
    os.makedirs(fig_dir, exist_ok=True)
    np.savez_compressed(os.path.join(fig_dir, f'{name}_cosmaps.npz'), **maps)
    x, _m = val_ds[i]
    rgb = denormalize(x.permute(1, 2, 0).numpy())
    panels = [('RGB', rgb, None), ('GT', mask_to_rgb(gt), None)]
    for pn, title in (('bce_aff', 'cos_p (BCE, Aff)'), ('region_aff', 'cos_p (Region, Aff)')):
        panels.append((title, maps.get(pn), 'cos'))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.4))
    for ax, (title, img, kind) in zip(axes, panels):
        if img is None:
            ax.text(0.5, 0.5, 'khong co cap nay', ha='center', va='center')
        elif kind == 'cos':
            im = ax.imshow(np.ma.masked_invalid(img), cmap='RdBu', vmin=-1, vmax=1, interpolation='nearest')
            fig.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'{name} — {label} (stride 4; trang = khong du ho tro)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f'{name}.png'), dpi=110)
    plt.close(fig)


def cmd_measure(args):
    ckpts, extra = parse_common(args)
    device = resolve_device_det(args.device)
    fig_path = args.figures_json or os.path.join(args.output_dir, 'figure_images.json')
    figure_ids = set()
    if FIGURE_CKPT in ckpts:
        if not os.path.exists(fig_path):
            raise SystemExit(f"Chua co {fig_path} — chay Tools/select_figure_images.py TRUOC (muc 2.4: chot tu GT "
                             f"truoc khi tinh).")
        figure_ids = {d['image_id'] for d in read_json(fig_path)['images']}
    for label, path in ckpts.items():
        done_path = os.path.join(args.output_dir, f'done_{label}.json')
        fp = file_fingerprint(path)
        pw = extra['pos_weight'].get(label)
        if os.path.exists(done_path) and not args.force:
            d = read_json(done_path)
            if d.get('checkpoint_fingerprint') == fp and d.get('pos_weight') == pw \
                    and d.get('n_images_requested') == args.max_images and d.get('batch_size') == args.batch_size:
                print(f"[{label}] da do xong ({done_path}) — bo qua (--force de do lai).")
                continue
        gpath = os.path.join(args.output_dir, f'gates_{label}.json')
        if not args.skip_gate_check:
            if not os.path.exists(gpath):
                raise SystemExit(f"[{label}] chua co {gpath} — chay 'gates' truoc (muc 4).")
            g = read_json(gpath)
            if g.get('overall') != 'PASS' or g.get('checkpoint_fingerprint') != fp or g.get('pos_weight') != pw:
                raise SystemExit(f"[{label}] gates chua PASS cho dung checkpoint/pos_weight nay ({gpath}).")
            if g.get('dry_run') and args.max_images is None:
                raise SystemExit(f"[{label}] gates moi chay o che do dry-run — chay lai 'gates' khong --max-images.")
        else:
            append_log(os.path.join(args.output_dir, 'run_log.txt'),
                       f"[CANH BAO] measure {label} chay voi --skip-gate-check")
        t0 = time.time()
        res = measure_one(label, args, path, extra['cfg'][label], device, pw, figure_ids)
        info = CHECKPOINTS[label]
        done = {'label': label, 'checkpoint_path': path, 'checkpoint_fingerprint': fp, 'iter': info['iter'],
                'seed': info['seed'], 'config': info['config'], 'model_type': info['model_type'],
                'mode': 'model.eval() — BN running stats, dropout/drop-path tat; fp32; 1 anh/backward',
                'pos_weight': pw, 'pos_weight_source': extra['pos_src'].get(label),
                'n_images_requested': args.max_images, 'n_perm': args.n_perm, 'device': str(device),
                'torch': torch.__version__,
                'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'git_commit': get_git_commit(), **res}
        done['wall_time_s'] = time.time() - t0
        write_json(done, done_path)
        append_log(os.path.join(args.output_dir, 'run_log.txt'),
                   f"[{done['created_at']}] measure {label}: {res['n_images']} anh, {done['wall_time_s']:.0f}s, "
                   f"gpu={done['gpu']} torch={done['torch']} pos_weight={pw} argv={' '.join(sys.argv)}")
        print(f"[{label}] xong trong {done['wall_time_s']:.0f}s -> {done_path}")


# ═══════════════════════════════ summarize ════════════════════════════════════

def _boot_counts(n, n_boot):
    idx = np.random.default_rng(SEED).integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    return counts


def summarize_label(label, od, n_boot):
    info = CHECKPOINTS[label]
    rows = []
    # ── muc tham so ──
    pb = pd.read_csv(os.path.join(od, f'per_batch_param_{label}.csv'))
    npar = pd.read_csv(os.path.join(od, f'null_param_{label}.csv'))
    for pn, sub in pb.groupby('pair', sort=False):
        cos = sub['cos'].to_numpy(dtype=np.float64)
        nb = len(cos)
        idx = np.random.default_rng(SEED).integers(0, nb, size=(n_boot, nb))
        meds = np.median(cos[idx], axis=1)
        nulls = npar[npar['pair'] == pn]['cos'].to_numpy(dtype=np.float64)
        rows.append(_criteria(label, 'param', pn, 'all', 'all', float(np.median(cos)),
                              *np.percentile(meds, [2.5, 97.5]),
                              *(np.percentile(nulls, [2.5, 97.5]) if len(nulls) else (np.nan, np.nan)),
                              float(np.mean(cos < 0)), float(np.median(sub['mag_ratio'])), 1.0, nb, TAU_PARAM))
    # ── muc dac trung ──
    st = pd.read_csv(os.path.join(od, f'per_image_strata_{label}.csv'))
    nf = pd.read_csv(os.path.join(od, f'null_feat_{label}.csv'))
    images = st['image_id'].drop_duplicates().tolist()
    n = len(images)
    counts = _boot_counts(n, n_boot)
    nf_sum = nf.groupby(['pair', 'axis', 'stratum', 'perm_id'])['sum_dot_perm'].sum()
    for (pn, ax, sname), sub in st.groupby(['pair', 'axis', 'stratum'], sort=False):
        sub = sub.set_index('image_id').loc[images]
        dot, n1, n2 = (sub[c].to_numpy(dtype=np.float64) for c in ('sum_dot', 'sum_n1', 'sum_n2'))
        Sd, S1, S2 = dot.sum(), n1.sum(), n2.sum()
        npos, nsup, nneg = sub['n_pos'].sum(), sub['n_support'].sum(), sub['n_neg'].sum()
        den = math.sqrt(S1 * S2) if S1 > 0 and S2 > 0 else 0.0
        obs = Sd / den if den > 0 else np.nan
        with np.errstate(invalid='ignore', divide='ignore'):
            bd = counts @ dot
            bn = np.sqrt((counts @ n1) * (counts @ n2))
            boot = np.where(bn > 0, bd / bn, np.nan)
        boot = boot[~np.isnan(boot)]
        ci = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
        try:
            nv = nf_sum.loc[(pn, ax, sname)].to_numpy(dtype=np.float64) / den if den > 0 else np.array([])
        except KeyError:
            nv = np.array([])
        nl = np.percentile(nv, [2.5, 97.5]) if len(nv) else (np.nan, np.nan)
        rows.append(_criteria(label, 'feat', pn, ax, sname, obs, *ci, *nl,
                              (nneg / nsup) if nsup else np.nan, math.sqrt(S2 / S1) if S1 > 0 else np.nan,
                              (nsup / npos) if npos else 0.0, n, TAU_FEAT, n_pos=int(npos)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(od, f'summary_{label}.csv'), index=False)
    return df


def _criteria(label, level, pair, axis, stratum, value, ci_lo, ci_hi, null_lo, null_hi, frac_neg, mag, cov,
              n_units, tau, n_pos=None):
    insufficient = (level == 'feat') and (not (cov >= SUPPORT_MIN_COV))
    c1 = bool(ci_hi < 0)
    c2 = bool(value < null_lo)
    c3 = bool(abs(value) >= tau) if not np.isnan(value) else False
    c4 = bool(MAG_LO <= mag <= MAG_HI) if not np.isnan(mag) else False
    return {'label': label, 'level': level, 'pair': pair, 'axis': axis, 'stratum': stratum,
            'value': value, 'value_kind': 'median_cos_batch' if level == 'param' else 'cosW_micro',
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'null_lo': null_lo, 'null_hi': null_hi,
            'frac_neg': frac_neg, 'mag_ratio': mag, 'support_cov': cov, 'n_pos': n_pos,
            'flag_insufficient': bool(insufficient), 'n_units': n_units, 'tau': tau,
            'crit1_ci_below_0': c1, 'crit2_below_null': c2, 'crit3_abs_ge_tau': c3, 'crit4_mag_in_range': c4,
            'confirmed_single_ckpt': bool(c1 and c2 and c3 and c4 and not insufficient)}


def decide_branch(dfs):
    """Bang 5.2 NGUYEN VAN tren Static s19@36k + Static s86@36k."""
    if not all(lb in dfs for lb in DECISION_CKPTS):
        return {'branch': None, 'text': f'CHUA AP DUOC bang 5.2 — can ca {list(DECISION_CKPTS)}', 'raw': {},
                'confirmed_both': []}
    a, b = (dfs[lb].set_index(['level', 'pair', 'axis', 'stratum']) for lb in DECISION_CKPTS)
    both = []
    for key in a.index:
        if key in b.index and a.loc[key, 'confirmed_single_ckpt'] and b.loc[key, 'confirmed_single_ckpt'] \
                and np.sign(a.loc[key, 'value']) == np.sign(b.loc[key, 'value']):
            both.append(key)
    bset = set(both)

    def conf(level, pair, axis='all', stratum='all'):
        return (level, pair, axis, stratum) in bset

    def eligible(axis, pair):
        keys = [k for k in a.index if k[0] == 'feat' and k[1] == pair and k[2] == axis]
        return [k for k in keys if not a.loc[k, 'flag_insufficient'] and
                not (k in b.index and b.loc[k, 'flag_insufficient'])]

    ba = 'bce_aff'
    G = conf('param', ba)
    ds_elig = eligible('dist', ba) + eligible('size', ba)
    ds_conf = [k for k in ds_elig if k in bset]
    L_raw = bool(ds_conf) and len(ds_conf) < len(ds_elig)
    cls_conf = [k for k in eligible('class', ba) if k in bset]
    C_raw = 1 <= len(cls_conf) <= 3
    ba_any = any(k[1] == ba for k in bset)
    far = ['d8-16', 'd16-32', 'd>32']
    RA_raw = conf('param', 'region_aff') or any(conf('feat', 'region_aff', 'dist', s) for s in far)
    lit = {
        'G': G,
        'L': (not G) and L_raw,
        'C': (not G) and (not L_raw) and C_raw,
        'RA': (not ba_any) and RA_raw,
        'N': len(bset) == 0,
    }
    raw = {'G': G, 'L': L_raw, 'C': C_raw, 'RA': RA_raw, 'N': len(bset) == 0}
    chosen = next((br for br in ('G', 'RA', 'L', 'C') if lit[br]), 'N' if lit['N'] else None)
    if chosen is None:
        text = ('Khong nhanh nao cua bang 5.2 dat theo dung dieu kien, nhung co xung dot xac nhan '
                '(vd chi o cap region_bce hoac moi tang) — xem danh sach xac nhan ca hai seed')
    else:
        text = {'G': 'G — Toan cuc', 'L': 'L — Khu tru khong gian', 'C': 'C — Khu tru theo lop',
                'RA': 'RA — Affinity xung dot voi nhiem vu chinh',
                'N': 'N — Khong xung dot tai diem hoi tu (chay Phan D truoc khi ket luan)'}[chosen]
    ablation = [br for br in ('G', 'RA', 'L', 'C') if raw[br] and br != chosen]
    return {'branch': chosen, 'text': text, 'literal': lit, 'raw': raw, 'ablation_variants': ablation,
            'confirmed_both': [list(k) for k in both],
            'L_detail': {'eligible': [list(k) for k in ds_elig], 'confirmed': [list(k) for k in ds_conf]},
            'C_detail': [list(k) for k in cls_conf]}


def cmd_summarize(args):
    od = args.output_dir
    present = [lb for lb in CHECKPOINTS if os.path.exists(os.path.join(od, f'done_{lb}.json'))]
    if not present:
        raise SystemExit(f"Khong co done_<ckpt>.json nao trong {od} — chay 'measure' truoc.")
    dfs, dones, gates = {}, {}, {}
    for lb in present:
        print(f"[{lb}] tong hop ...")
        dfs[lb] = summarize_label(lb, od, args.n_boot)
        dones[lb] = read_json(os.path.join(od, f'done_{lb}.json'))
        gp = os.path.join(od, f'gates_{lb}.json')
        gates[lb] = read_json(gp) if os.path.exists(gp) else None
    gh = os.path.join(args.holes_dir, 'gates_holes.json')
    gate6 = read_json(gh) if os.path.exists(gh) else None
    decision = decide_branch(dfs)
    write_json({'decision_52': decision, 'present': present,
                'missing': [lb for lb in CHECKPOINTS if lb not in present], 'missing_runs': MISSING_RUNS},
               os.path.join(od, 'summary_grad_conflict.json'))
    write_summary_md(od, dfs, dones, gates, gate6, decision, args)
    print(f"Nhanh 5.2: {decision['text']}")
    print(f"Da ghi summary_grad_conflict.md / .json va summary_<ckpt>.csv trong {od}")


def _f(x, nd=3):
    if x is None:
        return '—'
    try:
        if np.isnan(x):
            return '—'
    except TypeError:
        return str(x)
    return f'{x:.{nd}f}'


def write_summary_md(od, dfs, dones, gates, gate6, decision, args):
    L = []
    first = DECISION_CKPTS[0] if DECISION_CKPTS[0] in dones else next(iter(dones))
    d0 = dones[first]
    L.append(f"Checkpoint chinh: {first} · iter {d0['iter']} · seed {d0['seed']} · che do `model.eval()` "
             f"(BN running stats) · fp32 · batch muc tham so = {d0['batch_size']} · A_min chinh (Phan B) = 64 px")
    L.append('')
    L.append('# Phan A — Chan doan xung dot gradient')
    L.append('')
    L.append('## Checkpoint')
    L.append('')
    L.append('| Nhan | Vai tro | iter | seed | Cap do | Trong so | pos_weight | So anh | Thoi gian that | Cong 1–5,7 |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for lb, info in CHECKPOINTS.items():
        if lb in dones:
            d = dones[lb]
            g = gates.get(lb)
            gtxt = g['overall'] if g else 'khong co file gates'
            L.append(f"| {lb} | {info['role_a']} | {info['iter']} | {info['seed']} | {', '.join(d['pairs'])} | "
                     f"{d['weights']} | {d.get('pos_weight')} | {d['n_images']} | {d['wall_time_s'] / 60:.1f} ph | "
                     f"{gtxt} |")
        else:
            L.append(f"| {lb} | {info['role_a']} | {info['iter']} | {info['seed']} | — | — | — | — | CHUA DO | — |")
    for k, v in MISSING_RUNS.items():
        L.append(f"| {k} | — | — | — | — | — | — | — | {v} | — |")
    L.append('')
    L.append('## Cong kiem (muc 4)')
    L.append('')
    L.append('| Nhan | ' + ' | '.join(f'Cong {i}' for i in (1, 2, 3, 4, 5, 7)) + ' |')
    L.append('|---|' + '---|' * 6)
    for lb in dones:
        g = gates.get(lb)
        if not g:
            L.append(f'| {lb} | ' + ' | '.join(['—'] * 6) + ' |')
            continue
        by = {x['gate']: x['status'] for x in g['gates']}
        L.append(f'| {lb} | ' + ' | '.join(by.get(i, '—') for i in (1, 2, 3, 4, 5, 7)) + ' |')
    L.append('')
    L.append(f"Cong 6 (unit test lo, Phan B): **{gate6['status'] if gate6 else 'CHUA CHAY'}**")
    L.append('')
    L.append('## Nhanh phuong phap (bang 5.2, ap nguyen van tren Static s19@36k + s86@36k)')
    L.append('')
    L.append(f"**{decision['text']}**")
    L.append('')
    if decision.get('literal'):
        L.append('| Nhanh | Dieu kien theo bang (co loai tru) | Dieu kien tho (khong loai tru) |')
        L.append('|---|---|---|')
        for br in ('G', 'L', 'C', 'RA', 'N'):
            L.append(f"| {br} | {'DAT' if decision['literal'][br] else 'khong'} | "
                     f"{'DAT' if decision['raw'][br] else 'khong'} |")
        L.append('')
        L.append(f"Bien the ablation (nhanh khac cung dat, thu tu G > RA > L > C): "
                 f"{decision['ablation_variants'] or 'khong co'}")
        L.append('')
        L.append('Cac (muc, cap, truc, tang) xung dot xac nhan o CA HAI seed:')
        L.append('')
        if decision['confirmed_both']:
            for k in decision['confirmed_both']:
                L.append(f"- {k[0]} · {k[1]} · {k[2]} · {k[3]}")
        else:
            L.append('- (khong co)')
        L.append('')
    L.append('Tieu chi 5.1 (dong thoi): (1) CI 95% bootstrap co can tren < 0; (2) gia tri quan sat < can 2.5% cua null; '
             f'(3) |cos| ≥ τ (τ_param = {TAU_PARAM}, τ_feat = {TAU_FEAT}); (4) ti le do lon trong [{MAG_LO}, {MAG_HI}]; '
             '(5) dung o ca Static s19@36k va s86@36k (cung dau, moi ckpt dat 1–4). Tang co do phu ho tro < 1% '
             'bi danh dau "khong du ho tro" va loai khoi quyet dinh.')
    L.append('')
    L.append('## Mo ta 5.3 (khong dung de chon nhanh)')
    L.append('')
    L.extend(describe_53(dfs))
    L.append('')
    L.append('## Bang chi tiet theo checkpoint')
    L.append('')
    L.append('Muc tham so: gia tri = trung vi cos theo batch, CI bootstrap tren batch, null = cos cap batch lech (b ≠ b\'). '
             'Muc dac trung: cosW gop micro, CI bootstrap per-image, null = 20 hoan vi vi tri trong cung tang cung anh. '
             'C1–C4 = tieu chi 5.1 (1)–(4); ✔ = xac nhan tai checkpoint nay.')
    L.append('')
    for lb, df in dfs.items():
        L.append(f'### {lb}')
        L.append('')
        L.append('| Muc | Cap | Truc | Tang | Gia tri | CI 95% | Null 2.5–97.5% | Ti le cos<0 | Ti le do lon | '
                 'Do phu ho tro | C1 | C2 | C3 | C4 | Xac nhan |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for _, r in df.iterrows():
            cov = 'khong du ho tro' if r['flag_insufficient'] else _f(r['support_cov'])
            L.append(f"| {r['level']} | {r['pair']} | {r['axis']} | {r['stratum']} | {_f(r['value'])} | "
                     f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] | [{_f(r['null_lo'])}, {_f(r['null_hi'])}] | "
                     f"{_f(r['frac_neg'])} | {_f(r['mag_ratio'])} | {cov} | "
                     + ' | '.join('x' if r[c] else '' for c in ('crit1_ci_below_0', 'crit2_below_null',
                                                                  'crit3_abs_ge_tau', 'crit4_mag_in_range'))
                     + f" | {'✔' if r['confirmed_single_ckpt'] else ''} |")
        L.append('')
    L.append('## Ghi chu quy uoc')
    L.append('')
    L.append('- L_region tinh tren TUNG anh (Dice gop trong 1 anh), khong phai Dice gop batch 4 nhu luc train. '
             'G_k(b) = tong gradient theo anh trong batch.')
    L.append('- fp32, khong autocast (luc train dung AMP). `model.eval()`: BN running stats, dropout/drop-path tat.')
    L.append('- AffinityLoss cua repo tat dinh (khong lay mau ngau nhien); van dat torch.manual_seed(19 + idx) moi anh.')
    L.append('- Probe: so hang affinity tinh tren checkpoint chua tung train affinity (BCE λ=0.4/0.2, Baseline) voi '
             'trong so 0.4.')
    L.append('- File null tach 2: `null_param_<ckpt>.csv` (muc tham so) va `null_feat_<ckpt>.csv` (muc dac trung).')
    L.append('')
    with open(os.path.join(od, 'summary_grad_conflict.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def _row(df, level, pair, axis='all', stratum='all'):
    sub = df[(df['level'] == level) & (df['pair'] == pair) & (df['axis'] == axis) & (df['stratum'] == stratum)]
    return sub.iloc[0] if len(sub) else None


def _ci_txt(r):
    return '—' if r is None else f"{_f(r['value'])} [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]"


def describe_53(dfs):
    L = []
    L.append('**Probe BCE vs Static — cap (BCE, Aff):**')
    L.append('')
    L.append('| Checkpoint | Muc tham so (trung vi cos, CI) | Muc dac trung toan anh (cosW, CI) |')
    L.append('|---|---|---|')
    for lb in ('bce04_40k', 'bce02_40k', 'static_s19_36k', 'static_s86_36k', 'static_s86_40k'):
        if lb in dfs:
            L.append(f"| {lb} | {_ci_txt(_row(dfs[lb], 'param', 'bce_aff'))} | "
                     f"{_ci_txt(_row(dfs[lb], 'feat', 'bce_aff'))} |")
    L.append('')
    if 'bce04_40k' in dfs and all(lb in dfs for lb in DECISION_CKPTS):
        pr = _row(dfs['bce04_40k'], 'param', 'bce_aff')
        stronger = all(pr is not None and _row(dfs[lb], 'param', 'bce_aff') is not None
                       and pr['ci_hi'] < _row(dfs[lb], 'param', 'bce_aff')['ci_lo'] for lb in DECISION_CKPTS)
        L.append(f"Probe BCE λ=0.4 am hon ro (CI khong chong, ca hai seed Static) o muc tham so: "
                 f"**{'co' if stronger else 'khong'}** → "
                 + ("dien giai 'affinity da thoa hiep voi BCE trong huan luyen' — ung ho Phan D."
                    if stronger else 'khong co bang chung thoa hiep tu probe.'))
        L.append('')
    L.append('**AffOnly vs Static — cap (Region, Aff):**')
    L.append('')
    L.append('| Checkpoint | Muc tham so | Muc dac trung toan anh |')
    L.append('|---|---|---|')
    for lb in ('affonly_s19_40k', 'affonly_s86_40k', 'static_s19_36k', 'static_s86_36k', 'baseline_40k'):
        if lb in dfs:
            L.append(f"| {lb} | {_ci_txt(_row(dfs[lb], 'param', 'region_aff'))} | "
                     f"{_ci_txt(_row(dfs[lb], 'feat', 'region_aff'))} |")
    L.append('')
    pairs = [('affonly_s19_40k', 'static_s19_36k'), ('affonly_s86_40k', 'static_s86_36k')]
    for a, s in pairs:
        if a in dfs and s in dfs:
            ra, rs = _row(dfs[a], 'param', 'region_aff'), _row(dfs[s], 'param', 'region_aff')
            if ra is not None and rs is not None:
                flip = np.sign(ra['value']) != np.sign(rs['value'])
                L.append(f"- {a} → {s}: dau (muc tham so) {'DOI' if flip else 'giu nguyen'} khi co BCE"
                         + (' → co che ung vien cho Nhanh R cua Run 7.' if flip else '.'))
    L.append('')
    if 'static_s86_36k' in dfs and 'static_s86_40k' in dfs:
        a = dfs['static_s86_36k'].set_index(['level', 'pair', 'axis', 'stratum'])['confirmed_single_ckpt']
        b = dfs['static_s86_40k'].set_index(['level', 'pair', 'axis', 'stratum'])['confirmed_single_ckpt']
        common = a.index.intersection(b.index)
        diff = [k for k in common if bool(a[k]) != bool(b[k])]
        L.append(f"**Static s86 @36k vs @40k:** {len(diff)} (muc, cap, truc, tang) doi ket luan xac nhan giua hai moc"
                 + (' → CANH BAO do on dinh (khong chon moc co loi).' if diff else '.'))
        for k in diff[:20]:
            L.append(f"  - {k}: 36k={'xac nhan' if a[k] else 'khong'}, 40k={'xac nhan' if b[k] else 'khong'}")
    return L


# ═══════════════════════════════ CLI ══════════════════════════════════════════

def resolve_device_det(device_arg):
    from Tools.oracle_boundary_ceiling import resolve_device
    set_deterministic()
    return resolve_device(device_arg)


def parse_common(args):
    ckpts = parse_label_values(args.ckpt, '--ckpt')
    check_labels_known(ckpts, '--ckpt')
    for lb, p in ckpts.items():
        if not os.path.exists(p):
            raise SystemExit(f"[{lb}] khong thay file checkpoint: {p}")
    cfg_over = parse_label_values(args.config, '--config')
    dump = parse_label_values(args.dump_dir, '--dump-dir')
    expected = parse_label_values(args.expected_miou, '--expected-miou', float)
    raw_pw = parse_label_values(args.pos_weight, '--pos-weight')
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = {lb: load_cfg_for(lb, cfg_over.get(lb), args.data_root) for lb in ckpts}
    pw, src = resolve_pos_weights(list(ckpts), raw_pw, cfg, args.output_dir)
    return ckpts, {'cfg': cfg, 'dump': dump, 'expected': expected, 'pos_weight': pw, 'pos_src': src}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def ckpt_args(p):
        p.add_argument('--ckpt', action='append', required=True, metavar='LABEL=PATH',
                       help=f'Checkpoint .pth; lap lai. LABEL thuoc: {", ".join(CHECKPOINTS)}')
        p.add_argument('--pos-weight', action='append', default=[], metavar='LABEL=VALUE|auto',
                       help='pos_weight BCE trong log train (bat buoc cho ckpt co BCE); auto = uoc luong lai')
        p.add_argument('--dump-dir', action='append', default=[], metavar='LABEL=DIR',
                       help='Thu muc dump cu (masks/ + per_image_stats.csv) — cong 7, va cong 1 khi spec khong co hang so')
        p.add_argument('--expected-miou', action='append', default=[], metavar='LABEL=VALUE')
        p.add_argument('--config', action='append', default=[], metavar='LABEL=YAML',
                       help='Ghi de config mac dinh cua bang checkpoint (hiem khi can)')
        p.add_argument('--data-root', default=None)
        p.add_argument('--output-dir', default='output/grad_conflict')
        p.add_argument('--device', default=None)
        p.add_argument('--num-workers', type=int, default=2)
        p.add_argument('--gt-cache', default='output/_cache/gt1024', help="cache GT .npy ('' = tat)")
        p.add_argument('--max-images', type=int, default=None, help='dry-run: chi N anh dau')
        p.add_argument('--n-perm', type=int, default=20)
        p.add_argument('--force', action='store_true')

    p = sub.add_parser('gates', help='Cong 1,2,3,4,5,7')
    ckpt_args(p)
    p.add_argument('--continue-on-fail', action='store_true', help='Khong dung o checkpoint FAIL dau tien')

    p = sub.add_parser('measure', help='Do Phan A')
    ckpt_args(p)
    p.add_argument('--batch-size', type=int, default=8, help='Batch muc tham so (4 x 2 GPU luc train)')
    p.add_argument('--scratch-dir', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'grad_conflict_scratch')
                   if os.name != 'nt' else os.path.join(os.environ.get('TEMP', '.'), 'grad_conflict_scratch'))
    p.add_argument('--keep-scratch', action='store_true')
    p.add_argument('--skip-gate-check', action='store_true')
    p.add_argument('--figures-json', default=None)

    p = sub.add_parser('summarize', help='Gop + bang 5.2')
    p.add_argument('--output-dir', default='output/grad_conflict')
    p.add_argument('--holes-dir', default='output/spurious_holes')
    p.add_argument('--n-boot', type=int, default=10000)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    {'gates': cmd_gates, 'measure': cmd_measure, 'summarize': cmd_summarize}[args.cmd](args)


if __name__ == '__main__':
    main()
