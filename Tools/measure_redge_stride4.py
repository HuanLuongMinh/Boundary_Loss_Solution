"""Tools/measure_redge_stride4.py — do r_edge @ stride-4 PER-CLASS cho cong C1
cua Run 6 (docs/2-spec-do-r-edge-stride4-cong-C1.md).

Day la thong ke cua NHAN va cua duong tinh downsample, KHONG phai cua mo hinh:
  - KHONG load checkpoint nao
  - KHONG chay inference
  - CHI doc nhan ground-truth, chay extract_edge_gt (ham dung chung, import lai
    tu src/losses/boundary_bce.py — KHONG copy logic) voi 3 cach downsample
    khac nhau (V0/V1/V2, xem muc 1 cua spec), roi dem.

File nay DOC LAP — khong import bat ky script train_*.py nao (dung quy uoc cua
Tools/measure_edge_ratio.py), chi dung chung src/data (dataset/transforms,
khong sua) va src/losses/{boundary_bce,affinity}.py (chi doc, khong sua —
dung y muc 8 "KHONG DUOC DOI" cua spec).

Giao thuc EDGE_STATS (muc 0.1 + muc 8 cua spec, GIU NGUYEN):
  - TRAIN crops, 512x512, 800 crop = 100 batch/rank x 2 rank x batch_size 4
  - seed = 19, connectivity = 4, ignore_index = 255, dilation_radius = 0 (goc)
  - DistributedSampler(shuffle=True, seed=19) — mo phong 2 rank BANG CACH set
    lai global RNG (random/numpy/torch) truoc MOI rank, dung y het 2 tien
    trinh training that (moi rank tu set_seed(seed) mot lan luc khoi dong,
    KHONG phai 1 stream noi tiep) — xem ham iter_crop_batches().

Ba bien the V0/V1/V2 (muc 1 cua spec):
  - V0 (dang chay o affinity.py): mask -> nearest-downsample x4 -> extract_edge_gt
  - V1 (max-pool):                extract_edge_gt o full-res -> max-pool x4
  - V2 (max-pool + dilation):     nhu V1 nhung extract_edge_gt full-res dung
                                   dilation_radius=1 (tham so co san cua ham
                                   dung chung — KHONG can config
                                   AFFINITY_SAMPLING_DILATION moi, xem docstring
                                   AffinityLoss.__init__ ve viec luon truyen
                                   tuong minh dilation_radius)
Ca 3 bien the dung CHUNG 1 ban nhan da nearest-downsample (mask_ds) de xac
dinh LOP va VALIDITY o stride-4 — chi cach tinh edge_gt la khac nhau. Dinh
nghia nay duoc ghi lai nguyen van vao khoi "meta.definition" cua output JSON.

Usage (chay tren Kaggle, dataset chi mount o do — xem huong dan cuoi hoi thoai):
    python Tools/measure_redge_stride4.py \
        --data-root /kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap \
        --out-dir docs/results

    # Debug nhanh khong can dataset that (tu-kiem logic, KHONG phai so chinh thuc):
    python Tools/measure_redge_stride4.py --self-test
"""

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_train_transforms, get_val_transforms
import src.losses.boundary_bce as boundary_bce_mod
import src.losses.affinity as affinity_mod
from src.losses.boundary_bce import extract_edge_gt, compute_pos_weight

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = OpenEarthMapDataset.CLASSES  # 9 lop, thu tu co dinh cua dataset
NUM_CLASSES = len(CLASS_NAMES)

# So lieu da do duoc (muc 3.1 cua spec) — dung lam khoang chap nhan cua gate 3.1.
FULLRES_R_EDGE_EXPECTED = 0.0833
FULLRES_R_EDGE_TOL = 0.0006
FULLRES_POS_WEIGHT_EXPECTED = 11.00
FULLRES_POS_WEIGHT_TOL = 0.08
VAL_ORACLE_P_BAND_D0 = 0.0810
VAL_ORACLE_REL_TOL = 0.10

# Muc 4/5 cua spec — nguong "bao toan bien" va nguong quyet dinh gate.
PREDICTED_STRIDE4_FACTOR = 4.0
SEVERE_LOSS_FRACTION = 0.75  # < 75% muc bao toan = "tut sau"
G1_THRESHOLD = 0.25
G2_THRESHOLD = 0.30


# ───────────────────────── dataset path resolution ─────────────────────────
# Sao chep co y tu Tools/measure_edge_ratio.py (quy uoc "moi file tu chua",
# xem README.md) — khong import lai tu do de giu file nay doc lap.

def find_data_base(data_root: str) -> str:
    candidates = [
        data_root,
        os.path.join(data_root, 'OpenEarthMap_Mini'),
        os.path.join(data_root, 'OpenEarthMap_flat'),
        os.path.join(data_root, 'OpenEarthMap'),
        os.path.join(data_root, 'openearthmap'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'images', 'train')) or \
           os.path.isdir(os.path.join(c, 'images', 'val')):
            return c
    return data_root


def find_label_subdir(base: str) -> str:
    for name in ('labels', 'label'):
        if os.path.isdir(os.path.join(base, name)):
            return name
    return 'labels'


def resolve_root_and_mask_dir(root_dir: str, mask_dir: str):
    root_base = find_data_base(root_dir)
    if root_base != root_dir:
        print(f"Resolved ROOT_DIR: {root_dir} -> {root_base}")
    label_sub = find_label_subdir(root_base)
    head, _, tail = mask_dir.partition('/')
    if head in ('labels', 'label') and head != label_sub:
        fixed = label_sub + '/' + tail if tail else label_sub
        print(f"Resolved MASK_DIR: '{mask_dir}' -> '{fixed}'")
        mask_dir = fixed
    return root_base, mask_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return 'unknown'


# ───────────────────────── synthetic dataset (--self-test) ─────────────────
# Chi dung de tu-kiem logic script nay cuc bo, KHONG PHAI so chinh thuc cua
# cong C1 (xem canh bao in ra khi dung --self-test). Tao nhan tong hop co cau
# truc bien (stripe + duong mong 1-pixel gia lam Water/Road) de bai test co y
# nghia (khong toan random, se khong sinh bien nao ca).

class _SyntheticLabelDataset(Dataset):
    def __init__(self, n=64, size=512, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n
        self.masks = []
        band = size // NUM_CLASSES
        water_idx = CLASS_NAMES.index('Water')
        road_idx = CLASS_NAMES.index('Road')
        for _ in range(n):
            mask = np.zeros((size, size), dtype=np.int64)
            for c in range(NUM_CLASSES):
                mask[c * band:(c + 1) * band, :] = c
            idx = np.arange(size)
            water_row = (idx + rng.randint(0, size)) % size
            mask[water_row, idx] = water_idx
            road_col = (idx * 2 + rng.randint(0, size)) % size
            mask[idx, road_col] = road_idx
            self.masks.append(mask)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        mask = self.masks[idx]
        image = torch.zeros(3, mask.shape[0], mask.shape[1], dtype=torch.float32)
        return image, torch.from_numpy(mask).long()


# ──────────────────────────── crop sampling ─────────────────────────────────

def iter_crop_batches(train_ds, world_size: int, batch_size: int,
                       batches_per_rank: int, seed: int):
    """Sinh dung 800 crop (mac dinh) theo giao thuc EDGE_STATS: mo phong
    world_size rank, moi rank tu set_seed(seed) RIENG (giong het 1 tien trinh
    DDP moi tu seed lai) roi keo batches_per_rank batch dau tu
    DistributedSampler(shuffle=True, seed=seed) cua rank do."""
    for rank in range(world_size):
        set_seed(seed)
        sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                      shuffle=True, seed=seed)
        sampler.set_epoch(0)
        loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                             num_workers=0, drop_last=True)
        it = iter(loader)
        for _ in range(batches_per_rank):
            yield next(it)


# ──────────────────────────── core accumulation ─────────────────────────────

def new_state():
    return {
        'n_valid_full': 0.0, 'n_edge_full': 0.0,
        'n_valid_class': [0.0] * NUM_CLASSES,
        'n_edge_v0': [0.0] * NUM_CLASSES,
        'n_edge_v1': [0.0] * NUM_CLASSES,
        'n_edge_v2': [0.0] * NUM_CLASSES,
        'n_crops': 0,
    }


def process_batch(masks: torch.Tensor, ignore_index: int, connectivity: int,
                   dilation_v2: int, state: dict):
    """Cap nhat state (in-place) tu 1 batch mask (B,H,W). Tinh ca gate-3.1
    (full-res, dilation=0) LAN 3 bien the stride-4 trong 1 lan duyet — V1 tai
    su dung truc tiep edge_full0 (khong extract_edge_gt lai lan 2 cho V1)."""
    B, H, W = masks.shape
    assert H % 4 == 0 and W % 4 == 0, f"crop {H}x{W} khong chia het cho 4"
    Hf, Wf = H // 4, W // 4

    # Full-res, dilation=0 — dung cho gate 3.1 VA la nguon cua V1 (max-pool).
    edge_full0, valid_full = extract_edge_gt(masks, ignore_index, connectivity, dilation_radius=0)
    state['n_valid_full'] += valid_full.sum().item()
    state['n_edge_full'] += edge_full0.sum().item()

    # Nhan da nearest-downsample — dung chung cho LOP + VALIDITY o stride-4
    # cua CA 3 bien the (dung y het 2 dong dau AffinityLoss.forward()).
    mask_ds = F.interpolate(masks.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()
    valid_ds = (mask_ds != ignore_index)

    # V0 — dung dung ham+tham so affinity.py dang goi.
    edge_v0, _ = extract_edge_gt(mask_ds, ignore_index, connectivity, dilation_radius=0)

    # V1 — max-pool x4 cua edge_gt full-res (dilation=0), da co san o tren.
    edge_v1 = F.max_pool2d(edge_full0, kernel_size=4, stride=4)

    # V2 — full-res voi dilation_radius=dilation_v2, roi max-pool x4.
    edge_full2, _ = extract_edge_gt(masks, ignore_index, connectivity, dilation_radius=dilation_v2)
    edge_v2 = F.max_pool2d(edge_full2, kernel_size=4, stride=4)

    e0 = edge_v0.squeeze(1).bool() & valid_ds
    e1 = (edge_v1.squeeze(1) > 0.5) & valid_ds
    e2 = (edge_v2.squeeze(1) > 0.5) & valid_ds

    for ci in range(NUM_CLASSES):
        cm = (mask_ds == ci) & valid_ds
        nv = cm.sum().item()
        if nv == 0:
            continue
        state['n_valid_class'][ci] += nv
        state['n_edge_v0'][ci] += (cm & e0).sum().item()
        state['n_edge_v1'][ci] += (cm & e1).sum().item()
        state['n_edge_v2'][ci] += (cm & e2).sum().item()

    state['n_crops'] += B


# ──────────────────────────────── gates ──────────────────────────────────────

def run_gate_3_2(probe_masks: torch.Tensor, ignore_index: int, connectivity: int) -> dict:
    """Cong 3.2 — V0 phai la dung duong dang chay. Hai bang chung:
    (1) identity check: affinity.py.extract_edge_gt LA CUNG mot object ham voi
        boundary_bce.py.extract_edge_gt (khong phai ban cai lai);
    (2) chay that AffinityLoss.forward() tren 1 batch nhan that (fused_feature
        gia ngau nhien dung shape — KHONG can checkpoint/model that) va so
        n_affinity_centers voi phep tinh thu cong dung CHINH ham dung chung."""
    identity_ok = affinity_mod.extract_edge_gt is boundary_bce_mod.extract_edge_gt

    B, H, W = probe_masks.shape
    Hf, Wf = H // 4, W // 4
    loss_fn = affinity_mod.AffinityLoss(window_size=5, distance='cosine', margin=1.0,
                                         ignore_index=ignore_index, connectivity=connectivity,
                                         dilation_radius=0)
    fake_feat = torch.randn(B, 64, Hf, Wf)
    probe_loss = loss_fn(fake_feat, probe_masks)
    loss_finite = bool(torch.isfinite(probe_loss).item())

    mask_ds = F.interpolate(probe_masks.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()
    edge_gt_manual, _ = boundary_bce_mod.extract_edge_gt(mask_ds, ignore_index, connectivity, dilation_radius=0)
    n_centers_probe = int(edge_gt_manual.sum().item())

    return {
        'extract_edge_gt_identity': identity_ok,
        'probe_loss_finite': loss_finite,
        'probe_loss_value': float(probe_loss.item()),
        'probe_n_affinity_centers': n_centers_probe,
        'pass': bool(identity_ok and loss_finite),
    }


def run_gate_3_1(state: dict, w_max: float, allow_continue: bool) -> dict:
    n_valid_full = state['n_valid_full']
    n_edge_full = state['n_edge_full']
    r_edge_full = n_edge_full / max(n_valid_full, 1.0)
    n_nonedge_full = n_valid_full - n_edge_full
    pos_weight_raw = n_nonedge_full / max(n_edge_full, 1.0)
    pos_weight_clipped = compute_pos_weight(n_edge_full, n_nonedge_full, w_max)

    r_ok = abs(r_edge_full - FULLRES_R_EDGE_EXPECTED) <= FULLRES_R_EDGE_TOL
    pw_ok = abs(pos_weight_raw - FULLRES_POS_WEIGHT_EXPECTED) <= FULLRES_POS_WEIGHT_TOL
    passed = r_ok and pw_ok

    result = {
        'N_valid': n_valid_full, 'N_edge': n_edge_full, 'N_nonedge': n_nonedge_full,
        'r_edge': r_edge_full, 'pos_weight_raw': pos_weight_raw,
        'pos_weight_clipped': pos_weight_clipped,
        'expected_r_edge': FULLRES_R_EDGE_EXPECTED, 'expected_r_edge_tol': FULLRES_R_EDGE_TOL,
        'expected_pos_weight': FULLRES_POS_WEIGHT_EXPECTED, 'expected_pos_weight_tol': FULLRES_POS_WEIGHT_TOL,
        'pass': passed,
    }
    print(f"Gate 3.1: r_edge(full-res)={r_edge_full:.6f} (ky vong {FULLRES_R_EDGE_EXPECTED} +/- {FULLRES_R_EDGE_TOL}) "
          f"pos_weight={pos_weight_raw:.4f} (ky vong {FULLRES_POS_WEIGHT_EXPECTED} +/- {FULLRES_POS_WEIGHT_TOL}) "
          f"-> {'PASS' if passed else 'FAIL'}")
    if not passed and not allow_continue:
        print("Gate 3.1 FAIL — giao thuc lay mau da lech. DUNG, khong do tiep "
              "(dung --force de bo qua khi debug cuc bo, KHONG dung so ra tu do lam so chinh thuc).")
        sys.exit(1)
    elif not passed:
        print("CANH BAO: gate 3.1 FAIL nhung tiep tuc vi --force/--self-test — "
              "cac so stride-4 phia sau KHONG duoc coi la chinh thuc.")
    return result


def run_gate_3_3(state: dict, allow_continue: bool) -> dict:
    violations = []
    global_v1 = sum(state['n_edge_v1'])
    global_v2 = sum(state['n_edge_v2'])
    if global_v2 < global_v1:
        violations.append({'scope': 'global', 'v1': global_v1, 'v2': global_v2})
    for ci, cname in enumerate(CLASS_NAMES):
        v1c = state['n_edge_v1'][ci]
        v2c = state['n_edge_v2'][ci]
        if v2c < v1c:
            violations.append({'scope': cname, 'v1': v1c, 'v2': v2c})
    passed = len(violations) == 0
    print(f"Gate 3.3 (don dieu n_affinity_centers(V2) >= n_affinity_centers(V1)): "
          f"{'PASS' if passed else 'FAIL'} ({len(violations)} vi pham)")
    result = {'pass': passed, 'violations': violations}
    if not passed and not allow_continue:
        raise RuntimeError(f"Gate 3.3 vi pham (loi cai dat): {violations}")
    elif not passed:
        print(f"CANH BAO: gate 3.3 FAIL nhung tiep tuc vi --force/--self-test: {violations}")
    return result


def run_gate_3_4(val_ds, ignore_index: int, connectivity: int, batch_size: int) -> dict:
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    n_valid, n_edge = 0.0, 0.0
    n_images = 0
    for _, masks in loader:
        edge_gt, valid_mask = extract_edge_gt(masks, ignore_index, connectivity, dilation_radius=0)
        n_valid += valid_mask.sum().item()
        n_edge += edge_gt.sum().item()
        n_images += masks.shape[0]
    r_edge_val = n_edge / max(n_valid, 1.0)
    rel_diff = abs(r_edge_val - VAL_ORACLE_P_BAND_D0) / VAL_ORACLE_P_BAND_D0
    passed = rel_diff <= VAL_ORACLE_REL_TOL
    print(f"Gate 3.4 (val cross-check, {n_images} anh): r_edge(full-res, val)={r_edge_val:.6f} "
          f"vs oracle P_band@d=0={VAL_ORACLE_P_BAND_D0} (lech {rel_diff * 100:.1f}%) -> "
          f"{'PASS' if passed else 'CANH BAO (khong STOP)'}")
    return {
        'n_images': n_images, 'N_valid': n_valid, 'N_edge': n_edge, 'r_edge': r_edge_val,
        'oracle_p_band_d0': VAL_ORACLE_P_BAND_D0, 'rel_diff': rel_diff, 'pass': passed,
    }


# ─────────────────────────── section 4/5 (predictions + decision) ──────────

def per_class_r_edge(state: dict, key: str) -> dict:
    out = {}
    for ci, cname in enumerate(CLASS_NAMES):
        nv = state['n_valid_class'][ci]
        ne = state[key][ci]
        out[cname] = (ne / nv) if nv > 0 else 0.0
    return out


def build_predictions_section(gate31: dict, r_edge_v0_class: dict, r_edge_v0_global: float) -> dict:
    predicted_global = PREDICTED_STRIDE4_FACTOR * gate31['r_edge']
    severe_threshold = PREDICTED_STRIDE4_FACTOR * gate31['r_edge'] * SEVERE_LOSS_FRACTION
    severe_classes = [c for c, r in r_edge_v0_class.items() if r < severe_threshold]
    return {
        'predicted_r_edge_stride4': predicted_global,
        'severe_loss_threshold': severe_threshold,
        'actual_r_edge_stride4_global_v0': r_edge_v0_global,
        'classes_below_severe_threshold_v0': severe_classes,
        'water_v0': r_edge_v0_class.get('Water'),
        'road_v0': r_edge_v0_class.get('Road'),
    }


def build_decision_section(r_edge_v0_class: dict, r_edge_v1_class: dict, r_edge_v2_class: dict) -> dict:
    water_v0, road_v0 = r_edge_v0_class['Water'], r_edge_v0_class['Road']

    if water_v0 < G1_THRESHOLD and road_v0 < G1_THRESHOLD:
        base = 'G1'
    elif water_v0 >= G2_THRESHOLD and road_v0 >= G2_THRESHOLD:
        base = 'G2'
    else:
        base = 'G3'

    final = base
    chosen_variant = None
    reason = ''

    if base == 'G1':
        water_v1, road_v1 = r_edge_v1_class['Water'], r_edge_v1_class['Road']
        water_v2, road_v2 = r_edge_v2_class['Water'], r_edge_v2_class['Road']
        if water_v1 >= G2_THRESHOLD and road_v1 >= G2_THRESHOLD:
            final, chosen_variant = 'G1', 'V1'
            reason = 'V0: Water/Road < 0.25 (teo tin hieu duoc xac nhan). V1 dua ca hai ve >= 0.30 -> chay V1.'
        elif water_v2 >= G2_THRESHOLD and road_v2 >= G2_THRESHOLD:
            final, chosen_variant = 'G1', 'V2'
            reason = ('V0: Water/Road < 0.25. V1 KHONG du de dua ca hai ve >= 0.30, '
                       'V2 (max-pool+dilation) moi du -> chay V2, ghi ro can thiep HAI thanh phan.')
        else:
            final, chosen_variant = 'G3', None
            reason = ('V0: Water/Road < 0.25 nhung CA V1 lan V2 deu khong dua duoc ca hai ve >= 0.30 '
                       '-> khong ket luan duoc tu cong, ha xuong G3.')
    elif base == 'G2':
        reason = 'V0: Water va Road >= 0.30 -> chan doan teo tin hieu bi bac bo. HUY Run 6.'
    else:
        reason = 'V0: Water hoac Road roi vao bang 0.25-0.30 -> khong ket luan duoc tu cong nay.'

    return {
        'branch_from_v0': base, 'final_branch': final, 'recommended_variant': chosen_variant,
        'reason': reason, 'water_v0': water_v0, 'road_v0': road_v0,
        'g1_threshold': G1_THRESHOLD, 'g2_threshold': G2_THRESHOLD,
    }


# ──────────────────────────────── output ─────────────────────────────────────

def build_json_payload(args, gate31, gate32, gate33, gate34, state, predictions, decision) -> dict:
    variants = {}
    for vname, key in (('V0', 'n_edge_v0'), ('V1', 'n_edge_v1'), ('V2', 'n_edge_v2')):
        per_class = {}
        n_edge_total = 0.0
        for ci, cname in enumerate(CLASS_NAMES):
            nv = state['n_valid_class'][ci]
            ne = state[key][ci]
            per_class[cname] = {
                'r_edge': (ne / nv) if nv > 0 else 0.0,
                'n_affinity_centers': int(ne),
                'n_valid_px_stride4': int(nv),
            }
            n_edge_total += ne
        n_valid_total = sum(state['n_valid_class'])
        variants[vname] = {
            'r_edge_global': (n_edge_total / n_valid_total) if n_valid_total > 0 else 0.0,
            'n_affinity_centers_global': int(n_edge_total),
            'n_valid_px_stride4_global': int(n_valid_total),
            'per_class': per_class,
        }

    ratio_vs_v0 = {}
    for vname in ('V1', 'V2'):
        ratio_vs_v0[vname] = {}
        for cname in CLASS_NAMES:
            n0 = variants['V0']['per_class'][cname]['n_affinity_centers']
            nx = variants[vname]['per_class'][cname]['n_affinity_centers']
            ratio_vs_v0[vname][cname] = (nx / n0) if n0 > 0 else float('nan')

    return {
        'meta': {
            'measured_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'git_commit': get_git_commit(),
            'spec': 'docs/2-spec-do-r-edge-stride4-cong-C1.md',
            'no_checkpoint_no_gpu': True,
            'self_test': args.self_test,
            'protocol': {
                'split': 'train', 'crop_size': 512, 'world_size': args.world_size,
                'batches_per_rank': args.batches_per_rank, 'batch_size': args.batch_size,
                'n_crops': args.world_size * args.batches_per_rank * args.batch_size,
                'seed': args.seed,
                'sampling': 'DistributedSampler(shuffle=True, seed=seed) per rank giong het EDGE_STATS '
                             'trong train_dynamic_boundary.py/train_static_boundary.py, epoch=0, '
                             'global RNG (random/numpy/torch) duoc set_seed(seed) lai truoc MOI rank.',
            },
            'connectivity': args.connectivity, 'ignore_index': args.ignore_index,
            'dilation_radius_v0_v1': 0, 'dilation_radius_v2': args.dilation_v2,
            'class_names': CLASS_NAMES,
            'definition': (
                "r_edge stride-4 per-class: pixel trung tam hop le THUOC LOP c (theo nhan da "
                "nearest-downsample x4, mask_ds) VA co it nhat 1 hang xom 4-lien-thong hop le "
                "mang nhan khac (extract_edge_gt), chia cho tong pixel hop le thuoc lop c. "
                "LOP va VALIDITY o stride-4 luon lay tu mask_ds (dung chung ca 3 bien the) — "
                "chi cach tinh edge_gt khac nhau: "
                "V0 = extract_edge_gt(mask_ds, dilation_radius=0); "
                "V1 = max_pool4x( extract_edge_gt(mask_full_res, dilation_radius=0) ); "
                "V2 = max_pool4x( extract_edge_gt(mask_full_res, dilation_radius=1) ). "
                "n_affinity_centers = so pixel edge_gt=1 (da AND voi validity), theo lop cua mask_ds."
            ),
        },
        'gate_3_1_fullres_reproduction': gate31,
        'gate_3_2_v0_identity_check': gate32,
        'gate_3_3_monotonic': gate33,
        'gate_3_4_val_crosscheck': gate34,
        'variants': variants,
        'ratio_vs_v0': ratio_vs_v0,
        'predictions_section4': predictions,
        'decision_gate_section5': decision,
        'note_section6': (
            'Cong nay (C1) KHONG tu quyet Run 6. Ket qua G1/G2/G3 o tren phai gop voi cong C5b '
            '(oracle tren Static+BCE) theo bang gop 4 o o oracle-d-ket-qua-baseline-va-danh-gia.md '
            'muc 7 (file do khong ton tai trong repo checkout nay — chi trich lai bang quyet dinh '
            'tu docs/2-spec-do-r-edge-stride4-cong-C1.md muc 6): '
            'G1+Gap -> chay Run 6; G1+O_b -> van chay Run 6 (giu B song); '
            'G2+Gap -> khong chay Run 6 dang hien tai; G2+O_b -> HUY Run 6, chuyen sang (B) projection head.'
        ),
    }


def render_md(payload: dict) -> str:
    meta = payload['meta']
    lines = []
    lines.append('# r_edge @ stride-4 per-class — cong C1 cua Run 6')
    lines.append('')
    lines.append(f"Do luc: {meta['measured_at']}  |  git commit: `{meta['git_commit']}`"
                  f"{'  |  **SELF-TEST (khong phai so chinh thuc)**' if meta['self_test'] else ''}")
    lines.append('')
    p = meta['protocol']
    lines.append(f"Giao thuc: {p['n_crops']} crop {p['crop_size']}x{p['crop_size']} "
                  f"({p['batches_per_rank']} batch/rank x {p['world_size']} rank x batch_size {p['batch_size']}), "
                  f"seed={p['seed']}, connectivity={meta['connectivity']}, ignore_index={meta['ignore_index']}.")
    lines.append('')
    lines.append('## Dinh nghia da dung')
    lines.append('')
    lines.append(meta['definition'])
    lines.append('')

    lines.append('## Cong kiem (muc 3 cua spec)')
    lines.append('')
    g31 = payload['gate_3_1_fullres_reproduction']
    lines.append(f"- **3.1 Tai lap full-res**: r_edge={g31['r_edge']:.6f} "
                  f"(ky vong {g31['expected_r_edge']} +/- {g31['expected_r_edge_tol']}), "
                  f"pos_weight={g31['pos_weight_raw']:.4f} "
                  f"(ky vong {g31['expected_pos_weight']} +/- {g31['expected_pos_weight_tol']}) "
                  f"-> **{'PASS' if g31['pass'] else 'FAIL'}**")
    g32 = payload['gate_3_2_v0_identity_check']
    lines.append(f"- **3.2 V0 dung ham dang chay**: extract_edge_gt identity="
                  f"{g32['extract_edge_gt_identity']}, probe forward finite={g32['probe_loss_finite']} "
                  f"-> **{'PASS' if g32['pass'] else 'FAIL'}**")
    g33 = payload['gate_3_3_monotonic']
    lines.append(f"- **3.3 Don dieu V2>=V1**: {len(g33['violations'])} vi pham "
                  f"-> **{'PASS' if g33['pass'] else 'FAIL'}**")
    g34 = payload.get('gate_3_4_val_crosscheck')
    if g34:
        lines.append(f"- **3.4 Val cross-check** ({g34['n_images']} anh): r_edge={g34['r_edge']:.6f} "
                      f"vs oracle {g34['oracle_p_band_d0']} (lech {g34['rel_diff'] * 100:.1f}%) "
                      f"-> **{'PASS' if g34['pass'] else 'CANH BAO'}**")
    else:
        lines.append('- **3.4 Val cross-check**: bo qua (--skip-val)')
    lines.append('')

    lines.append('## Bang chinh — 9 lop x 3 bien the x 2 dai luong')
    lines.append('')
    header = '| Lop | ' + ' | '.join(f'{v} r_edge' for v in ('V0', 'V1', 'V2')) + \
             ' | ' + ' | '.join(f'{v} n_centers' for v in ('V0', 'V1', 'V2')) + \
             ' | n_valid_stride4 |'
    lines.append(header)
    lines.append('|' + '---|' * (1 + 3 + 3 + 1))
    for cname in meta['class_names']:
        row = [cname]
        for v in ('V0', 'V1', 'V2'):
            row.append(f"{payload['variants'][v]['per_class'][cname]['r_edge']:.4f}")
        for v in ('V0', 'V1', 'V2'):
            row.append(str(payload['variants'][v]['per_class'][cname]['n_affinity_centers']))
        row.append(str(payload['variants']['V0']['per_class'][cname]['n_valid_px_stride4']))
        lines.append('| ' + ' | '.join(row) + ' |')
    row = ['**TOAN CUC**']
    for v in ('V0', 'V1', 'V2'):
        row.append(f"**{payload['variants'][v]['r_edge_global']:.4f}**")
    for v in ('V0', 'V1', 'V2'):
        row.append(f"**{payload['variants'][v]['n_affinity_centers_global']}**")
    row.append(f"**{payload['variants']['V0']['n_valid_px_stride4_global']}**")
    lines.append('| ' + ' | '.join(row) + ' |')
    lines.append('')

    lines.append('## Ti le so V0 (n_affinity_centers)')
    lines.append('')
    lines.append('| Lop | V1/V0 | V2/V0 |')
    lines.append('|---|---|---|')
    for cname in meta['class_names']:
        r1 = payload['ratio_vs_v0']['V1'][cname]
        r2 = payload['ratio_vs_v0']['V2'][cname]
        lines.append(f"| {cname} | {r1:.3f} | {r2:.3f} |")
    lines.append('')

    pred = payload['predictions_section4']
    lines.append('## Du doan dinh luong (muc 4)')
    lines.append('')
    lines.append(f"Du doan (bao toan bien): r_edge@stride4 ~= 4 x r_edge_fullres = "
                  f"{pred['predicted_r_edge_stride4']:.4f}")
    lines.append(f"Thuc do (V0, toan cuc) = {pred['actual_r_edge_stride4_global_v0']:.4f}")
    lines.append(f"Water (V0) = {pred['water_v0']:.4f}  |  Road (V0) = {pred['road_v0']:.4f}")
    if pred['classes_below_severe_threshold_v0']:
        lines.append(f"Lop tut duoi nguong 'mat supervision' "
                      f"(< {pred['severe_loss_threshold']:.4f}): "
                      f"{', '.join(pred['classes_below_severe_threshold_v0'])}")
    else:
        lines.append("Khong co lop nao tut duoi nguong 'mat supervision'.")
    lines.append('')

    dec = payload['decision_gate_section5']
    lines.append('## Bang quyet dinh cong (muc 5) — doc tren V0, Water & Road')
    lines.append('')
    lines.append(f"Water(V0)={dec['water_v0']:.4f}  Road(V0)={dec['road_v0']:.4f}  "
                  f"(nguong G1 < {dec['g1_threshold']}, nguong G2 >= {dec['g2_threshold']})")
    lines.append('')
    variant_suffix = f" (chay {dec['recommended_variant']})" if dec['recommended_variant'] else ''
    lines.append(f"**Nhanh tu V0: {dec['branch_from_v0']}  ->  Ket luan cuoi: {dec['final_branch']}"
                  f"{variant_suffix}**")
    lines.append('')
    lines.append(dec['reason'])
    lines.append('')
    lines.append('## Ghi chu muc 6 — cong nay khong tu quyet Run 6')
    lines.append('')
    lines.append(payload['note_section6'])
    lines.append('')
    return '\n'.join(lines)


# ──────────────────────────────────── main ───────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-root', default='/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap')
    ap.add_argument('--train-img-dir', default='images/train')
    ap.add_argument('--train-mask-dir', default='labels/train')
    ap.add_argument('--val-root', default=None, help='Mac dinh = --data-root')
    ap.add_argument('--val-img-dir', default='images/val')
    ap.add_argument('--val-mask-dir', default='labels/val')
    ap.add_argument('--out-dir', default='docs/results')
    ap.add_argument('--skip-val', action='store_true', help='Bo qua cong 3.4 (tuy chon).')
    ap.add_argument('--seed', type=int, default=19, help='GIU NGUYEN = 19 cho so chinh thuc.')
    ap.add_argument('--batches-per-rank', type=int, default=100, help='GIU NGUYEN = 100 cho so chinh thuc.')
    ap.add_argument('--world-size', type=int, default=2, help='GIU NGUYEN = 2 cho so chinh thuc.')
    ap.add_argument('--batch-size', type=int, default=4, help='GIU NGUYEN = 4 cho so chinh thuc.')
    ap.add_argument('--connectivity', type=int, default=4, choices=[4, 8])
    ap.add_argument('--ignore-index', type=int, default=255)
    ap.add_argument('--dilation-v2', type=int, default=1, help='dilation_radius cua V2 (muc 1 spec).')
    ap.add_argument('--w-max', type=float, default=20.0)
    ap.add_argument('--val-batch-size', type=int, default=4)
    ap.add_argument('--force', action='store_true',
                     help='Bo qua STOP cua gate 3.1/3.3 — CHI de debug, khong dung so ra lam so chinh thuc.')
    ap.add_argument('--self-test', action='store_true',
                     help='Dung nhan TONG HOP (khong can dataset that) de tu-kiem logic script cuc bo. '
                          'KHONG PHAI so chinh thuc — tu dong bo qua STOP cua gate 3.1 va gate 3.4.')
    return ap.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    if args.self_test:
        print("=" * 70)
        print("SELF-TEST: dung nhan TONG HOP, khong phai dataset OpenEarthMap that.")
        print("So ra CHI de kiem logic pipeline (monotonic, shape, output json/md),")
        print("KHONG duoc dung lam so chinh thuc cua cong C1.")
        print("=" * 70)
        args.world_size = min(args.world_size, 2)
        args.batches_per_rank = min(args.batches_per_rank, 4)
        args.batch_size = min(args.batch_size, 4)
        train_ds = _SyntheticLabelDataset(n=64, size=512, seed=args.seed)
        val_ds = _SyntheticLabelDataset(n=16, size=1024, seed=args.seed + 1) if not args.skip_val else None
    else:
        root_dir, train_mask_dir = resolve_root_and_mask_dir(args.data_root, args.train_mask_dir)
        train_ds = OpenEarthMapDataset(root_dir=root_dir, img_dir=args.train_img_dir,
                                        mask_dir=train_mask_dir, transform=get_train_transforms())
        print(f"Train dataset: {len(train_ds)} anh — {root_dir}/{args.train_img_dir}")
        val_ds = None
        if not args.skip_val:
            val_root = args.val_root or args.data_root
            val_root, val_mask_dir = resolve_root_and_mask_dir(val_root, args.val_mask_dir)
            val_ds = OpenEarthMapDataset(root_dir=val_root, img_dir=args.val_img_dir,
                                          mask_dir=val_mask_dir, transform=get_val_transforms())
            print(f"Val dataset: {len(val_ds)} anh — {val_root}/{args.val_img_dir}")

    n_crops = args.world_size * args.batches_per_rank * args.batch_size
    print(f"Se lay {n_crops} crop = {args.batches_per_rank} batch/rank x {args.world_size} rank x "
          f"batch_size {args.batch_size}, seed={args.seed}")

    gen = iter_crop_batches(train_ds, args.world_size, args.batch_size, args.batches_per_rank, args.seed)
    first_images, first_masks = next(gen)

    print("Chay gate 3.2 (V0 dung ham dang chay) tren batch dau tien...")
    gate32 = run_gate_3_2(first_masks, args.ignore_index, args.connectivity)
    print(f"Gate 3.2: identity={gate32['extract_edge_gt_identity']} "
          f"probe_finite={gate32['probe_loss_finite']} -> {'PASS' if gate32['pass'] else 'FAIL'}")
    if not gate32['pass'] and not (args.force or args.self_test):
        print("Gate 3.2 FAIL — affinity.py khong dung dung extract_edge_gt dung chung. DUNG.")
        sys.exit(1)

    state = new_state()
    process_batch(first_masks, args.ignore_index, args.connectivity, args.dilation_v2, state)
    n_done = args.batch_size
    for _, masks in gen:
        process_batch(masks, args.ignore_index, args.connectivity, args.dilation_v2, state)
        n_done += masks.shape[0]
        if n_done % 200 == 0:
            print(f"  ... {n_done}/{n_crops} crop da xu ly ({time.time() - t0:.1f}s)")

    print(f"Da xu ly {state['n_crops']} crop.")

    allow_continue = args.force or args.self_test
    gate31 = run_gate_3_1(state, args.w_max, allow_continue)
    gate33 = run_gate_3_3(state, allow_continue)

    gate34 = None
    if val_ds is not None:
        gate34 = run_gate_3_4(val_ds, args.ignore_index, args.connectivity, args.val_batch_size)

    r_edge_v0_class = per_class_r_edge(state, 'n_edge_v0')
    r_edge_v1_class = per_class_r_edge(state, 'n_edge_v1')
    r_edge_v2_class = per_class_r_edge(state, 'n_edge_v2')
    n_valid_total = sum(state['n_valid_class'])
    n_edge_v0_total = sum(state['n_edge_v0'])
    r_edge_v0_global = (n_edge_v0_total / n_valid_total) if n_valid_total > 0 else 0.0

    predictions = build_predictions_section(gate31, r_edge_v0_class, r_edge_v0_global)
    decision = build_decision_section(r_edge_v0_class, r_edge_v1_class, r_edge_v2_class)

    payload = build_json_payload(args, gate31, gate32, gate33, gate34, state, predictions, decision)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, 'redge_stride4.json')
    md_path = os.path.join(args.out_dir, 'redge_stride4.md')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(render_md(payload))

    print('')
    print('=' * 70)
    print(f"Da ghi {json_path}")
    print(f"Da ghi {md_path}")
    variant_suffix = f" (chay {decision['recommended_variant']})" if decision['recommended_variant'] else ''
    print(f"Ket luan (muc 5): {decision['branch_from_v0']} -> {decision['final_branch']}{variant_suffix}")
    print(f"Tong thoi gian: {time.time() - t0:.1f}s")
    print('=' * 70)


if __name__ == '__main__':
    main()
