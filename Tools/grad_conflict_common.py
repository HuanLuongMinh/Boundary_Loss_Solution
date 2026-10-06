"""Tools/grad_conflict_common.py — Phan dung chung cho
docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md (Phan A/B/C).

Chua:
  - CHECKPOINTS: bang checkpoint CO DINH (thuoc tinh cua tung run — config,
    model-type, iter, seed, trong so hieu dung, cap gradient hop le, mIoU-9 ky
    vong). Duong dan file .pth KHONG nam o day — truyen luc chay
    (--ckpt LABEL=path).
  - gt_components(): thanh phan GT 8-lien thong theo lop — DUNG CHUNG giua
    Phan A (per_component_small_*.csv) va Phan B (holes_*.csv, cot
    gt_comp_id) de Phan C ghep duoc theo comp_id.
  - gt_strata_maps(): ban do tang (khoang cach toi bien GT, lop, kich thuoc
    thanh phan) o do phan giai full-res roi ha ve stride 4 bang NEAREST (cung
    quy uoc AffinityLoss).
  - Tien ich nho: parse LABEL=value, git commit, hash file, ghi JSON.

Khong sua bat ky file nao trong src/ — chi import.
"""

import hashlib
import json
import os
import subprocess

import numpy as np
from scipy.ndimage import distance_transform_edt, label as cc_label

# ─────────────────────────── Hang so cua spec ────────────────────────────────

CLASS_NAMES = ['Background', 'Bareland', 'Rangeland', 'Developed',
               'Road', 'Tree', 'Water', 'Agriculture', 'Building']
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255
SEED = 19

# Muc 2.3 — tang khoang cach toi bien GT (px full-res): [0,4] (4,8] (8,16] (16,32] >32
DIST_EDGES = (4, 8, 16, 32)
DIST_STRATA = ['d0-4', 'd4-8', 'd8-16', 'd16-32', 'd>32']
# Muc 2.3 — kich thuoc thanh phan GT (dien tich full-res): nho <4096, vua 4096-65536, lon >65536
SIZE_SMALL_MAX = 4096
SIZE_LARGE_MIN = 65536
SIZE_STRATA = ['small', 'medium', 'large']
AXES = {
    'all': ['all'],
    'dist': DIST_STRATA,
    'class': CLASS_NAMES,
    'size': SIZE_STRATA,
}

TERMS = ('region', 'bce', 'aff')
ALL_PAIRS = (('bce', 'aff'), ('region', 'aff'), ('region', 'bce'))


def pair_name(pair):
    return f'{pair[0]}_{pair[1]}'


CFG_STATIC = 'configs/unet_former_resnet18_static_boundary/static_boundary.yaml'
CFG_BCE = 'configs/unet_former_resnet18_bce_edge/bce_edge.yaml'
CFG_AFFONLY = 'configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml'
CFG_BASELINE = 'configs/unet_former_resnet18_combineLoss/baseline.yaml'

# Trong so hieu dung (muc 2.1 spec): w_region=1, w_bce=alpha*lambda1, w_aff=alpha*lambda2.
# 'trained_total' mo ta L_total THAT luc train (dung cho cong 2) — so hang probe
# (affinity chua tung train) KHONG nam trong trained_total.
CHECKPOINTS = {
    'static_s19_36k': dict(
        default_file='best_model_Static_S19.pth', model_type='static_boundary', config=CFG_STATIC,
        iter=36000, seed=19, w_bce=0.4, w_aff=0.4, has_bce=True, aff_probe=False,
        pairs=[('bce', 'aff'), ('region', 'aff'), ('region', 'bce')],
        expected_miou9=0.6540486, role_a='A1 (bat buoc)', role_b='bat buoc',
        trained_total='static'),
    'static_s86_36k': dict(
        default_file='best_miou_Static_s86.pth', model_type='static_boundary', config=CFG_STATIC,
        iter=36000, seed=86, w_bce=0.4, w_aff=0.4, has_bce=True, aff_probe=False,
        pairs=[('bce', 'aff'), ('region', 'aff'), ('region', 'bce')],
        expected_miou9=0.6557, role_a='A1 (bat buoc)', role_b='bat buoc',
        trained_total='static'),
    'static_s86_40k': dict(
        default_file='final_iter40000_static_s86.pth', model_type='static_boundary', config=CFG_STATIC,
        iter=40000, seed=86, w_bce=0.4, w_aff=0.4, has_bce=True, aff_probe=False,
        pairs=[('bce', 'aff'), ('region', 'aff'), ('region', 'bce')],
        expected_miou9=None, role_a='A2 (nen co)', role_b='nen co',
        trained_total='static'),
    'bce04_40k': dict(
        default_file='best_model_BCE_04.pth', model_type='bce_edge', config=CFG_BCE,
        iter=40000, seed=19, w_bce=0.4, w_aff=0.4, has_bce=True, aff_probe=True,
        pairs=[('bce', 'aff'), ('region', 'aff'), ('region', 'bce')],
        expected_miou9=0.6566, role_a='A2 (nen co) — probe', role_b='bat buoc',
        trained_total='bce'),
    'bce02_40k': dict(
        default_file='best_model_BCE_02.pth', model_type='bce_edge', config=CFG_BCE,
        iter=40000, seed=19, w_bce=0.2, w_aff=0.4, has_bce=True, aff_probe=True,
        pairs=[('bce', 'aff'), ('region', 'aff'), ('region', 'bce')],
        expected_miou9=0.6490, role_a='bo sung (ngoai spec) — probe', role_b='bo sung',
        trained_total='bce'),
    'affonly_s19_40k': dict(
        default_file='best_miou_aff_s19.pth', model_type='baseline', config=CFG_AFFONLY,
        iter=40000, seed=19, w_bce=None, w_aff=0.4, has_bce=False, aff_probe=False,
        pairs=[('region', 'aff')],
        expected_miou9=None, role_a='A2 (nen co) — chi (Region, Aff)', role_b='bat buoc',
        trained_total='affonly'),
    'affonly_s86_40k': dict(
        default_file='best_miou_aff_s86.pth', model_type='baseline', config=CFG_AFFONLY,
        iter=40000, seed=86, w_bce=None, w_aff=0.4, has_bce=False, aff_probe=False,
        pairs=[('region', 'aff')],
        expected_miou9=None, role_a='A2 (nen co) — chi (Region, Aff)', role_b='bat buoc',
        trained_total='affonly'),
    'baseline_40k': dict(
        default_file='best_model_baseline.pth', model_type='baseline', config=CFG_BASELINE,
        iter=40000, seed=19, w_bce=None, w_aff=0.4, has_bce=False, aff_probe=True,
        pairs=[('region', 'aff')],
        expected_miou9=0.6550926, role_a='A3 (tuy chon) — probe (Region, Aff)', role_b='bat buoc (moc)',
        trained_total='baseline'),
}

# Checkpoint spec co nhac nhung KHONG co file (Checkpoint_summary.txt) — bao cao ghi "khong co".
MISSING_RUNS = {
    'run3b_40k': 'Run 3b (alpha_aff 0.2) — khong co checkpoint',
    'run4_40k': 'Run 4 (dynamic) — khong co checkpoint (spec cung bo)',
}

DECISION_CKPTS = ('static_s19_36k', 'static_s86_36k')   # muc 5.1 dieu kien 5
FIGURE_CKPT = 'static_s19_36k'                          # muc 2.4


def terms_of(label):
    info = CHECKPOINTS[label]
    return ('region', 'bce', 'aff') if info['has_bce'] else ('region', 'aff')


def term_weights(label, overrides=None):
    info = CHECKPOINTS[label]
    w = {'region': 1.0, 'aff': info['w_aff']}
    if info['has_bce']:
        w['bce'] = info['w_bce']
    if overrides:
        w.update(overrides)
    return w


# ─────────────────────────── Thanh phan / tang GT ────────────────────────────

STRUCT_8 = np.ones((3, 3), dtype=bool)
STRUCT_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def gt_components(gt):
    """gt (H,W) nhan 0..8 (255 = ignore). Thanh phan 8-lien thong THEO LOP.

    Tra ve (comp_map, table):
      comp_map (H,W) int32, 0 = khong thuoc thanh phan nao (ignore), id >= 1
        duy nhat trong anh. Thu tu id: lop 0..8, trong moi lop theo thu tu
        scipy.ndimage.label — tat dinh.
      table: dict comp_id -> (class_id, area_fullres).
    """
    comp_map = np.zeros(gt.shape, dtype=np.int32)
    table = {}
    offset = 0
    for c in range(NUM_CLASSES):
        lab, n = cc_label(gt == c, structure=STRUCT_8)
        if n == 0:
            continue
        areas = np.bincount(lab.ravel(), minlength=n + 1)
        sel = lab > 0
        comp_map[sel] = lab[sel] + offset
        for j in range(1, n + 1):
            table[offset + j] = (c, int(areas[j]))
        offset += n
    return comp_map, table


def size_stratum_index(area):
    if area < SIZE_SMALL_MAX:
        return 0
    if area <= SIZE_LARGE_MIN:
        return 1
    return 2


def gt_edge_np(gt, ignore_index=IGNORE_INDEX):
    """Bien GT 4-lien thong — CUNG dinh nghia extract_edge_gt (src/losses/boundary_bce.py):
    pixel valid co it nhat 1 hang xom 4-lien thong valid mang nhan khac."""
    import torch
    from src.losses.boundary_bce import extract_edge_gt
    edge, _ = extract_edge_gt(torch.from_numpy(gt.astype(np.int64))[None],
                              ignore_index=ignore_index, connectivity=4, dilation_radius=0)
    return edge[0, 0].numpy().astype(bool)


def downsample_nearest(arr, out_hw):
    """Ha mau NEAREST dung quy uoc F.interpolate(mode='nearest') ma AffinityLoss dung."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(arr)).to(torch.float64)[None, None]
    out = F.interpolate(t, size=out_hw, mode='nearest')[0, 0].numpy()
    return out.astype(arr.dtype)


def gt_strata_maps(gt, out_hw):
    """Tinh ban do tang tren GT full-res roi ha ve out_hw (stride 4) bang nearest.

    Tra ve dict (moi gia tri (h,w) int64, -1 = loai — pixel ignore):
      'all', 'dist', 'class', 'size', 'comp' (comp_id, 0 = khong co),
    va 'comp_table' (dict comp_id -> (class, area)).
    """
    valid = gt != IGNORE_INDEX
    edge = gt_edge_np(gt)
    if edge.any():
        dist = distance_transform_edt(~edge)
    else:
        dist = np.full(gt.shape, np.inf)
    dist_idx = np.digitize(dist, DIST_EDGES, right=True).astype(np.int64)   # 0..4

    comp_map, table = gt_components(gt)
    area_map = np.zeros(gt.shape, dtype=np.int64)
    if table:
        areas = np.zeros(max(table) + 1, dtype=np.int64)
        for cid, (_c, a) in table.items():
            areas[cid] = a
        area_map = areas[comp_map]
    size_idx = np.where(area_map < SIZE_SMALL_MAX, 0, np.where(area_map <= SIZE_LARGE_MIN, 1, 2))

    class_idx = gt.astype(np.int64)
    all_idx = np.zeros(gt.shape, dtype=np.int64)
    out = {}
    for name, arr in (('all', all_idx), ('dist', dist_idx), ('class', class_idx), ('size', size_idx)):
        arr = np.where(valid, arr, -1)
        out[name] = downsample_nearest(arr.astype(np.int64), out_hw)
    out['comp'] = downsample_nearest(comp_map.astype(np.int64), out_hw)
    out['comp_table'] = table
    return out


# ─────────────────────────────── Tien ich ─────────────────────────────────────

def parse_label_values(items, flag_name, cast=str):
    """['a=1', 'b=2'] -> {'a': cast('1'), ...}. Bao loi ro neu sai dinh dang."""
    out = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"{flag_name} phai co dang LABEL=gia_tri, nhan: '{item}'")
        k, v = item.split('=', 1)
        k = k.strip()
        if k in out:
            raise ValueError(f"{flag_name}: nhan '{k}' bi truyen 2 lan")
        out[k] = cast(v.strip())
    return out


def check_labels_known(labels, flag_name):
    unknown = [lb for lb in labels if lb not in CHECKPOINTS]
    if unknown:
        raise ValueError(f"{flag_name}: nhan khong co trong bang checkpoint cua spec: {unknown}. "
                         f"Nhan hop le: {list(CHECKPOINTS)}")


def get_git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        return 'unknown'


def file_fingerprint(path):
    """Kich thuoc + sha1 cua 4MB dau va 4MB cuoi — du de nhan dien file ckpt ma khong doc het."""
    st = os.stat(path)
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        h.update(f.read(4 << 20))
        if st.st_size > (8 << 20):
            f.seek(-(4 << 20), os.SEEK_END)
            h.update(f.read(4 << 20))
    return f"{st.st_size}:{h.hexdigest()}"


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def read_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def append_log(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(text.rstrip('\n') + '\n')


def load_gt_cached(val_ds, idx, cache_dir):
    """GT 1024 (sau get_val_transforms — Resize nearest cho mask) cua anh idx, cache .npy.
    Tra ve (image_tensor_or_None, gt uint8/int). Doc lai tu cache neu co (chi GT)."""
    name = os.path.splitext(os.path.basename(val_ds.samples[idx][0]))[0]
    path = os.path.join(cache_dir, f'{name}.npy') if cache_dir else None
    if path and os.path.exists(path):
        return name, np.load(path)
    _img, mask = val_ds[idx]
    gt = mask.numpy() if hasattr(mask, 'numpy') else np.asarray(mask)
    gt = gt.astype(np.int64)
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(path, gt.astype(np.uint8) if gt.max() < 256 else gt)
    return name, gt
