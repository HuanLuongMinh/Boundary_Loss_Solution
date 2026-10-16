"""Tools/refine_common.py — Phan dung chung cho docs/S4-spec-P0-R0-ban-giao-claude-code.md
(P0: JSON hau ky run Static chay lai; R0: P-CRF).

Chua:
  - REFINE_CKPTS: 5 checkpoint cua R0 (§3.1) + 2 snapshot cua P0 (§2.1) — thuoc tinh lay tu
    CHECKPOINTS / PROBE_EXTRA cua Tools/grad_conflict_common.py (KHONG dinh nghia lai).
  - load_model(): nap state_dict (checkpoint thuong hoac snapshot Phan D) vao dung class model,
    strict=True, model.eval(), fp32.
  - forward_softmax(): forward 1 batch -> softmax fp32 (C,H,W) tung anh.
  - load_rgb_1024(): anh RGB goc uint8, resize 1024 (CUNG phep resize cua get_val_transforms, chua chuan hoa)
    — dung cho kernel song phuong cua CRF.
  - bounded_imap(): imap co gioi han so task dang bay (softmax 1024x1024x9 fp32 = 37.7 MB/anh —
    Pool.imap thuong se nap het generator vao RAM).

Khong sua src/ hay cong cu danh gia — chi import.
"""

import os
import sys
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import CHECKPOINTS, CFG_STATIC, NUM_CLASSES  # noqa: E402

# ─────────────────────────────── Checkpoint ───────────────────────────────────

# §3.1 (R0). 'static_s19r_36k' = snapshot Phan D @36000 (§2.1), dung chung voi P0.
R0_LABELS = ('static_s19_36k', 'static_s86_36k', 'baseline_40k', 'bce04_40k', 'static_s19r_36k')
DECISION_LABELS = ('static_s19_36k', 'static_s86_36k')            # §3.6
P0_LABELS = ('static_s19r_36k', 'static_s19r_40k')                # §2.1

# §2.1 — fingerprint (kich thuoc + sha1 4 MB dau + 4 MB cuoi, = grad_conflict_common.file_fingerprint)
P0_SNAPSHOTS = {
    'static_s19r_36k': dict(file='iter_36000.pth', iter=36000,
                            fingerprint='47002106:cf74b5adfb7a29371263482fc74e3768dfb67150', csv_miou9=0.6404),
    'static_s19r_40k': dict(file='iter_40000.pth', iter=40000,
                            fingerprint='47002106:d363a0a36dac08c2b74df13769dbbad49fb4df7d', csv_miou9=0.6548),
}
P0_CONFIG_MD5 = 'd7985f64d90402dda025a7bd702b719e'


def ckpt_info(label):
    """Thuoc tinh 1 nhan: model_type, config, default_file (None voi snapshot), expected_miou9, iter, seed."""
    if label in CHECKPOINTS:
        c = CHECKPOINTS[label]
        return dict(model_type=c['model_type'], config=c['config'], default_file=c['default_file'],
                     expected_miou9=c['expected_miou9'], iter=c['iter'], seed=c['seed'], snapshot=False)
    if label in P0_SNAPSHOTS:
        s = P0_SNAPSHOTS[label]
        return dict(model_type='static_boundary', config=CFG_STATIC, default_file=None,
                    expected_miou9=s['csv_miou9'], iter=s['iter'], seed=19, snapshot=True,
                    snapshot_file=s['file'], fingerprint=s['fingerprint'])
    raise KeyError(f"nhan khong biet: {label}")


def resolve_ckpt_path(label, ck_dir, snap_dir):
    info = ckpt_info(label)
    if info['snapshot']:
        if not snap_dir:
            raise SystemExit(f"[{label}] can --snapshot-dir (thu muc snapshots/ cua Phan D)")
        return os.path.join(snap_dir, info['snapshot_file'])
    if not ck_dir:
        raise SystemExit(f"[{label}] can --ckpt-dir (dataset checkpoint)")
    return os.path.join(ck_dir, info['default_file'])


# ─────────────────────────────── Model ────────────────────────────────────────

def set_deterministic():
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_arg=None):
    from Tools.oracle_boundary_ceiling import resolve_device as _rd
    set_deterministic()
    return _rd(device_arg)


def load_model(label, path, cfg, device):
    """Nap checkpoint/snapshot vao DUNG class model (build_eval_model cua eval_boundary_metrics), strict=True."""
    import torch
    from Tools.eval_boundary_metrics import build_eval_model
    from src.utils.phanD_snapshots import load_snapshot_state
    info = ckpt_info(label)
    model = build_eval_model(cfg, info['model_type'])
    sd, _meta = load_snapshot_state(path)          # checkpoint thuong: tra lai nguyen state_dict
    model.load_state_dict(sd)
    model = model.to(device).float().eval()
    torch.set_grad_enabled(False)
    return model


def model_logits(model, model_type, images):
    out = model(images)
    if model_type == 'bce_edge':
        return out[0]
    if model_type == 'static_boundary':
        return out[0]
    return out


def forward_softmax(model, model_type, images):
    """images (B,3,H,W) da chuan hoa tren device -> (logits fp32 (B,C,H,W) tren device, softmax numpy (B,C,H,W) fp32)."""
    import torch
    with torch.no_grad():
        logits = model_logits(model, model_type, images.float()).float()
        prob = torch.softmax(logits, dim=1)
    return logits, prob.cpu().numpy().astype(np.float32, copy=False)


# ─────────────────────────────── Du lieu ──────────────────────────────────────

def load_rgb_1024(img_path, size=1024):
    """RGB goc uint8 (H,W,3), resize ve size x size bang CHINH A.Resize cua get_val_transforms
    (cung noi suy) nhung KHONG chuan hoa — dau vao kernel song phuong (§3.2)."""
    import albumentations as A
    import rasterio
    with rasterio.open(img_path) as src:
        image = src.read()
    image = np.transpose(image, (1, 2, 0))
    if image.shape[2] > 3:
        image = image[:, :, :3]
    image = image.astype(np.uint8)
    out = A.Resize(height=size, width=size)(image=image)['image']
    return np.ascontiguousarray(out.astype(np.uint8))


def dataset_items(ds, indices):
    """Lap (ten, anh chuan hoa tensor (3,H,W), GT numpy int64 (H,W), duong dan anh) theo indices."""
    for i in indices:
        img, mask = ds[i]
        path = ds.samples[i][0]
        yield os.path.splitext(os.path.basename(path))[0], img, mask.numpy().astype(np.int64), path


def iter_softmax(model, model_type, ds, indices, device, batch_size=2):
    """Forward theo batch -> lap (ten, softmax (C,H,W) fp32, GT (H,W) int64, duong dan anh)."""
    import torch
    buf = []

    def flush():
        imgs = torch.stack([b[1] for b in buf]).to(device)
        _logits, prob = forward_softmax(model, model_type, imgs)
        for k, (name, _img, gt, path) in enumerate(buf):
            yield name, prob[k], gt, path

    for item in dataset_items(ds, indices):
        buf.append(item)
        if len(buf) == batch_size:
            yield from flush()
            buf = []
    if buf:
        yield from flush()


# ─────────────────────────────── Song song ────────────────────────────────────

def bounded_imap(pool, fn, tasks, max_inflight):
    """Nhu pool.imap (giu thu tu) nhung chi giu toi da max_inflight task chua xong -> RAM co gioi han.
    pool=None -> chay tuan tu trong tien trinh hien tai."""
    if pool is None:
        for t in tasks:
            yield fn(t)
        return
    q = deque()
    for t in tasks:
        q.append(pool.apply_async(fn, (t,)))
        if len(q) >= max_inflight:
            yield q.popleft().get()
    while q:
        yield q.popleft().get()


def make_pool(workers):
    if workers <= 1:
        return None
    from multiprocessing import get_context
    return get_context('spawn').Pool(workers, initializer=_worker_init)


def _worker_init():
    import torch
    torch.set_num_threads(1)


def per_image_stats(pred, gt, name, boundary_distances):
    """Thong ke du 1 anh bang DUNG Tools/per_image_dump.compute_per_image_stats (connectivity 4, dilation 0,
    BF theta 2 px) — tu nhan du doan (pseudo-logits, argmax tra lai dung pred)."""
    import torch
    from Tools.per_image_dump import compute_per_image_stats
    from Tools.oracle_boundary_ceiling import labels_to_pseudo_logits
    from Tools.grad_conflict_common import CLASS_NAMES, IGNORE_INDEX
    rows, _ = compute_per_image_stats(labels_to_pseudo_logits(pred, NUM_CLASSES), torch.from_numpy(gt)[None],
                                      [name], CLASS_NAMES, IGNORE_INDEX, 4, 0, tuple(boundary_distances), 2)
    return rows[0]


def save_png(arr, path):
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint8), mode='L').save(path)


def file_md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
