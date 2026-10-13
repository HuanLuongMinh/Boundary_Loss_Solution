"""src/utils/phanD_snapshots.py — Tien ich cho Phan D
(docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md §1.1, cong D1/D2/D5).

Dung chung boi src/train_static_boundary_weighted.py va src/train_static_boundary.py.
MOI tinh nang chi bat qua co CLI (mac dinh OFF) — khong truyen co thi train y het cu.

  SnapshotSaver  — torch.save(model.module.state_dict()) tai cac moc cua lich, CHI rank 0,
                   khong barrier, khong forward/backward, khong goi ham ngau nhien.
  LossLogger     — rank 0 ghi CSV loss tung iter (gia tri da la float san trong `parts`).
  verify_snapshot_load — cong D2: nap snapshot vao model MOI, so trung bit voi model trong bo nho.
"""

import csv
import hashlib
import os
import time

import torch

# §1.1 — 35 moc: {0,250,...,1000} U {1500,...,8000 buoc 500} U {10000,...,40000 buoc 2000}
SCHEDULE_DEFAULT = sorted(set(range(0, 1001, 250)) | set(range(1500, 8001, 500)) | set(range(10000, 40001, 2000)))
# D5 — lich gian: {0-1000 buoc 250; 1000-8000 buoc 1000; 8000-40000 buoc 4000}
SCHEDULE_SPARSE = sorted(set(range(0, 1001, 250)) | set(range(1000, 8001, 1000)) | set(range(8000, 40001, 4000)))


def parse_schedule(spec):
    """'default' | 'sparse' | '0,250,500' -> list int tang dan. None/'' -> None (tat)."""
    if not spec:
        return None
    if spec == 'default':
        return list(SCHEDULE_DEFAULT)
    if spec == 'sparse':
        return list(SCHEDULE_SPARSE)
    return sorted({int(x) for x in spec.split(',') if x.strip()})


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def snapshot_path(snapshot_dir, t):
    return os.path.join(snapshot_dir, f'iter_{t}.pth')


class SnapshotSaver:
    """Moc 0 = trong so TRUOC buoc optimizer dau tien; moc t > 0 = SAU buoc optimizer cua iter t."""

    def __init__(self, schedule, snapshot_dir, seed, config_md5, is_main):
        self.schedule = set(schedule)
        self.dir = snapshot_dir
        self.seed = seed
        self.config_md5 = config_md5
        self.is_main = is_main
        if is_main:
            os.makedirs(snapshot_dir, exist_ok=True)

    def prune_after(self, last_iter):
        """Resume tu latest_checkpoint (iter = last_iter): snapshot co moc > last_iter thuoc quy dao CU
        (bi ngat) — xoa de khong tron 2 quy dao; chung se duoc luu lai tren quy dao resume."""
        if not self.is_main:
            return []
        removed = []
        for t in sorted(self.schedule):
            p = snapshot_path(self.dir, t)
            if t > last_iter and os.path.exists(p):
                os.remove(p)
                removed.append(t)
        return removed

    def maybe_save(self, module, t):
        if not self.is_main or t not in self.schedule:
            return None
        path = snapshot_path(self.dir, t)
        if os.path.exists(path):          # resume: moc da luu thi giu nguyen
            return path
        payload = {'state_dict': module.state_dict(), 'iter': t, 'seed': self.seed,
                   'config_md5': self.config_md5}
        tmp = path + '.tmp'
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return path


class LossLogger:
    """CSV: iter, wall_s, lr, l_region, l_bce, l_affinity, l_total (rank 0, batch cuc bo cua rank 0)."""

    FIELDS = ['iter', 'wall_s', 'lr', 'l_region', 'l_bce', 'l_affinity', 'l_total']

    def __init__(self, path, is_main):
        self.is_main = is_main
        self.f = None
        if is_main:
            os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
            self.f = open(path, 'w', newline='', encoding='utf-8')
            self.w = csv.writer(self.f)
            self.w.writerow(self.FIELDS)
            self.t0 = time.time()

    def log(self, iteration, lr, parts):
        if self.f is None:
            return
        self.w.writerow([iteration, f'{time.time() - self.t0:.3f}', repr(float(lr))] +
                        [repr(float(parts[k])) for k in ('l_region', 'l_bce', 'l_affinity', 'l_total')])
        if iteration % 50 == 0:
            self.f.flush()

    def close(self):
        if self.f is not None:
            self.f.close()
            self.f = None


def _transform_of(dataset):
    while hasattr(dataset, 'dataset') and not hasattr(dataset, 'transform'):   # Subset (dry-run)
        dataset = dataset.dataset
    return getattr(dataset, 'transform', None)


def augment_worker_init(worker_id):
    """--seed-augment: albumentations 2.x KHONG dung seed toan cuc (random/np/torch) — moi Compose tu lay
    entropy cua OS, nen 2 lan chay cung seed van crop/flip khac nhau. Ham nay (worker_init_fn) seed
    transform cua worker tu torch.initial_seed() — gia tri PyTorch da dat cho worker = base_seed + worker_id,
    voi base_seed rut tu torch RNG cua tien trinh chinh => tat dinh theo SEED, khac nhau giua worker/epoch.
    Chinh sach augmentation KHONG doi."""
    from torch.utils.data import get_worker_info
    info = get_worker_info()
    t = _transform_of(info.dataset)
    if t is not None and hasattr(t, 'set_random_seed'):
        t.set_random_seed(torch.initial_seed() % (2 ** 32))


def seed_augment_main_process(dataset, seed):
    """--seed-augment voi num_workers=0 (transform chay o tien trinh chinh): seed 1 lan tu SEED."""
    t = _transform_of(dataset)
    if t is not None and hasattr(t, 'set_random_seed'):
        t.set_random_seed(int(seed))


def load_snapshot_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'state_dict' in obj and 'iter' in obj:
        return obj['state_dict'], {k: v for k, v in obj.items() if k != 'state_dict'}
    return obj, {}


def verify_snapshot_load(path, build_fresh, live_module):
    """Cong D2: nap `path` vao model MOI (build_fresh()), so TRUNG BIT moi tham so + buffer
    voi `live_module` (model trong bo nho cua run ON). Tra ve dict ket qua."""
    sd, meta = load_snapshot_state(path)
    fresh = build_fresh()
    fresh.load_state_dict(sd, strict=True)
    live = live_module.state_dict()
    mine = fresh.state_dict()
    mismatched = []
    for k, v in live.items():
        a = v.detach().cpu()
        b = mine[k].detach().cpu()
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            mismatched.append(k)
    missing = sorted(set(live) ^ set(mine))
    ok = not mismatched and not missing
    return {'gate': 'D2', 'status': 'PASS' if ok else 'FAIL', 'snapshot': path, 'meta': meta,
            'n_tensors': len(live), 'n_mismatched': len(mismatched), 'mismatched': mismatched[:20],
            'key_set_diff': missing[:20], 'threshold': 'trung bit (torch.equal) moi tham so + buffer'}
