"""Tools/oracle_boundary_ceiling.py — Do tran tuyet doi cua moi loss thuan bien
(oracle-d), theo dung dinh nghia o docs/spec-oracle-d-tran-cai-thien-bien.md.

Voi moi checkpoint duoc truyen vao, script sua "hoan hao" toan bo pixel trong
dai rong d quanh bien GT (oracle_boundary_d) hoac NGOAI dai (oracle_interior_d),
roi do mIoU tang bao nhieu so voi khong sua gi (variant "none"). Con so nay la
tran cua moi loss thuan bien — khong cau hinh nao vuot qua duoc.

KHONG sua bat ky file da co (extract_edge_gt, SegmentationMetrics,
eval_boundary_metrics.py, visualizer.py) — file nay CHI IMPORT lai, dung dinh
nghia/pipeline cua chung nguyen ven (muc 9 cua spec).

So checkpoint chay trong 1 lan goi KHONG co dinh — moi checkpoint can du 5 co
lap lai: --checkpoint/--config/--model-type/--label/--iter (+ tuy chon
--expected-miou). So checkpoint N = so lan --checkpoint xuat hien tren dong
lenh; xem epilog --help hoac USAGE_EXAMPLE ben duoi de vi du day du.

Uu tien CPU tren Kaggle (dung ngan sach CPU cua spec) — mac dinh --device tu
chon giong het eval_boundary_metrics.py (dung cuda neu build PyTorch tuong
thich, tu roi ve cpu neu khong vd P100 sm_60); truyen thang --device cpu neu
muon ep CPU tuyet doi.

Vi du chay 3 checkpoint bat buoc cua spec (muc 1.1) — xem USAGE_EXAMPLE.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yaml
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader

# Cho phep chay `python Tools/oracle_boundary_ceiling.py` tu repo root ma
# khong can cai package — giong cach eval_boundary_metrics.py da lam.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_val_transforms
from src.losses.boundary_bce import extract_edge_gt
from src.utils.metrics import SegmentationMetrics
from src.utils.visualizer import mask_to_rgb, denormalize
from Tools.eval_boundary_metrics import resolve_dataset_paths, build_eval_model


USAGE_EXAMPLE = """
Vi du — chay dung 3 checkpoint bat buoc cua
docs/spec-oracle-d-tran-cai-thien-bien.md muc 1.1 (N tu suy ra tu so lan
--checkpoint, KHONG hard-code):

    python Tools/oracle_boundary_ceiling.py ^
        --checkpoint work_dirs/phase1/run1_baseline/best_model.pth ^
            --config configs/unet_former_resnet18_combineLoss/baseline.yaml ^
            --model-type baseline --label baseline --iter 40000 --expected-miou 0.6551 ^
        --checkpoint work_dirs/phase1/run2b_bce_lambda04/best_model.pth ^
            --config configs/unet_former_resnet18_bce_edge/bce_edge.yaml ^
            --model-type bce_edge --label bce_lambda04 --iter 40000 --expected-miou 0.6183 ^
        --checkpoint work_dirs/phase1/run3_static_alpha04/best_model.pth ^
            --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml ^
            --model-type static_boundary --label static_alpha04 --iter 36000 --expected-miou 0.6540 ^
        --data-root /kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap ^
        --device cpu

Muon chay 1, 2 hay 5 checkpoint thi lap/bot dung bay nhieu bo 5 co
(--checkpoint/--config/--model-type/--label/--iter[/--expected-miou]) theo
dung THU TU tuong ung 1 checkpoint — script tu dem N tu so lan --checkpoint.
--expected-miou co the bo han (bo qua cong 5.1 moi checkpoint) hoac truyen
"skip" cho tung checkpoint rieng le trong khi van kiem cho cac checkpoint khac.
"""


# ─────────────────────────── CLI / validate ────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=USAGE_EXAMPLE)
    ap.add_argument('--checkpoint', action='append', default=[], required=True,
                     help='best_model.pth cua 1 checkpoint. Lap lai --checkpoint de chay nhieu checkpoint.')
    ap.add_argument('--config', action='append', default=[], required=True,
                     help='YAML da dung lue train checkpoint tuong ung (cung thu tu voi --checkpoint).')
    ap.add_argument('--model-type', dest='model_type', action='append', default=[], required=True,
                     choices=['baseline', 'bce_edge', 'static_boundary'],
                     help='Loai model cua checkpoint tuong ung (cung thu tu voi --checkpoint).')
    ap.add_argument('--label', action='append', default=[], required=True,
                     help='Ten ngan, duy nhat cho checkpoint nay — dung lam ten thu muc cache/output/cot bang.')
    ap.add_argument('--iter', action='append', default=[], type=int, required=True,
                     help='Iteration cua checkpoint (bat buoc co trong moi bang, muc 1.2 spec).')
    ap.add_argument('--expected-miou', dest='expected_miou', action='append', default=[],
                     help='mIoU-9 da biet truoc cua checkpoint (dung cho cong 5.1). '
                          'Truyen "skip" de bo qua rieng 1 checkpoint. Bo han co nay = bo qua cong 5.1 cho tat ca.')
    ap.add_argument('--data-root', default=None, help='Override DATASET.ROOT_DIR/VAL_ROOT_DIR cho moi checkpoint.')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--device', default=None, help="'cuda'/'cpu' — mac dinh tu chon (uu tien CPU tren Kaggle P100).")
    ap.add_argument('--distances', default='0,1,2,4,8', help='Cac muc d cho oracle BOUNDARY, phan cach dau phay.')
    ap.add_argument('--interior-distances', dest='interior_distances', default='1,2,4',
                     help='Cac muc d cho oracle INTERIOR, phan cach dau phay.')
    ap.add_argument('--gate-tol', dest='gate_tol', type=float, default=1e-3, help='Nguong cong 5.1.')
    ap.add_argument('--n-overlay', dest='n_overlay', type=int, default=3, help='So anh overlay cho cong 5.4.')
    ap.add_argument('--cache-dir', dest='cache_dir', default='cache/oracle')
    ap.add_argument('--output-dir', dest='output_dir', default='docs/results/oracle')
    return ap.parse_args()


def parse_int_list(s):
    return [int(x) for x in s.split(',') if x.strip() != '']


def validate_run_specs(args):
    """Args (hoac bat ky object co cung thuoc tinh, vd SimpleNamespace trong
    test) -> list[dict], 1 dict/checkpoint. So checkpoint N = len(args.checkpoint)
    — day la noi "dem so checkpoint can chay" thuc su xay ra."""
    n = len(args.checkpoint)
    if n == 0:
        raise ValueError("Can it nhat 1 bo --checkpoint/--config/--model-type/--label/--iter.")

    for flag_name, values in (
        ('--config', args.config), ('--model-type', args.model_type),
        ('--label', args.label), ('--iter', args.iter),
    ):
        if len(values) != n:
            raise ValueError(
                f"So lan truyen {flag_name} ({len(values)}) khac so lan --checkpoint ({n}) — "
                f"moi checkpoint can DU 5 co, cung thu tu: "
                f"--checkpoint/--config/--model-type/--label/--iter.")

    if args.expected_miou and len(args.expected_miou) != n:
        raise ValueError(
            f"--expected-miou duoc truyen {len(args.expected_miou)} lan nhung co {n} checkpoint — "
            f"phai truyen dung 0 lan (bo qua cong 5.1 cho tat ca) hoac dung {n} lan "
            f"(dung 'skip' cho checkpoint muon bo qua rieng le).")

    if len(set(args.label)) != n:
        raise ValueError(f"--label co gia tri trung lap: {args.label} — moi checkpoint can 1 label duy nhat.")

    specs = []
    for i in range(n):
        expected = None
        if args.expected_miou:
            raw = args.expected_miou[i]
            expected = None if str(raw).strip().lower() == 'skip' else float(raw)
        specs.append({
            'checkpoint': args.checkpoint[i], 'config': args.config[i],
            'model_type': args.model_type[i], 'label': args.label[i],
            'iter': args.iter[i], 'expected_miou': expected,
        })
    return specs


# ─────────────────────── Oracle core (pure, testable) ──────────────────────

def apply_oracle_boundary(y: np.ndarray, y_hat: np.ndarray, dist: np.ndarray, d: int) -> np.ndarray:
    """Muc 2 buoc 3-4 cua spec: trong dai (dist<=d) lay nhan dung, ngoai dai giu du doan."""
    return np.where(dist <= d, y, y_hat)


def apply_oracle_interior(y: np.ndarray, y_hat: np.ndarray, dist: np.ndarray, d: int) -> np.ndarray:
    """Muc 2.2: oracle nghich dao — sua hoan hao NGOAI dai, giu nguyen TRONG dai."""
    return np.where(dist <= d, y_hat, y)


def band_pixel_counts(y: np.ndarray, y_hat: np.ndarray, dist: np.ndarray, d: int, ignore_index: int):
    """Muc 2.3: tu so/mau so cua P_band(d) va E_band(d), dem PIXEL (khong trung binh anh)."""
    valid = y != ignore_index
    band = (dist <= d) & valid
    wrong = (y_hat != y) & valid
    return int(band.sum()), int(valid.sum()), int((wrong & band).sum()), int(wrong.sum())


def labels_to_pseudo_logits(y_pred_array: np.ndarray, num_classes: int) -> torch.Tensor:
    """(H,W) mang nhan (khong phai logits) -> (1,C,H,W) pseudo-logits ma argmax
    tra lai DUNG y_pred_array. Chi de tai dung SegmentationMetrics.update()
    (chi nhan logits) ma KHONG viet lai bat ky cong thuc mIoU/confusion-matrix
    nao o day — dung dung thu thuat one-hot*10-5 da co san trong self-test cua
    src/utils/boundary_metrics.py (make_logits_from_labels)."""
    clipped = np.clip(y_pred_array, 0, num_classes - 1).astype(np.int64)
    labels = torch.from_numpy(clipped)
    onehot = torch.nn.functional.one_hot(labels, num_classes).permute(2, 0, 1).float()
    return (onehot * 10.0 - 5.0).unsqueeze(0)


def accumulate_variant(seg_metrics: SegmentationMetrics, y_pred_array: np.ndarray,
                        y_true_tensor: torch.Tensor, num_classes: int) -> None:
    pseudo_logits = labels_to_pseudo_logits(y_pred_array, num_classes)
    seg_metrics.update(pseudo_logits, y_true_tensor)


# ─────────────────────────── Gate checks (muc 5) ────────────────────────────

def gate_5_1_check(none_miou9: float, expected, tol: float):
    """None -> khong kiem (chua truyen --expected-miou). Raise RuntimeError neu FAIL."""
    if expected is None:
        return None
    diff = abs(none_miou9 - expected)
    if diff >= tol:
        raise RuntimeError(
            f"CONG 5.1 FAIL: none.mIoU-9={none_miou9:.4f} lech {diff:.4f} so voi "
            f"expected={expected:.4f} (nguong {tol}). DUNG — moi so oracle vo gia tri.")
    return diff


def gate_5_2_check(deltas):
    """deltas = [delta_miou9 tai d tang dan]. Raise neu khong don dieu."""
    for a, b in zip(deltas, deltas[1:]):
        if b < a - 1e-9:
            raise RuntimeError(f"CONG 5.2 FAIL: Delta khong don dieu theo d ({deltas}) — loi cai dat band.")
    return True


def gate_5_3_check(all_miou9: float, tol: float = 1e-6):
    if abs(all_miou9 - 1.0) >= tol:
        raise RuntimeError(f"CONG 5.3 FAIL: variant 'all' phai cho mIoU=1.0, do duoc {all_miou9}.")
    return True


def run_gates(spec, results, distances, gate_tol):
    label = spec['label']

    diff = gate_5_1_check(results['none']['miou9'], spec.get('expected_miou'), gate_tol)
    if diff is None:
        print(f"[{label}] Cong 5.1 CHUA XAC NHAN (khong truyen --expected-miou) — "
              f"tu doi chieu thu cong voi benchmark_results.csv truoc khi tin ket qua.")
    else:
        print(f"[{label}] Cong 5.1 PASS (none.mIoU-9={results['none']['miou9']:.4f}, lech {diff:.4f}).")

    sorted_d = sorted(distances)
    deltas = [results[f'oracle_boundary_d{d}']['delta_miou9'] for d in sorted_d]
    gate_5_2_check(deltas)
    print(f"[{label}] Cong 5.2 PASS (Delta don dieu tai d={sorted_d}: {[round(x, 4) for x in deltas]}).")

    gate_5_3_check(results['all']['miou9'])
    print(f"[{label}] Cong 5.3 PASS (all.mIoU-9={results['all']['miou9']:.8f}).")

    if 2 in distances:
        p2 = results['oracle_boundary_d2']['P_band']
        if not (0.10 <= p2 <= 0.45):
            print(f"[{label}] CANH BAO cong 5.5: P_band(2)={p2:.4f} ngoai [0.10,0.45] — nghi ngo cai dat band.")
        else:
            print(f"[{label}] Cong 5.5 PASS (P_band(2)={p2:.4f}).")
    else:
        print(f"[{label}] d=2 khong nam trong --distances, bo qua cong 5.5.")


# ───────────────────────── Dataset / device helpers ─────────────────────────

def load_cfg(config_path: str, data_root):
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if data_root:
        cfg['DATASET']['ROOT_DIR'] = data_root
        cfg['DATASET']['VAL_ROOT_DIR'] = data_root
    resolve_dataset_paths(cfg['DATASET'])
    return cfg


def build_val_dataset(cfg):
    ds = cfg['DATASET']
    return OpenEarthMapDataset(
        root_dir=ds['VAL_ROOT_DIR'], img_dir=ds['VAL_IMG_DIR'], mask_dir=ds['VAL_MASK_DIR'],
        split_file=ds.get('VAL_SPLIT_FILE'), transform=get_val_transforms(),
    )


def image_ids_of(dataset):
    return [os.path.splitext(os.path.basename(img_path))[0] for img_path, _ in dataset.samples]


def resolve_device(device_arg):
    """Y HET logic tu-chon-CPU cua eval_boundary_metrics.py::main() — lap lai
    (khong import duoc vi no nam inline trong main(), khong phai ham rieng)."""
    device = torch.device(device_arg) if device_arg else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            (torch.zeros(1, device=device) + 1).sum().item()
        except Exception as e:
            print(f"CUDA khong dung duoc voi build PyTorch hien tai ({type(e).__name__}: {e}) "
                  f"— chuyen sang CPU.")
            device = torch.device('cpu')
    print(f"Device: {device}")
    return device


# ───────────────────────────── Pha 0 — cache dist/GT ─────────────────────────

def save_band_overlay(image_tensor, y, band, save_path, image_id, d):
    image = denormalize(image_tensor.permute(1, 2, 0).numpy())
    gt_rgb = mask_to_rgb(y)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image); axes[0].set_title('Anh goc')
    axes[1].imshow(gt_rgb); axes[1].set_title('GT (mau lop)')
    axes[2].imshow(gt_rgb)
    axes[2].imshow(np.ma.masked_where(~band, band), cmap='autumn', alpha=0.55)
    axes[2].set_title(f'Dai band d={d} (do) chong GT')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'{image_id} — cong 5.4 overlay', fontsize=11)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def prepare_dist_gt_cache(first_cfg, cache_dir, output_dir, n_overlay, distances):
    """Chay dung 1 lan, KHONG can model — dist chi phu thuoc GT (muc 1.1/3.3
    cua spec). Overlay cong 5.4 cung chi phu thuoc GT+band (khong phu thuoc
    checkpoint) nen sinh o day, dung chung cho moi checkpoint."""
    val_ds = build_val_dataset(first_cfg)
    canonical_ids = image_ids_of(val_ds)

    boundary_cfg = first_cfg.get('BOUNDARY_LOSS', {})
    connectivity = boundary_cfg.get('CONNECTIVITY', 4)
    dilation_radius = boundary_cfg.get('DILATION_RADIUS', 0)
    num_classes = first_cfg['TRAIN']['NUM_CLASSES']

    dist_dir = os.path.join(cache_dir, '_dist')
    gt_dir = os.path.join(cache_dir, '_gt')
    overlay_dir = os.path.join(output_dir, '_overlays')
    os.makedirs(dist_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    missing = [i for i in canonical_ids
               if not (os.path.exists(os.path.join(dist_dir, i + '.npy'))
                       and os.path.exists(os.path.join(gt_dir, i + '.npy')))]
    overlays_needed = n_overlay > 0 and sum(
        1 for i in canonical_ids[:n_overlay]
        if not os.path.exists(os.path.join(overlay_dir, f'overlay_{i}.png'))
    ) > 0

    if not missing and not overlays_needed:
        print(f"Pha 0: cache dist/GT + overlay da du cho {len(canonical_ids)} anh, bo qua.")
        return canonical_ids, connectivity, dilation_radius, num_classes

    d_for_overlay = 2 if 2 in distances else (max(distances) if distances else 2)
    if overlays_needed:
        os.makedirs(overlay_dir, exist_ok=True)

    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    t0 = time.time()
    for idx, (images, masks) in enumerate(loader):
        image_id = canonical_ids[idx]
        gt_path = os.path.join(gt_dir, image_id + '.npy')
        dist_path = os.path.join(dist_dir, image_id + '.npy')

        if not (os.path.exists(gt_path) and os.path.exists(dist_path)):
            edge_gt, _valid = extract_edge_gt(
                masks, ignore_index=OpenEarthMapDataset.IGNORE_INDEX,
                connectivity=connectivity, dilation_radius=dilation_radius)
            edge_np = edge_gt[0, 0].numpy().astype(bool)
            dist = distance_transform_edt(~edge_np).astype(np.float32)
            y = masks[0].numpy().astype(np.uint8)
            np.save(gt_path, y)
            np.save(dist_path, dist)

        overlay_path = os.path.join(overlay_dir, f'overlay_{image_id}.png')
        if idx < n_overlay and not os.path.exists(overlay_path):
            y = np.load(gt_path)
            dist = np.load(dist_path)
            save_band_overlay(images[0], y, dist <= d_for_overlay, overlay_path, image_id, d_for_overlay)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(canonical_ids):
            print(f"  Pha 0 (dist/GT cache): {idx + 1}/{len(canonical_ids)} anh ({time.time() - t0:.1f}s)", end='\r')
    print()
    return canonical_ids, connectivity, dilation_radius, num_classes


def verify_boundary_cfg_matches(cfg, connectivity, dilation_radius, label):
    boundary_cfg = cfg.get('BOUNDARY_LOSS', {})
    c2 = boundary_cfg.get('CONNECTIVITY', 4)
    d2 = boundary_cfg.get('DILATION_RADIUS', 0)
    if (c2, d2) != (connectivity, dilation_radius):
        raise RuntimeError(
            f"[{label}] BOUNDARY_LOSS connectivity/dilation ({c2},{d2}) khac checkpoint dau tien "
            f"({connectivity},{dilation_radius}) — DUNG, khong tu doan.")


# ──────────────────────── Pha 1..N — cache prediction ────────────────────────

def ensure_pred_cache(cfg, spec, canonical_ids, cache_dir, device, batch_size, num_workers):
    val_ds = build_val_dataset(cfg)
    ids_now = image_ids_of(val_ds)
    if ids_now != canonical_ids:
        raise RuntimeError(
            f"[{spec['label']}] danh sach anh val khac checkpoint dau tien — DUNG, kiem tra lai "
            f"config/val split (dung val_clean_*.txt, khong dung val_2000_fixed.txt).")

    pred_dir = os.path.join(cache_dir, spec['label'], 'preds')
    os.makedirs(pred_dir, exist_ok=True)
    missing = [i for i in canonical_ids if not os.path.exists(os.path.join(pred_dir, i + '.npy'))]
    if not missing:
        print(f"[{spec['label']}] cache prediction da du ({len(canonical_ids)} anh), bo qua inference.")
        return

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    model = build_eval_model(cfg, spec['model_type']).to(device)
    state_dict = torch.load(spec['checkpoint'], map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[{spec['label']}] da nap checkpoint: {spec['checkpoint']}")

    idx = 0
    t0 = time.time()
    with torch.no_grad():
        for images, _masks in val_loader:
            images = images.to(device, non_blocking=True)
            if spec['model_type'] == 'bce_edge':
                logits, _edge_logits = model(images)
            elif spec['model_type'] == 'static_boundary':
                logits, _edge_logits, _fused = model(images)
            else:
                logits = model(images)

            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            for b in range(preds.shape[0]):
                np.save(os.path.join(pred_dir, canonical_ids[idx] + '.npy'), preds[b])
                idx += 1
            print(f"  [{spec['label']}] inference: {idx}/{len(canonical_ids)} anh "
                  f"({time.time() - t0:.1f}s)", end='\r')
    print()
    del model


# ───────────────────────── Oracle computation per checkpoint ────────────────

def compute_checkpoint_oracle(spec, canonical_ids, cache_dir, distances, interior_distances,
                               num_classes, ignore_index, class_names):
    pred_dir = os.path.join(cache_dir, spec['label'], 'preds')
    dist_dir = os.path.join(cache_dir, '_dist')
    gt_dir = os.path.join(cache_dir, '_gt')

    variant_names = (['none']
                      + [f'oracle_boundary_d{d}' for d in sorted(distances)]
                      + [f'oracle_interior_d{d}' for d in sorted(interior_distances)]
                      + ['all'])
    metrics_by_variant = {name: SegmentationMetrics(num_classes, ignore_index) for name in variant_names}
    band_sums = {d: [0, 0, 0, 0] for d in distances}  # p_num, p_den, e_num, e_den

    for image_id in canonical_ids:
        y = np.load(os.path.join(gt_dir, image_id + '.npy')).astype(np.int64)
        y_hat = np.load(os.path.join(pred_dir, image_id + '.npy')).astype(np.int64)
        dist = np.load(os.path.join(dist_dir, image_id + '.npy'))
        y_t = torch.from_numpy(y).unsqueeze(0)

        accumulate_variant(metrics_by_variant['none'], y_hat, y_t, num_classes)
        accumulate_variant(metrics_by_variant['all'], y, y_t, num_classes)

        for d in distances:
            y_ob = apply_oracle_boundary(y, y_hat, dist, d)
            accumulate_variant(metrics_by_variant[f'oracle_boundary_d{d}'], y_ob, y_t, num_classes)
            p_num, p_den, e_num, e_den = band_pixel_counts(y, y_hat, dist, d, ignore_index)
            s = band_sums[d]
            s[0] += p_num; s[1] += p_den; s[2] += e_num; s[3] += e_den

        for d in interior_distances:
            y_oi = apply_oracle_interior(y, y_hat, dist, d)
            accumulate_variant(metrics_by_variant[f'oracle_interior_d{d}'], y_oi, y_t, num_classes)

    results = {}
    for name in variant_names:
        r = metrics_by_variant[name].compute()
        per_class = r['per_class_iou']
        results[name] = {
            'miou9': r['mIoU'],
            'miou8': float(np.mean(per_class[1:])),
            'per_class_iou': {cn: float(v) for cn, v in zip(class_names, per_class)},
        }

    base = results['none']
    for name in variant_names:
        if name == 'none':
            continue
        results[name]['delta_miou9'] = results[name]['miou9'] - base['miou9']
        results[name]['delta_miou8'] = results[name]['miou8'] - base['miou8']

    for d in distances:
        p_num, p_den, e_num, e_den = band_sums[d]
        results[f'oracle_boundary_d{d}']['P_band'] = (p_num / p_den) if p_den else float('nan')
        results[f'oracle_boundary_d{d}']['E_band'] = (e_num / e_den) if e_den else float('nan')

    return results


# ──────────────────────────────── Output ────────────────────────────────────

def _fmt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.4f}'


def render_oracle_table_md(spec, results, distances, interior_distances, class_names):
    lines = [
        f"# Oracle table — {spec['label']} (iter {spec['iter']})", '',
        f"Checkpoint: `{spec['checkpoint']}`", f"Config: `{spec['config']}`",
        f"model-type: `{spec['model_type']}`", '',
        '## Bang 1 — duong cong tran', '',
        '| Bien the | mIoU-9 | Delta (diem) | mIoU-8 | Delta-8 | P_band | E_band |',
        '|---|---|---|---|---|---|---|',
    ]
    none_r = results['none']
    lines.append(f"| none (baseline) | {none_r['miou9']:.4f} | — | {none_r['miou8']:.4f} | — | — | — |")
    for d in sorted(distances):
        r = results[f'oracle_boundary_d{d}']
        lines.append(f"| oracle boundary d={d} | {r['miou9']:.4f} | {r['delta_miou9'] * 100:.2f}d | "
                      f"{r['miou8']:.4f} | {r['delta_miou8'] * 100:.2f}d | {_fmt(r.get('P_band'))} | "
                      f"{_fmt(r.get('E_band'))} |")
    for d in sorted(interior_distances):
        r = results[f'oracle_interior_d{d}']
        lines.append(f"| oracle interior d={d} | {r['miou9']:.4f} | {r['delta_miou9'] * 100:.2f}d | "
                      f"{r['miou8']:.4f} | {r['delta_miou8'] * 100:.2f}d | — | — |")

    lines += ['', '## Bang 2 — muc tang oracle per-class tai d=2 (giam dan)', '']
    if 'oracle_boundary_d2' in results:
        d2_pc = results['oracle_boundary_d2']['per_class_iou']
        none_pc = none_r['per_class_iou']
        ranked = sorted(((cn, d2_pc[cn] - none_pc[cn]) for cn in class_names), key=lambda kv: -kv[1])
        lines += ['| Lop | IoU none | IoU oracle d=2 | Delta (diem) |', '|---|---|---|---|']
        for cn, dv in ranked:
            lines.append(f"| {cn} | {none_pc[cn]:.4f} | {d2_pc[cn]:.4f} | {dv * 100:.2f}d |")
    else:
        lines.append('_d=2 khong nam trong --distances cua lan chay nay, bo qua Bang 2._')

    return '\n'.join(lines) + '\n'


def plot_oracle_curve(results, distances, interior_distances, save_path, label):
    fig, ax = plt.subplots(figsize=(7, 5))
    xb = sorted(distances)
    yb = [results[f'oracle_boundary_d{d}']['delta_miou9'] * 100 for d in xb]
    ax.plot(xb, yb, marker='o', label='Boundary-oracle Delta mIoU-9 (diem)')
    if interior_distances:
        xi = sorted(interior_distances)
        yi = [results[f'oracle_interior_d{d}']['delta_miou9'] * 100 for d in xi]
        ax.plot(xi, yi, marker='s', label='Interior-oracle Delta mIoU-9 (diem)')
    ax.set_xlabel('d (pixel)')
    ax.set_ylabel('Delta mIoU-9 (diem)')
    ax.set_title(f'Oracle curve — {label}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def write_checkpoint_outputs(spec, results, distances, interior_distances, class_names, output_dir, n_images):
    out_dir = os.path.join(output_dir, spec['label'])
    os.makedirs(out_dir, exist_ok=True)

    payload = {
        'checkpoint': spec['checkpoint'], 'checkpoint_iter': spec['iter'],
        'model_type': spec['model_type'], 'config': spec['config'],
        'n_images': n_images, 'class_names': class_names, 'variants': results,
    }
    with open(os.path.join(out_dir, 'oracle_results.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, 'oracle_table.md'), 'w', encoding='utf-8') as f:
        f.write(render_oracle_table_md(spec, results, distances, interior_distances, class_names))

    plot_oracle_curve(results, distances, interior_distances,
                       os.path.join(out_dir, 'oracle_curve.png'), spec['label'])
    print(f"[{spec['label']}] da ghi ket qua vao {out_dir}")


def write_comparison_md(all_results, distances, output_dir):
    """Muc 6.1 — bang O_b/Gap tong the va per-class tai d=2, TU DONG theo N
    checkpoint da chay (khong hard-code 3 hang)."""
    if 2 not in distances:
        print("d=2 khong co trong --distances, bo qua comparison.md (muc 6.1 can d=2).")
        return

    lines = [
        '# So sanh O_b / Gap tai d=2 — moi checkpoint da chay', '',
        '| Cau hinh (label) | iter | M(c) mIoU-9 | O_b(c) @d=2 | Gap(c)=O_b-M (diem) |',
        '|---|---|---|---|---|',
    ]
    for label, entry in all_results.items():
        r = entry['results']
        m = r['none']['miou9']
        ob = r['oracle_boundary_d2']['miou9']
        lines.append(f"| {label} | {entry['iter']} | {m:.4f} | {ob:.4f} | {(ob - m) * 100:.2f}d |")

    class_names = list(next(iter(all_results.values()))['results']['none']['per_class_iou'].keys())
    lines += ['', '## Per-class O_b / Gap tai d=2', '',
              '| Lop | ' + ' | '.join(f"{lbl} O_b" for lbl in all_results)
              + ' | ' + ' | '.join(f"{lbl} Gap" for lbl in all_results) + ' |',
              '|' + '---|' * (1 + 2 * len(all_results))]
    for cn in class_names:
        obs, gaps = [], []
        for entry in all_results.values():
            r = entry['results']
            m_c = r['none']['per_class_iou'][cn]
            ob_c = r['oracle_boundary_d2']['per_class_iou'][cn]
            obs.append(f"{ob_c:.4f}")
            gaps.append(f"{(ob_c - m_c) * 100:.2f}d")
        lines.append(f"| {cn} | " + ' | '.join(obs) + ' | ' + ' | '.join(gaps) + ' |')

    out_path = os.path.join(output_dir, 'comparison.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Da ghi {out_path}")


# ──────────────────────────────────── main ───────────────────────────────────

def main():
    args = parse_args()
    try:
        specs = validate_run_specs(args)
    except ValueError as e:
        print(f"LOI THAM SO: {e}")
        sys.exit(1)

    print(f"Se chay N={len(specs)} checkpoint: {[s['label'] for s in specs]}")

    distances = parse_int_list(args.distances)
    interior_distances = parse_int_list(args.interior_distances)
    device = resolve_device(args.device)

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    first_cfg = load_cfg(specs[0]['config'], args.data_root)
    canonical_ids, connectivity, dilation_radius, num_classes = prepare_dist_gt_cache(
        first_cfg, args.cache_dir, args.output_dir, args.n_overlay, distances)
    print(f"Val set: {len(canonical_ids)} anh (canonical, lay tu checkpoint dau tien '{specs[0]['label']}').")

    class_names = OpenEarthMapDataset.CLASSES
    ignore_index = OpenEarthMapDataset.IGNORE_INDEX

    all_results = {}
    for spec in specs:
        print(f"\n=== Checkpoint [{spec['label']}] (iter {spec['iter']}, model-type={spec['model_type']}) ===")
        cfg = load_cfg(spec['config'], args.data_root)
        try:
            verify_boundary_cfg_matches(cfg, connectivity, dilation_radius, spec['label'])
            ensure_pred_cache(cfg, spec, canonical_ids, args.cache_dir, device, args.batch_size, args.num_workers)
            results = compute_checkpoint_oracle(spec, canonical_ids, args.cache_dir, distances,
                                                 interior_distances, num_classes, ignore_index, class_names)
            run_gates(spec, results, distances, args.gate_tol)
        except RuntimeError as e:
            print(f"[{spec['label']}] {e}")
            sys.exit(1)

        write_checkpoint_outputs(spec, results, distances, interior_distances, class_names,
                                  args.output_dir, len(canonical_ids))
        all_results[spec['label']] = {'iter': spec['iter'], 'checkpoint': spec['checkpoint'],
                                       'model_type': spec['model_type'], 'results': results}
        print(f"[{spec['label']}] xong. none.mIoU-9={results['none']['miou9']:.4f}, "
              f"oracle_boundary_d2.mIoU-9={results.get('oracle_boundary_d2', {}).get('miou9', float('nan')):.4f}")

    write_comparison_md(all_results, distances, args.output_dir)
    print(f"\nHoan tat toan bo {len(specs)} checkpoint. Xem {args.output_dir}/comparison.md va "
          f"{args.output_dir}/<label>/oracle_table.md cho tung checkpoint.")


if __name__ == '__main__':
    main()
