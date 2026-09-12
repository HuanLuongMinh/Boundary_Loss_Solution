"""Tools/agriculture_fake_fragments.py — Phan D (docs/spec-per-image-bootstrap-
eval-v2-ban-giao-claude-code.md muc 8): dem thanh phan lien thong "gia" tren
lop Agriculture, dung chung mask PNG da dump o Phan A/B (--dump-preds), 0
inference them.

File MOI, DOC LAP — chi DOC PNG mask + CSV thong ke du (Tools/per_image_dump.py)
+ dung lai OpenEarthMapDataset/get_val_transforms de dung lai GT DUNG resolution
voi mask PNG (mask PNG o resolution SAU transform, khong phai .tif goc — xem
Tools/test_agriculture_fake_fragments.py).

Dinh nghia "manh gia" (muc 8.3 buoc 4): 1 connected component trong PRED ma
ti le dien tich overlap voi TOAN BO vung Agriculture cua GT (hop cua moi
component GT) < nguong (mac dinh 5%) — tuong duong "khong overlap voi bat ky
component nao trong GT qua nguong do" (kiem tra overlap voi HOP cac component
GT cho cung ket qua nhi phan hoac ti le voi kiem tung component GT rieng le,
vi phan tram la tren dien tich component PRED, khong phai GT).

Chay: `python Tools/agriculture_fake_fragments.py --config <yaml> \\
    --mask-dir baseline=output/dump/run1_baseline_iter40000/masks \\
    --mask-dir bce04=output/dump/run2b_bce04_iter40000/masks \\
    --mask-dir static_s19=output/dump/run3_static_s19_iter36000/masks \\
    --mask-dir run3b_aff02=output/dump/run3b_aff02_iter40000/masks \\
    --per-image-stats output/dump/run1_baseline_iter40000/per_image_stats.csv`

Tu-test: `python Tools/test_agriculture_fake_fragments.py`.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_val_transforms
from Tools.eval_boundary_metrics import resolve_dataset_paths


AGRICULTURE_CLASS_ID = OpenEarthMapDataset.CLASSES.index('Agriculture')


def connectivity_structure(connectivity: int) -> np.ndarray:
    """scipy.ndimage.label dung 'structure' de dinh nghia hang xom — 4-connectivity
    (mac dinh toan du an, xem BoundaryMetrics) = cross-shaped, KHONG cheo."""
    if connectivity == 4:
        return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    elif connectivity == 8:
        return np.ones((3, 3), dtype=np.int64)
    raise ValueError(f"connectivity phai la 4 hoac 8, nhan: {connectivity}")


def count_fake_fragments(pred_mask: np.ndarray, gt_mask: np.ndarray, connectivity: int,
                          overlap_threshold: float):
    """pred_mask/gt_mask: (H,W) bool, True = Agriculture. Tra ve dict:
    n_components_pred, n_components_gt, n_fake_components, fake_area_total."""
    structure = connectivity_structure(connectivity)
    pred_labels, n_pred = cc_label(pred_mask, structure=structure)
    _gt_labels, n_gt = cc_label(gt_mask, structure=structure)

    n_fake = 0
    fake_area_total = 0
    for comp_id in range(1, n_pred + 1):
        comp_mask = pred_labels == comp_id
        comp_area = int(comp_mask.sum())
        if comp_area == 0:
            continue
        overlap = int(np.logical_and(comp_mask, gt_mask).sum())
        overlap_frac = overlap / comp_area
        if overlap_frac < overlap_threshold:
            n_fake += 1
            fake_area_total += comp_area

    return {
        'n_components_pred': int(n_pred), 'n_components_gt': int(n_gt),
        'n_fake_components': int(n_fake), 'fake_area_total': int(fake_area_total),
    }


def build_gt_agriculture_masks(config_path, data_root=None):
    """Tra ve dict ten_anh -> (H,W) bool GT Agriculture, o DUNG resolution
    sau get_val_transforms() — CUNG resolution voi mask PNG da dump (Phan A
    dump prediction argmax sau khi qua model, dau vao model da qua transform
    nay), khong phai .tif goc."""
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if data_root:
        cfg['DATASET']['ROOT_DIR'] = data_root
        cfg['DATASET']['VAL_ROOT_DIR'] = data_root
    resolve_dataset_paths(cfg['DATASET'])

    ds = cfg['DATASET']
    val_ds = OpenEarthMapDataset(
        root_dir=ds['VAL_ROOT_DIR'], img_dir=ds['VAL_IMG_DIR'], mask_dir=ds['VAL_MASK_DIR'],
        split_file=ds.get('VAL_SPLIT_FILE'), transform=get_val_transforms(),
    )
    gt_by_name = {}
    for i in range(len(val_ds)):
        name = os.path.splitext(os.path.basename(val_ds.samples[i][0]))[0]
        _img, mask_t = val_ds[i]
        gt_by_name[name] = (mask_t.numpy() == AGRICULTURE_CLASS_ID)
    return gt_by_name


def images_with_agriculture(per_image_stats_paths):
    """Muc 8.3: chi xet anh co class_present_Agriculture=1 trong GT — lay tu
    CSV Phan B (co the truyen NHIEU CSV, vi class_present_Agriculture la
    thuoc tinh cua GT, khong doi giua cac checkpoint — kiem tra nhat quan,
    khong tin mu 1 file)."""
    reference = None
    reference_label = None
    for label, path in per_image_stats_paths.items():
        df = pd.read_csv(path, usecols=['image', 'class_present_Agriculture'])
        flags = dict(zip(df['image'], df['class_present_Agriculture']))
        if reference is None:
            reference, reference_label = flags, label
            continue
        if flags != reference:
            mismatched = {k: (reference.get(k), flags.get(k)) for k in flags
                          if flags.get(k) != reference.get(k)}
            raise AssertionError(
                f"class_present_Agriculture khac nhau giua '{reference_label}' va '{label}' — "
                f"day la thuoc tinh cua GT, PHAI giong het giua moi checkpoint (cung val split?). "
                f"Vi du lech: {dict(list(mismatched.items())[:5])}")
    return sorted(name for name, present in reference.items() if present == 1)


def run(mask_dirs, gt_by_name, agriculture_images, connectivity, overlap_threshold):
    rows = []
    for name in agriculture_images:
        if name not in gt_by_name:
            raise KeyError(f"Anh '{name}' co trong CSV per-image-stats nhung khong co trong val "
                            f"dataset dung --config — kiem tra lai --config/--data-root co dung val "
                            f"split voi luc dump khong.")
        gt_mask = gt_by_name[name]
        for config_label, mask_dir in mask_dirs.items():
            png_path = os.path.join(mask_dir, name + '.png')
            if not os.path.isfile(png_path):
                raise FileNotFoundError(f"Khong thay mask PNG: {png_path}")
            pred_full = np.array(Image.open(png_path))
            if pred_full.shape != gt_mask.shape:
                raise ValueError(
                    f"[{config_label}/{name}] shape mask PNG {pred_full.shape} != shape GT "
                    f"{gt_mask.shape} — --config truyen vao co dung voi luc dump khong?")
            pred_mask = pred_full == AGRICULTURE_CLASS_ID
            stats = count_fake_fragments(pred_mask, gt_mask, connectivity, overlap_threshold)
            stats['image'] = name
            stats['config'] = config_label
            rows.append(stats)
    return pd.DataFrame(rows)


def write_summary_md(df, overlap_threshold, connectivity, path):
    lines = [
        '# Dem manh gia Agriculture — tong hop 4 cau hinh',
        '',
        f"Nguong overlap coi la 'manh gia': < {overlap_threshold * 100:.1f}% dien tich component pred "
        f"overlap voi vung Agriculture GT. Connectivity = {connectivity}.",
        '',
        '| Cau hinh | So anh | Trung binh manh gia/anh | Tong manh gia | Tong dien tich manh gia (px) |'
        ' Trung binh so component pred | Trung binh so component GT |',
        '|---|---|---|---|---|---|---|',
    ]
    for config, g in df.groupby('config'):
        lines.append(
            f"| {config} | {len(g)} | {g['n_fake_components'].mean():.3f} | "
            f"{g['n_fake_components'].sum()} | {g['fake_area_total'].sum()} | "
            f"{g['n_components_pred'].mean():.3f} | {g['n_components_gt'].mean():.3f} |")
    lines += ['', '_Cau hinh nao co it manh gia nhat (trung binh/anh) duoc coi la it "ruot vung gia" '
              'nhat — doi chieu voi bang oracle C5b (Agriculture +2.36d cho Static, chu yeu hieu ung '
              'ruot vung — muc 8.1 spec) de xem co khop thu tu khong._']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def parse_label_path(items, flag_name):
    out = {}
    for item in items:
        if '=' not in item:
            raise ValueError(f"{flag_name} phai co dang LABEL=PATH, nhan: {item}")
        label, path = item.split('=', 1)
        out[label] = path
    return out


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True,
                     help='YAML bat ky trong 4 cau hinh (chi dung phan DATASET — resolve val split/resize, '
                          'khong dung kien truc model).')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--mask-dir', action='append', default=[], required=True,
                     help='LABEL=thu_muc_mask_png — lap lai cho tung cau hinh (khuyen nghi dung 4: '
                          'baseline, bce04, static_s19, run3b_aff02 — muc 8.2 spec).')
    ap.add_argument('--per-image-stats', action='append', default=[], required=True,
                     help='LABEL=duong_dan.csv — CSV Phan A/B (dung class_present_Agriculture). '
                          'Lap lai de tu kiem tra nhat quan giua cac checkpoint (khuyen nghi truyen du).')
    ap.add_argument('--overlap-threshold', dest='overlap_threshold', type=float, default=0.05,
                     help="Ti le dien tich (tren tong dien tich component pred) duoi nguong nay -> "
                          "'manh gia' (muc 8.3 buoc 4, mac dinh 5%% nhu vi du trong spec).")
    ap.add_argument('--connectivity', type=int, default=4, choices=[4, 8])
    ap.add_argument('--output-dir', dest='output_dir', default='output/agriculture')
    return ap.parse_args()


def main():
    args = parse_args()
    mask_dirs = parse_label_path(args.mask_dir, '--mask-dir')
    stats_paths = parse_label_path(args.per_image_stats, '--per-image-stats')

    agriculture_images = images_with_agriculture(stats_paths)
    print(f"So anh co Agriculture trong GT: {len(agriculture_images)} "
          f"(nhat quan giua {len(stats_paths)} CSV da truyen).")

    gt_by_name = build_gt_agriculture_masks(args.config, args.data_root)

    df = run(mask_dirs, gt_by_name, agriculture_images, args.connectivity, args.overlap_threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'agriculture_fake_fragments.csv')
    md_path = os.path.join(args.output_dir, 'agriculture_fake_fragments_summary.md')
    df.to_csv(csv_path, index=False)
    write_summary_md(df, args.overlap_threshold, args.connectivity, md_path)

    print(f"\n=== Tong hop ({len(agriculture_images)} anh co Agriculture) ===")
    for config, g in df.groupby('config'):
        print(f"  {config:16s} trung binh manh gia/anh={g['n_fake_components'].mean():.3f}  "
              f"tong={g['n_fake_components'].sum()}  tong dien tich={g['fake_area_total'].sum()}px")
    print(f"\nDa ghi {csv_path} va {md_path}.")


if __name__ == '__main__':
    main()
