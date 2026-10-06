"""Tools/select_figure_images.py — Chot 6 anh minh hoa cua muc 2.4
docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md, CHI TU GT, TRUOC khi tinh
gradient:

  - 2 anh co dien tich GT Agriculture lon nhat,
  - 2 anh co dien tich GT Water lon nhat,
  - 2 anh co so thanh phan GT Building "nho" (8-lien thong, < 4096 px) nhieu nhat.

Trung anh giua cac tieu chi -> lay anh ke tiep cua tieu chi sau (6 anh rieng
biet). Hoa -> sap theo ten anh. Ket qua ghi vao figure_images.json va KHONG
duoc ghi de (muc 2.4: "Khong doi danh sach sau khi thay ban do") — chay lai se
dung, tru khi --force.

Chay (repo root):
    python Tools/select_figure_images.py --data-root /kaggle/input/... \\
        --output output/grad_conflict/figure_images.json
"""

import argparse
import datetime
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (
    CLASS_NAMES, CFG_STATIC, SIZE_SMALL_MAX, gt_components, get_git_commit, write_json, load_gt_cached,
)


def figure_scores(gt):
    agri = CLASS_NAMES.index('Agriculture')
    water = CLASS_NAMES.index('Water')
    bld = CLASS_NAMES.index('Building')
    _cm, table = gt_components(gt)
    n_small_bld = sum(1 for c, a in table.values() if c == bld and a < SIZE_SMALL_MAX)
    return {
        'agriculture_area': int((gt == agri).sum()),
        'water_area': int((gt == water).sum()),
        'n_small_building_components': int(n_small_bld),
    }


def select(scores_by_image, n_per=2):
    """scores_by_image: dict name -> dict score. Tra ve list dict (thu tu tieu chi)."""
    chosen = []
    used = set()
    for key, crit in (('agriculture_area', 'Agriculture area lon nhat'),
                      ('water_area', 'Water area lon nhat'),
                      ('n_small_building_components', 'so thanh phan Building nho nhieu nhat')):
        ranked = sorted(scores_by_image.items(), key=lambda kv: (-kv[1][key], kv[0]))
        k = 0
        for name, sc in ranked:
            if k >= n_per:
                break
            if name in used:
                continue
            chosen.append({'image_id': name, 'criterion': crit, 'score_key': key, 'score': sc[key],
                           'rank_in_criterion': k + 1})
            used.add(name)
            k += 1
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=CFG_STATIC, help='YAML de doc DATASET (moi config dung chung val split)')
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--output', default='output/grad_conflict/figure_images.json')
    ap.add_argument('--gt-cache', default='output/_cache/gt1024', help="Thu muc cache GT .npy ('' = tat)")
    ap.add_argument('--max-images', type=int, default=None, help='Chi dung cho dry-run')
    ap.add_argument('--force', action='store_true', help='Ghi de danh sach da chot (KHONG nen dung)')
    args = ap.parse_args()

    if os.path.exists(args.output) and not args.force:
        print(f"Da co {args.output} — danh sach da chot, KHONG chon lai (muc 2.4). Dung --force neu that su can.")
        return

    from Tools.oracle_boundary_ceiling import load_cfg, build_val_dataset
    cfg = load_cfg(args.config, args.data_root)
    val_ds = build_val_dataset(cfg)
    n = len(val_ds) if args.max_images is None else min(args.max_images, len(val_ds))
    scores = {}
    for i in range(n):
        name, gt = load_gt_cached(val_ds, i, args.gt_cache or None)
        scores[name] = figure_scores(np.asarray(gt))
        print(f"  ... {i + 1}/{n}", end='\r')
    print()
    chosen = select(scores)
    write_json({
        'spec': 'docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md muc 2.4',
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': get_git_commit(),
        'n_images_scanned': n,
        'small_component_max_area': SIZE_SMALL_MAX,
        'images': chosen,
    }, args.output)
    for c in chosen:
        print(f"  {c['image_id']:30s} {c['criterion']} = {c['score']}")
    print(f"Da chot {len(chosen)} anh: {args.output}")


if __name__ == '__main__':
    main()
