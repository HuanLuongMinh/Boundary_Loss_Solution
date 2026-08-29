"""Tools/measure_edge_ratio.py — đo tỉ lệ edge-pixel r_edge ở mức TOÀN DATASET
(1 pass đầy đủ, deterministic, seed=19) cho thực nghiệm "Baseline vs BCE Loss".

Đây là bước đo "chính thức" 1 lần theo đúng docs/workflow_2.md mục 3.1 điểm 3
và checklist mục 9 ("Đã xác nhận... trước khi chạy Run 2") — KHÔNG bắt buộc
phải chạy trước src/train_bce_edge.py (script train tự ước lượng pos_weight
nếu không tìm thấy file JSON do tool này sinh ra, xem BOUNDARY_LOSS.EDGE_STATS_FILE
trong config). Chạy tool này khi muốn có con số r_edge/pos_weight "authoritative"
để ghi vào báo cáo/summary.txt thay vì dùng ước lượng bounded lúc khởi động.

File này ĐỘC LẬP — không import bất kỳ script train_*.py nào, chỉ dùng chung
src/data (dataset/transforms, không sửa) và src/losses/boundary_bce.py (mới,
không sửa).

Usage:
    python Tools/measure_edge_ratio.py \
        --config configs/unet_former_resnet18_bce_edge/bce_edge.yaml \
        --out configs/unet_former_resnet18_bce_edge/edge_ratio_stats.json

    # Ghi đè data root nếu Kaggle mount khác path trong YAML:
    python Tools/measure_edge_ratio.py --config <cfg> --data-root /kaggle/input/openearthmap --out <path>

    # Giới hạn số batch (debug nhanh, KHÔNG dùng số ra từ đây làm con số chính
    # thức cho báo cáo — chỉ full epoch (mặc định, --max-batches bỏ trống) mới
    # là thống kê "chính thức"):
    python Tools/measure_edge_ratio.py --config <cfg> --out <path> --max-batches 50
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_train_transforms
from src.losses.boundary_bce import compute_edge_ratio_stats


# ── Dataset path resolution (duplicated from src/train_unet_former_resnet18.py
# on purpose — mỗi script trong project này tự chứa, không phụ thuộc lẫn
# nhau, xem quy ước "file độc lập" trong README.md) ─────────────────────────

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


def resolve_dataset_paths(ds: dict) -> None:
    root_base = find_data_base(ds['ROOT_DIR'])
    if root_base != ds['ROOT_DIR']:
        print(f"Resolved ROOT_DIR: {ds['ROOT_DIR']} -> {root_base}")
        ds['ROOT_DIR'] = root_base
    label_sub = find_label_subdir(root_base)
    cur = ds['TRAIN_MASK_DIR']
    head, _, tail = cur.partition('/')
    if head in ('labels', 'label') and head != label_sub:
        fixed = label_sub + '/' + tail if tail else label_sub
        print(f"Resolved TRAIN_MASK_DIR: '{cur}' -> '{fixed}'")
        ds['TRAIN_MASK_DIR'] = fixed


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to YAML config (bce_edge.yaml)')
    parser.add_argument('--data-root', default=None, help='Override DATASET.ROOT_DIR')
    parser.add_argument('--seed', type=int, default=None,
                        help='Mặc định: đọc TRAIN.SEED trong config (=19 cho bce_edge.yaml)')
    parser.add_argument('--out', default=None,
                        help='Đường dẫn JSON output (mặc định: <config_dir>/edge_ratio_stats.json)')
    parser.add_argument('--connectivity', type=int, default=None, choices=[4, 8],
                        help='Mặc định: đọc BOUNDARY_LOSS.CONNECTIVITY trong config')
    parser.add_argument('--dilation-radius', type=int, default=None,
                        help='Mặc định: đọc BOUNDARY_LOSS.DILATION_RADIUS trong config')
    parser.add_argument('--w-max', type=float, default=None,
                        help='Mặc định: đọc BOUNDARY_LOSS.POS_WEIGHT_MAX trong config')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size chỉ dùng cho việc đọc dữ liệu ở đây, không liên quan '
                             'BATCH_SIZE_PER_GPU lúc train thật')
    parser.add_argument('--max-batches', type=int, default=None,
                        help='Giới hạn số batch (debug) — bỏ trống = 1 epoch ĐẦY ĐỦ (khuyến nghị, '
                             'đây mới là con số "chính thức" theo docs/workflow_2.md)')
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    ds = cfg['DATASET']
    bl = cfg.get('BOUNDARY_LOSS', {})

    seed = args.seed if args.seed is not None else cfg.get('TRAIN', {}).get('SEED', 19)
    connectivity = args.connectivity if args.connectivity is not None else bl.get('CONNECTIVITY', 4)
    dilation_radius = args.dilation_radius if args.dilation_radius is not None else bl.get('DILATION_RADIUS', 0)
    w_max = args.w_max if args.w_max is not None else bl.get('POS_WEIGHT_MAX', 20.0)
    ignore_index = bl.get('IGNORE_INDEX', 255)

    set_seed(seed)
    print(f"seed={seed}  connectivity={connectivity}  dilation_radius={dilation_radius}  "
          f"ignore_index={ignore_index}  w_max={w_max}")

    if args.data_root:
        ds['ROOT_DIR'] = args.data_root
    resolve_dataset_paths(ds)

    train_ds = OpenEarthMapDataset(
        root_dir=ds['ROOT_DIR'], img_dir=ds['TRAIN_IMG_DIR'],
        mask_dir=ds['TRAIN_MASK_DIR'], split_file=ds.get('TRAIN_SPLIT_FILE'),
        transform=get_train_transforms(),
    )
    print(f"Train dataset: {len(train_ds)} ảnh — {ds['ROOT_DIR']}/{ds['TRAIN_IMG_DIR']}")

    # num_workers=0: đơn giản, deterministic, single-process — đủ nhanh cho 1
    # lần đo thống kê, không cần tối ưu throughput như lúc train thật.
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False)

    scope = 'FULL EPOCH (chính thức)' if args.max_batches is None else f'{args.max_batches} batch đầu (debug, KHÔNG chính thức)'
    print(f"Đang đo r_edge — {scope} ...")

    stats = compute_edge_ratio_stats(
        loader, ignore_index=ignore_index, connectivity=connectivity,
        dilation_radius=dilation_radius, w_max=w_max, max_batches=args.max_batches,
    )
    stats.update({
        'seed': seed,
        'ignore_index': ignore_index,
        'config': os.path.abspath(args.config),
        'dataset_root': ds['ROOT_DIR'],
        'split': 'train',
        'batch_size': args.batch_size,
        'full_epoch': args.max_batches is None,
        'measured_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    })

    print('')
    print('=' * 60)
    print(' EDGE-PIXEL RATIO STATISTICS')
    print('=' * 60)
    for k in ('N_valid', 'N_edge', 'N_nonedge', 'r_edge', 'r_nonedge',
              'pos_weight', 'connectivity', 'dilation_radius', 'n_batches'):
        print(f"  {k:<16}: {stats[k]}")
    print('=' * 60)

    out_path = args.out or os.path.join(os.path.dirname(os.path.abspath(args.config)),
                                        'edge_ratio_stats.json')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu: {out_path}")
    print("Trỏ BOUNDARY_LOSS.EDGE_STATS_FILE trong config tới file này để "
          "train_bce_edge.py dùng đúng pos_weight 'chính thức' này thay vì tự ước lượng.")


if __name__ == '__main__':
    main()
