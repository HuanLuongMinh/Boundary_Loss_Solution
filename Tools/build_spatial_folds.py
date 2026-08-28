"""
build_spatial_folds.py — Sinh 5-Fold Spatial Cross-Validation cho OpenEarthMap,
dùng cho Giai đoạn 2 (Main Benchmark) của đề tài "Dynamic Boundary-Aware Loss
via Adaptive BCE-Affinity Coupling" (idead_research.md, mục 3 — "5 Folds phân
vùng không gian độc lập").

Ý TƯỞNG:
    OpenEarthMap gồm 97 vùng địa lý (region) trên 44 quốc gia — mỗi ảnh có
    tiền tố tên vùng trong filename (vd "svaneti_44", "christchurch_4",
    "al_qurnah_12"). Bài báo gốc OpenEarthMap (Xia et al., WACV 2023) cũng
    chia train/val/test theo TỪNG VÙNG (tỉ lệ 6:1:3) chính vì lý do chống rò
    rỉ không gian (spatial leakage): nếu 2 ảnh liền kề của cùng 1 vùng lọt
    vào cả train và val, mô hình dễ "học thuộc" đặc điểm cục bộ (loại nhà,
    loại cây, độ phân giải máy ảnh...) thay vì học tổng quát, khiến chỉ số
    val bị đánh giá quá lạc quan.

    Script này áp dụng đúng nguyên tắc đó cho 5-fold CV: gộp toàn bộ ảnh
    THẬT (không tính bản augment) từ cả images/train và images/val, gom
    theo vùng, rồi CHIA NGUYÊN VẸN TỪNG VÙNG (không cắt đôi 1 vùng) vào 5
    fold sao cho tổng số ảnh mỗi fold cân bằng nhất có thể (thuật toán
    greedy LPT — Longest Processing Time first: xếp vùng đông ảnh nhất
    trước, mỗi lần gán vùng hiện tại vào fold đang có ít ảnh nhất).

    Với mỗi fold i (1..5): val = ảnh thuộc các vùng được gán cho fold i;
    train = ảnh thuộc các vùng còn lại (4 fold kia). Ghi ra
    fold{i}_train.txt / fold{i}_val.txt.

Usage:
    python Tools/build_spatial_folds.py \
        --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
        --output-dir /kaggle/working/unetformer-resnet18-combinedloss-openearthmap \
        --n-folds 5
"""

import argparse
import os
import re
from collections import defaultdict


def scan_images(directory, suffix='.tif'):
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith(suffix)
    )


def find_data_base(data_root):
    candidates = [
        data_root,
        os.path.join(data_root, 'OpenEarthMap_Mini'),
        os.path.join(data_root, 'OpenEarthMap_flat'),
        os.path.join(data_root, 'OpenEarthMap'),
        os.path.join(data_root, 'openearthmap'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'images', 'val')) or \
           os.path.isdir(os.path.join(c, 'images', 'train')):
            return c
    raise FileNotFoundError(f'Could not find images/train or images/val under: {candidates}')


def find_label_subdir(base):
    for name in ('labels', 'label'):
        if os.path.isdir(os.path.join(base, name)):
            return name
    return 'labels'


_REGION_RE = re.compile(r'_\d+$')


def region_of(name: str) -> str:
    """'svaneti_44' -> 'svaneti'; 'al_qurnah_12' -> 'al_qurnah'.
    Chỉ cắt hậu tố '_<số>' cuối cùng, an toàn với tên vùng có gạch dưới."""
    return _REGION_RE.sub('', name)


def greedy_balanced_folds(region_counts: dict, n_folds: int):
    """LPT (Longest Processing Time first) bin-packing: xếp vùng đông ảnh
    nhất trước, mỗi lần gán vào fold đang có tổng số ảnh nhỏ nhất.
    Trả về list[n_folds] các set tên vùng."""
    folds = [set() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for region, count in sorted(region_counts.items(), key=lambda kv: -kv[1]):
        i = fold_sizes.index(min(fold_sizes))
        folds[i].add(region)
        fold_sizes[i] += count
    return folds, fold_sizes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', default=None,
                   help='Nơi ghi fold{i}_train.txt / fold{i}_val.txt (mặc định: <data-root>/dataset)')
    p.add_argument('--n-folds', type=int, default=5)
    p.add_argument('--img-suffix', default='.tif')
    p.add_argument('--include-val-pool', action='store_true', default=True,
                   help='Gộp cả images/val (ảnh thật, không tính bản augment) vào pool trước '
                        'khi chia fold — mặc định BẬT vì Phase 2 cần tối đa dữ liệu thật.')
    args = p.parse_args()

    base = find_data_base(args.data_root)
    label_sub = find_label_subdir(base)

    pool = {}  # name -> 'train' | 'val' (chỉ để log nguồn gốc, không ảnh hưởng chia fold)
    for split in ('train', 'val'):
        img_dir   = os.path.join(base, 'images', split)
        label_dir = os.path.join(base, label_sub, split)
        names = scan_images(img_dir, args.img_suffix)
        valid = [n for n in names
                if os.path.isfile(os.path.join(label_dir, n + args.img_suffix))]
        print(f'Scanned images/{split}: {len(names)} found, {len(valid)} with valid label')
        for n in valid:
            pool[n] = split

    if not pool:
        raise RuntimeError('No valid image-label pairs found in images/train or images/val.')

    # Gom theo vùng
    region_to_names = defaultdict(list)
    for name in pool:
        region_to_names[region_of(name)].append(name)
    region_counts = {r: len(v) for r, v in region_to_names.items()}

    print(f'\nTotal pool: {len(pool)} real (non-augmented) images across '
          f'{len(region_to_names)} distinct regions')

    folds, fold_sizes = greedy_balanced_folds(region_counts, args.n_folds)

    output_dir = args.output_dir or os.path.join(args.data_root, 'dataset')
    fold_dir = os.path.join(output_dir, 'spatial_folds')
    os.makedirs(fold_dir, exist_ok=True)

    print(f'\n=== Fold assignment (region-stratified, {args.n_folds}-fold) ===')
    for i, (regions, size) in enumerate(zip(folds, fold_sizes), start=1):
        val_names = sorted(n for r in regions for n in region_to_names[r])
        train_names = sorted(n for n in pool if region_of(n) not in regions)

        val_path   = os.path.join(fold_dir, f'fold{i}_val.txt')
        train_path = os.path.join(fold_dir, f'fold{i}_train.txt')
        with open(val_path, 'w') as f:
            f.write('\n'.join(f"images/{pool[n]}/{n}" for n in val_names) + '\n')
        with open(train_path, 'w') as f:
            f.write('\n'.join(f"images/{pool[n]}/{n}" for n in train_names) + '\n')

        print(f'  Fold {i}: {len(regions):>3} regions | val={len(val_names):>5} imgs | '
              f'train={len(train_names):>5} imgs  -> {val_path}')

    # Sanity check: mỗi vùng chỉ xuất hiện ở đúng 1 fold (không rò rỉ không gian)
    seen = {}
    leak = 0
    for i, regions in enumerate(folds, start=1):
        for r in regions:
            if r in seen:
                leak += 1
                print(f'  !! LEAK: region {r!r} in both fold {seen[r]} and fold {i}')
            seen[r] = i
    print(f'\nSpatial-leakage check: {leak} region(s) duplicated across folds '
          f'({"OK — none" if leak == 0 else "FIX NEEDED"})')
    print(f'Output dir: {fold_dir}')


if __name__ == '__main__':
    main()
