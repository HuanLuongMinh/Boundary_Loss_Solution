"""
build_clean_val_split.py — Tạo validation split KHÔNG PADDING bằng ảnh
augment (hflip/vflip/rot90/...), chỉ gồm ảnh gốc thật sự, dùng cho Phase 1
(Fixed 80/20 split) của đề tài "Dynamic Boundary-Aware Loss via Adaptive
BCE-Affinity Coupling" (idead_research.md).

LÝ DO CẦN SCRIPT NÀY (đọc kỹ trước khi dùng):
    dataset/val_2000_fixed.txt hiện có trong repo được sinh bởi
    Tools/prepare_splits.py, và trong ~2000 dòng đó có tới ~1462 dòng
    (≈73%) là ảnh AUGMENTED (hậu tố "_augN": hflip/vflip/rot90/rot180/
    rot270/hflip_rot90 của cùng 538 ảnh gốc) — được thêm vào chỉ để "đệm"
    cho đủ 2000 mẫu val phục vụ các track ablation CNN/backbone cũ (nơi
    mục tiêu là ước lượng mIoU ổn định, ít nhạy với việc vài mẫu bị lặp
    dưới dạng lật/xoay).

    Với nghiên cứu boundary loss (đo Boundary IoU/BF-Score/ASD ở dải
    1-3 pixel), việc lặp lại ảnh dưới dạng lật/xoay KHÔNG tạo thêm thông
    tin biên mới — hình học biên chỉ bị phản chiếu/xoay chứ không đổi độ
    khó — nên 2000 "mẫu" thực chất chỉ có ~538 cấu hình biên độc lập.
    Tính mean/std hoặc so sánh có ý nghĩa thống kê trên tập có
    pseudo-replication như vậy sẽ đánh giá sai độ tin cậy (giả lập cỡ mẫu
    lớn hơn thực tế, làm hẹp giả tạo khoảng tin cậy khi so Bảng 1).

    => Script này quét lại chính thư mục images/val gốc (KHÔNG copy/augment
    gì thêm), lọc ảnh có label hợp lệ, rồi ghi toàn bộ (hoặc N ảnh đầu tiên
    sau khi shuffle theo seed) vào 1 file split — không có hậu tố "_aug".

Usage:
    python Tools/build_clean_val_split.py \
        --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
        --output-dir /kaggle/working/unetformer-resnet18-combinedloss-openearthmap

    # Giới hạn số lượng (vd chỉ lấy 500 ảnh val thật, khớp tỉ lệ 6:1:3 gốc của paper):
    python Tools/build_clean_val_split.py --data-root ... --max-size 500
"""

import argparse
import os
import random


def scan_images(directory, suffix='.tif'):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Directory not found: {directory}')
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
        if os.path.isdir(os.path.join(c, 'images', 'val')):
            return c
    raise FileNotFoundError(
        f'Could not find images/val under any of: {candidates}')


def find_label_subdir(base):
    for name in ('labels', 'label'):
        if os.path.isdir(os.path.join(base, name)):
            return name
    return 'labels'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', default=None,
                   help='Nơi ghi val_clean_fixed.txt (mặc định: <data-root>/dataset)')
    p.add_argument('--seed', type=int, default=19)
    p.add_argument('--img-suffix', default='.tif')
    p.add_argument('--max-size', type=int, default=None,
                   help='Nếu đặt, chỉ lấy N ảnh (shuffle theo seed) thay vì toàn bộ '
                        'ảnh val thật hiện có — KHÔNG BAO GIỜ augment để bù thêm.')
    args = p.parse_args()

    base = find_data_base(args.data_root)
    label_sub = find_label_subdir(base)
    val_dir   = os.path.join(base, 'images', 'val')
    label_dir = os.path.join(base, label_sub, 'val')

    print(f'Scanning: {val_dir}')
    all_names = scan_images(val_dir, args.img_suffix)
    valid_names = [n for n in all_names
                  if os.path.isfile(os.path.join(label_dir, n + args.img_suffix))]
    n_skipped = len(all_names) - len(valid_names)
    if n_skipped:
        print(f'  WARNING: {n_skipped} image(s) skipped — no matching label found')
    print(f'  {len(valid_names)} valid original (non-augmented) image-label pairs found')

    random.seed(args.seed)
    names = list(valid_names)
    random.shuffle(names)

    if args.max_size is not None:
        if len(names) < args.max_size:
            print(f'  WARNING: requested --max-size={args.max_size} but only '
                  f'{len(names)} real images exist — using all {len(names)} '
                  f'(script KHÔNG augment để bù, theo đúng chủ đích tránh pseudo-replication).')
        names = names[:args.max_size]

    output_dir = args.output_dir or os.path.join(args.data_root, 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    tag = f'{len(names)}' if args.max_size is None else f'{args.max_size}'
    out_path = os.path.join(output_dir, f'val_clean_{tag}.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(f'images/val/{n}' for n in names) + '\n')

    print(f'\n=== Done ===')
    print(f'  Total (real, non-augmented) samples : {len(names)}')
    print(f'  Output file                         : {out_path}')
    print(f'\n  So sánh với dataset/val_2000_fixed.txt cũ: file đó có 2000 dòng nhưng chỉ')
    print(f'  ~538 ảnh gốc độc lập (~73% là bản lật/xoay của cùng ảnh) — file mới này')
    print(f'  không có hiện tượng đó, phù hợp hơn cho đánh giá boundary-level metrics.')


if __name__ == '__main__':
    main()
