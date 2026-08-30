"""Tools/eval_boundary_metrics.py — Retro-fit mIoU + Boundary IoU/BF-Score/ASD
cho 1 checkpoint `best_model.pth` ĐÃ TRAIN XONG (Run 1 baseline hoặc Run 2 BCE
edge), KHÔNG train lại. Đây là bản THAM KHẢO, đặt vào thư mục Tools/ đúng
convention hiện có (build_clean_val_split.py, measure_edge_ratio.py, ...).

Vì sao script này tồn tại: `train_bce_edge.py`/`train_unet_former_resnet18.py`
chỉ tính SegmentationMetrics (mIoU) trong lúc train — `BoundaryMetrics`
(src/utils/boundary_metrics.py, gửi kèm trước đó) chưa được cắm vào 2 script
đó. Thay vì train lại toàn bộ 40k iteration chỉ để có thêm 2 chỉ số, script
này chạy 1 LƯỢT FORWARD PASS (không backward, không cần GPU — CPU chạy được,
chỉ chậm hơn) qua val set bằng checkpoint đã có sẵn, tính lại mIoU (để đối
chiếu đúng con số đã có trong benchmark_results.csv, xác nhận checkpoint nạp
đúng) VÀ tính thêm Boundary IoU(d)/BF-Score/ASD mà lúc train chưa có.

Cách dùng (chạy từ REPO ROOT — cả trên Kaggle notebook lẫn máy cá nhân):

    # Run 1 — baseline
    python Tools/eval_boundary_metrics.py \\
        --config configs/unet_former_resnet18_combineLoss/baseline.yaml \\
        --checkpoint /path/to/best_model.pth \\
        --model-type baseline \\
        --output docs/results/run1_boundary_metrics.json

    # Run 2 — BCE edge (bắt buộc --model-type bce_edge, khác class model)
    python Tools/eval_boundary_metrics.py \\
        --config configs/unet_former_resnet18_bce_edge/bce_edge.yaml \\
        --checkpoint /path/to/best_model.pth \\
        --model-type bce_edge \\
        --output docs/results/run2_lambda04_boundary_metrics.json

    # Nếu dataset mount ở path khác path mặc định trong config (giống DATA_ROOT
    # của scripts/run_baseline.sh) — override trực tiếp, không cần sửa YAML:
        --data-root /kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap
    # (máy cá nhân: trỏ thẳng vào thư mục bạn tải images/val + labels/val về)

Không cần GPU: --device mặc định tự chọn 'cuda' nếu có, tự rơi về 'cpu' nếu
không — script CHẠY ĐÚNG trên CPU (chỉ chậm hơn), không có bước nào bắt buộc
GPU. `PRETRAINED` trong config bị ép về False trước khi dựng model (dòng ~90)
để tránh tải trọng số ImageNet không cần thiết qua mạng — checkpoint đã có sẽ
ghi đè 100% tham số ngay sau đó.
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

# Cho phép chạy `python Tools/eval_boundary_metrics.py` từ repo root mà không
# cần cài package — thêm repo root vào sys.path (giống cách các script khác
# trong Tools/ đã làm).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_val_transforms
from src.utils.metrics import SegmentationMetrics
from src.utils.boundary_metrics import BoundaryMetrics
from src.train_unet_former_resnet18 import resolve_dataset_paths, build_model


def build_eval_model(cfg: dict, model_type: str):
    """model_type='baseline' -> UNetFormer thuần (build_model). model_type=
    'bce_edge' -> BCEEdgeUNetFormer (base UNetFormer + BoundaryHead) — BẮT
    BUỘC dùng đúng wrapper này cho checkpoint Run 2, nếu không load_state_dict
    sẽ báo lỗi thiếu/thừa key (checkpoint Run 2 có cả key 'base.*' lẫn
    'boundary_head.*')."""
    cfg['MODEL']['PRETRAINED'] = False  # khỏi tải ImageNet weight qua mạng — sắp bị ghi đè hết
    if model_type == 'baseline':
        return build_model(cfg)
    elif model_type == 'bce_edge':
        # Import trễ (không import ở đầu file) vì train_bce_edge.py có thể kéo
        # theo code phụ thuộc DDP/argparse — chỉ cần đúng lúc dùng.
        from src.train_bce_edge import BCEEdgeUNetFormer
        return BCEEdgeUNetFormer(cfg)
    else:
        raise ValueError(f"--model-type phải là 'baseline' hoặc 'bce_edge', nhận: {model_type}")


@torch.no_grad()
def run_eval(model, model_type: str, loader, device, num_classes: int,
             ignore_index: int, connectivity: int, dilation_radius: int):
    model.eval()
    seg_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    boundary_metrics = BoundaryMetrics(
        num_classes=num_classes, ignore_index=ignore_index,
        boundary_distances=(1, 3, 5), bf_tolerance=2,
        connectivity=connectivity, dilation_radius=dilation_radius,
    )

    n_images = 0
    t0 = time.time()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks_dev = masks.to(device, non_blocking=True)

        if model_type == 'bce_edge':
            logits, _edge_logits = model(images)  # không cần edge_logits cho metric này
        else:
            logits = model(images)

        seg_metrics.update(logits, masks_dev)
        boundary_metrics.update(logits, masks_dev)
        n_images += images.shape[0]
        print(f"  ... {n_images} ảnh đã xử lý ({time.time() - t0:.1f}s)", end='\r')

    print()
    return seg_metrics.compute(), boundary_metrics.compute(), n_images, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='Đúng YAML đã dùng lúc train run này')
    ap.add_argument('--checkpoint', required=True, help='Đường dẫn best_model.pth')
    ap.add_argument('--model-type', required=True, choices=['baseline', 'bce_edge'])
    ap.add_argument('--data-root', default=None, help='Override DATASET.ROOT_DIR/VAL_ROOT_DIR (tuỳ chọn)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--device', default=None, help="'cuda'/'cpu' — mặc định tự chọn")
    ap.add_argument('--output', default=None, help='Lưu kết quả JSON ra đây (tuỳ chọn)')
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.data_root:
        cfg['DATASET']['ROOT_DIR'] = args.data_root
        cfg['DATASET']['VAL_ROOT_DIR'] = args.data_root
    resolve_dataset_paths(cfg['DATASET'])  # tự dò path/tên thư mục label, giống lúc train

    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ds = cfg['DATASET']
    val_ds = OpenEarthMapDataset(
        root_dir=ds['VAL_ROOT_DIR'], img_dir=ds['VAL_IMG_DIR'], mask_dir=ds['VAL_MASK_DIR'],
        transform=get_val_transforms(),
    )
    print(f"Val set: {len(val_ds)} ảnh ({ds['VAL_ROOT_DIR']}/{ds['VAL_IMG_DIR']})")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=(device.type == 'cuda'))

    model = build_eval_model(cfg, args.model_type).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    # strict=True (mặc định): tự raise RuntimeError ngay nếu key thiếu/thừa —
    # đây chính là cách phát hiện sớm nếu lỡ chọn nhầm --model-type (baseline
    # checkpoint chỉ có key model gốc, bce_edge checkpoint có thêm
    # 'boundary_head.*') thay vì im lặng nạp sai.
    model.load_state_dict(state_dict)
    print(f"Đã nạp checkpoint: {args.checkpoint}")

    boundary_cfg = cfg.get('BOUNDARY_LOSS', {})
    connectivity = boundary_cfg.get('CONNECTIVITY', 4)
    dilation_radius = boundary_cfg.get('DILATION_RADIUS', 0)
    num_classes = cfg['TRAIN']['NUM_CLASSES']
    ignore_index = OpenEarthMapDataset.IGNORE_INDEX

    seg_result, boundary_result, n_images, elapsed = run_eval(
        model, args.model_type, val_loader, device, num_classes, ignore_index,
        connectivity, dilation_radius)

    print(f"\n=== Kết quả ({n_images} ảnh, {elapsed:.1f}s, {elapsed / max(n_images, 1):.3f}s/ảnh) ===")
    print(f"mIoU (9 lớp):        {seg_result['mIoU']:.4f}")
    for d in (1, 3, 5):
        print(f"Boundary IoU d={d}:    {boundary_result[f'boundary_iou_d{d}']:.4f}")
    print(f"BF-Score:            {boundary_result['bf_score']:.4f} "
          f"(precision={boundary_result['bf_precision']:.4f}, recall={boundary_result['bf_recall']:.4f})")
    print(f"ASD:                  {boundary_result['asd']:.4f} pixel")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({
                'checkpoint': args.checkpoint, 'config': args.config, 'model_type': args.model_type,
                'n_images': n_images, 'elapsed_sec': elapsed,
                'segmentation': seg_result, 'boundary': boundary_result,
            }, f, indent=2)
        print(f"\nĐã lưu: {args.output}")


if __name__ == '__main__':
    main()
