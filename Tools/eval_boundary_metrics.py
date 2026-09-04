"""Tools/eval_boundary_metrics.py — Retro-fit mIoU + Boundary IoU/BF-Score/ASD
cho 1 checkpoint `best_model.pth` ĐÃ TRAIN XONG (Run 1 baseline hoặc Run 2 BCE
edge), KHÔNG train lại. File MỚI, đặt vào Tools/ đúng convention hiện có
(Tools/measure_edge_ratio.py, Tools/build_clean_val_split.py, ...).

Vì sao script này tồn tại: src/train_bce_edge.py chỉ tính SegmentationMetrics
(mIoU) trong lúc train — BoundaryMetrics (src/utils/boundary_metrics.py, mới)
chưa được cắm vào validate() sống của script đó (cố tình KHÔNG sửa
train_bce_edge.py trong lần retro-fit này). Thay vì train lại toàn bộ 40k
iteration chỉ để có thêm chỉ số, script này chạy 1 LƯỢT FORWARD PASS (không
backward, không cần GPU — CPU chạy được, chỉ chậm hơn) qua val set bằng
checkpoint đã có sẵn, tính lại mIoU (để đối chiếu đúng con số đã có trong
benchmark_results.csv/summary.txt, xác nhận checkpoint nạp đúng) VÀ tính thêm
Boundary IoU(d)/BF-Score/ASD mà lúc train chưa có.

Chạy ĐỘC LẬP cho từng checkpoint (không có wrapper gộp Run 1 + Run 2 trong 1
lần chạy) — mỗi lần 1 --checkpoint, đúng cách bạn sẽ dùng trên Kaggle sau khi
tải best_model.pth về.

Cách dùng (chạy từ REPO ROOT — cả trên Kaggle notebook lẫn máy cá nhân):

    # Run 1 — baseline
    python Tools/eval_boundary_metrics.py \\
        --config configs/unet_former_resnet18_combineLoss/baseline.yaml \\
        --checkpoint /path/to/best_model.pth \\
        --model-type baseline \\
        --output docs/results/run1_boundary_metrics.json

    # Run 2 — BCE edge, λ=0.4 (bắt buộc --model-type bce_edge, khác class model —
    # checkpoint có cả key 'base.*' lẫn 'boundary_head.*')
    python Tools/eval_boundary_metrics.py \\
        --config configs/unet_former_resnet18_bce_edge/bce_edge.yaml \\
        --checkpoint /path/to/best_model.pth \\
        --model-type bce_edge \\
        --output docs/results/run2_lambda04_boundary_metrics.json

    # Run 3 (hoặc Run 3b) — Static Boundary, --model-type static_boundary
    # (StaticBoundaryUNetFormer, src/train_static_boundary.py — Run 3b dùng
    # chung đúng class này, cùng key state_dict, xem
    # docs/run3b_spec_lambda2_05.md mục 7). JSON output đã có sẵn per-class
    # Boundary IoU (boundary_iou_dX_per_class) và ASD 2 chiều
    # (asd_pred_to_gt/asd_gt_to_pred) — không cần script riêng cho việc phụ
    # mục 7 của spec.
    python Tools/eval_boundary_metrics.py \\
        --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml \\
        --checkpoint /path/to/best_model.pth \\
        --model-type static_boundary \\
        --output docs/results/run3_boundary_metrics.json

    # Nếu dataset mount ở path khác path mặc định trong config — override trực
    # tiếp, không cần sửa YAML:
        --data-root /kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap

Không cần GPU: --device mặc định tự chọn 'cuda' nếu có, tự rơi về 'cpu' nếu
không — script CHẠY ĐÚNG trên CPU (chỉ chậm hơn, ResNet-18 nhẹ), không có
bước nào bắt buộc GPU. `PRETRAINED` trong config bị ép về False trước khi
dựng model để tránh tải trọng số ImageNet không cần thiết qua mạng —
checkpoint đã có sẽ ghi đè 100% tham số ngay sau đó.

Lưu ý về d dùng cho Boundary IoU: docs/workflow_2.md mục 4 mô tả spec tổng
quát d ∈ {1,3,5} (Cheng et al.), nhưng src/train_bce_edge.py::
write_bce_edge_summary() (đã tồn tại, không sửa) in cố định 3 nhãn dòng
"Best Boundary IoU @1/@2/@4" trong summary.txt — nên --boundary-distances mặc
định ở đây là "1,2,4" để số tính ra khớp đúng nhãn đã có sẵn trong file đó
(Tools/patch_bce_edge_summary.py dùng trực tiếp key boundary_iou_d1/d2/d4 từ
JSON này). Có thể đổi --boundary-distances nếu chỉ cần số cho mục đích khác
(vd báo cáo riêng theo đúng spec Cheng et al. 1,3,5).
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
from src.models.unet_former_resnet18 import build_model
from src.utils.metrics import SegmentationMetrics
from src.utils.boundary_metrics import BoundaryMetrics


# ── Dataset path resolution (duplicated on purpose từ
# src/train_unet_former_resnet18.py / src/train_bce_edge.py / Tools/
# measure_edge_ratio.py — mỗi script trong project này tự chứa, không phụ
# thuộc lẫn nhau, đúng quy ước "file độc lập" đã dùng xuyên suốt repo) ───────

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

    val_root_in = ds.get('VAL_ROOT_DIR', ds['ROOT_DIR'])
    val_root_base = find_data_base(val_root_in)
    if val_root_base != val_root_in:
        print(f"Resolved VAL_ROOT_DIR: {val_root_in} -> {val_root_base}")
    ds['VAL_ROOT_DIR'] = val_root_base

    label_sub = find_label_subdir(root_base)
    for key, root in (('TRAIN_MASK_DIR', root_base), ('VAL_MASK_DIR', val_root_base)):
        cur = ds[key]
        head, _, tail = cur.partition('/')
        if head in ('labels', 'label') and head != label_sub:
            fixed = label_sub + '/' + tail if tail else label_sub
            print(f"Resolved {key}: '{cur}' -> '{fixed}' (detected '{label_sub}/' under {root})")
            ds[key] = fixed


def build_eval_model(cfg: dict, model_type: str):
    """model_type='baseline' -> UNetFormer thuần (build_model). model_type=
    'bce_edge' -> BCEEdgeUNetFormer (base UNetFormer + BoundaryHead), import
    trực tiếp từ src.train_bce_edge (tái sử dụng đúng class đã train ra
    checkpoint, đảm bảo khớp 100% key state_dict 'base.*'/'boundary_head.*'
    — không định nghĩa lại wrapper này ở đây). model_type='static_boundary'
    -> StaticBoundaryUNetFormer (Run 3 VÀ Run 3b dùng chung 1 kiến trúc wrapper
    giống hệt nhau — base UNetFormer + BoundaryHead, cùng key state_dict —
    import từ src.train_static_boundary, thêm bổ sung THUẦN ADDITIVE cho việc
    phụ mục 7 docs/run3b_spec_lambda2_05.md: eval hậu kỳ per-class Boundary
    IoU cho checkpoint Run 3 mà không cần train lại). KHÔNG đổi hành vi
    'baseline'/'bce_edge' đã có."""
    cfg['MODEL']['PRETRAINED'] = False  # khỏi tải ImageNet weight qua mạng — sắp bị ghi đè hết
    if model_type == 'baseline':
        return build_model(cfg)
    elif model_type == 'bce_edge':
        # Import trễ (không import ở đầu file) vì src/train_bce_edge.py có thể
        # kéo theo code phụ thuộc torch.distributed — chỉ cần đúng lúc dùng,
        # và import module không tự chạy main() (được bọc trong __main__ guard).
        from src.train_bce_edge import BCEEdgeUNetFormer
        return BCEEdgeUNetFormer(cfg)
    elif model_type == 'static_boundary':
        from src.train_static_boundary import StaticBoundaryUNetFormer
        return StaticBoundaryUNetFormer(cfg)
    else:
        raise ValueError(f"--model-type phải là 'baseline'/'bce_edge'/'static_boundary', nhận: {model_type}")


@torch.no_grad()
def run_eval(model, model_type: str, loader, device, num_classes: int,
             ignore_index: int, connectivity: int, dilation_radius: int,
             boundary_distances, bf_tolerance: int):
    model.eval()
    seg_metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    boundary_metrics = BoundaryMetrics(
        num_classes=num_classes, ignore_index=ignore_index,
        boundary_distances=boundary_distances, bf_tolerance=bf_tolerance,
        connectivity=connectivity, dilation_radius=dilation_radius,
    )

    n_images = 0
    t0 = time.time()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks_dev = masks.to(device, non_blocking=True)

        if model_type == 'bce_edge':
            logits, _edge_logits = model(images)  # không cần edge_logits cho metric này
        elif model_type == 'static_boundary':
            logits, _edge_logits, _fused_feature = model(images)  # không cần cho metric này
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
    ap.add_argument('--model-type', required=True, choices=['baseline', 'bce_edge', 'static_boundary'])
    ap.add_argument('--data-root', default=None, help='Override DATASET.ROOT_DIR/VAL_ROOT_DIR (tuỳ chọn)')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--device', default=None, help="'cuda'/'cpu' — mặc định tự chọn")
    ap.add_argument('--boundary-distances', default='1,2,4',
                    help="Danh sách d (pixel) cho Boundary IoU, phân cách bởi dấu phẩy. "
                         "Mặc định '1,2,4' khớp đúng 3 nhãn dòng 'Best Boundary IoU @1/@2/@4' "
                         "đã có sẵn trong src/train_bce_edge.py::write_bce_edge_summary() "
                         "(khác spec tổng quát d=1,3,5 của docs/workflow_2.md mục 4 — xem "
                         "docstring đầu file này).")
    ap.add_argument('--bf-tolerance', type=int, default=2,
                    help='Ngưỡng khoảng cách (pixel) khi so khớp biên GT/pred cho BF-Score')
    ap.add_argument('--output', default=None, help='Lưu kết quả JSON ra đây (tuỳ chọn)')
    args = ap.parse_args()

    boundary_distances = tuple(int(x) for x in args.boundary_distances.split(','))

    with open(args.config, encoding='utf-8') as f:
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
        split_file=ds.get('VAL_SPLIT_FILE'), transform=get_val_transforms(),
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
        connectivity, dilation_radius, boundary_distances, args.bf_tolerance)

    print(f"\n=== Kết quả ({n_images} ảnh, {elapsed:.1f}s, {elapsed / max(n_images, 1):.3f}s/ảnh) ===")
    print(f"mIoU (9 lớp):        {seg_result['mIoU']:.4f}")
    for d in boundary_distances:
        print(f"Boundary IoU d={d}:    {boundary_result[f'boundary_iou_d{d}']:.4f}")
    print(f"BF-Score:            {boundary_result['bf_score']:.4f} "
          f"(precision={boundary_result['bf_precision']:.4f}, recall={boundary_result['bf_recall']:.4f})")
    print(f"ASD:                  {boundary_result['asd']:.4f} pixel")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'checkpoint': args.checkpoint, 'config': args.config, 'model_type': args.model_type,
                'device': str(device), 'n_images': n_images, 'elapsed_sec': elapsed,
                'boundary_distances': list(boundary_distances), 'bf_tolerance': args.bf_tolerance,
                'connectivity': connectivity, 'dilation_radius': dilation_radius,
                'segmentation': seg_result, 'boundary': boundary_result,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nĐã lưu: {args.output}")


if __name__ == '__main__':
    main()
