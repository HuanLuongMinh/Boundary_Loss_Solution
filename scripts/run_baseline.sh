#!/usr/bin/env bash
# scripts/run_baseline.sh — 1 lệnh chạy toàn bộ baseline UNetFormer (ResNet-18):
# cài requirements rồi train TRỰC TIẾP trên TOÀN BỘ images/train + images/val
# đã có sẵn trên dataset — KHÔNG split, KHÔNG subset, KHÔNG augment-pad, không
# qua bất kỳ file split/danh sách trung gian nào — 40.000 iteration, val mỗi
# 4.000 iteration, seed=19. Dataset root được resolve ĐỘNG ngay trong train
# script (tự dò path/tên thư mục label — xem resolve_dataset_paths() trong
# src/train_unet_former_resnet18.py), không cần bước tiền xử lý riêng.
#
# File này ĐỘC LẬP với run_unet_former_resnet18_combineLoss_experiment.sh ở
# thư mục gốc (bản sweep 3 config 500/1000/1500 ảnh cũ) — không đụng, không
# đổi file đó. scripts/ là nơi orchestration MỚI, tách biệt với các thực
# nghiệm boundary-loss nghiên cứu sẽ làm sau.
#
# Cách dùng:
#   bash scripts/run_baseline.sh              # train full dataset (40k iter)
#   bash scripts/run_baseline.sh --dry-run    # smoke-test: 4 ảnh, 5 iteration
#
# Nếu Kaggle mount dataset ở path khác mặc định trong config, set DATA_ROOT
# trước khi gọi (ghi đè ROOT_DIR/VAL_ROOT_DIR trực tiếp, không tạo file gì):
#   DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_baseline.sh

set -e  # dừng ngay nếu có lỗi

# ── Cấu hình đường dẫn ───────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="configs/unet_former_resnet18_combineLoss/baseline.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN="${1:-}"

echo "========================================================"
echo " UNetFormer Baseline (ResNet-18 + GLTB) — full dataset, no split"
if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo " [DRY-RUN] Chỉ chạy 5 iteration / 4 ảnh để kiểm tra luồng"
fi
echo "========================================================"

# ── Bước 1: Cài requirements ─────────────────────────────────────────────────
echo ""
echo "[1/2] Cài đặt requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>&1 \
    | grep -v -E "pip's dependency resolver|requires .*(incompatible|which is not installed)" \
    || true
echo "      Done."

# ── Bước 2: Chạy training (dataset root resolve động bên trong train script) ──
echo ""
echo "[2/2] Bắt đầu training: $CONFIG"
echo ""

cd "$SCRIPT_DIR"

EXTRA_ARGS=()
[[ -n "$DATA_ROOT" ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ "$DRY_RUN" == "--dry-run" ]] && EXTRA_ARGS+=(--dry-run)

torchrun --nproc_per_node=2 src/train_unet_former_resnet18.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành: UNetFormer Baseline (ResNet-18)"
echo " Xem kết quả tại: OUTPUT.WORK_DIR trong $CONFIG"
echo "   - benchmark_results.csv   (số liệu mIoU/loss/per-class từng lần val)"
echo "   - train_log.txt           (log đầy đủ quá trình chạy)"
echo "   - final_summary.txt       (best mIoU ở lần val nào + per-class IoU + tổng thời gian)"
echo "   - learning_curves.png     (val mIoU + val Loss qua các lần val)"
echo "   - vis/ , vis/boundary/    (visualizer segmentation + boundary sau mỗi lần val)"
echo "========================================================"
