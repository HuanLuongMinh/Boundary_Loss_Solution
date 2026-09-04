#!/usr/bin/env bash
# scripts/run_static_boundary_weighted.sh — Run 3b (docs/run3b_spec_lambda2_05.md):
# L_total = L_region + alpha * (lambda1*L_BCE_edge + lambda2*L_Affinity),
# lambda1=1.0 (giữ nguyên Run 3), lambda2=0.5 (BIẾN DUY NHẤT thay đổi so với
# Run 3 — alpha_affinity hiệu dụng 0.4 -> 0.2).
#
# Script MỚI, ĐỘC LẬP HOÀN TOÀN với scripts/run_static_boundary.sh (Run 3) và
# mọi script run_*.sh trước đó — KHÔNG gọi các file đó, KHÔNG đụng
# config/work_dir của các run trước. Baseline/Run 2/Run 3 vẫn chạy lại y hệt
# bất kỳ lúc nào.
#
# Cách dùng:
#   bash scripts/run_static_boundary_weighted.sh                          # alpha=0.4, lambda1=1.0, lambda2=0.5 (mặc định trong config)
#   bash scripts/run_static_boundary_weighted.sh --dry-run                 # smoke-test: 5 iteration, 4 ảnh
#   bash scripts/run_static_boundary_weighted.sh --lambda2 0.3             # ablation lambda2 -> WORK_DIR tự thêm hậu tố
#   bash scripts/run_static_boundary_weighted.sh --lambda2 0.3 --dry-run   # thứ tự tham số không quan trọng
#
# Kiểm tra tương thích ngược (mục 6.3 spec) — trỏ CONFIG vào config Run 3 cũ
# (không có LAMBDA1_STATIC/LAMBDA2_STATIC) để xác nhận default 1.0/1.0:
#   CONFIG=configs/unet_former_resnet18_static_boundary/static_boundary.yaml \
#     bash scripts/run_static_boundary_weighted.sh --dry-run
#
# Nếu Kaggle mount dataset ở path khác mặc định trong config, set DATA_ROOT
# trước khi gọi (ghi đè ROOT_DIR/VAL_ROOT_DIR trực tiếp, không tạo file gì):
#   DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_static_boundary_weighted.sh

set -e  # dừng ngay nếu có lỗi

# ── Cấu hình đường dẫn ───────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="${CONFIG:-configs/unet_former_resnet18_static_boundary/run3b_static_lambda2_05.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse tham số: --dry-run (cờ), --alpha/--lambda1/--lambda2 (override số
#    thực, tuỳ chọn), thứ tự không quan trọng ─────────────────────────────────
DRY_RUN=""
ALPHA=""
LAMBDA1=""
LAMBDA2=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --alpha)   ALPHA="$2"; shift 2 ;;
        --lambda1) LAMBDA1="$2"; shift 2 ;;
        --lambda2) LAMBDA2="$2"; shift 2 ;;
        *) echo "Cảnh báo: bỏ qua tham số không nhận dạng được: $1" >&2; shift ;;
    esac
done

echo "========================================================"
echo " UNetFormer — Run 3b (lambda2=0.5, docs/run3b_spec_lambda2_05.md)"
echo " L_total = L_region + alpha * (lambda1*L_BCE_edge + lambda2*L_Affinity)"
[[ -n "$ALPHA"   ]] && echo " alpha override: $ALPHA"
[[ -n "$LAMBDA1" ]] && echo " lambda1 override: $LAMBDA1"
[[ -n "$LAMBDA2" ]] && echo " lambda2 override: $LAMBDA2"
if [[ -n "$DRY_RUN" ]]; then
    echo " [DRY-RUN] Chỉ chạy 5 iteration / 4 ảnh để kiểm tra luồng"
fi
echo " Config: $CONFIG"
echo "========================================================"

# ── Bước 1: Cài requirements ─────────────────────────────────────────────────
echo ""
echo "[1/2] Cài đặt requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>&1 \
    | grep -v -E "pip's dependency resolver|requires .*(incompatible|which is not installed)" \
    || true
echo "      Done."

# ── Bước 2: Chạy training ────────────────────────────────────────────────────
echo ""
echo "[2/2] Bắt đầu training: $CONFIG"
echo ""

cd "$SCRIPT_DIR"

EXTRA_ARGS=()
[[ -n "$DATA_ROOT" ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "$DRY_RUN"   ]] && EXTRA_ARGS+=(--dry-run)
[[ -n "$ALPHA"     ]] && EXTRA_ARGS+=(--alpha "$ALPHA")
[[ -n "$LAMBDA1"   ]] && EXTRA_ARGS+=(--lambda1 "$LAMBDA1")
[[ -n "$LAMBDA2"   ]] && EXTRA_ARGS+=(--lambda2 "$LAMBDA2")

torchrun --nproc_per_node=2 src/train_static_boundary_weighted.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành: Run 3b (lambda2=0.5)"
echo " Xem kết quả tại: OUTPUT.WORK_DIR trong $CONFIG (thêm hậu tố ablation nếu có override)"
echo "   - benchmark_results.csv     (10 dòng/vòng, đủ cột mục 4 spec — miou9/miou8, per-class"
echo "                                 BIoU d1/d2/d4, ASD 2 chiều, lambda1/lambda2/alpha (+effective))"
echo "   - checkpoint_index.json     (map best_miou/best_bfscore/best_bareland/final -> file/iter/round/metrics)"
echo "   - best_miou.pth / best_bfscore.pth / best_bareland.pth / final_iter<N>.pth  (raw state_dict)"
echo "   - summary.txt               (khối 4-checkpoint + mean±std 4 vòng cuối + hệ số effective)"
echo "   - final_summary.txt         (tóm tắt best mIoU, cùng format Run 1/2/3)"
echo "   - train_log.txt             (log đầy đủ quá trình chạy)"
echo "   - learning_curves.png       (val mIoU-9 + val L_total qua các lần val)"
echo "   - vis/ , vis/boundary/      (visualizer segmentation + boundary sau mỗi lần val)"
echo "   - sanity/                   (overlay edge_gt — sanity check, tự kiểm tra bằng mắt)"
echo "========================================================"
