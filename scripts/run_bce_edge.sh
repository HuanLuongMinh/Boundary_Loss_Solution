#!/usr/bin/env bash
# scripts/run_bce_edge.sh — thực nghiệm "Baseline vs BCE Loss" (Run 2,
# docs/idea_research.md Bảng 1): L_total = L_seg + lambda_edge * L_edge.
#
# Script MỚI, ĐỘC LẬP HOÀN TOÀN với scripts/run_baseline.sh — KHÔNG gọi file
# đó, KHÔNG đụng config/work_dir của baseline. Baseline vẫn chạy lại y hệt
# bất kỳ lúc nào bằng scripts/run_baseline.sh như trước.
#
# Cách dùng:
#   bash scripts/run_bce_edge.sh                  # lambda_edge=0.4 (mặc định trong bce_edge.yaml)
#   bash scripts/run_bce_edge.sh 0.2               # lambda_edge=0.2 (ablation) -> WORK_DIR tự thêm hậu tố _lambda0.20
#   bash scripts/run_bce_edge.sh --dry-run         # smoke-test: 5 iteration, 4 ảnh, lambda_edge=0.4
#   bash scripts/run_bce_edge.sh 0.2 --dry-run     # smoke-test với lambda_edge=0.2 (thứ tự tham số không quan trọng)
#
# Nếu Kaggle mount dataset ở path khác mặc định trong config, set DATA_ROOT
# trước khi gọi (ghi đè ROOT_DIR/VAL_ROOT_DIR trực tiếp, không tạo file gì):
#   DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_bce_edge.sh

set -e  # dừng ngay nếu có lỗi

# ── Cấu hình đường dẫn ───────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="configs/unet_former_resnet18_bce_edge/bce_edge.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse tham số: --dry-run (cờ) và 1 số thực tuỳ chọn = lambda_edge override,
#    thứ tự không quan trọng (vd "0.2 --dry-run" hoặc "--dry-run 0.2" đều được) ──
DRY_RUN=""
LAMBDA_EDGE=""
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN="--dry-run"
    elif [[ "$arg" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        LAMBDA_EDGE="$arg"
    else
        echo "Cảnh báo: bỏ qua tham số không nhận dạng được: $arg" >&2
    fi
done

echo "========================================================"
echo " UNetFormer — Baseline vs BCE Loss (Run 2)"
echo " L_total = L_seg + lambda_edge * L_edge"
if [[ -n "$LAMBDA_EDGE" ]]; then
    echo " lambda_edge override: $LAMBDA_EDGE  (mặc định trong config: 0.4)"
fi
if [[ -n "$DRY_RUN" ]]; then
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

# ── Bước 2: Chạy training ────────────────────────────────────────────────────
echo ""
echo "[2/2] Bắt đầu training: $CONFIG"
echo ""

cd "$SCRIPT_DIR"

EXTRA_ARGS=()
[[ -n "$DATA_ROOT" ]]    && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "$DRY_RUN" ]]      && EXTRA_ARGS+=(--dry-run)
[[ -n "$LAMBDA_EDGE" ]]  && EXTRA_ARGS+=(--lambda-edge "$LAMBDA_EDGE")

torchrun --nproc_per_node=2 src/train_bce_edge.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành: Baseline vs BCE Loss"
echo " Xem kết quả tại: OUTPUT.WORK_DIR trong $CONFIG (thêm hậu tố _lambdaX.XX nếu có override)"
echo "   - benchmark_results.csv   (mIoU, l_seg/l_edge/l_total, per-class IoU từng lần val)"
echo "   - summary.txt             (báo cáo đầy đủ: edge stats, pos_weight, pre-flight checks)"
echo "   - final_summary.txt       (tóm tắt best mIoU, cùng format baseline)"
echo "   - train_log.txt           (log đầy đủ quá trình chạy)"
echo "   - learning_curves.png     (val mIoU + val L_total qua các lần val)"
echo "   - vis/ , vis/boundary/    (visualizer segmentation + boundary sau mỗi lần val)"
echo "   - sanity/                 (overlay edge_gt — sanity check #6, tự kiểm tra bằng mắt)"
echo "========================================================"
