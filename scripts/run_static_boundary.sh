#!/usr/bin/env bash
# scripts/run_static_boundary.sh — thực nghiệm "+Static Boundary" (Run 3,
# docs/idea_research.md Bảng 1): L_total = L_region + alpha * (L_BCE_edge + L_Affinity),
# lambda1=lambda2=1 cố định.
#
# Script MỚI, ĐỘC LẬP HOÀN TOÀN với scripts/run_baseline.sh và
# scripts/run_bce_edge.sh — KHÔNG gọi 2 file đó, KHÔNG đụng config/work_dir
# của 2 run trước. Baseline và Run 2 vẫn chạy lại y hệt bất kỳ lúc nào.
#
# Cách dùng:
#   bash scripts/run_static_boundary.sh                  # alpha=0.4 (mặc định trong static_boundary.yaml)
#   bash scripts/run_static_boundary.sh 0.2               # alpha=0.2 (ablation) -> WORK_DIR tự thêm hậu tố _alpha0.20
#   bash scripts/run_static_boundary.sh --dry-run         # smoke-test: 5 iteration, 4 ảnh, alpha=0.4
#   bash scripts/run_static_boundary.sh 0.2 --dry-run     # smoke-test với alpha=0.2 (thứ tự tham số không quan trọng)
#
# Nếu Kaggle mount dataset ở path khác mặc định trong config, set DATA_ROOT
# trước khi gọi (ghi đè ROOT_DIR/VAL_ROOT_DIR trực tiếp, không tạo file gì):
#   DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_static_boundary.sh

set -e  # dừng ngay nếu có lỗi

# ── Cấu hình đường dẫn ───────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="configs/unet_former_resnet18_static_boundary/static_boundary.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse tham số: --dry-run (cờ) và 1 số thực tuỳ chọn = alpha override,
#    thứ tự không quan trọng (vd "0.2 --dry-run" hoặc "--dry-run 0.2" đều được) ──
DRY_RUN=""
ALPHA=""
for arg in "$@"; do
    if [[ "$arg" == "--dry-run" ]]; then
        DRY_RUN="--dry-run"
    elif [[ "$arg" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        ALPHA="$arg"
    else
        echo "Cảnh báo: bỏ qua tham số không nhận dạng được: $arg" >&2
    fi
done

echo "========================================================"
echo " UNetFormer — +Static Boundary (Run 3)"
echo " L_total = L_region + alpha * (L_BCE_edge + L_Affinity)  [lambda1=lambda2=1]"
if [[ -n "$ALPHA" ]]; then
    echo " alpha override: $ALPHA  (mặc định trong config: 0.4)"
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
[[ -n "$DATA_ROOT" ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "$DRY_RUN" ]]   && EXTRA_ARGS+=(--dry-run)
[[ -n "$ALPHA" ]]     && EXTRA_ARGS+=(--alpha "$ALPHA")

torchrun --nproc_per_node=2 src/train_static_boundary.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành: +Static Boundary"
echo " Xem kết quả tại: OUTPUT.WORK_DIR trong $CONFIG (thêm hậu tố _alphaX.XX nếu có override)"
echo "   - benchmark_results.csv   (mIoU, l_region/l_bce/l_affinity/l_total, boundary metrics, per-class IoU từng lần val)"
echo "   - summary.txt             (báo cáo đầy đủ: edge stats, pos_weight, affinity config, boundary metrics, pre-flight checks)"
echo "   - final_summary.txt       (tóm tắt best mIoU, cùng format baseline/Run 2)"
echo "   - train_log.txt           (log đầy đủ quá trình chạy)"
echo "   - learning_curves.png     (val mIoU + val L_total qua các lần val)"
echo "   - vis/ , vis/boundary/    (visualizer segmentation + boundary sau mỗi lần val)"
echo "   - sanity/                 (overlay edge_gt — sanity check #6, tự kiểm tra bằng mắt)"
echo "========================================================"
