#!/usr/bin/env bash
# scripts/run_dynamic_boundary.sh — Run 4 (docs/spec-run4-dynamic-weighting.md):
# L_total = L_region + alpha * (lambda1(t)*L_BCE_edge + lambda2(t)*L_Affinity),
# lambda1(t)=1.0 phang, lambda2(t): 0 (warmup 10%) -> ramp tuyen tinh (10%-30%)
# -> 1.0 (hold, 30%-100%). BIEN DUY NHAT so voi Run 3: DYNAMIC_WEIGHTS false -> true.
#
# Script MỚI, ĐỘC LẬP HOÀN TOÀN với scripts/run_static_boundary.sh (Run 3),
# scripts/run_static_boundary_weighted.sh (Run 3b) và mọi script run_*.sh
# trước đó — KHÔNG gọi các file đó, KHÔNG đụng config/work_dir của các run
# trước. Baseline/Run 2/Run 3/Run 3b vẫn chạy lại y hệt bất kỳ lúc nào.
#
# Cách dùng:
#   bash scripts/run_dynamic_boundary.sh                    # alpha=0.4, schedule=ramp_hold (mặc định trong config)
#   bash scripts/run_dynamic_boundary.sh --dry-run           # smoke-test: 100 iteration, val mỗi 25 (mục 5 spec)
#   bash scripts/run_dynamic_boundary.sh --alpha 0.2          # ablation alpha -> WORK_DIR tự thêm hậu tố
#   bash scripts/run_dynamic_boundary.sh --alpha 0.2 --dry-run
#
# Kiểm tra tương thích ngược (mục 5.3 spec) — trỏ CONFIG vào config Run 3 cũ
# (DYNAMIC_WEIGHTS: false) để xác nhận lambda1=lambda2=1.0 cố định, bỏ qua
# schedule hoàn toàn:
#   CONFIG=configs/unet_former_resnet18_static_boundary/static_boundary.yaml \
#     bash scripts/run_dynamic_boundary.sh --dry-run
#
# Nếu Kaggle mount dataset ở path khác mặc định trong config, set DATA_ROOT
# trước khi gọi (ghi đè ROOT_DIR/VAL_ROOT_DIR trực tiếp, không tạo file gì):
#   DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_dynamic_boundary.sh

set -e  # dừng ngay nếu có lỗi

# ── Cấu hình đường dẫn ───────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="${CONFIG:-configs/unet_former_resnet18_dynamic_boundary/run4_dynamic_ramp_hold.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse tham số: --dry-run (cờ), --alpha (override số thực, tuỳ chọn),
#    thứ tự không quan trọng. Schedule (SCHEDULE/LAMBDA1_CONST/LAMBDA2_END/
#    WARMUP_FRAC/RAMP_END_FRAC) chỉ đọc từ config — không có CLI override,
#    đúng thiết kế train_dynamic_boundary.py ────────────────────────────────
DRY_RUN=""
ALPHA=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --alpha)   ALPHA="$2"; shift 2 ;;
        *) echo "Cảnh báo: bỏ qua tham số không nhận dạng được: $1" >&2; shift ;;
    esac
done

echo "========================================================"
echo " UNetFormer — Run 4 (Dynamic Boundary-Aware Loss, ramp -> hold)"
echo " docs/spec-run4-dynamic-weighting.md"
echo " L_total = L_region + alpha * (lambda1(t)*L_BCE_edge + lambda2(t)*L_Affinity)"
[[ -n "$ALPHA" ]] && echo " alpha override: $ALPHA"
if [[ -n "$DRY_RUN" ]]; then
    echo " [DRY-RUN] Chỉ chạy 100 iteration, val mỗi 25 (mục 5 spec) để kiểm tra luồng"
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

# ── Bước 2: Kiểm tra GPU (bắt buộc theo mục 6 spec — KHÔNG chạy training thật
#    nếu dính P100, sm_60 không tương thích build PyTorch hiện tại) ──────────
if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "Kiểm tra GPU ..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
        echo "      GPU: $GPU_NAME"
        if [[ "$GPU_NAME" == *"P100"* ]]; then
            echo "Lỗi: phát hiện Tesla P100 (sm_60, không tương thích build PyTorch hiện tại)."
            echo "     Đợi phiên Kaggle khác (2x T4) rồi chạy lại."
            exit 1
        fi
    else
        echo "      Cảnh báo: không tìm thấy nvidia-smi — bỏ qua kiểm tra GPU."
    fi
fi

# ── Bước 3: Chạy training ────────────────────────────────────────────────────
echo ""
echo "[2/2] Bắt đầu training: $CONFIG"
echo ""

cd "$SCRIPT_DIR"

EXTRA_ARGS=()
[[ -n "$DATA_ROOT" ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "$DRY_RUN"   ]] && EXTRA_ARGS+=(--dry-run)
[[ -n "$ALPHA"     ]] && EXTRA_ARGS+=(--alpha "$ALPHA")

torchrun --nproc_per_node=2 src/train_dynamic_boundary.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành: Run 4 (Dynamic Boundary-Aware Loss)"
echo " Xem kết quả tại: OUTPUT.WORK_DIR trong $CONFIG (thêm hậu tố ablation nếu có override)"
echo "   - benchmark_results.csv     (10 dòng/vòng, đủ cột — miou9/miou8, per-class BIoU d1/d2/d4,"
echo "                                 ASD 2 chiều, lambda1/lambda2/alpha (+effective) đổi theo vòng)"
echo "   - lambda_schedule_log.csv   (iter/lambda1/lambda2/effective/4 loss, mỗi 200 iter — hình lịch trình)"
echo "   - checkpoint_index.json     (map best_miou/best_bfscore/best_bareland/final -> file/iter/round/metrics)"
echo "   - best_miou.pth / best_bfscore.pth / best_bareland.pth / final_iter<N>.pth  (raw state_dict)"
echo "   - summary.txt               (khối DYNAMIC SCHEDULE + 4-checkpoint + mean±std 4 vòng cuối)"
echo "   - final_summary.txt         (tóm tắt best mIoU, cùng format Run 1/2/3/3b)"
echo "   - train_log.txt             (log đầy đủ quá trình chạy)"
echo "   - learning_curves.png, schedule_curves.png (mIoU/loss + lambda1/lambda2 theo iteration)"
echo "   - vis/ , vis/boundary/      (visualizer segmentation + boundary sau mỗi lần val)"
echo "   - sanity/                   (overlay edge_gt — sanity check, tự kiểm tra bằng mắt)"
echo "========================================================"
