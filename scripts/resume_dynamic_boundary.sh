#!/usr/bin/env bash
# scripts/resume_dynamic_boundary.sh — resume Run 4 (Dynamic Boundary-Aware
# Loss) sau khi Kaggle session bị ngắt VÀ /kaggle/working đã bị xoá (session
# mới, không còn latest_checkpoint.pth). Nếu vẫn còn trong cùng session,
# không cần script này — bash scripts/run_dynamic_boundary.sh (cùng tham số
# override nếu có) sẽ tự auto-resume từ checkpoint có sẵn.
#
# Script MỚI, ĐỘC LẬP HOÀN TOÀN với scripts/resume_static_boundary.sh (Run 3),
# scripts/resume_static_boundary_weighted.sh (Run 3b) và mọi script
# resume_*.sh trước đó — KHÔNG gọi các file đó, KHÔNG đụng work_dir của các
# run trước.
#
# Cách dùng:
#   bash scripts/resume_dynamic_boundary.sh                        # mặc định (alpha=0.4)
#   bash scripts/resume_dynamic_boundary.sh --alpha 0.2             # resume đúng run ablation alpha=0.2
#   bash scripts/resume_dynamic_boundary.sh --path /kaggle/working/.../latest_checkpoint.pth   # bỏ qua hỏi
#
# Upload trực tiếp trong 1 cell notebook (thay vì hỏi đường dẫn qua terminal):
#   %run "Tools/get_resume_checkpoint.py" --browser --default-path <WORK_DIR>/latest_checkpoint.pth
#   # sau khi thấy "✔ Đã lưu checkpoint" thì chạy lại: bash scripts/resume_dynamic_boundary.sh [tham số]

set -e

DATA_ROOT="${DATA_ROOT:-}"
WORK_BASE="${WORK_BASE:-/kaggle/working/unetformer-resnet18-dynamic-boundary-openearthmap}"
CONFIG="${CONFIG:-configs/unet_former_resnet18_dynamic_boundary/run4_dynamic_ramp_hold.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse tham số: --alpha (phải khớp run gốc muốn resume), và --path
#    <checkpoint> để bỏ qua bước hỏi ─────────────────────────────────────────
ALPHA=""
EXTRA_PATH_VALUE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)  EXTRA_PATH_VALUE="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        *) echo "Cảnh báo: bỏ qua tham số không nhận dạng được: $1" >&2; shift ;;
    esac
done

WORK_DIR="$WORK_BASE/work_dirs/run4_dynamic_ramp_hold"
OVERRIDE_ARGS=()
if [[ -n "$ALPHA" ]]; then
    WORK_DIR="${WORK_DIR}_alpha$(printf '%.2f' "$ALPHA")"
    OVERRIDE_ARGS+=(--alpha "$ALPHA")
fi

echo "========================================================"
echo " Resume training — Run 4 (Dynamic Boundary-Aware Loss)"
[[ -n "$ALPHA" ]] && echo " alpha: $ALPHA"
echo "========================================================"

# ── Bước 1: Cài requirements ─────────────────────────────────────────────────
echo ""
echo "[1/3] Cài đặt requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>&1 \
    | grep -v -E "pip's dependency resolver|requires .*(incompatible|which is not installed)" \
    || true
echo "      Done."

# ── Bước 2: Xác định checkpoint để resume ────────────────────────────────────
echo ""
echo "[2/3] Xác định checkpoint để resume ..."
mkdir -p "$WORK_DIR"
DEFAULT_CKPT="$WORK_DIR/latest_checkpoint.pth"

set +e
if [[ -n "$EXTRA_PATH_VALUE" ]]; then
    CKPT_PATH="$(python "$SCRIPT_DIR/Tools/get_resume_checkpoint.py" \
        --default-path "$DEFAULT_CKPT" --path "$EXTRA_PATH_VALUE")"
else
    CKPT_PATH="$(python "$SCRIPT_DIR/Tools/get_resume_checkpoint.py" \
        --default-path "$DEFAULT_CKPT")"
fi
GET_CKPT_STATUS=$?
set -e

if [[ $GET_CKPT_STATUS -ne 0 || -z "$CKPT_PATH" ]]; then
    echo "Lỗi: không xác định được checkpoint hợp lệ để resume."
    exit 1
fi
echo "      Checkpoint: $CKPT_PATH"

# ── Bước 3: Resume training ───────────────────────────────────────────────────
echo ""
echo "[3/3] Resume training: $CONFIG"
echo "      Kết quả lưu tại: $WORK_DIR"
echo ""

cd "$SCRIPT_DIR"
EXTRA_ARGS=()
[[ -n "$DATA_ROOT" ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
torchrun --nproc_per_node=2 src/train_dynamic_boundary.py \
    --config "$CONFIG" --resume "$CKPT_PATH" "${OVERRIDE_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành (resume): Run 4 (Dynamic Boundary-Aware Loss)"
echo " Xem kết quả tại: $WORK_DIR"
echo "========================================================"
