#!/usr/bin/env bash
# scripts/resume_static_boundary.sh — resume "+Static Boundary" (Run 3) sau
# khi Kaggle session bị ngắt VÀ /kaggle/working đã bị xoá (session mới, không
# còn latest_checkpoint.pth). Nếu vẫn còn trong cùng session, không cần script
# này — bash scripts/run_static_boundary.sh (cùng tham số alpha nếu có) sẽ tự
# auto-resume từ checkpoint có sẵn.
#
# Script MỚI, ĐỘC LẬP HOÀN TOÀN với scripts/resume_baseline.sh và
# scripts/resume_bce_edge.sh — KHÔNG gọi 2 file đó, KHÔNG đụng work_dir của
# 2 run trước.
#
# Cách dùng:
#   bash scripts/resume_static_boundary.sh                        # alpha=0.4 (mặc định)
#   bash scripts/resume_static_boundary.sh 0.2                     # resume đúng run ablation alpha=0.2
#   bash scripts/resume_static_boundary.sh 0.2 --path /kaggle/working/.../latest_checkpoint.pth   # bỏ qua hỏi
#
# Upload trực tiếp trong 1 cell notebook (thay vì hỏi đường dẫn qua terminal):
#   %run "Tools/get_resume_checkpoint.py" --browser --default-path <WORK_DIR>/latest_checkpoint.pth
#   # sau khi thấy "✔ Đã lưu checkpoint" thì chạy lại: bash scripts/resume_static_boundary.sh [alpha]

set -e

DATA_ROOT="${DATA_ROOT:-}"
WORK_BASE="${WORK_BASE:-/kaggle/working/unetformer-resnet18-static-boundary-openearthmap}"
CONFIG="configs/unet_former_resnet18_static_boundary/static_boundary.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse tham số: 1 số thực tuỳ chọn = alpha (phải khớp run gốc muốn
#    resume), và --path <checkpoint> để bỏ qua bước hỏi ─────────────────────
ALPHA=""
EXTRA_PATH_VALUE=""
ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    a="${ARGS[$i]}"
    if [[ "$a" == "--path" ]]; then
        i=$((i + 1))
        EXTRA_PATH_VALUE="${ARGS[$i]:-}"
    elif [[ "$a" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        ALPHA="$a"
    fi
    i=$((i + 1))
done

WORK_DIR="$WORK_BASE/work_dirs/static_boundary"
ALPHA_ARGS=()
if [[ -n "$ALPHA" ]]; then
    WORK_DIR="${WORK_DIR}_alpha$(printf '%.2f' "$ALPHA")"
    ALPHA_ARGS+=(--alpha "$ALPHA")
fi

echo "========================================================"
echo " Resume training — +Static Boundary"
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
torchrun --nproc_per_node=2 src/train_static_boundary.py \
    --config "$CONFIG" --resume "$CKPT_PATH" "${ALPHA_ARGS[@]}" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành (resume): +Static Boundary"
echo " Xem kết quả tại: $WORK_DIR"
echo "========================================================"
