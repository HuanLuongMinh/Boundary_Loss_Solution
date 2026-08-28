#!/usr/bin/env bash
# scripts/resume_baseline.sh — resume baseline UNetFormer (ResNet-18) sau khi
# Kaggle session bị ngắt VÀ /kaggle/working đã bị xoá (session mới, không còn
# latest_checkpoint.pth). Nếu vẫn còn trong cùng session, không cần script
# này — bash scripts/run_baseline.sh sẽ tự auto-resume từ checkpoint có sẵn.
#
# Quy trình: cài requirements → hỏi đường dẫn checkpoint đã upload thủ công
# lên /kaggle/working (qua panel "Data" hoặc File Browser của Kaggle) →
# resume training (dataset root resolve động ngay trong train script, không
# qua file split/danh sách trung gian nào).
#
# Cách dùng:
#   bash scripts/resume_baseline.sh
#   bash scripts/resume_baseline.sh --path /kaggle/working/.../latest_checkpoint.pth   # bỏ qua hỏi
#
# Upload trực tiếp trong 1 cell notebook (thay vì hỏi đường dẫn qua terminal):
#   %run "Tools/get_resume_checkpoint.py" --browser --default-path <WORK_DIR>/latest_checkpoint.pth
#   # sau khi thấy "✔ Đã lưu checkpoint" thì chạy lại: bash scripts/resume_baseline.sh

set -e

DATA_ROOT="${DATA_ROOT:-}"
WORK_BASE="${WORK_BASE:-/kaggle/working/unetformer-resnet18-combinedloss-openearthmap}"
CONFIG="configs/unet_former_resnet18_combineLoss/baseline.yaml"
WORK_DIR="$WORK_BASE/work_dirs/baseline"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXTRA_PATH_VALUE=""
if [[ "${1:-}" == "--path" ]]; then
    EXTRA_PATH_VALUE="${2:-}"
fi

echo "========================================================"
echo " Resume training — UNetFormer Baseline (ResNet-18)"
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
torchrun --nproc_per_node=2 src/train_unet_former_resnet18.py \
    --config "$CONFIG" --resume "$CKPT_PATH" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoàn thành (resume): UNetFormer Baseline (ResNet-18)"
echo " Xem kết quả tại: $WORK_DIR"
echo "========================================================"
