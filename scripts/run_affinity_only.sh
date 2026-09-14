#!/usr/bin/env bash
# scripts/run_affinity_only.sh — Run 7, "Affinity-only"
# (docs/spec-run7-affinity-only-ban-giao-claude-code.md):
# L_total = L_region + alpha * lambda2 * L_Affinity — KHONG co so hang BCE
# nao (khong phai chi nhan trong so 0, xem docstring src/train_affinity_only.py).
#
# Script MOI, DOC LAP HOAN TOAN voi moi scripts/run_*.sh truoc do — KHONG goi
# cac file do, KHONG dung config/work_dir cua run nao khac. Baseline/Run 2/
# Run 3/3b/4/5 van chay lai y het bat ky luc nao.
#
# KHAC BIET QUAN TRONG so voi cac script run_*.sh khac: --seed la BAT BUOC
# (config Run 7 KHONG co TRAIN.SEED mac dinh — muc 1.3 spec). Script nay tu
# kiem tra va dung SOM (truoc ca buoc pip install) neu thieu --seed, thay vi
# de torchrun khoi dong roi moi bao loi.
#
# Cach dung (1 lenh duy nhat cho toan bo phan train cua thuc nghiem nay):
#   bash scripts/run_affinity_only.sh --seed 19                    # train that, alpha=0.4 (config mac dinh)
#   bash scripts/run_affinity_only.sh --seed 19 --dry-run           # smoke-test: MAX_ITERS=100
#   bash scripts/run_affinity_only.sh --seed 86 --dry-run           # seed thu 2 (khuyen nghi, vai tro nhu Run 5)
#   bash scripts/run_affinity_only.sh --seed 19 --alpha 0.3         # ablation nhanh -> WORK_DIR tu them hau to
#   bash scripts/run_affinity_only.sh --seed 19 --work-dir work_dirs/foo  # doi WORK_DIR tuong minh
#
# Lieu phu (muc 1.2 spec, CHI chay sau khi da co ket qua lieu 0.4 o >=1 seed):
#   CONFIG=configs/unet_former_resnet18_affinity_only/run7b_affinity_only_lambda02.yaml \
#     bash scripts/run_affinity_only.sh --seed 19 --dry-run
#
# Neu Kaggle mount dataset o path khac mac dinh trong config, set DATA_ROOT
# truoc khi goi (ghi de ROOT_DIR/VAL_ROOT_DIR truc tiep, khong tao file gi):
#   DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_affinity_only.sh --seed 19
#
# Script nay CHI train (buoc "1 lan duy nhat" nang nhat, ~2.5-3.5h/seed). Dump
# preds/per-image-stats (Tools/eval_boundary_metrics.py) va bootstrap CI
# (Tools/bootstrap_boundary_ci_run7.py) la 2 buoc RIENG, nhe, chay SAU khi co
# checkpoint — xem docs/run7-huong-dan-chay.md muc 5-6 (khong gop vao day vi
# 2 buoc do can duong dan checkpoint cu, ngoai tam kiem soat cua 1 lenh train).

set -e  # dung ngay neu co loi

# ── Cau hinh duong dan ───────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="${CONFIG:-configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Parse tham so: --dry-run (co), --seed (BAT BUOC), --alpha/--lambda2
#    (override so thuc, tuy chon), --work-dir (override duong dan), thu tu
#    khong quan trong ──────────────────────────────────────────────────────
DRY_RUN=""
ALPHA=""
LAMBDA2=""
SEED=""
WORK_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)  DRY_RUN="--dry-run"; shift ;;
        --alpha)    ALPHA="$2"; shift 2 ;;
        --lambda2)  LAMBDA2="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --work-dir) WORK_DIR_OVERRIDE="$2"; shift 2 ;;
        *) echo "Canh bao: bo qua tham so khong nhan dang duoc: $1" >&2; shift ;;
    esac
done

# ── Seed BAT BUOC — dung SOM, truoc ca pip install (muc 1.3.2 spec) ─────────
if [[ -z "$SEED" ]]; then
    echo "LOI: --seed la bat buoc cho Run 7 (config nay khong co TRAIN.SEED co dinh)." >&2
    echo "     Vi du: bash scripts/run_affinity_only.sh --seed 19" >&2
    exit 1
fi

echo "========================================================"
echo " UNetFormer — Run 7 (Affinity-only, docs/spec-run7-affinity-only-ban-giao-claude-code.md)"
echo " L_total = L_region + alpha * lambda2 * L_Affinity  (khong co nhanh BCE)"
echo " Seed: $SEED"
[[ -n "$ALPHA"   ]] && echo " alpha override: $ALPHA"
[[ -n "$LAMBDA2" ]] && echo " lambda2 override: $LAMBDA2"
[[ -n "$WORK_DIR_OVERRIDE" ]] && echo " work-dir override: $WORK_DIR_OVERRIDE"
if [[ -n "$DRY_RUN" ]]; then
    echo " [DRY-RUN] MAX_ITERS=100 de kiem tra luong (muc 3 spec)"
fi
echo " Config: $CONFIG"
echo "========================================================"

# ── Buoc 1: Cai requirements ─────────────────────────────────────────────────
echo ""
echo "[1/2] Cai dat requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>&1 \
    | grep -v -E "pip's dependency resolver|requires .*(incompatible|which is not installed)" \
    || true
echo "      Done."

# ── Buoc 2: Chay training ────────────────────────────────────────────────────
echo ""
echo "[2/2] Bat dau training: $CONFIG  (seed=$SEED)"
echo ""

cd "$SCRIPT_DIR"

EXTRA_ARGS=(--seed "$SEED")
[[ -n "$DATA_ROOT" ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "$DRY_RUN"   ]] && EXTRA_ARGS+=(--dry-run)
[[ -n "$ALPHA"     ]] && EXTRA_ARGS+=(--alpha "$ALPHA")
[[ -n "$LAMBDA2"   ]] && EXTRA_ARGS+=(--lambda2 "$LAMBDA2")
[[ -n "$WORK_DIR_OVERRIDE" ]] && EXTRA_ARGS+=(--work-dir "$WORK_DIR_OVERRIDE")

torchrun --nproc_per_node=2 src/train_affinity_only.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo ""
echo "========================================================"
echo " Hoan thanh: Run 7 (Affinity-only, seed=$SEED)"
echo " Xem ket qua tai: OUTPUT.WORK_DIR trong $CONFIG, {SEED} da duoc thay bang $SEED"
echo "   - benchmark_results.csv     (10 dong/vong, du cot: per-class IoU, ASD 2 chieu,"
echo "                                 bf_precision/recall, BIoU d1/d2/d4, lambda1/lambda2/"
echo "                                 alpha/alpha_bce_effective(luon 0.0)/alpha_affinity_effective)"
echo "   - checkpoint_index.json     (map best_miou/best_bfscore/best_bareland/final -> file/iter/round/seed/metrics)"
echo "   - best_miou.pth / best_bfscore.pth / best_bareland.pth / final_iter<N>.pth  (raw state_dict,"
echo "                                 tuong thich Tools/eval_boundary_metrics.py --model-type baseline)"
echo "   - summary.txt               (khoi 4-checkpoint + mean+/-std ddof=1 4 vong cuoi + khoi seed + khoi lieu)"
echo "   - final_summary.txt         (tom tat best mIoU, cung format Run 1/2/3/3b)"
echo "   - train_log.txt             (log day du qua trinh chay)"
echo "   - learning_curves.png       (val mIoU-9 + val L_total qua cac lan val)"
echo "   - vis/ , vis/boundary/      (visualizer segmentation + boundary sau moi lan val)"
echo ""
echo " Buoc tiep theo (RIENG, xem docs/run7-huong-dan-chay.md muc 5-6):"
echo "   1. Dump preds/per-image-stats cho best_miou.pth VA final_iter40000.pth qua"
echo "      Tools/eval_boundary_metrics.py --model-type baseline"
echo "   2. Bootstrap CI qua Tools/bootstrap_boundary_ci_run7.py (can duong dan"
echo "      per_image_stats.csv cua run1_baseline/run2b_bce04/run3_static_s19 da dump tu truoc)"
echo "========================================================"
