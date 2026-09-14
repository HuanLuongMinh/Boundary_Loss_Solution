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
# kiem tra va dung SOM (truoc ca buoc pip install) neu thieu --seed.
#
# 1 lenh duy nhat, CHAY LIEN TUC KHONG DUNG DE HOI:
#   [1] train (dry-run hoac that)
#   [2] TU DONG dump preds/per-image-stats cho CA best_miou VA final checkpoint
#       (Tools/eval_boundary_metrics.py --model-type baseline) — doc thang tu
#       checkpoint_index.json vua train ra, khong can go tay duong dan/iteration.
#   [3] TU DONG dump 3 checkpoint CU (--baseline-ckpt/--bce04-ckpt/--static-ckpt,
#       truyen truc tiep file .pth luc goi script) — config/model-type/iter cua
#       3 cai nay la thuoc tinh CO DINH cua cac run do (khong doi), da hardcode
#       san ben duoi. Idempotent: neu per_image_stats.csv da ton tai tu lan
#       chay truoc (checkpoint cu "da xu ly"), TU BO QUA dump lai, dung lai file
#       cu — khong chay lai eval mot cach thua thai.
#   [4] TU DONG bootstrap CI (Tools/bootstrap_boundary_ci_run7.py): cap #7/#8/#9
#       (so voi 3 checkpoint cu buoc [3], #9 la "Phep kiem chinh cua run nay",
#       muc 4.2 spec) cong cap #10 (run7b vs run7 cung seed)/#11 (lien-seed).
#       Thieu checkpoint cu nao thi CAP DO tu bo qua (khong loi), cac cap con
#       lai van chay binh thuong.
#
# Cach dung — truyen 3 checkpoint .pth cu ngay khi goi (chi can 1 lan, tu lan
# sau se tu dung lai CSV da dump, KHONG can truyen lai --*-ckpt nua):
#   bash scripts/run_affinity_only.sh --seed 19 \
#       --baseline-ckpt /path/to/run1_baseline/best_model.pth \
#       --bce04-ckpt    /path/to/run2b_bce04/best_model.pth \
#       --static-ckpt   /path/to/run3_static_s19/best_model.pth
#
#   bash scripts/run_affinity_only.sh --seed 19                    # lan sau: khong can --*-ckpt (da co CSV tu lan truoc)
#   bash scripts/run_affinity_only.sh --seed 19 --dry-run           # smoke-test: MAX_ITERS=100 + dump + bootstrap
#   bash scripts/run_affinity_only.sh --seed 86 --dry-run           # seed thu 2 (khuyen nghi, vai tro nhu Run 5)
#                                                                    #   -> co seed 19+86 xong, bootstrap tu tinh duoc cap lien-seed #11
#   bash scripts/run_affinity_only.sh --seed 19 --alpha 0.3         # ablation nhanh -> WORK_DIR tu them hau to
#   bash scripts/run_affinity_only.sh --seed 19 --work-dir work_dirs/foo  # doi WORK_DIR tuong minh
#
# Lieu phu (muc 1.2 spec, CHI chay sau khi da co ket qua lieu 0.4 o >=1 seed)
# — chay xong, cap #10 (0.2 vs 0.4, cung seed) se tu tinh duoc trong buoc [4]:
#   CONFIG=configs/unet_former_resnet18_affinity_only/run7b_affinity_only_lambda02.yaml \
#     bash scripts/run_affinity_only.sh --seed 19 --dry-run
#
# Neu Kaggle mount dataset o path khac mac dinh trong config, set DATA_ROOT
# truoc khi goi (ghi de ROOT_DIR/VAL_ROOT_DIR truc tiep, khong tao file gi):
#   DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_affinity_only.sh --seed 19
#
# Muon so sanh them voi 4 checkpoint cu con lai (run3b_aff02/run4_dynamic/
# run5_static_s86_best/run5_static_s86_final) — KHONG nam trong 1 lenh nay (chi
# 3 cai bat buoc cho cap #7/8/9 moi co san o day) — chay tay
# Tools/bootstrap_boundary_ci_run7.py voi --checkpoint/--pair, xem
# docs/run7-huong-dan-chay.md.

set -e  # dung ngay neu co loi
set -o pipefail  # loi trong torchrun (dau ben trai `| tee`) khong duoc "tee" lam an di

# ── Cau hinh duong dan ───────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-}"
CONFIG="${CONFIG:-configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUMP_DIR="${DUMP_DIR:-output/dump}"
BOOTSTRAP_OUT_DIR="${BOOTSTRAP_OUT_DIR:-output/bootstrap_run7}"

# 3 checkpoint CU can cho cap #7/#8/#9 (muc 4.2 spec) — TRUYEN FILE .pth qua
# --baseline-ckpt/--bce04-ckpt/--static-ckpt (hoac env BASELINE_CKPT/BCE04_CKPT/
# STATIC_CKPT) luc goi script. config/model-type/iteration la thuoc tinh CO
# DINH cua 3 run nay (khong doi qua thoi gian), hardcode san — KHONG can nguoi
# dung tu nho/tu truyen.
BASELINE_CKPT="${BASELINE_CKPT:-}"
BCE04_CKPT="${BCE04_CKPT:-}"
STATIC_CKPT="${STATIC_CKPT:-}"
BASELINE_CONFIG="configs/unet_former_resnet18_combineLoss/baseline.yaml"
BCE04_CONFIG="configs/unet_former_resnet18_bce_edge/bce_edge.yaml"
STATIC_CONFIG="configs/unet_former_resnet18_static_boundary/static_boundary.yaml"
BASELINE_MODEL_TYPE="baseline"
BCE04_MODEL_TYPE="bce_edge"
STATIC_MODEL_TYPE="static_boundary"
BASELINE_ITER=40000
BCE04_ITER=40000
STATIC_ITER=36000

# ── Parse tham so ─────────────────────────────────────────────────────────
DRY_RUN=""
ALPHA=""
LAMBDA2=""
SEED=""
WORK_DIR_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)       DRY_RUN="--dry-run"; shift ;;
        --alpha)         ALPHA="$2"; shift 2 ;;
        --lambda2)       LAMBDA2="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --work-dir)      WORK_DIR_OVERRIDE="$2"; shift 2 ;;
        --baseline-ckpt) BASELINE_CKPT="$2"; shift 2 ;;
        --bce04-ckpt)    BCE04_CKPT="$2"; shift 2 ;;
        --static-ckpt)   STATIC_CKPT="$2"; shift 2 ;;
        *) echo "Canh bao: bo qua tham so khong nhan dang duoc: $1" >&2; shift ;;
    esac
done

# ── Seed BAT BUOC — dung SOM, truoc ca pip install (muc 1.3.2 spec) ─────────
if [[ -z "$SEED" ]]; then
    echo "LOI: --seed la bat buoc cho Run 7 (config nay khong co TRAIN.SEED co dinh)." >&2
    echo "     Vi du: bash scripts/run_affinity_only.sh --seed 19" >&2
    exit 1
fi

# run7 vs run7b — suy tu CONFIG, dung de dat ten thu muc dump dung pattern
# ma Tools/bootstrap_boundary_ci_run7.py quet (run7_affinity_only_seed*_* /
# run7b_affinity_only_lambda02_seed*_*).
if [[ "$CONFIG" == *run7b* ]]; then
    RUN_PREFIX="run7b_affinity_only_lambda02"
else
    RUN_PREFIX="run7_affinity_only"
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
echo "[1/5] Cai dat requirements ..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>&1 \
    | grep -v -E "pip's dependency resolver|requires .*(incompatible|which is not installed)" \
    || true
echo "      Done."

# ── Buoc 2: Chay training (tee ra log tam de doc lai RUN7_WORK_DIR=...) ─────
echo ""
echo "[2/5] Bat dau training: $CONFIG  (seed=$SEED)"
echo ""

cd "$SCRIPT_DIR"

EXTRA_ARGS=(--seed "$SEED")
[[ -n "$DATA_ROOT" ]] && EXTRA_ARGS+=(--data-root "$DATA_ROOT")
[[ -n "$DRY_RUN"   ]] && EXTRA_ARGS+=(--dry-run)
[[ -n "$ALPHA"     ]] && EXTRA_ARGS+=(--alpha "$ALPHA")
[[ -n "$LAMBDA2"   ]] && EXTRA_ARGS+=(--lambda2 "$LAMBDA2")
[[ -n "$WORK_DIR_OVERRIDE" ]] && EXTRA_ARGS+=(--work-dir "$WORK_DIR_OVERRIDE")

TRAIN_LOG="$(mktemp)"
torchrun --nproc_per_node=2 src/train_affinity_only.py --config "$CONFIG" "${EXTRA_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"

WORK_DIR="$(grep '^RUN7_WORK_DIR=' "$TRAIN_LOG" | tail -1 | cut -d'=' -f2-)"
rm -f "$TRAIN_LOG"

echo ""
echo "========================================================"
echo " Hoan thanh training: Run 7 (Affinity-only, seed=$SEED)"
echo " WORK_DIR: ${WORK_DIR:-<khong doc duoc — xem log ben tren>}"
echo "========================================================"

# ── Buoc 3: TU DONG dump preds/per-image-stats cho best_miou + final ────────
if [[ -z "$WORK_DIR" ]]; then
    echo ""
    echo "CANH BAO: khong doc duoc RUN7_WORK_DIR tu log training — bo qua buoc dump/bootstrap" >&2
    echo "          tu dong. Dump thu cong theo docs/run7-huong-dan-chay.md muc 5." >&2
    exit 0
fi

CKPT_INDEX="$WORK_DIR/checkpoint_index.json"
if [[ ! -f "$CKPT_INDEX" ]]; then
    echo ""
    echo "CANH BAO: khong thay $CKPT_INDEX — bo qua buoc dump/bootstrap tu dong." >&2
    exit 0
fi

echo ""
echo "[3/5] Tu dong dump preds/per-image-stats cho best_miou + final (--model-type baseline) ..."

read -r BEST_FILE BEST_ITER FINAL_FILE FINAL_ITER <<< "$(python -c "
import json
d = json.load(open('$CKPT_INDEX', encoding='utf-8'))
best = d.get('best_miou')
final = d.get('final')
print(best['file'] if best else 'NONE', best['iter'] if best else 0,
      final['file'] if final else 'NONE', final['iter'] if final else 0)
")"

dump_one() {
    local ckpt_file="$1" iter="$2" tag="$3"
    if [[ "$ckpt_file" == "NONE" ]]; then
        echo "  Bo qua ($tag): checkpoint_index.json khong co entry nay." >&2
        return
    fi
    local run_name="${RUN_PREFIX}_seed${SEED}_${tag}"
    local out_dir="$DUMP_DIR/${run_name}"
    echo "  -> $run_name  (checkpoint=$ckpt_file, iter=$iter)"
    python Tools/eval_boundary_metrics.py \
        --config "$CONFIG" \
        --checkpoint "$WORK_DIR/$ckpt_file" \
        --model-type baseline \
        --dump-preds "$out_dir/masks" \
        --dump-per-image-stats "$out_dir/per_image_stats.csv" \
        --run-name "$run_name" --checkpoint-iter "$iter" \
        --dump-only
}

dump_one "$BEST_FILE" "$BEST_ITER" "best"
dump_one "$FINAL_FILE" "$FINAL_ITER" "final${FINAL_ITER}"

echo "      Da dump xong -> $DUMP_DIR/${RUN_PREFIX}_seed${SEED}_best , ${RUN_PREFIX}_seed${SEED}_final${FINAL_ITER}"
echo "      (kiem tra: --model-type baseline load checkpoint Run 7 dung khong bi loi state_dict)"

# ── Buoc 4: TU DONG dump 3 checkpoint CU tu file .pth da truyen qua --*-ckpt
#    (muc [3] docstring dau file) — idempotent: bo qua neu per_image_stats.csv
#    da co tu lan chay truoc, khong chay lai eval thua thai. ─────────────────
echo ""
echo "[4/5] Tu dong dump 3 checkpoint cu (--baseline-ckpt/--bce04-ckpt/--static-ckpt) ..."

dump_old_checkpoint() {
    local label="$1" ckpt_path="$2" cfg="$3" model_type="$4" iter="$5"
    local out_dir="$DUMP_DIR/${label}_iter${iter}"
    local csv_path="$out_dir/per_image_stats.csv"
    if [[ -f "$csv_path" ]]; then
        echo "  -> $label: da dump tu lan chay truoc, dung lai $csv_path"
        return
    fi
    if [[ -z "$ckpt_path" ]]; then
        echo "  Bo qua $label: chua truyen --*-ckpt (va chua co CSV cu) — cap can no se tu bo qua." >&2
        return
    fi
    if [[ ! -f "$ckpt_path" ]]; then
        echo "  Canh bao $label: khong thay checkpoint '$ckpt_path' — cap can no se tu bo qua." >&2
        return
    fi
    echo "  -> $label: dump tu $ckpt_path (config=$cfg, model-type=$model_type, iter=$iter)"
    python Tools/eval_boundary_metrics.py \
        --config "$cfg" --checkpoint "$ckpt_path" --model-type "$model_type" \
        --dump-preds "$out_dir/masks" --dump-per-image-stats "$csv_path" \
        --run-name "$label" --checkpoint-iter "$iter" --dump-only
}

dump_old_checkpoint run1_baseline   "$BASELINE_CKPT" "$BASELINE_CONFIG" "$BASELINE_MODEL_TYPE" "$BASELINE_ITER"
dump_old_checkpoint run2b_bce04     "$BCE04_CKPT"    "$BCE04_CONFIG"    "$BCE04_MODEL_TYPE"    "$BCE04_ITER"
dump_old_checkpoint run3_static_s19 "$STATIC_CKPT"   "$STATIC_CONFIG"   "$STATIC_MODEL_TYPE"   "$STATIC_ITER"

# ── Buoc 5: TU DONG bootstrap CI — cap #7/#8/#9 (so voi 3 checkpoint cu buoc
#    [4], neu da dump duoc) cong cap #10/#11 (Run 7/7b noi bo, tu tinh khi du
#    dieu kien). Thieu checkpoint cu nao thi CAP DO tu bo qua (khong loi — xem
#    Tools/bootstrap_boundary_ci_run7.py), cac cap con lai van chay. ─────────
echo ""
echo "[5/5] Tu dong bootstrap CI (Tools/bootstrap_boundary_ci_run7.py) ..."

OLD_CKPT_ARGS=()
for entry in "run1_baseline:$BASELINE_ITER" "run2b_bce04:$BCE04_ITER" "run3_static_s19:$STATIC_ITER"; do
    label="${entry%%:*}"; iter="${entry#*:}"
    csv_path="$DUMP_DIR/${label}_iter${iter}/per_image_stats.csv"
    if [[ -f "$csv_path" ]]; then
        OLD_CKPT_ARGS+=(--checkpoint "${label}=${csv_path}")
    else
        echo "  Canh bao: khong co CSV cho '$label' (${csv_path}) — cap can no (#7/#8/#9) se tu bo qua." >&2
    fi
done

python Tools/bootstrap_boundary_ci_run7.py \
    "${OLD_CKPT_ARGS[@]}" \
    --dump-dir "$DUMP_DIR" \
    --n-boot "${BOOT_N:-10000}" --seed "${BOOT_SEED:-19}" \
    --output-dir "$BOOTSTRAP_OUT_DIR"
echo "      Manifest -> $DUMP_DIR/checkpoints_manifest_run7.json"
echo "      Ket qua (neu co cap nao tinh duoc) -> $BOOTSTRAP_OUT_DIR/bootstrap_ci_run7.csv / .md"
echo "      Bang lien-seed (neu >=2 seed) -> $BOOTSTRAP_OUT_DIR/run7_interseed_amplitude.md"
echo ""
echo " Muon so sanh them voi 4 checkpoint cu con lai (run3b_aff02/run4_dynamic/"
echo " run5_static_s86_best/run5_static_s86_final): chay tay Tools/bootstrap_"
echo " boundary_ci_run7.py voi them --checkpoint LABEL=<csv> --pair <nhan_run7>=<LABEL>"
echo " (xem docs/run7-huong-dan-chay.md)."
echo "========================================================"
