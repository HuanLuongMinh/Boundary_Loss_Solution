#!/usr/bin/env bash
# scripts/run_grad_conflict.sh — Chan doan xung dot gradient (Phan A) + lo gia (Phan B, B2) + noi (Phan C)
# theo docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md. Huong dan day du:
# docs/huong-dan-chay-grad-conflict-va-lo-gia.md
#
# KHONG train, KHONG sua src/losses/. Checkpoint, dump va pos_weight truyen LUC CHAY
# (khong hard-code duong dan). Nhan checkpoint (co dinh, xem Tools/grad_conflict_common.py):
#   static_s19_36k static_s86_36k static_s86_40k bce04_40k bce02_40k
#   affonly_s19_40k affonly_s86_40k baseline_40k
#
# Cach dung (1 lenh chay het):
#   bash scripts/run_grad_conflict.sh \
#     --ckpt static_s19_36k=/kaggle/input/ckpts/best_model_Static_S19.pth \
#     --dump static_s19_36k=/kaggle/input/dumps/run3_static_s19_iter36000 \
#     --pos-weight static_s19_36k=11.0392 \
#     ... (lap lai cho tung checkpoint) ...
#
# Chon tap con:
#   --stages "figures gates A summarize B C B2"   (mac dinh: tat ca, theo dung thu tu nay)
#   --ckpts  "static_s19_36k static_s86_36k"      (mac dinh: moi nhan da truyen --ckpt / --dump)
#   --dry-run N        chi N anh dau, output vao <out>_dryrun/ (khong lan voi ket qua that)
#   --device cuda:1    chon GPU (vd chay 2 tien trinh song song tren T4 x2)
#   --out output       thu muc goc output (mac dinh: output)
#   --data-root DIR    ghi de DATASET.ROOT_DIR/VAL_ROOT_DIR
#   --force            chay lai du da co ket qua
#
# Stage:
#   figures   Tools/select_figure_images.py — chot 6 anh minh hoa tu GT (1 lan, khong ghi de)
#   gates     Tools/grad_conflict_probe.py gates  (cong 1,2,3,4,5,7) cho tung ckpt co --ckpt
#   A         Tools/grad_conflict_probe.py measure cho tung ckpt co --ckpt
#   summarize Tools/grad_conflict_probe.py summarize (bang 5.1/5.2/5.3)
#   B         Tools/spurious_holes.py selftest (cong 6) + count + compare (bang 5.4, cong 3.3)
#   C         Tools/link_holes_conflict.py (tu bo qua neu Phan B = "khong do loss")
#   B2        Tools/spurious_holes.py fill-test cho Static s19/s86 @36k (tu bo qua neu "hiem")

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

ALL_LABELS="static_s19_36k static_s86_36k static_s86_40k bce04_40k bce02_40k affonly_s19_40k affonly_s86_40k baseline_40k"
declare -A CKPT DUMP POSW
STAGES="figures gates A summarize B C B2"
CKPTS=""
DRY=""
DEVICE=""
OUT="output"
DATA_ROOT="${DATA_ROOT:-}"
FORCE=""

kv() {  # "label=value" -> kiem nhan hop le
  local item="$1" flag="$2"
  local k="${item%%=*}"
  if [[ "$item" != *=* ]] || [[ " $ALL_LABELS " != *" $k "* ]]; then
    echo "LOI: $flag '$item' — can dang LABEL=gia_tri, LABEL thuoc: $ALL_LABELS" >&2; exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)       kv "$2" --ckpt;       CKPT["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --dump)       kv "$2" --dump;       DUMP["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --pos-weight) kv "$2" --pos-weight; POSW["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --stages)     STAGES="$2"; shift 2 ;;
    --ckpts)      CKPTS="$2"; shift 2 ;;
    --dry-run)    DRY="$2"; shift 2 ;;
    --device)     DEVICE="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --data-root)  DATA_ROOT="$2"; shift 2 ;;
    --force)      FORCE="--force"; shift ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "LOI: tham so khong biet: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CKPTS" ]]; then
  for lb in $ALL_LABELS; do
    if [[ -n "${CKPT[$lb]:-}" || -n "${DUMP[$lb]:-}" ]]; then CKPTS="$CKPTS $lb"; fi
  done
fi
CKPTS="$(echo $CKPTS)"
[[ -n "$CKPTS" ]] || { echo "LOI: khong co checkpoint/dump nao (--ckpt / --dump)." >&2; exit 1; }

SUFFIX=""; DRYARG=()
if [[ -n "$DRY" ]]; then SUFFIX="_dryrun"; DRYARG=(--max-images "$DRY"); fi
GDIR="$OUT/grad_conflict$SUFFIX"
HDIR="$OUT/spurious_holes$SUFFIX"
GT_CACHE="$OUT/_cache/gt1024"
mkdir -p "$GDIR" "$HDIR"
COMMON=()
[[ -n "$DATA_ROOT" ]] && COMMON+=(--data-root "$DATA_ROOT")
DEVARG=()
[[ -n "$DEVICE" ]] && DEVARG=(--device "$DEVICE")
LOG="$GDIR/run_log.txt"

echo "== run_grad_conflict.sh  $(date -Iseconds)"
echo "   stages : $STAGES"
echo "   ckpts  : $CKPTS"
echo "   output : $GDIR  |  $HDIR  ${DRY:+(dry-run $DRY anh)}"
echo "[$(date -Iseconds)] run_grad_conflict.sh stages='$STAGES' ckpts='$CKPTS' dry='${DRY}'" >> "$LOG"

has_stage() { [[ " $STAGES " == *" $1 "* ]]; }

ckpt_args() {   # in ra mang tham so --ckpt/--dump-dir/--pos-weight cho cac nhan co checkpoint
  ARGS=()
  for lb in $CKPTS; do
    [[ -n "${CKPT[$lb]:-}" ]] || continue
    ARGS+=(--ckpt "$lb=${CKPT[$lb]}")
    [[ -n "${DUMP[$lb]:-}" ]] && ARGS+=(--dump-dir "$lb=${DUMP[$lb]}")
    [[ -n "${POSW[$lb]:-}" ]] && ARGS+=(--pos-weight "$lb=${POSW[$lb]}")
  done
}

mask_dir() {    # <dump>/masks neu co, nguoc lai chinh <dump>
  if [[ -d "$1/masks" ]]; then echo "$1/masks"; else echo "$1"; fi
}

if has_stage figures; then
  echo "== [figures] chot 6 anh minh hoa tu GT"
  python Tools/select_figure_images.py "${COMMON[@]}" --output "$GDIR/figure_images.json" \
    --gt-cache "$GT_CACHE" "${DRYARG[@]}"
fi

ckpt_args
if [[ ${#ARGS[@]} -gt 0 ]]; then
  if has_stage gates; then
    echo "== [gates] cong 1,2,3,4,5,7"
    python Tools/grad_conflict_probe.py gates "${ARGS[@]}" "${COMMON[@]}" "${DEVARG[@]}" "${DRYARG[@]}" \
      --output-dir "$GDIR" --gt-cache "$GT_CACHE" $FORCE
  fi
  if has_stage A; then
    echo "== [A] do xung dot gradient"
    python Tools/grad_conflict_probe.py measure "${ARGS[@]}" "${COMMON[@]}" "${DEVARG[@]}" "${DRYARG[@]}" \
      --output-dir "$GDIR" --gt-cache "$GT_CACHE" $FORCE
  fi
else
  if has_stage gates || has_stage A; then echo "   (khong co --ckpt nao -> bo qua gates/A)"; fi
fi

if has_stage summarize; then
  echo "== [summarize] bang 5.1 / 5.2 / 5.3"
  python Tools/grad_conflict_probe.py summarize --output-dir "$GDIR" --holes-dir "$HDIR"
fi

if has_stage B; then
  echo "== [B] cong 6 + dem lo gia + bootstrap"
  python Tools/spurious_holes.py selftest --output-dir "$HDIR"
  MASKS=(); STATS=()
  for lb in $CKPTS baseline_40k; do
    [[ -n "${DUMP[$lb]:-}" ]] || continue
    [[ " ${MASKS[*]} " == *" $lb="* ]] && continue
    MASKS+=(--masks "$lb=$(mask_dir "${DUMP[$lb]}")")
    [[ -f "${DUMP[$lb]}/per_image_stats.csv" ]] && STATS+=(--per-image-stats "$lb=${DUMP[$lb]}/per_image_stats.csv")
  done
  if [[ -z "${DUMP[baseline_40k]:-}" ]]; then
    echo "LOI: Phan B can --dump baseline_40k=<dir> (lo gia moi tinh so voi Baseline)." >&2; exit 1
  fi
  python Tools/spurious_holes.py count "${MASKS[@]}" "${STATS[@]}" "${COMMON[@]}" "${DRYARG[@]}" \
    --output-dir "$HDIR" --gt-cache "$GT_CACHE" $FORCE
  python Tools/spurious_holes.py compare --output-dir "$HDIR"
fi

if has_stage C; then
  echo "== [C] noi lo gia voi xung dot gradient"
  python Tools/link_holes_conflict.py --grad-dir "$GDIR" --holes-dir "$HDIR"
fi

if has_stage B2; then
  echo "== [B2] thu hau xu ly lap lo"
  FILL=()
  for lb in static_s19_36k static_s86_36k; do
    [[ -n "${DUMP[$lb]:-}" ]] && FILL+=(--masks "$lb=$(mask_dir "${DUMP[$lb]}")")
  done
  if [[ ${#FILL[@]} -gt 0 ]]; then
    python Tools/spurious_holes.py fill-test "${FILL[@]}" "${COMMON[@]}" "${DRYARG[@]}" \
      --output-dir "$HDIR" --gt-cache "$GT_CACHE" $FORCE
  else
    echo "   (khong co dump Static s19/s86 -> bo qua B2)"
  fi
fi

echo "== Xong. Ket qua: $GDIR/summary_grad_conflict.md  |  $HDIR/pairs_bootstrap.md"
