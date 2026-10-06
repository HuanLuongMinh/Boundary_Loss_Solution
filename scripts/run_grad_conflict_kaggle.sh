#!/usr/bin/env bash
# scripts/run_grad_conflict_kaggle.sh — Lenh gon de chay toan bo chan doan xung dot gradient + lo gia
# (docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md) tren Kaggle.
#
# Ban chi can truyen THU MUC chua 8 checkpoint (dung ten trong docs/Checkpoint_summary.txt) va THU MUC
# chua cac dump cu. Script tu ghep checkpoint <-> nhan <-> config/model-type (bang co dinh trong
# Tools/grad_conflict_common.py) <-> dump, roi goi scripts/run_grad_conflict.sh.
#
#   bash scripts/run_grad_conflict_kaggle.sh \
#     --ckpt-dir  /kaggle/input/my-ckpts \
#     --dump-root /kaggle/input/my-dumps \
#     --data-root /kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap \
#     --pos-weight static_s19_36k=11.04 --pos-weight static_s86_36k=10.88 --pos-weight static_s86_40k=10.88 \
#     --pos-weight bce04_40k=11.00 --pos-weight bce02_40k=11.00
#
# Tuy chon:
#   --ckpt LABEL=PATH      ghi de file checkpoint cua 1 nhan (ten file khac mac dinh)
#   --dump LABEL=DIR       ghi de thu muc dump cua 1 nhan
#   --make-missing-dumps   tao dump con thieu (vd BCE lambda=0.2) bang Tools/eval_boundary_metrics.py
#                          vao output/dump/<ten>/ truoc khi chay (can cho cong 7)
#   --dry-run N            chay thu N anh, output vao output/*_dryrun/
#   --parallel             T4 x2: chia checkpoint cho cuda:0 va cuda:1, sau do gop (summarize B C B2)
#   --stages "..."         truyen thang cho run_grad_conflict.sh (mac dinh: figures gates A summarize B C B2)
#   --ckpts "..."          chi chay cac nhan nay
#   --device cuda:0        chon GPU (khi khong --parallel)
#   --out DIR              thu muc output goc (mac dinh: output)
#   --force                chay lai du da co ket qua
#   --print-only           chi in lenh se chay, khong chay
#
# pos_weight: lay dong "pos_weight used" trong summary.txt (hoac log train) cua tung run co BCE.
# Khong co -> truyen LABEL=auto (uoc luong lai, xap xi). Thieu pos_weight cho ckpt co BCE -> script dung.

set -e
set -o pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ── Bang ghep: nhan -> ten file checkpoint (Checkpoint_summary.txt) -> ten thu muc dump mac dinh ──
LABELS="static_s19_36k static_s86_36k static_s86_40k bce04_40k bce02_40k affonly_s19_40k affonly_s86_40k baseline_40k"
declare -A FILE=(
  [static_s19_36k]=best_model_Static_S19.pth
  [static_s86_36k]=best_miou_Static_s86.pth
  [static_s86_40k]=final_iter40000_static_s86.pth
  [bce04_40k]=best_model_BCE_04.pth
  [bce02_40k]=best_model_BCE_02.pth
  [affonly_s19_40k]=best_miou_aff_s19.pth
  [affonly_s86_40k]=best_miou_aff_s86.pth
  [baseline_40k]=best_model_baseline.pth
)
declare -A DUMPNAME=(
  [static_s19_36k]=run3_static_s19_iter36000
  [static_s86_36k]=run5_static_s86_best_iter36000
  [static_s86_40k]=run5_static_s86_final_iter40000
  [bce04_40k]=run2b_bce04_iter40000
  [bce02_40k]=run2_bce02_iter40000
  [affonly_s19_40k]=run7_affinity_only_seed19_best
  [affonly_s86_40k]=run7_affinity_only_seed86_best
  [baseline_40k]=run1_baseline_iter40000
)
# Dung cho --make-missing-dumps (thuoc tinh co dinh cua tung run)
declare -A CFG=(
  [static_s19_36k]=configs/unet_former_resnet18_static_boundary/static_boundary.yaml
  [static_s86_36k]=configs/unet_former_resnet18_static_boundary/static_boundary.yaml
  [static_s86_40k]=configs/unet_former_resnet18_static_boundary/static_boundary.yaml
  [bce04_40k]=configs/unet_former_resnet18_bce_edge/bce_edge.yaml
  [bce02_40k]=configs/unet_former_resnet18_bce_edge/bce_edge.yaml
  [affonly_s19_40k]=configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml
  [affonly_s86_40k]=configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml
  [baseline_40k]=configs/unet_former_resnet18_combineLoss/baseline.yaml
)
declare -A MTYPE=(
  [static_s19_36k]=static_boundary [static_s86_36k]=static_boundary [static_s86_40k]=static_boundary
  [bce04_40k]=bce_edge [bce02_40k]=bce_edge
  [affonly_s19_40k]=baseline [affonly_s86_40k]=baseline [baseline_40k]=baseline
)
declare -A ITER=(
  [static_s19_36k]=36000 [static_s86_36k]=36000 [static_s86_40k]=40000 [bce04_40k]=40000
  [bce02_40k]=40000 [affonly_s19_40k]=40000 [affonly_s86_40k]=40000 [baseline_40k]=40000
)
HAS_BCE="static_s19_36k static_s86_36k static_s86_40k bce04_40k bce02_40k"

CKPT_DIR=""; DUMP_ROOT=""; DATA_ROOT="${DATA_ROOT:-}"; OUT="output"
DRY=""; PARALLEL=""; STAGES=""; ONLY=""; DEVICE=""; FORCE=""; PRINT=""; MAKE_DUMPS=""
declare -A CK DP PW

die() { echo "LOI: $*" >&2; exit 1; }
chk_label() { [[ " $LABELS " == *" ${1%%=*} "* && "$1" == *=* ]] || die "$2 '$1' — can LABEL=gia_tri, LABEL thuoc: $LABELS"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-dir)   CKPT_DIR="$2"; shift 2 ;;
    --dump-root)  DUMP_ROOT="$2"; shift 2 ;;
    --data-root)  DATA_ROOT="$2"; shift 2 ;;
    --ckpt)       chk_label "$2" --ckpt;       CK["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --dump)       chk_label "$2" --dump;       DP["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --pos-weight) chk_label "$2" --pos-weight; PW["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --dry-run)    DRY="$2"; shift 2 ;;
    --parallel)   PARALLEL=1; shift ;;
    --stages)     STAGES="$2"; shift 2 ;;
    --ckpts)      ONLY="$2"; shift 2 ;;
    --device)     DEVICE="$2"; shift 2 ;;
    --out)        OUT="$2"; shift 2 ;;
    --force)      FORCE=1; shift ;;
    --print-only) PRINT=1; shift ;;
    --make-missing-dumps) MAKE_DUMPS=1; shift ;;
    -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
    *) die "tham so khong biet: $1" ;;
  esac
done

# ── Ghep checkpoint + dump cho tung nhan ──
echo "== Ghep checkpoint / dump"
USE=""
for lb in ${ONLY:-$LABELS}; do
  [[ " $LABELS " == *" $lb "* ]] || die "--ckpts: nhan khong hop le '$lb'"
  ck="${CK[$lb]:-}"
  [[ -z "$ck" && -n "$CKPT_DIR" && -f "$CKPT_DIR/${FILE[$lb]}" ]] && ck="$CKPT_DIR/${FILE[$lb]}"
  dp="${DP[$lb]:-}"
  if [[ -z "$dp" ]]; then
    for cand in "$DUMP_ROOT/${DUMPNAME[$lb]}" "$OUT/dump/${DUMPNAME[$lb]}"; do
      [[ -n "$DUMP_ROOT" || "$cand" == "$OUT/dump/"* ]] || continue
      if [[ -d "$cand" ]]; then dp="$cand"; break; fi
    done
  fi
  if [[ -z "$dp" && -n "$MAKE_DUMPS" && -n "$ck" ]]; then
    dp="$OUT/dump/${DUMPNAME[$lb]}"
    echo "   [$lb] tao dump con thieu -> $dp"
    cmd=(python Tools/eval_boundary_metrics.py --config "${CFG[$lb]}" --checkpoint "$ck" --model-type "${MTYPE[$lb]}"
         --dump-preds "$dp/masks" --dump-per-image-stats "$dp/per_image_stats.csv"
         --run-name "${DUMPNAME[$lb]}" --checkpoint-iter "${ITER[$lb]}" --dump-only)
    [[ -n "$DATA_ROOT" ]] && cmd+=(--data-root "$DATA_ROOT")
    if [[ -n "$PRINT" ]]; then echo "   + ${cmd[*]}"; else "${cmd[@]}"; fi
  fi
  printf "   %-16s ckpt: %-60s dump: %s\n" "$lb" "${ck:-(KHONG CO)}" "${dp:-(KHONG CO)}"
  if [[ -n "$ck" && " $HAS_BCE " == *" $lb "* && -z "${PW[$lb]:-}" ]]; then
    die "[$lb] co BCE nhung thieu --pos-weight $lb=<gia tri log | auto>"
  fi
  [[ -n "$ck" ]] && CK[$lb]="$ck" || unset "CK[$lb]"
  [[ -n "$dp" ]] && DP[$lb]="$dp" || unset "DP[$lb]"
  [[ -n "$ck" || -n "$dp" ]] && USE="$USE $lb"
done
USE="$(echo $USE)"
[[ -n "$USE" ]] || die "khong tim thay checkpoint/dump nao — kiem --ckpt-dir / --dump-root"
[[ -n "${DP[baseline_40k]:-}" ]] || echo "   CANH BAO: thieu dump baseline_40k -> Phan B se dung (lo gia moi tinh so voi Baseline)."

build_args() {   # $1 = danh sach nhan
  ARGS=()
  for lb in $1; do
    [[ -n "${CK[$lb]:-}" ]] && ARGS+=(--ckpt "$lb=${CK[$lb]}")
    [[ -n "${DP[$lb]:-}" ]] && ARGS+=(--dump "$lb=${DP[$lb]}")
    [[ -n "${PW[$lb]:-}" ]] && ARGS+=(--pos-weight "$lb=${PW[$lb]}")
  done
  [[ -n "$DATA_ROOT" ]] && ARGS+=(--data-root "$DATA_ROOT")
  [[ -n "$DRY" ]] && ARGS+=(--dry-run "$DRY")
  [[ -n "$FORCE" ]] && ARGS+=(--force)
  ARGS+=(--out "$OUT")
  return 0
}

run() {
  echo "+ bash scripts/run_grad_conflict.sh $*"
  [[ -n "$PRINT" ]] || bash scripts/run_grad_conflict.sh "$@"
}

if [[ -z "$PARALLEL" ]]; then
  build_args "$USE"
  [[ -n "$STAGES" ]] && ARGS+=(--stages "$STAGES")
  [[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")
  run "${ARGS[@]}"
else
  # Chia luan phien de can tai: cuda:0 va cuda:1. figures chay TRUOC (1 lan), gop chay SAU.
  G0=""; G1=""; i=0
  for lb in $USE; do
    [[ -n "${CK[$lb]:-}" ]] || continue
    if (( i % 2 == 0 )); then G0="$G0 $lb"; else G1="$G1 $lb"; fi
    i=$((i + 1))
  done
  build_args "$USE"; run "${ARGS[@]}" --stages "figures"
  mkdir -p "$OUT"
  build_args "$G0"; A0=("${ARGS[@]}")
  build_args "$G1"; A1=("${ARGS[@]}")
  echo "   cuda:0 <-$G0"
  echo "   cuda:1 <-$G1"
  if [[ -n "$PRINT" ]]; then
    run "${A0[@]}" --stages "gates A" --device cuda:0
    run "${A1[@]}" --stages "gates A" --device cuda:1
  else
    run "${A0[@]}" --stages "gates A" --device cuda:0 > "$OUT/parallel_cuda0.log" 2>&1 & P0=$!
    run "${A1[@]}" --stages "gates A" --device cuda:1 > "$OUT/parallel_cuda1.log" 2>&1 & P1=$!
    echo "   dang chay song song — log: $OUT/parallel_cuda0.log, $OUT/parallel_cuda1.log"
    S0=0; S1=0
    wait $P0 || S0=$?
    wait $P1 || S1=$?
    if (( S0 != 0 || S1 != 0 )); then
      tail -n 20 "$OUT/parallel_cuda0.log" "$OUT/parallel_cuda1.log"
      die "mot tien trinh GPU loi (cuda:0=$S0, cuda:1=$S1) — xem log o tren (cong FAIL -> DUNG, muc 4 spec)"
    fi
  fi
  build_args "$USE"; run "${ARGS[@]}" --stages "${STAGES:-summarize B C B2}"
fi
