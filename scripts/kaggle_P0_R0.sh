#!/usr/bin/env bash
# scripts/kaggle_P0_R0.sh — S4: P0 (JSON hau ky run Static chay lai) + R0 (P-CRF), MOT LENH chay het.
# Spec: docs/S4-spec-P0-R0-ban-giao-claude-code.md   Huong dan: docs/huong-dan-kaggle-P0-R0.md
#
#   cai thu vien + pydensecrf -> self-test -> DRY-RUN (cong chan: P0 + R0 tren vai anh, uoc tinh thoi gian)
#   -> P0 (fingerprint, forward 2 snapshot, JSON + per-image + masks, cong P0-g1/g2, so CSV)
#   -> R0 (g1 -> g2/g3 -> luoi 27 to hop tren 32 anh train -> chon (g4) -> val 384 anh -> bootstrap, §3.6, §3.8)
#   -> summary_P0_R0.md
#
# Notebook: Accelerator GPU T4 (forward nhanh; CRF chay tren CPU), Internet On. Add Input:
#   1. dataset OpenEarthMap (images/{train,val}, label/{train,val})
#   2. dataset checkpoint (huanluong/checkpoint-sumary: best_model_Static_S19.pth, best_miou_Static_s86.pth,
#      best_model_baseline.pth, best_model_BCE_04.pth)
#   3. OUTPUT notebook Phan D (notebook5ab020caa1): output/phanD/snapshots/iter_{36000,40000}.pth,
#      probe_ids.txt, figure_images.json, logs/benchmark_results.csv
# Chay bang Save Version -> Save & Run All (Commit).
#   Cell 1:  !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
#   Cell 2:  !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_P0_R0.sh
#
# Tuy chon:
#   --dryrun            CHI chay kiem nhanh (cai dat, self-test, dry-run P0 + R0 tren vai anh) roi dung
#   --skip-dryrun       bo dry-run (vd version truoc da dry-run PASS)
#   --stages "p0 r0"    chi chay phan nay (mac dinh "p0 r0"; summary luon chay)
#   --resume            chep output/refine cua version truoc (Add Input) roi chay tiep (stage da xong bo qua)
#   --workers N         so tien trinh CRF (mac dinh = so CPU)
#   --dry-n N           so anh val cua dry-run (mac dinh 4)
#   --input-mode M      softmax (mac dinh, §3.2) | gt_prob (du phong — chi khi softmax khong kha thi)
# Bien moi truong (ghi de tu do): CK, DATA, SNAP (thu muc snapshots/), PHD (thu muc output/phanD), SKIP_PIP=1

source "$(dirname "${BASH_SOURCE[0]}")/kaggle_common.sh"
die() { echo; echo "  [LOI]   $*" >&2; echo "==> DUNG. Xem docs/huong-dan-kaggle-P0-R0.md (Phu luc)" >&2; exit 1; }

DRY_ONLY=""; SKIP_DRY=""; STAGES="p0 r0"; RESUME=""; WORKERS="$(nproc 2>/dev/null || echo 4)"; DRYN=4; MODE="softmax"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dryrun|--dry-run) DRY_ONLY=1 ;;
    --skip-dryrun) SKIP_DRY=1 ;;
    --stages)      shift; STAGES="$1" ;;
    --resume)      RESUME=1 ;;
    --workers)     shift; WORKERS="$1" ;;
    --dry-n)       shift; DRYN="$1" ;;
    --input-mode)  shift; MODE="$1" ;;
    *) die "tham so khong biet: $1" ;;
  esac
  shift
done
has() { [[ " $STAGES " == *" $1 "* ]]; }
export PYTHONUNBUFFERED=1 NO_ALBUMENTATIONS_UPDATE=1

T0=$(date +%s)
RF="$OUT/refine"; DRY="$OUT/refine_dryrun"
mkdir -p "$RF"

install_deps

step "GPU"
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  ok "$(python -c "import torch; print(torch.cuda.get_device_name(0))") — forward tren GPU, CRF tren $WORKERS CPU"
else
  echo "  [!]     khong co GPU — forward chay tren CPU (cham hon ~1h, ket qua khong doi)"
fi

step "Tim duong dan"
[[ -z "${CK:-}" ]] && CK="$(dirname "$(find /kaggle/input -name best_model_Static_S19.pth 2>/dev/null | head -1 || true)")"
if [[ -z "${DATA:-}" ]]; then
  v="$(find /kaggle/input -maxdepth 7 -type d -path '*images/val' 2>/dev/null | head -1 || true)"
  [[ -n "$v" ]] && DATA="$(dirname "$(dirname "$v")")"
fi
# Snapshot: uu tien .../snapshots/iter_36000.pth (output Phan D), neu khong co thi lay iter_36000.pth bat ky
# (dataset tu tao, vd Boundary_PhanD de phang: moi file nam chung 1 thu muc).
if [[ -z "${SNAP:-}" ]]; then
  s="$(find /kaggle/input -name iter_36000.pth -path '*snapshots*' 2>/dev/null | head -1 || true)"
  [[ -n "$s" ]] || s="$(find /kaggle/input -name iter_36000.pth 2>/dev/null | head -1 || true)"
  SNAP="$(dirname "$s")"
fi
# Thu muc Phan D (probe_ids.txt, figure_images.json, benchmark_results.csv): cung thu muc snapshot (de phang),
# thu muc cha cua snapshots/ (output Phan D), hoac tim probe_ids.txt.
if [[ -z "${PHD:-}" ]]; then
  if [[ -f "$SNAP/probe_ids.txt" ]]; then PHD="$SNAP"
  elif [[ -f "$(dirname "$SNAP")/probe_ids.txt" ]]; then PHD="$(dirname "$SNAP")"
  else PHD="$(dirname "$(find /kaggle/input -name probe_ids.txt -ipath '*phand*' 2>/dev/null | head -1 || true)")"
  fi
fi
echo "  CK   = $CK"; echo "  DATA = $DATA"; echo "  SNAP = $SNAP"; echo "  PHD  = $PHD"
[[ -n "${DATA:-}" && -d "$DATA/images/val" && -d "$DATA/images/train" ]] || die "khong tim thay dataset OpenEarthMap (images/train, images/val)"
[[ -n "$SNAP" && "$SNAP" != "." && -f "$SNAP/iter_36000.pth" && -f "$SNAP/iter_40000.pth" ]] \
  || die "khong tim thay iter_36000.pth + iter_40000.pth (cung 1 thu muc) — Add Input output Phan D / dataset Boundary_PhanD"
PROBE="$PHD/probe_ids.txt"; FIG="$PHD/figure_images.json"
BENCH="$PHD/logs/benchmark_results.csv"; [[ -f "$BENCH" ]] || BENCH="$PHD/benchmark_results.csv"
[[ -f "$PROBE" ]] || die "khong co probe_ids.txt cua Phan D ($PROBE)"
[[ -f "$BENCH" ]] && ok "benchmark_results.csv: $BENCH" || { echo "  [!]     khong co benchmark_results.csv — §2.4 chi in gia tri hau ky"; BENCH=""; }
if has r0; then
  [[ -n "$CK" && "$CK" != "." && -d "$CK" ]] || die "khong tim thay dataset checkpoint (best_model_Static_S19.pth)"
  for f in best_model_Static_S19.pth best_miou_Static_s86.pth best_model_baseline.pth best_model_BCE_04.pth; do
    [[ -f "$CK/$f" ]] && ok "$f" || die "thieu checkpoint $CK/$f"
  done
fi
ok "probe_ids.txt: $(wc -l < "$PROBE") anh"
if [[ ! -f "$FIG" ]]; then
  FIG="$RF/figure_images.json"
  echo "  (khong co figure_images.json cua Phan D -> chon lai tu GT val, quy tac Buoc 1: $FIG)"
  python Tools/select_figure_images.py --data-root "$DATA" --output "$FIG" --gt-cache "" || die "khong chon duoc figure_images.json"
fi
ok "figure_images.json: $FIG"

CRF_OK=0
if has r0; then
  step "Cai pydensecrf (§3.3)"
  if ! python -c "import pydensecrf.densecrf" 2>/dev/null; then
    pip install -q cython 2>&1 | tail -2 || true
    pip install -q git+https://github.com/lucasb-eyer/pydensecrf.git 2>&1 | tail -5 || true
    if ! python -c "import pydensecrf.densecrf" 2>/dev/null; then
      echo "  (lan 1 that bai — thu lai cung repo voi Cython < 3, --no-build-isolation)"
      pip install -q "cython<3" setuptools wheel 2>&1 | tail -2 || true
      pip install -q --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git 2>&1 | tail -5 || true
    fi
  fi
  if python Tools/r0_pcrf.py crf-info; then
    CRF_OK=1; ok "pydensecrf: $(pip freeze 2>/dev/null | grep -i pydensecrf)"
  else
    die "cai pydensecrf KHONG duoc — theo §3.3 dung, KHONG tu viet CRF thay the. (Chi chay P0: --stages p0)"
  fi
fi

step "Self-test (CPU, du lieu tong hop)"
ST_ARGS=(); [[ $CRF_OK -eq 1 ]] && ST_ARGS=(--require-crf)
python Tools/test_refine.py "${ST_ARGS[@]}" > "$RF/selftest.log" 2>&1 || { tail -n 30 "$RF/selftest.log"; die "self-test FAIL"; }
ok "$(tail -n 1 "$RF/selftest.log")"

if [[ -n "$RESUME" ]]; then
  step "Resume: chep output/refine cua version truoc"
  old="$(find /kaggle/input -type d -path "*Boundary_Loss_Solution/$RF" 2>/dev/null | head -1 || true)"
  [[ -n "$old" ]] || die "--resume nhung khong thay $RF trong /kaggle/input — Add Input output version truoc"
  cp -rn "$old"/. "$RF"/ && ok "da chep $old"
fi

run_p0() { local d="$1"; shift
  local a=(--snapshot-dir "$SNAP" --data-root "$DATA" --output-dir "$d/p0_rerun")
  [[ -n "$BENCH" ]] && a+=(--bench-csv "$BENCH")
  python Tools/p0_rerun_eval.py "${a[@]}" "$@" 2>&1 | tee -a "$d/p0.log"
}
run_r0() { local d="$1"; shift
  python Tools/r0_pcrf.py run --ckpt-dir "$CK" --snapshot-dir "$SNAP" --data-root "$DATA" --probe-ids "$PROBE" \
      --figure-images "$FIG" --output-dir "$d/r0_crf" --workers "$WORKERS" --input-mode "$MODE" "$@" 2>&1 | tee -a "$d/r0.log"
}

# ── DRY-RUN: cong chan ─────────────────────────────────────────────────────────
if [[ -z "$SKIP_DRY" ]]; then
  step "DRY-RUN — $DRYN anh val, 2 anh train, bootstrap 200 lan (output: $DRY)"
  rm -rf "$DRY"; mkdir -p "$DRY"
  RC=0
  if has p0; then run_p0 "$DRY" --max-images "$DRYN" --force || RC=1; fi
  if has r0; then run_r0 "$DRY" --max-val "$DRYN" --max-probe 2 --n-boot 200 --force || RC=1; fi
  python Tools/refine_summary.py --root "$DRY" > /dev/null || true
  [[ $RC -eq 0 ]] || die "DRY-RUN FAIL — xem $DRY/p0.log, $DRY/r0.log (khong chay that)"
  ok "DRY-RUN PASS ($(( ($(date +%s) - T0) / 60 )) phut tu dau)"
  if [[ -n "$DRY_ONLY" ]]; then
    echo; echo "==> --dryrun: XONG kiem nhanh. Doc $DRY/summary_P0_R0.md (uoc tinh thoi gian chay that o muc R0)."
    exit 0
  fi
elif [[ -n "$DRY_ONLY" ]]; then
  die "--dryrun va --skip-dryrun mau thuan"
fi

# ── CHAY THAT ─────────────────────────────────────────────────────────────────
RC_P0=0; RC_R0=0
if has p0; then
  step "P0 — JSON hau ky run Static chay lai (2 snapshot x 384 anh val)"
  run_p0 "$RF" || RC_P0=$?
  [[ $RC_P0 -eq 0 ]] && ok "P0 xong, cong PASS" || echo "  [!]     P0: CONG FAIL / loi (rc=$RC_P0) — van chay R0 (doc lap); xem $RF/p0_rerun/gates_p0.json"
fi
if has r0; then
  step "R0 — P-CRF (5 checkpoint; $WORKERS tien trinh CRF; che do $MODE)"
  run_r0 "$RF" || RC_R0=$?
  [[ $RC_R0 -eq 0 ]] && ok "R0 xong, cong PASS" || echo "  [!]     R0: CONG FAIL / loi (rc=$RC_R0) — xem $RF/r0.log, $RF/r0_crf/gates/"
fi

step "Tong hop"
python Tools/refine_summary.py --root "$RF" || true
[[ -f "$RF/summary_P0_R0.md" ]] && sed -n '1,60p' "$RF/summary_P0_R0.md"

echo
echo "==> P0+R0: XONG sau $(( ($(date +%s) - T0) / 60 )) phut. (P0 rc=$RC_P0, R0 rc=$RC_R0)"
echo "    Doc: Output -> Boundary_Loss_Solution/$RF/summary_P0_R0.md, r0_crf/crf_val.md, p0_rerun/p0_compare_csv.md"
[[ $RC_P0 -eq 0 && $RC_R0 -eq 0 ]] || exit 1
exit 0
