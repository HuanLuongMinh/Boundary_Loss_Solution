#!/usr/bin/env bash
# scripts/kaggle_phanD_phien2.sh — PHAN D, PHIEN 2 CHAY DOC LAP: do snapshot + tong hop (bang 6.3).
# Spec: docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md §2, §6   Huong dan: docs/huong-dan-kaggle-phanD.md (muc 6)
#
# Dung khi Phien 1 (train + snapshot) da xong o mot version truoc. KHONG train lai, KHONG can dataset checkpoint.
#
# Notebook: Accelerator GPU T4 x2 (hoac 1 GPU — cham gap doi), Internet On.
# Add Input: (1) dataset OpenEarthMap; (2) OUTPUT cua version da chay Phien 1
#            (Your Work -> Notebooks -> version do), HOAC Kaggle Dataset chua thu muc phanD/ (snapshots/ + probe_ids.txt).
#   Cell 1:  !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
#   Cell 2:  !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_phanD_phien2.sh
#
# Script tu tim trong /kaggle/input: thu muc phanD cu (co probe_ids.txt), thu muc snapshots/ (co iter_36000.pth),
# dataset (images/train). Chep ket qua nho cua Phien 1 (gates_dryrun.json, meta, post/, logs/, va gram/feat/done/
# da do do dang neu co) vao output/phanD; snapshot doc TAI CHO (khong chep 1.7 GB).
#
# Tuy chon:
#   --summarize-only      chi tong hop lai (vd da do du 35 snapshot o version truoc)
#   --prev DIR            thu muc phanD cu (mac dinh: tu tim)
#   --snapshot-dir DIR    thu muc iter_<t>.pth (mac dinh: <prev>/snapshots hoac tu tim)
# Bien moi truong: DATA=<thu muc dataset co images/train>, SCRATCH=<thu muc memmap tam, mac dinh /tmp/phanD_scratch>

source "$(dirname "${BASH_SOURCE[0]}")/kaggle_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/phanD_phien2_core.sh"
die() { echo; echo "  [LOI]   $*" >&2; echo "==> DUNG. Xem docs/huong-dan-kaggle-phanD.md (muc 6 + Phu luc)" >&2; exit 1; }

MEASURE=1; PREV="${PREV:-}"; SNAP_DIR="${SNAP_DIR:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --summarize-only) MEASURE="" ;;
    --prev)           shift; PREV="$1" ;;
    --snapshot-dir)   shift; SNAP_DIR="$1" ;;
    *) die "tham so khong biet: $1" ;;
  esac
  shift
done

T0=$(date +%s)
PD="$REPO/$OUT/phanD"
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/phanD_scratch}"
mkdir -p "$PD/logs"

install_deps

step "Kiem GPU"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || die "khong co GPU — Settings -> Accelerator: GPU T4 x2"
NGPU=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
[[ "$NGPU" -ge 1 ]] || die "torch khong thay GPU"
python -c "import torch; (torch.zeros(1, device='cuda') + 1).item()" || die "GPU khong chay duoc voi ban PyTorch nay (vd P100) — chon T4"
ok "$NGPU GPU — do tren $(( NGPU >= 2 ? 2 : 1 )) GPU song song"

step "Tim du lieu dau vao trong /kaggle/input"
if [[ -z "$DATA" ]]; then
  v="$(find /kaggle/input -maxdepth 7 -type d -path '*images/train' 2>/dev/null | head -1)"
  [[ -n "$v" ]] && DATA="$(dirname "$(dirname "$v")")"
fi
[[ -n "$DATA" && -d "$DATA/images/train" ]] || die "khong tim thay dataset OpenEarthMap (images/train) — kiem Add Input"
if [[ -z "$PREV" ]]; then
  p="$(find /kaggle/input -type f -name probe_ids.txt -path '*phanD*' 2>/dev/null | head -1)"
  [[ -n "$p" ]] && PREV="$(dirname "$p")"
fi
[[ -n "$PREV" && -f "$PREV/probe_ids.txt" ]] || die "khong tim thay output Phan D cu (phanD/probe_ids.txt) — Add Input version Phien 1"
if [[ -z "$SNAP_DIR" ]]; then
  if ls "$PREV/snapshots"/iter_*.pth >/dev/null 2>&1; then
    SNAP_DIR="$PREV/snapshots"
  else
    s="$(find /kaggle/input -type f -name iter_36000.pth 2>/dev/null | head -1)"
    [[ -n "$s" ]] && SNAP_DIR="$(dirname "$s")"
  fi
fi
[[ -n "$SNAP_DIR" ]] && ls "$SNAP_DIR"/iter_*.pth >/dev/null 2>&1 || die "khong tim thay thu muc snapshot (iter_<t>.pth)"
echo "  DATA     = $DATA"
echo "  PREV     = $PREV"
echo "  SNAP_DIR = $SNAP_DIR ($(ls "$SNAP_DIR"/iter_*.pth | wc -l) file)"
[[ -f "$SNAP_DIR/iter_36000.pth" ]] || die "thieu iter_36000.pth trong $SNAP_DIR (spec §1.1 bat buoc)"

step "Chep ket qua nho cua Phien 1 -> $PD (khong chep snapshot/checkpoint)"
if [[ "$(cd "$PREV" && pwd)" != "$(cd "$PD" && pwd)" ]]; then
  for item in probe_ids.txt probe_ids_meta.json gates_dryrun.json meta.json train_script.txt schedule.txt \
              figure_images.json d0_script_diff.md d0_static.json affinity_support.md affinity_support.json \
              gates post logs gram feat done run_log.txt measure_errors.txt; do
    [[ -e "$PREV/$item" ]] && cp -rn "$PREV/$item" "$PD/" && echo "  [chep]  $item"
  done
fi
[[ $(wc -l < "$PD/probe_ids.txt") -eq 32 ]] || die "probe_ids.txt khong du 32 dong"
if [[ -f phanD/probe_ids.txt ]]; then
  cmp -s phanD/probe_ids.txt "$PD/probe_ids.txt" || die "probe_ids.txt cua Phien 1 KHAC ban da commit phanD/probe_ids.txt"
  ok "probe_ids.txt khop ban da commit"
fi
python - "$PD" <<'EOF'
import json, os, sys
pd = sys.argv[1]
p = os.path.join(pd, 'gates_dryrun.json')
if not os.path.exists(p):
    print("  [!]     khong co gates_dryrun.json — bao cao se ghi 'dien giai duoc: ?'")
else:
    g = json.load(open(p))
    post = g.get('post', {})
    print(f"  dry-run: {g.get('overall_dryrun')} | script train: {g.get('meta', {}).get('train_script')} | "
          f"sanity §5: {post.get('sanity5', {}).get('status')} | D6: {post.get('d6', {}).get('status')} | "
          f"dien giai duoc: {post.get('interpretable')}")
n = len([f for f in os.listdir(os.path.join(pd, 'done'))]) if os.path.isdir(os.path.join(pd, 'done')) else 0
print(f"  snapshot da do xong tu truoc: {n}")
EOF

if [[ -n "$MEASURE" ]]; then
  phanD_measure
fi
phanD_summarize

echo
VERD=$(python -c "import json; d=json.load(open('$PD/summary_phanD.json')); print(d['decision']['verdict'], '| dien giai duoc:', d.get('interpretable'))" 2>/dev/null || echo "(chua tong hop)")
echo "==> PHAN D — PHIEN 2: XONG sau $(( ($(date +%s) - T0) / 60 )) phut. Verdict: $VERD"
echo "    Ket qua: Output -> Boundary_Loss_Solution/output/phanD/summary_phanD.md (+ traj_param.csv, figures/, gram/)"
exit 0
