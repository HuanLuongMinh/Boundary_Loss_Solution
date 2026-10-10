#!/usr/bin/env bash
# scripts/kaggle_common.sh — Ham dung chung cho 3 script phien Kaggle:
#   scripts/kaggle_session1_check.sh   (CPU)  chuan bi + kiem tra, khong chay do
#   scripts/kaggle_session2_partA.sh   (GPU)  Phan A
#   scripts/kaggle_session3_partB.sh   (CPU)  Phan B, C, B2 + tong hop
# Khong chay truc tiep file nay — no duoc `source` boi cac script tren.
#
# Duong dan tu do trong /kaggle/input; ghi de bang bien moi truong neu can:
#   CK=<thu muc 8 checkpoint + 4 summary>  DATA=<thu muc chua images/val>

set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
DUMP_EXTRACT="${DUMP_EXTRACT:-/tmp/dump_repo}"
OUT="${OUT:-output}"

CKPT_FILES="best_model_Static_S19.pth best_miou_Static_s86.pth final_iter40000_static_s86.pth best_model_BCE_04.pth best_model_BCE_02.pth best_miou_aff_s19.pth best_miou_aff_s86.pth best_model_baseline.pth"
SUMMARY_FILES="summary_static_s19.txt summary_static_s86.txt summary_bce04.txt summary_bce02.txt"
LABELS="static_s19_36k static_s86_36k static_s86_40k bce04_40k bce02_40k affonly_s19_40k affonly_s86_40k baseline_40k"

step() { echo; echo "==================== $* ===================="; }
ok()   { echo "  [OK]    $*"; }
die()  { echo; echo "  [LOI]   $*" >&2; echo "==> DUNG. Xem huong dan: docs/huong-dan-kaggle-tung-buoc-grad-conflict.md (Phu luc A)" >&2; exit 1; }

install_deps() {
  step "Cai thu vien (requirements.txt)"
  local mods="torch, timm, albumentations, rasterio, scipy, pandas"
  if [[ -n "${SKIP_PIP:-}" ]]; then
    echo "  (SKIP_PIP=1 — chi dung khi test local: bo qua pip install va kiem rasterio)"
    mods="torch, timm, albumentations, scipy, pandas"
  else
    pip install -q -r requirements.txt 2>&1 | grep -v -E "^\s*$|dependency resolver|incompatible" || true
  fi
  python -c "import $mods; print('  torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '| so GPU:', torch.cuda.device_count())" \
    || die "thieu thu vien sau khi pip install"
}

find_paths() {
  step "Tim duong dan dataset"
  if [[ -z "${CK:-}" ]]; then
    CK="$(dirname "$(find /kaggle/input -name best_model_Static_S19.pth 2>/dev/null | head -1)")"
  fi
  if [[ -z "${DATA:-}" ]]; then
    local v; v="$(find /kaggle/input -maxdepth 6 -type d -path '*images/val' 2>/dev/null | head -1)"
    [[ -n "$v" ]] && DATA="$(dirname "$(dirname "$v")")"
  fi
  echo "  CK   = ${CK:-(khong tim thay)}"
  echo "  DATA = ${DATA:-(khong tim thay)}"
  [[ -n "${CK:-}" && "$CK" != "." && -d "$CK" ]] || die "khong tim thay thu muc checkpoint (co best_model_Static_S19.pth) — kiem Add Input"
  [[ -n "${DATA:-}" && -d "$DATA/images/val" ]] || die "khong tim thay dataset OpenEarthMap (thu muc images/val) — kiem Add Input"
  printf 'CK=%q\nDATA=%q\n' "$CK" "$DATA" > /tmp/paths.env
  export CK DATA
}

check_ckpts() {
  step "Kiem 8 checkpoint + 4 file summary (pos_weight)"
  for f in $CKPT_FILES; do
    [[ -f "$CK/$f" ]] && ok "$f ($(du -h "$CK/$f" | cut -f1))" || die "thieu checkpoint $CK/$f"
  done
  for f in $SUMMARY_FILES; do
    [[ -f "$CK/$f" ]] || die "thieu $CK/$f (doi ten summary.txt theo buoc 0.3 cua huong dan)"
    ok "$f -> $(grep -m1 'pos_weight used' "$CK/$f" | sed 's/  */ /g')"
  done
}

join_dump() {
  step "Ghep + kiem sha256 + giai nen dump tu repo (data/dump_parts -> $DUMP_EXTRACT)"
  [[ -f data/dump_parts/manifest.json ]] || die "khong co data/dump_parts/manifest.json trong repo — da push data/dump_parts chua?"
  python Tools/dump_archive.py join --parts-dir data/dump_parts --dest "$DUMP_EXTRACT" | tee /tmp/join.log \
    || die "ghep/giai nen dump that bai"
  DP="$(sed -n 's/^DUMP_ROOT=//p' /tmp/join.log | tail -1)"
  [[ -d "$DP" ]] || die "khong xac dinh duoc thu muc dump sau khi giai nen"
  export DP
}

check_val_match() {
  step "Doi chieu ten anh dump <-> anh val cua dataset"
  python - "$DP" "$DATA" <<'EOF' || die "danh sach anh dump khac anh val — dang gan nham ban dataset OpenEarthMap?"
import os, sys
dp, data = sys.argv[1:]
val = sorted(f[:-4] for f in os.listdir(os.path.join(data, 'images/val')) if f.lower().endswith('.tif'))
png = sorted(f[:-4] for f in os.listdir(os.path.join(dp, 'run3_static_s19_iter36000/masks')))
print(f"  val tif = {len(val)}, dump png = {len(png)}, trung khop = {val == png}")
sys.exit(0 if val == png and len(val) == 384 else 1)
EOF
}

# Kiem gates_*.json / done_*.json trong 1 thu muc. $1 = thu muc, $2 = so ckpt toi thieu, $3 = "gpu" de doi device cuda
verify_part_a() {
  python - "$1" "$2" "${3:-}" <<'EOF'
import glob, json, os, sys
d, need, want_gpu = sys.argv[1], int(sys.argv[2]), sys.argv[3] == 'gpu'
gates = sorted(glob.glob(os.path.join(d, 'gates_*.json')))
dones = sorted(glob.glob(os.path.join(d, 'done_*.json')))
bad = 0
for g in gates:
    j = json.load(open(g, encoding='utf-8'))
    fails = [f"cong {x['gate']}" for x in j['gates'] if x['status'] not in ('PASS', 'SKIP_DRYRUN')]
    print(f"  gates  {j['label']:16s} {j['overall']:5s} {'(' + ', '.join(fails) + ')' if fails else ''}")
    bad += j['overall'] != 'PASS'
for p in dones:
    j = json.load(open(p, encoding='utf-8'))
    dev = j.get('device', '?')
    print(f"  done   {j['label']:16s} {j['n_images']:4d} anh  {j['wall_time_s'] / 60:6.1f} ph  device={dev}  gpu={j.get('gpu')}")
    if want_gpu and not str(dev).startswith('cuda'):
        print(f"         ^ KHONG chay tren GPU!"); bad += 1
if len(dones) < need:
    print(f"  chi co {len(dones)}/{need} checkpoint da do xong"); bad += 1
sys.exit(1 if bad else 0)
EOF
}

# Chep output cua 1 version notebook truoc (da Add Input) vao repo. $1 = mo ta
copy_previous_output() {
  local old
  old="$(find /kaggle/input -type d -path '*Boundary_Loss_Solution/output' 2>/dev/null | head -1)"
  [[ -n "$old" ]] || return 1
  echo "  output version truoc: $old"
  mkdir -p "$OUT"
  cp -rn "$old"/. "$OUT"/
  return 0
}
