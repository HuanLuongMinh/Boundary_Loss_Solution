#!/usr/bin/env bash
# scripts/kaggle_K2.sh — K2: KIEM LO GIA (Kaggle CPU, khong ton quota GPU, khong can checkpoint).
# Ke hoach: docs/K2-ke-hoach-kiem-lo-gia-va-quyet-dinh-huong-giai-phap.md   Huong dan: docs/huong-dan-kaggle-K2.md
#
#   K2-a  Tools/hole_oracle_fill.py   oracle lap lo co chon loc: 4 bien the x A_fill {64,256} x 4 checkpoint,
#                                     cong K-a1, K-a2, luat §4.2 -> output/hole_oracle/oracle_fill.md
#   K2-b  Tools/hole_audit_sample.py  mau ~146 lo gia (Static s19@36k) -> audit_sample.csv, audit_crops/H###.png,
#                                     audit_sample_rating.csv (trong) -> goi audit_cham.zip de tai ve cham
#
# Notebook: Accelerator None (CPU), Internet On. Add Input: dataset OpenEarthMap (chi can images/val + label/val).
#           (tuy chon) OUTPUT cua version Phien 3 Buoc 1 (co spurious_holes/fill_test.csv, holes_static_s19_36k.csv)
#           -> cong K-a1 / K-b0 doi chieu them tung dong.
# Chay bang Save Version -> Save & Run All (Commit).
#   Cell 1:  !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
#   Cell 2:  !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_K2.sh
#
# Mask du doan lay tu dump DA COMMIT trong repo (data/dump_parts -> /tmp/dump_repo), giong Phien 3 Buoc 1.
#
# Tuy chon:
#   --stages "a b"     chi chay K2-a (a) hoac K2-b (b); mac dinh ca hai
#   --dry-run N        chay thu N anh -> output/hole_oracle_dryrun/ (cong K-a1, K-b0 = SKIP)
#   --workers N        so tien trinh (mac dinh = so CPU)
#   --resume           chep output/hole_oracle cua version truoc (Add Input) roi chay tiep (nhan da do / mau da chot giu nguyen)
# Bien moi truong: DATA=<thu muc dataset co images/val>

source "$(dirname "${BASH_SOURCE[0]}")/kaggle_common.sh"
die() { echo; echo "  [LOI]   $*" >&2; echo "==> DUNG. Xem docs/huong-dan-kaggle-K2.md (Phu luc)" >&2; exit 1; }

STAGES="a b"; DRY=""; WORKERS="$(nproc 2>/dev/null || echo 4)"; RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stages)  shift; STAGES="$1" ;;
    --dry-run) shift; DRY="$1" ;;
    --workers) shift; WORKERS="$1" ;;
    --resume)  RESUME=1 ;;
    *) die "tham so khong biet: $1" ;;
  esac
  shift
done

T0=$(date +%s)
HO="$OUT/hole_oracle"; EXTRA=()
if [[ -n "$DRY" ]]; then HO="$OUT/hole_oracle_dryrun"; EXTRA=(--max-images "$DRY"); fi
GTC="${TMPDIR:-/tmp}/k2_gt1024"
mkdir -p "$HO"

install_deps

step "Tim dataset OpenEarthMap"
if [[ -z "${DATA:-}" ]]; then
  v="$(find /kaggle/input -maxdepth 7 -type d -path '*images/val' 2>/dev/null | head -1 || true)"
  [[ -n "$v" ]] && DATA="$(dirname "$(dirname "$v")")"
fi
[[ -n "${DATA:-}" && -d "$DATA/images/val" ]] || die "khong tim thay dataset OpenEarthMap (images/val) — kiem Add Input"
ok "DATA = $DATA"

REF_FT="$(find /kaggle/input -type f -name fill_test.csv -path '*spurious_holes*' 2>/dev/null | head -1 || true)"
REF_HO="$(find /kaggle/input -type f -name holes_static_s19_36k.csv -path '*spurious_holes*' 2>/dev/null | head -1 || true)"
[[ -n "$REF_FT" ]] && ok "fill_test.csv Buoc 1: $REF_FT (K-a1 doi chieu them)" || echo "  (khong co fill_test.csv Buoc 1 — K-a1 dung gia tri trong ke hoach)"
[[ -n "$REF_HO" ]] && ok "holes_static_s19_36k.csv Buoc 1: $REF_HO (K-b0 doi chieu them)" || echo "  (khong co holes_static_s19_36k.csv — K-b0 dung so dem trong ke hoach)"

if [[ -n "$RESUME" ]]; then
  step "Resume: chep output K2 cua version truoc"
  old="$(find /kaggle/input -type d -path "*Boundary_Loss_Solution/$HO" 2>/dev/null | head -1 || true)"
  [[ -n "$old" ]] || die "--resume nhung khong thay $HO trong /kaggle/input — Add Input output version truoc"
  cp -rn "$old"/. "$HO"/ && ok "da chep $old"
fi

step "Self-test (CPU, du lieu tong hop)"
python Tools/test_hole_oracle.py > "$HO/selftest.log" 2>&1 || { tail -n 30 "$HO/selftest.log"; die "self-test FAIL"; }
ok "$(tail -n 1 "$HO/selftest.log")"

join_dump
check_val_match
ok "nguon mask: $DP"

RC_A=0; RC_B=0
if [[ " $STAGES " == *" a "* ]]; then
  step "K2-a — oracle lap lo (4 checkpoint x 2 A_fill x 4 bien the, $WORKERS tien trinh)"
  args=(measure --dump-root "$DP" --data-root "$DATA" --gt-cache "$GTC" --output-dir "$HO" --workers "$WORKERS" "${EXTRA[@]}")
  [[ -n "$REF_FT" ]] && args+=(--ref-fill-test "$REF_FT")
  python Tools/hole_oracle_fill.py "${args[@]}" 2>&1 | tee "$HO/k2a.log" || RC_A=$?
  [[ -f "$HO/gates_oracle.json" ]] || die "K2-a khong tao duoc gates_oracle.json — xem $HO/k2a.log"
  [[ $RC_A -eq 0 ]] && ok "K2-a xong, cong PASS" || echo "  [!]     K2-a: CONG FAIL — van chay K2-b; xem $HO/oracle_fill.md muc Cong"
fi

if [[ " $STAGES " == *" b "* ]]; then
  step "K2-b — chon mau + crop kiem bang mat"
  args=(--dump-root "$DP" --data-root "$DATA" --gt-cache "$GTC" --output-dir "$HO" --workers "$WORKERS" "${EXTRA[@]}")
  [[ -n "$REF_HO" ]] && args+=(--ref-holes "$REF_HO")
  python Tools/hole_audit_sample.py "${args[@]}" 2>&1 | tee "$HO/k2b.log" || RC_B=$?
  if [[ $RC_B -eq 0 ]]; then
    # Goi cham: crop + huong dan + file cham trong. KHONG kem audit_sample.csv (khoa mau).
    (cd "$HO" && rm -f audit_cham.zip && python -c "
import glob, zipfile
with zipfile.ZipFile('audit_cham.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob('audit_crops/*')) + ['audit_sample_rating.csv']:
        z.write(f)
") || die "khong tao duoc audit_cham.zip"
    ok "goi cham: $HO/audit_cham.zip ($(du -h "$HO/audit_cham.zip" | cut -f1))"
  else
    echo "  [!]     K2-b FAIL (xem $HO/k2b.log)"
  fi
fi

step "Ket qua"
python - "$HO" <<'EOF'
import json, os, sys
ho = sys.argv[1]
p = os.path.join(ho, 'gates_oracle.json')
if os.path.exists(p):
    g = json.load(open(p, encoding='utf-8'))
    for x in g['gates']:
        print(f"  K2-a {x['gate']:5s}: {x['status']:11s} {x['detail']}")
    lv = g['lever_42']
    print(f"  K2-a §4.2 don bay lo gia: {'CO' if lv['lever_exists'] else ('KHONG' if lv['lever_exists'] is False else 'chua ket luan')}"
          + ('   (DRY-RUN — khong dung duoc)' if g.get('dryrun') else ''))
p = os.path.join(ho, 'gates_audit.json')
if os.path.exists(p):
    g = json.load(open(p, encoding='utf-8'))['gate_kb0']
    print(f"  K2-b K-b0 : {g['status']:11s} {g['detail']}")
s = os.path.join(ho, 'audit_sample.csv')
if os.path.exists(s):
    import pandas as pd
    df = pd.read_csv(s, keep_default_na=False)
    print(f"  K2-b mau : {len(df)} muc ({(df['duplicate_of'] != '').sum()} lap an), "
          f"{len([f for f in os.listdir(os.path.join(ho, 'audit_crops')) if f.endswith('.png')])} crop")
EOF

echo
echo "==> K2: XONG sau $(( ($(date +%s) - T0) / 60 )) phut. (K2-a rc=$RC_A, K2-b rc=$RC_B)"
echo "    Doc: Output -> Boundary_Loss_Solution/$HO/oracle_fill.md ; tai audit_cham.zip de cham (KHONG mo audit_sample.csv)"
[[ $RC_A -eq 0 && $RC_B -eq 0 ]] || exit 1
exit 0
