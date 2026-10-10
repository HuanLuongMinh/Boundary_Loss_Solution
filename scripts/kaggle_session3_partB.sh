#!/usr/bin/env bash
# scripts/kaggle_session3_partB.sh — PHIEN 3 (Kaggle CPU, khong ton quota GPU): Phan B, C, B2 + tong hop.
#
# Notebook: Accelerator None, Internet On, Add Input: dataset OpenEarthMap + dataset checkpoint
#           + OUTPUT cua version phien 2 (Add Input -> Your Work -> Notebooks -> notebook GPU).
# Chay bang Save Version -> Save & Run All (Commit).
#   Cell 1:  !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
#   Cell 2:  !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_session3_partB.sh
#
# Cac buoc: cai thu vien -> tim duong dan -> giai nen dump tu repo -> chep output Phan A (phien 2)
# -> kiem 8 checkpoint Phan A da PASS -> Phan B (cong 6, dem lo, bootstrap, bang 5.4, cong tan suat)
# -> Phan C (neu B != "khong do loss") -> Phan B2 (neu khong "hiem") -> tong hop lai bao cao Phan A.

source "$(dirname "${BASH_SOURCE[0]}")/kaggle_common.sh"
T0=$(date +%s)

install_deps
find_paths
join_dump

step "Chep output Phan A tu version phien 2"
copy_previous_output || die "khong tim thay output phien 2 trong /kaggle/input — Add Input -> Your Work -> notebook GPU"
verify_part_a "$OUT/grad_conflict" 8 || die "output phien 2 chua du 8 checkpoint PASS"
[[ -d "$OUT/dump/run2_bce02_iter40000/masks" ]] && ok "dump BCE lambda=0.2 (tao o phien 2)" \
  || echo "  [CANH BAO] khong co dump BCE lambda=0.2 — cap mo ta X2 se bi bo qua"

step "Phan B, C, B2"
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" --dump-extract "$DUMP_EXTRACT" --stages "B C B2" 2>&1 | tee part_B.log \
  || die "Phan B/C/B2 that bai — xem part_B.log"

step "Tong hop lai bao cao Phan A (them dong Cong 6)"
python Tools/grad_conflict_probe.py summarize --output-dir "$OUT/grad_conflict" --holes-dir "$OUT/spurious_holes" \
  || die "summarize that bai"

step "Ket qua"
python - "$OUT" <<'EOF'
import json, os, sys
out = sys.argv[1]
a = json.load(open(os.path.join(out, 'grad_conflict/summary_grad_conflict.json'), encoding='utf-8'))
b = json.load(open(os.path.join(out, 'spurious_holes/pairs_bootstrap.json'), encoding='utf-8'))
print(f"  Phan A — nhanh bang 5.2 : {a['decision_52']['text']}")
print(f"  Phan B — verdict bang 5.4: {b['verdict_54']['text']}")
print(f"  Phan B — cong tan suat   : {b['frequency_gate']['status']}")
pc = os.path.join(out, 'spurious_holes/part_c_link.json')
print(f"  Phan C                   : {json.load(open(pc, encoding='utf-8'))['conclusion'] if os.path.exists(pc) else 'khong chay'}")
print(f"  Phan B2                  : {'co fill_test.md' if os.path.exists(os.path.join(out, 'spurious_holes/fill_test.md')) else 'khong chay'}")
EOF

echo
echo "==> PHIEN 3: XONG sau $(( ($(date +%s) - T0) / 60 )) phut."
echo "    Tai ve: Output -> Boundary_Loss_Solution/output/ (grad_conflict/summary_grad_conflict.md, spurious_holes/pairs_bootstrap.md)"
