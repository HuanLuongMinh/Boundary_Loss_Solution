#!/usr/bin/env bash
# scripts/kaggle_session1_check.sh — PHIEN 1 (Kaggle CPU, khong ton quota GPU): chuan bi + kiem tra.
# KHONG chay do. Muc dich: phat hien moi loi du lieu/duong dan TRUOC khi mo phien GPU.
#
# Notebook: Accelerator None, Internet On, Add Input: dataset OpenEarthMap + dataset checkpoint.
#   Cell 1:  !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
#   Cell 2:  !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_session1_check.sh
#
# Cac buoc: cai thu vien -> tim duong dan -> 8 checkpoint + 4 summary -> giai nen dump tu repo
# -> doi chieu anh val -> chay test -> in truoc lenh se chay o phien 2 (bang ghep + pos_weight).
# Dong cuoi "==> PHIEN 1: TAT CA OK" = san sang sang phien 2.

source "$(dirname "${BASH_SOURCE[0]}")/kaggle_common.sh"

install_deps
find_paths
check_ckpts
join_dump
check_val_match

step "Chay test (du lieu tong hop, ~1 phut)"
python Tools/test_spurious_holes.py 2>&1 | tail -1 | grep -q "PASS" || die "Tools/test_spurious_holes.py FAIL"
ok "Tools/test_spurious_holes.py"
python Tools/test_grad_conflict_probe.py 2>&1 | tail -1 | grep -q "PASS" || die "Tools/test_grad_conflict_probe.py FAIL"
ok "Tools/test_grad_conflict_probe.py"

step "Xem truoc lenh se chay o phien 2 (KHONG chay that)"
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" --dump-extract "$DUMP_EXTRACT" --make-missing-dumps --print-only \
  | tee /tmp/preview.log | grep -vE '^\+ |^   \+ '
grep -q "(KHONG CO)" /tmp/preview.log && die "co nhan thieu checkpoint/dump (dong '(KHONG CO)' o tren)"
n_pw=$(grep -c "pos_weight = " /tmp/preview.log || true)
[[ "$n_pw" == "5" ]] || die "doc duoc $n_pw/5 gia tri pos_weight"
ok "8 checkpoint duoc ghep, 5 pos_weight doc tu summary"

echo
echo "==> PHIEN 1: TAT CA OK — tat phien (Stop session) va sang PHIEN 2 (GPU)."
