#!/usr/bin/env bash
# scripts/kaggle_session2_partA.sh — PHIEN 2 (Kaggle GPU T4 x2, TON QUOTA): Phan A.
#
# Notebook: Accelerator GPU T4 x2, Internet On, Add Input: dataset OpenEarthMap + dataset checkpoint.
# Chay bang Save Version -> Save & Run All (Commit).
#   Cell 1:  !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
#   Cell 2:  !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_session2_partA.sh
#
# Cac buoc:
#   [1] cai thu vien, kiem GPU, tim duong dan, kiem checkpoint/summary, giai nen dump
#   [2] CHAY THU 4 anh (tao dump BCE lambda=0.2 + chot 6 anh minh hoa + cong kiem + do) — "cong chan":
#       co cong FAIL hoac khong chay tren GPU -> DUNG, khong ton quota cho 384 anh
#   [3] CHAY THAT 384 anh (2 GPU song song neu co T4 x2) + tong hop summary_grad_conflict.md
#   [4] xac nhan: 8 checkpoint PASS, chay tren cuda, in dau bao cao
#
# Tuy chon:
#   --dryrun-only   chi chay [1]-[2] (de do thoi gian/anh truoc)
#   --skip-dryrun   bo [2] (vd chay lai sau khi da chay thu thanh cong)
#   --resume        chep output cua version truoc (Add Input -> Your Work -> notebook do) roi chay tiep;
#                   checkpoint da co done_*.json se duoc bo qua

source "$(dirname "${BASH_SOURCE[0]}")/kaggle_common.sh"

DRYRUN=1; FULL=1; RESUME=""
for a in "$@"; do
  case "$a" in
    --dryrun-only) FULL="" ;;
    --skip-dryrun) DRYRUN="" ;;
    --resume)      RESUME=1 ;;
    *) die "tham so khong biet: $a" ;;
  esac
done
T0=$(date +%s)

install_deps
step "Kiem GPU"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || die "khong co GPU — Settings -> Accelerator: GPU T4 x2"
NGPU=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
[[ "$NGPU" -ge 1 ]] || die "torch khong thay GPU"
python -c "import torch; (torch.zeros(1, device='cuda') + 1).item()" || die "GPU khong chay duoc voi ban PyTorch nay (vd P100) — chon T4"
PAR=""; [[ "$NGPU" -ge 2 ]] && PAR="--parallel"
ok "$NGPU GPU — che do: ${PAR:-1 GPU tuan tu}"

find_paths
check_ckpts
join_dump

if [[ -n "$RESUME" ]]; then
  step "Chep output version truoc (--resume)"
  copy_previous_output || die "--resume nhung khong tim thay output version truoc trong /kaggle/input"
  ls "$OUT/grad_conflict"/done_*.json 2>/dev/null | sed 's/^/  da xong: /' || true
fi

if [[ -n "$DRYRUN" ]]; then
  step "[2] CHAY THU 4 anh — cong chan"
  bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" --dump-extract "$DUMP_EXTRACT" \
       --make-missing-dumps --dry-run 4 --stages "figures gates A" 2>&1 | tee dryrun_A.log \
    || die "chay thu that bai — xem dryrun_A.log va $OUT/grad_conflict_dryrun/gates_*.json"
  step "[2] Ket qua chay thu"
  verify_part_a "$OUT/grad_conflict_dryrun" 8 gpu || die "chay thu co cong FAIL / khong chay tren GPU"
  python - "$OUT/grad_conflict_dryrun" <<'EOF'
import glob, json, sys
t = [json.load(open(p))['wall_time_s'] for p in glob.glob(sys.argv[1] + '/done_*.json')]
print(f"  tong thoi gian do 4 anh x {len(t)} ckpt = {sum(t)/60:.1f} ph -> uoc tinh 384 anh ~ {sum(t)/4*384/3600:.1f} gio GPU (1 GPU), chua tinh cong kiem")
EOF
  ok "chay thu dat — tiep tuc chay that"
fi

if [[ -n "$FULL" ]]; then
  step "[3] CHAY THAT 384 anh ${PAR:+(2 GPU song song)}"
  bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" --dump-extract "$DUMP_EXTRACT" \
       --make-missing-dumps $PAR --stages "summarize" 2>&1 | tee full_A.log \
    || die "chay that that bai — xem full_A.log, $OUT/parallel_cuda*.log, $OUT/grad_conflict/gates_*.json"
  step "[4] Xac nhan ket qua Phan A"
  verify_part_a "$OUT/grad_conflict" 8 gpu || die "chua du 8 checkpoint PASS tren GPU"
  [[ -f "$OUT/grad_conflict/summary_grad_conflict.md" ]] || die "khong co summary_grad_conflict.md"
  echo; sed -n '1,45p' "$OUT/grad_conflict/summary_grad_conflict.md"
fi

echo
echo "==> PHIEN 2: XONG sau $(( ($(date +%s) - T0) / 60 )) phut."
[[ -n "$FULL" ]] && echo "    Ket qua: Output -> Boundary_Loss_Solution/output/grad_conflict/summary_grad_conflict.md" \
                 && echo "    Tiep theo: PHIEN 3 (CPU) — Add Input output cua version nay."
exit 0
