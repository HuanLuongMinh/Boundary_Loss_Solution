#!/usr/bin/env bash
# scripts/kaggle_phanD.sh — PHAN D (Kaggle GPU T4 x2, TON QUOTA): quy dao xung dot gradient, Static s19.
# Spec: docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md   Huong dan: docs/huong-dan-kaggle-phanD.md
#
# MOT LENH chay het, khong dung giua chung: dry-run (cong chan) -> Phien 1 (train 40k + snapshot)
# -> kiem sau train (sanity §5, D6) -> Phien 2 (do 35 snapshot) -> tong hop (bang 6.3).
#
# Notebook: Accelerator GPU T4 x2, Internet On, Add Input: dataset OpenEarthMap + dataset checkpoint
# (can best_model_Static_S19.pth + summary_static_s19.txt). Chay bang Save Version -> Save & Run All (Commit).
#   Cell 1:  !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
#   Cell 2:  !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_phanD.sh
#
# Stage (mac dinh chay tat ca, theo thu tu):
#   prep       cai thu vien, kiem 2xT4, tim duong dan, kiem checkpoint/summary, md5 config,
#              sinh probe_ids.txt (32 anh train) + figure_images.json
#   dryrun     R0, D0 (chon script train), D1, D2, D3, D4, D5, §3 — CONG CHAN: co cong FAIL -> DUNG
#   train      Phien 1: torchrun 2 GPU, 40.000 iter, luu snapshot theo lich (tu resume neu bi ngat)
#   post       sanity §5 (mIoU-9 @36k) + D6 — FAIL chi GAN CO "khong dien giai", VAN chay tiep
#   measure    Phien 2: do moi snapshot (2 GPU song song, moi GPU mot nua so snapshot)
#   summarize  traj_param.csv, traj_feat_strata.csv, bang 6.3, mo ta 6.4, 2 hinh, summary_phanD.md
#
# Tuy chon:
#   --dryrun-only        chi prep + dryrun (xem gates_dryrun.json va thoi gian D5 truoc)
#   --skip-dryrun        bo dryrun (vd version truoc da dry-run PASS — dung kem --resume)
#   --resume             chep output cua version truoc (Add Input -> Your Work -> notebook do) roi chay tiep:
#                        train tu resume tu latest_checkpoint.pth, snapshot/do da co thi bo qua
#   --stages "a b ..."   chi chay cac stage nay (vd --stages "measure summarize")

source "$(dirname "${BASH_SOURCE[0]}")/kaggle_common.sh"

die() { echo; echo "  [LOI]   $*" >&2; echo "==> DUNG. Xem docs/huong-dan-kaggle-phanD.md (Phu luc — loi thuong gap)" >&2; exit 1; }

STAGES="prep dryrun train post measure summarize"
RESUME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dryrun-only) STAGES="prep dryrun" ;;
    --skip-dryrun) STAGES="${STAGES/dryrun/}" ;;
    --resume)      RESUME=1 ;;
    --stages)      shift; STAGES="prep $1" ;;
    *) die "tham so khong biet: $1" ;;
  esac
  shift
done
has() { [[ " $STAGES " == *" $1 "* ]]; }

T0=$(date +%s)
REF_COMMIT="8d52089"                                    # commit Buoc 1 (logic do da qua 7 cong)
CFG="configs/unet_former_resnet18_static_boundary/static_boundary.yaml"
PD="$REPO/$OUT/phanD"                                   # moi output Phan D
GD="$PD/gates"                                          # file JSON tung cong dry-run
DR="$PD/dryrun"                                         # 4-5 run 500 iter + R0 + D3
SCRATCH="${TMPDIR:-/tmp}/phanD_scratch"
W_SCRIPT="src/train_static_boundary_weighted.py"
O_SCRIPT="src/train_static_boundary.py"
mkdir -p "$PD" "$GD" "$DR" "$PD/logs" "$PD/post"

# ── chung ─────────────────────────────────────────────────────────────────────
gpu_setup() {
  step "Kiem GPU"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv || die "khong co GPU — Settings -> Accelerator: GPU T4 x2"
  NGPU=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")
  [[ "$NGPU" -ge 2 ]] || die "can 2 GPU (train DDP 2xT4, batch 4/GPU) — dang co $NGPU"
  python -c "import torch; (torch.zeros(1, device='cuda') + 1).item()" || die "GPU khong chay duoc voi ban PyTorch nay (vd P100) — chon T4 x2"
  ok "$NGPU GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
}

script_of() { [[ "$1" == "orig" ]] && echo "$O_SCRIPT" || echo "$W_SCRIPT"; }

# run_500 <nhan> <orig|weighted> <port> [co them...]: 500 iter, seed 19, 2 GPU, khong validation
run_500() {
  local name="$1" which="$2" port="$3"; shift 3
  local wd="$DR/$name"
  rm -rf "$wd"; mkdir -p "$wd"
  echo "  -> run $name ($(script_of "$which"), 500 iter) ..."
  local t=$(date +%s)
  torchrun --nproc_per_node=2 --master_port="$port" "$(script_of "$which")" --config "$CFG" --data-root "$DATA" \
      --work-dir "$wd" --stop-at-iter 500 --loss-log "$wd/loss.csv" "$@" > "$DR/$name.log" 2>&1 \
    || { tail -40 "$DR/$name.log"; die "run $name that bai — xem $DR/$name.log"; }
  [[ -f "$wd/loss.csv" ]] || die "run $name khong ghi loss.csv"
  ok "$name xong sau $(( $(date +%s) - t ))s — $(grep -o 'pos_weight=[0-9.]*' "$wd/train_log.txt" | tail -1)"
}

assemble() {
  python Tools/phanD_dryrun.py assemble --gate-dir "$GD" --meta "$PD/meta.json" \
      ${1:+--post "$1"} --out "$PD/gates_dryrun.json"
}

if [[ -n "$RESUME" ]]; then
  step "Chep output version truoc (--resume)"
  copy_previous_output || die "--resume nhung khong tim thay output version truoc trong /kaggle/input"
  ls "$PD" 2>/dev/null | sed 's/^/  da co: /' || true
fi

# ══════════════════════════════════ prep ═════════════════════════════════════
install_deps
gpu_setup
find_paths
step "Kiem checkpoint + summary can cho Phan D"
[[ -f "$CK/best_model_Static_S19.pth" ]] && ok "best_model_Static_S19.pth ($(du -h "$CK/best_model_Static_S19.pth" | cut -f1))" \
  || die "thieu $CK/best_model_Static_S19.pth"
[[ -f "$CK/summary_static_s19.txt" ]] || die "thieu $CK/summary_static_s19.txt"
PW_LINE=$(grep -m1 'pos_weight used' "$CK/summary_static_s19.txt" | sed 's/  */ /g')
ok "summary_static_s19.txt -> $PW_LINE"
echo "$PW_LINE" | grep -q '11.0014' || die "summary_static_s19.txt khong co pos_weight 11.0014 — dat nham file?"
[[ -d "$DATA/images/train" ]] || die "dataset khong co images/train"

if has prep; then
  step "Meta: md5 config, commit"
  CFG_MD5=$(md5sum "$CFG" | cut -d' ' -f1)
  python - "$PD/meta.json" "$CFG" "$CFG_MD5" "$CK" "$DATA" <<'EOF'
import json, os, subprocess, sys, datetime, torch
out, cfg, md5, ck, data = sys.argv[1:]
old = json.load(open(out)) if os.path.exists(out) else {}
old.update({'config': cfg, 'config_md5': md5, 'seed': 19, 'ckpt_dir': ck, 'data_root': data,
            'git_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
            'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'torch': torch.__version__, 'eval_mode': 'model.eval() — BN running stats; fp32; 1 anh/backward',
            'note_config': 'S1 §7 ghi run3_static_boundary.yaml — repo chi co static_boundary.yaml (file sinh '
                           'best_model_Static_S19.pth theo done_static_s19_36k.json Buoc 1); md5 ghi o tren',
            'prep_at': datetime.datetime.now().isoformat(timespec='seconds')})
old.setdefault('train_script', None)
json.dump(old, open(out, 'w'), indent=2, ensure_ascii=False)
EOF
  ok "config $CFG md5=$CFG_MD5"

  step "§2.1 — Tap do: 32 anh train (luu TRUOC phien 1)"
  CHECK=""; [[ -f phanD/probe_ids.txt ]] && CHECK="--check phanD/probe_ids.txt"
  python Tools/grad_conflict_probe.py phanD-probe-ids --config "$CFG" --data-root "$DATA" \
      --output "$PD/probe_ids.txt" $CHECK || die "sinh probe_ids.txt that bai"
  [[ $(wc -l < "$PD/probe_ids.txt") -eq 32 ]] || die "probe_ids.txt khong du 32 dong"
  ok "probe_ids.txt: $(head -3 "$PD/probe_ids.txt" | xargs) ... (sha1 $(sha1sum "$PD/probe_ids.txt" | cut -c1-12))"

  step "figure_images.json (2 anh val dau tien dung cho §3)"
  if [[ ! -f "$PD/figure_images.json" ]]; then
    python Tools/select_figure_images.py --config "$CFG" --data-root "$DATA" --output "$PD/figure_images.json" \
        --gt-cache "$OUT/_cache/gt1024" || die "select_figure_images that bai"
  fi
  ok "$(python -c "import json; print([d['image_id'] for d in json.load(open('$PD/figure_images.json'))['images']][:2])")"
fi

# ══════════════════════════════════ dryrun ═══════════════════════════════════
if has dryrun; then
  if [[ -f "$PD/gates_dryrun.json" ]] && python -c "import json,sys; sys.exit(0 if json.load(open('$PD/gates_dryrun.json'))['overall_dryrun']=='PASS' else 1)"; then
    step "DRY-RUN da PASS truoc do ($PD/gates_dryrun.json) — bo qua"
  else
    step "[R0] Hoi quy probe: code cu ($REF_COMMIT) vs code moi, static_s19_36k, 4 anh val"
    REFWT="${TMPDIR:-/tmp}/phanD_ref"
    if [[ ! -d "$REFWT" ]]; then
      git worktree add --detach "$REFWT" "$REF_COMMIT" >/dev/null 2>&1 \
        || die "khong tao duoc worktree commit $REF_COMMIT (clone shallow?) — clone lai khong dung --depth"
    fi
    for side in old new; do
      dir="$REPO"; [[ "$side" == "old" ]] && dir="$REFWT"
      rm -rf "$DR/r0_$side"
      ( cd "$dir" && python Tools/grad_conflict_probe.py measure \
          --ckpt "static_s19_36k=$CK/best_model_Static_S19.pth" --pos-weight static_s19_36k=11.0014 \
          --data-root "$DATA" --output-dir "$DR/r0_$side" --max-images 4 --batch-size 2 --skip-gate-check \
          --figures-json "$PD/figure_images.json" --gt-cache '' --device cuda:0 \
          --scratch-dir "$SCRATCH/r0_$side" \
        && python Tools/grad_conflict_probe.py summarize --output-dir "$DR/r0_$side" \
          --holes-dir "$DR/r0_$side/_noholes" --n-boot 2000 ) > "$DR/r0_$side.log" 2>&1 \
        || { tail -30 "$DR/r0_$side.log"; die "R0: chay probe ($side) that bai — xem $DR/r0_$side.log"; }
    done
    rm -f "$DR"/r0_*/run_log.txt
    python Tools/phanD_dryrun.py regress --old "$DR/r0_old" --new "$DR/r0_new" --out "$GD/R0.json" \
      || die "R0 FAIL — mo rong grad_conflict_probe.py da doi ket qua che do cu ($GD/R0.json)"

    step "[D0] Chon script train: phan tich tinh 5 muc + 500 iter (W-a, W-b, O-a)"
    python Tools/phanD_dryrun.py script-diff --out "$PD/d0_script_diff.md" --static-json "$PD/d0_static.json" \
      || die "D0: phan tich tinh khong chay duoc"
    run_500 W-a weighted 29511
    run_500 W-b weighted 29512
    run_500 O-a orig     29513
    set +e
    python Tools/phanD_dryrun.py compare --gate D0 --selection \
        --name 'chon script: weighted vs goc (500 iter, seed 19)' \
        --a "$DR/W-a/loss.csv" --b "$DR/W-b/loss.csv" --c "$DR/O-a/loss.csv" \
        --iter1-exact --static-json "$PD/d0_static.json" \
        --pos-weight-log "$DR/W-a/train_log.txt" --out "$GD/D0.json"
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      TRAIN_WHICH=weighted; A=W-a; B=W-b
      ok "D0 PASS -> dung $W_SCRIPT (chi them --save-snapshots)"
    elif [[ $rc -eq 3 ]]; then
      TRAIN_WHICH=orig; A=O-a; B=O-b
      echo "  [!]     D0 FAIL -> chuyen sang script goc $O_SCRIPT (+ --report-ckpts); chay them O-b"
      run_500 O-b orig 29514
    else
      die "D0: pos_weight cua W-a khac 11.0014 — xem $GD/D0.json"
    fi
    echo "$TRAIN_WHICH" > "$PD/train_script.txt"

    step "[D1+D2] Run ON (snapshot 0/250/500) bang script da chon + nap lai snapshot iter_500"
    run_500 ON "$TRAIN_WHICH" 29515 --save-snapshots 0,250,500 --snapshot-dir "$DR/ON/snapshots" \
        --verify-snapshot-load "$GD/D2.json"
    python Tools/phanD_dryrun.py compare --gate D1 --name 'luu snapshot khong doi huan luyen' \
        --a "$DR/$A/loss.csv" --b "$DR/$B/loss.csv" --c "$DR/ON/loss.csv" \
        --pos-weight-log "$DR/ON/train_log.txt" --out "$GD/D1.json" || die "D1 FAIL — xem $GD/D1.json"
    [[ -f "$GD/D2.json" ]] || die "khong co $GD/D2.json"
    grep -q '"status": "PASS"' "$GD/D2.json" && ok "D2 PASS (trung bit)" || die "D2 FAIL — xem $GD/D2.json"
    for t in 0 250 500; do [[ -f "$DR/ON/snapshots/iter_$t.pth" ]] || die "thieu snapshot iter_$t.pth"; done

    step "[D3] Gram dung (iter_500.pth, 32 anh do) + do thoi gian/snapshot"
    python Tools/grad_conflict_probe.py phanD-gates --gate d3 --snapshot "$DR/ON/snapshots/iter_500.pth" \
        --data-root "$DATA" --image-list "$PD/probe_ids.txt" --output-dir "$DR/d3" --device cuda:0 \
        --scratch-dir "$SCRATCH" --json-out "$GD/D3.json" > "$DR/d3.log" 2>&1 \
      || { tail -30 "$DR/d3.log"; die "D3 FAIL — xem $GD/D3.json / $DR/d3.log"; }
    ok "D3 PASS — $(python -c "import json; v=json.load(open('$GD/D3.json'))['value']; print(f\"{v['s_per_snapshot']:.0f} s/snapshot, {v['s_per_image']:.2f} s/anh, P={v['n_shared_params']}\")")"

    step "[D4] Cong 2/3/5 cua S1 tren 4 anh do (train), iter_500.pth"
    python Tools/grad_conflict_probe.py phanD-gates --gate d4 --snapshot "$DR/ON/snapshots/iter_500.pth" \
        --data-root "$DATA" --image-list "$PD/probe_ids.txt" --output-dir "$DR/d4" --device cuda:0 \
        --json-out "$GD/D4.json" > "$DR/d4.log" 2>&1 || { tail -30 "$DR/d4.log"; die "D4 FAIL — xem $GD/D4.json"; }
    ok "D4 PASS"

    step "[D5] Thoi gian"
    python Tools/phanD_dryrun.py timing --loss-log "$DR/ON/loss.csv" --d3-json "$GD/D3.json" --n-gpu "$NGPU" \
        --out "$GD/D5.json" --schedule-out "$PD/schedule.txt" || die "D5 FAIL — ca lich gian van vuot 10h"

    step "[§3] Vung ho tro affinity (mo ta, khong la cong)"
    python Tools/affinity_support.py --ckpt "$CK/best_model_Static_S19.pth" --figures-json "$PD/figure_images.json" \
        --data-root "$DATA" --device cuda:0 --output-dir "$PD" > "$DR/affinity_support.log" 2>&1 \
      && ok "affinity_support.md" || echo "  [!]     §3 loi (khong chan) — xem $DR/affinity_support.log"

    python - "$PD/meta.json" "$TRAIN_WHICH" "$(script_of "$TRAIN_WHICH")" "$(cat "$PD/schedule.txt")" <<'EOF'
import json, sys
p, which, script, sched = sys.argv[1:]
m = json.load(open(p)); m.update({'train_script': script, 'train_which': which, 'snapshot_schedule': sched})
json.dump(m, open(p, 'w'), indent=2, ensure_ascii=False)
EOF
    step "Tong hop dry-run"
    assemble
    python -c "import json,sys; sys.exit(0 if json.load(open('$PD/gates_dryrun.json'))['overall_dryrun']=='PASS' else 1)" \
      || die "DRY-RUN FAIL — xem $PD/gates_dryrun.json; KHONG chay phien 1"
    ok "DRY-RUN: tat ca cong PASS — tiep tuc Phien 1"
  fi
fi

# ══════════════════════════════════ train (Phien 1) ══════════════════════════
TRAIN_WHICH=$(cat "$PD/train_script.txt" 2>/dev/null || echo weighted)
SCHED=$(cat "$PD/schedule.txt" 2>/dev/null || echo default)
WORK="$PD/work"
if has train; then
  step "PHIEN 1 — train 40.000 iter, $(script_of "$TRAIN_WHICH"), snapshot lich '$SCHED'"
  [[ -f "$PD/train_script.txt" ]] || echo "  [!]     khong co ket qua D0 (bo qua dry-run?) — mac dinh $W_SCRIPT"
  python - "$PD/meta.json" "$TRAIN_WHICH" "$(script_of "$TRAIN_WHICH")" "$SCHED" <<'EOF'
import json, os, sys
p, which, script, sched = sys.argv[1:]
m = json.load(open(p)) if os.path.exists(p) else {}
m.update({'train_script': script, 'train_which': which, 'snapshot_schedule': sched})
json.dump(m, open(p, 'w'), indent=2, ensure_ascii=False)
EOF
  EXTRA=(); [[ "$TRAIN_WHICH" == "orig" ]] && EXTRA+=(--report-ckpts)
  [[ -f "$WORK/latest_checkpoint.pth" ]] && echo "  (co latest_checkpoint.pth — script train tu resume)"
  t=$(date +%s)
  set +e
  torchrun --nproc_per_node=2 --master_port=29520 "$(script_of "$TRAIN_WHICH")" --config "$CFG" --data-root "$DATA" \
      --work-dir "$WORK" --save-snapshots "$SCHED" --snapshot-dir "$PD/snapshots" "${EXTRA[@]}" 2>&1 \
      | tee -a "$PD/logs/train_stdout.log" \
      | grep -E --line-buffered 'Val round|New best|Final|Phan D|LOI|Error|Traceback|Early stopping|pos_weight|\[ *[0-9]+000/'
  rc=${PIPESTATUS[0]}
  set -e
  [[ $rc -eq 0 ]] || { tail -40 "$PD/logs/train_stdout.log"; die "train that bai (ma $rc) — xem $PD/logs/train_stdout.log"; }
  ok "train xong sau $(( ($(date +%s) - t) / 60 )) phut"

  step "Kiem output phien 1"
  python - "$WORK" "$PD/snapshots" "$SCHED" <<'EOF' || die "thieu checkpoint/snapshot sau train"
import json, os, sys
sys.path.insert(0, '.')
from src.utils.phanD_snapshots import parse_schedule
work, sd, sched = sys.argv[1:]
idx = json.load(open(os.path.join(work, 'checkpoint_index.json')))
for k in ('best_miou', 'best_bfscore', 'best_bareland', 'final'):
    e = idx.get(k)
    assert e and os.path.exists(os.path.join(work, e['file'])), f'thieu checkpoint {k}'
    print(f"  [OK]    {k:14s} {e['file']:22s} iter {e['iter']}")
assert idx['final']['iter'] == 40000, f"final o iter {idx['final']['iter']} (early stopping?) — khong du 40000"
want = parse_schedule(sched)
have = [t for t in want if os.path.exists(os.path.join(sd, f'iter_{t}.pth'))]
miss = sorted(set(want) - set(have))
print(f"  [OK]    snapshot {len(have)}/{len(want)}" + (f" — THIEU {miss}" if miss else ''))
assert not miss and 36000 in have
EOF
  mkdir -p "$PD/checkpoints"
  python - "$WORK" "$PD/checkpoints" <<'EOF'
import json, os, shutil, sys
work, dst = sys.argv[1:]
idx = json.load(open(os.path.join(work, 'checkpoint_index.json')))
for k in ('best_miou', 'best_bfscore', 'best_bareland', 'final'):
    shutil.copy2(os.path.join(work, idx[k]['file']), os.path.join(dst, idx[k]['file']))
shutil.copy2(os.path.join(work, 'checkpoint_index.json'), dst)
EOF
  for f in train_log.txt benchmark_results.csv benchmark_results_ext.csv summary.txt final_summary.txt checkpoint_index.json; do
    [[ -f "$WORK/$f" ]] && cp "$WORK/$f" "$PD/logs/"
  done
  ok "4 checkpoint -> $PD/checkpoints ; log + CSV -> $PD/logs"
fi

# ══════════════════════════════════ post ═════════════════════════════════════
if has post; then
  step "Sau phien 1: D6 (nap dung snapshot 36k, val 384 anh) + sanity §5"
  [[ -f "$PD/snapshots/iter_36000.pth" ]] || die "khong co snapshots/iter_36000.pth"
  BENCH="$WORK/benchmark_results.csv"
  EXP=$(python - "$BENCH" <<'EOF'
import csv, sys
for r in csv.DictReader(open(sys.argv[1])):
    if int(r['iter']) == 36000:
        print(r.get('miou9') or r.get('mIoU'))
EOF
)
  [[ -n "$EXP" ]] || die "khong thay vong validation 36000 trong $BENCH"
  python Tools/grad_conflict_probe.py phanD-gates --gate d6 --snapshot "$PD/snapshots/iter_36000.pth" \
      --data-root "$DATA" --expected-miou "$EXP" --output-dir "$PD/post" --device cuda:0 --no-fail-exit \
      --json-out "$PD/post/D6.json" > "$PD/logs/d6.log" 2>&1 || { tail -30 "$PD/logs/d6.log"; die "D6 khong chay duoc"; }
  python Tools/phanD_dryrun.py post --bench-csv "$BENCH" --ext-csv "$WORK/benchmark_results_ext.csv" \
      --d6-json "$PD/post/D6.json" --out "$PD/post/post.json"
  assemble "$PD/post/post.json" > /dev/null || true
fi

# ══════════════════════════════════ measure (Phien 2) ════════════════════════
if has measure; then
  NS=$(ls "$PD/snapshots"/iter_*.pth 2>/dev/null | wc -l)
  [[ "$NS" -gt 0 ]] || die "khong co snapshot trong $PD/snapshots"
  step "PHIEN 2 — do $NS snapshot x 32 anh tren 2 GPU song song"
  t=$(date +%s)
  pids=()
  for k in 0 1; do
    python Tools/grad_conflict_probe.py phanD-measure --snapshot-dir "$PD/snapshots" --shard "$k/2" \
        --data-root "$DATA" --image-list "$PD/probe_ids.txt" --output-dir "$PD" --device "cuda:$k" \
        --scratch-dir "$SCRATCH/m$k" > "$PD/logs/measure_cuda$k.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [[ $fail -eq 0 ]] || { tail -20 "$PD"/logs/measure_cuda*.log; die "do snapshot that bai — xem $PD/logs/measure_cuda*.log"; }
  ND=$(ls "$PD/done"/iter_*.json 2>/dev/null | wc -l)
  [[ "$ND" -eq "$NS" ]] || die "moi do $ND/$NS snapshot"
  ok "do xong $ND snapshot sau $(( ($(date +%s) - t) / 60 )) phut"
fi

# ══════════════════════════════════ summarize ════════════════════════════════
if has summarize; then
  step "Tong hop quy dao + bang 6.3"
  python Tools/phanD_traj.py --output-dir "$PD" --gates-json "$PD/gates_dryrun.json" || die "tong hop that bai"
  echo; sed -n '1,30p' "$PD/summary_phanD.md"
fi

echo
VERD=$(python -c "import json; d=json.load(open('$PD/summary_phanD.json')); print(d['decision']['verdict'], '| dien giai duoc:', d.get('interpretable'))" 2>/dev/null || echo "(chua tong hop)")
echo "==> PHAN D: XONG sau $(( ($(date +%s) - T0) / 60 )) phut. Verdict: $VERD"
echo "    Ket qua: Output -> Boundary_Loss_Solution/output/phanD/summary_phanD.md (+ gates_dryrun.json, traj_param.csv, figures/)"
exit 0
