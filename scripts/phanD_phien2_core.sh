#!/usr/bin/env bash
# scripts/phanD_phien2_core.sh — Ham dung chung cho PHIEN 2 cua Phan D (do snapshot + tong hop).
# Duoc `source` boi scripts/kaggle_phanD.sh (stage measure/summarize) va scripts/kaggle_phanD_phien2.sh
# (chay Phien 2 doc lap). Khong chay truc tiep.
#
# Can bien: PD (thu muc output Phan D), DATA (dataset, co images/train), SNAP_DIR (thu muc iter_<t>.pth),
#           NGPU (so GPU), SCRATCH (thu muc memmap tam). Dung step/ok/die cua kaggle_common.sh.

# Dung luong trong (GB) cua thu muc chua $1
free_gb() { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4 / 1048576}'; }

# In N dong cuoi cua TUNG file log (khong dung `tail -N f1 f2` — cu phap cu bi tu choi khi nhieu file)
show_logs() {
  local n="$1"; shift
  for f in "$@"; do
    [[ -f "$f" ]] || continue
    echo "  ---------- $f (${n} dong cuoi) ----------"
    tail -n "$n" "$f" | tr '\r' '\n' | grep -v '^\s*$' | tail -n "$n" | sed 's/^/  | /'
  done
}

phanD_measure() {
  local ns nd nshard t fail k pids=() logs=()
  ns=$(ls "$SNAP_DIR"/iter_*.pth 2>/dev/null | wc -l)
  [[ "$ns" -gt 0 ]] || die "khong co iter_<t>.pth trong $SNAP_DIR"
  [[ -f "$PD/probe_ids.txt" && $(wc -l < "$PD/probe_ids.txt") -eq 32 ]] || die "thieu/hong $PD/probe_ids.txt (32 dong)"
  nshard=$(( NGPU >= 2 ? 2 : 1 ))

  step "PHIEN 2 — do $ns snapshot x 32 anh tren $nshard GPU (snapshot: $SNAP_DIR)"
  mkdir -p "$SCRATCH" "$PD/logs"
  local need=$(( 5 * nshard )) have
  have=$(free_gb "$SCRATCH")
  echo "  dia trong: scratch $SCRATCH = ${have} GB (can ~${need} GB: ~4.6 GB memmap/GPU) | output $PD = $(free_gb "$PD") GB"
  [[ -n "$have" && "$have" -ge "$need" ]] || die "khong du dia cho memmap ($have GB < $need GB) — dat SCRATCH=<thu muc khac>"
  free -g 2>/dev/null | sed -n '1,2p' | sed 's/^/  /' || true

  t=$(date +%s)
  for (( k = 0; k < nshard; k++ )); do
    logs+=("$PD/logs/measure_cuda$k.log")
    python Tools/grad_conflict_probe.py phanD-measure --snapshot-dir "$SNAP_DIR" --shard "$k/$nshard" \
        --data-root "$DATA" --image-list "$PD/probe_ids.txt" --output-dir "$PD" --device "cuda:$k" \
        --gt-cache "$OUT/_cache/gt1024_train_s$k" --scratch-dir "$SCRATCH/m$k" > "$PD/logs/measure_cuda$k.log" 2>&1 &
    pids+=($!)
  done
  fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  nd=$(ls "$PD/done"/iter_*.json 2>/dev/null | wc -l)
  if [[ $fail -ne 0 || "$nd" -lt "$ns" ]]; then
    show_logs 40 "${logs[@]}"
    [[ -f "$PD/measure_errors.txt" ]] && { echo "  ---------- $PD/measure_errors.txt ----------"; sed 's/^/  | /' "$PD/measure_errors.txt" | tail -n 60; }
    die "do snapshot chua xong: $nd/$ns (xem log o tren, $PD/logs/measure_cuda*.log). Snapshot da do xong DUOC GIU — chay lai Phien 2 se do tiep phan con thieu"
  fi
  ok "do xong $nd/$ns snapshot sau $(( ($(date +%s) - t) / 60 )) phut"
}

phanD_summarize() {
  step "Tong hop quy dao + bang 6.3"
  python Tools/phanD_traj.py --output-dir "$PD" --gates-json "$PD/gates_dryrun.json" || die "tong hop that bai"
  echo; sed -n '1,30p' "$PD/summary_phanD.md"
}
