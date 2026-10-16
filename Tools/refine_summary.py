"""Tools/refine_summary.py — summary_P0_R0.md (docs/S4-spec-P0-R0-ban-giao-claude-code.md §5, §6):
gop cong P0 (gates_p0.json, p0_compare_csv.csv) va R0 (gates_r0.json, crf_selected.json, crf_val.csv).
Chi doc file da co — thieu phan nao thi ghi "chua chay".

  python Tools/refine_summary.py --root output/refine
"""

import argparse
import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import get_git_commit, read_json  # noqa: E402


def _load(p):
    return read_json(p) if os.path.exists(p) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='output/refine')
    args = ap.parse_args(argv)
    root = args.root
    p0, r0 = os.path.join(root, 'p0_rerun'), os.path.join(root, 'r0_crf')
    gp0, gr0 = _load(os.path.join(p0, 'gates_p0.json')), _load(os.path.join(r0, 'gates_r0.json'))
    sel = _load(os.path.join(r0, 'crf_selected.json'))
    dry = bool((gp0 or {}).get('dryrun') or (gr0 or {}).get('dryrun'))

    L = ['# Tom tat P0 + R0 (S4)', '',
         f"Tao {datetime.datetime.now().isoformat(timespec='seconds')} · git {get_git_commit()}", '']
    if dry:
        L += ['> **DRY-RUN** — chi kiem duong chay, so lieu KHONG dung duoc.', '']

    L += ['## P0 — JSON hau ky run Static s19 chay lai', '']
    if gp0 is None:
        L += ['_Chua chay (khong co p0_rerun/gates_p0.json)._', '']
    else:
        L += [f"Cong tong: **{gp0['overall']}**", '', '| Cong | Trang thai | Chi tiet |', '|---|---|---|']
        L += [f"| {g['gate']} | **{g['status']}** | {g.get('detail', '')} |" for g in gp0['gates']]
        cp = os.path.join(p0, 'p0_compare_csv.csv')
        if os.path.exists(cp):
            c = pd.read_csv(cp)
            L += ['', '§2.4 — lech tuyet doi lon nhat so voi CSV val trong train (chi tiet: `p0_rerun/p0_compare_csv.md`):', '',
                  '| Nhan | Chi so lech nhat | |Δ| |', '|---|---|---|']
            for lb, g in c.dropna(subset=['abs_diff']).groupby('label'):
                r = g.loc[g['abs_diff'].idxmax()]
                L.append(f"| {lb} | {r['metric']} | {r['abs_diff']:.5f} |")
        L.append('')

    L += ['## R0 — P-CRF', '']
    if gr0 is None:
        L += ['_Chua co bao cao (khong co r0_crf/gates_r0.json)._', '']
    else:
        rule = gr0['rule_36']
        crf = gr0.get('crf_backend', {})
        L += [f"> {gr0['rule_note']}", '',
              f"- Che do dau vao: **{gr0['input_mode']}**; pydensecrf {crf.get('version')} @ {crf.get('commit')}",
              f"- Cong tong: **{gr0['overall']}**", '',
              '| Cong | Trang thai | Chi tiet |', '|---|---|---|']
        L += [f"| {g['gate']} | **{g['status']}** | {g.get('detail', '')} |" for g in gr0['gates']]
        from Tools.r0_pcrf import CODE_TEXT, _f
        L += ['', f"**Ma §3.6 (mo ta): {CODE_TEXT[rule['code']]}** — do nhay BIoU 9 lop: "
                  f"{CODE_TEXT[gr0['rule_36_sensitivity_biou9']['code']]}", '']
        if rule.get('per_seed'):
            L += ['| Seed | Δ p→g | Δ BIoU8@2 | Δ g→p | Δ BF-R | (i) (ii) (iii) (iv) | Ma seed |',
                  '|---|---|---|---|---|---|---|']
            for lb, v in rule['per_seed'].items():
                c = v['cond']
                L.append(f"| {lb} | {_f(v['d_asd_pred_to_gt'])} | {_f(v['d_biou_d2'], 5)} | {_f(v['d_asd_gt_to_pred'])} | "
                         f"{_f(v['d_bf_recall'])} | {' '.join('✓' if c[k] else '✗' for k in ('i', 'ii', 'iii', 'iv'))} | "
                         f"{CODE_TEXT[v['seed_code']]} |")
        if sel:
            L += ['', 'To hop chon tren 32 anh train: ' +
                  ', '.join(f"{lb} `{s['name']}`" for lb, s in sel['selected'].items())]
        if gr0.get('eta_full_run'):
            e = gr0['eta_full_run']
            L += ['', f"Uoc tinh chay that (tu dry-run): luoi ~{e['grid_min']:.0f} ph, val ~{e['val_min']:.0f} ph."]
        L += ['', 'Chi tiet: `r0_crf/crf_val.md` (bang §3.6, moi chi so, mo ta §3.8), `r0_crf/crf_val.csv`, '
                  '`r0_crf/crf_selected.json`, `r0_crf/run_log.txt`.', '']

    L += ['## File can gui (§6)', '',
          '`summary_P0_R0.md`, `p0_rerun/gates_p0.json`, `r0_crf/gates_r0.json`, `r0_crf/crf_val.md`, '
          '`r0_crf/crf_selected.json`, `r0_crf/run_log.txt` (+ `p0_rerun/run_log.txt`).', '']
    path = os.path.join(root, 'summary_P0_R0.md')
    os.makedirs(root, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"Da ghi {path}")


if __name__ == '__main__':
    main()
