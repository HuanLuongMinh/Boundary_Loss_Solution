"""Tools/phanD_dryrun.py — Cong dry-run + kiem sau train cua Phan D
(docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md §4, §5; cong D0 do nguoi dung yeu cau them).

Subcommand:
  script-diff   D0 (1): phan tinh tu dong (Tools/phanD_d0_static.py) -> d0_script_diff.md + JSON, kem `diff -u`.
  compare       D0 (2) / D1: so loss tung iter cua 3 run 500 iter (A = OFF-a, B = OFF-b, C = run can kiem).
                PASS khi max|C - A| <= max|B - A|; neu A == B (trung bit) thi yeu cau C == A.
                D0 them: --iter1-exact (loss iter 1 cua A va C trung bit), --static-json (phan tinh PASS).
  timing        D5: s/iter (loss log run ON) + s/snapshot (D3) -> du bao phien 1/2; > 10h -> lich gian.
  post          Sanity §5 + D6 sau phien 1 -> muc `post` (+ co interpretable).
  assemble      Gop cac file cong rieng le -> gates_dryrun.json.

Moi subcommand ghi JSON; ma thoat != 0 khi cong FAIL (tru `post` — sanity FAIL chi gan co, khong dung).
"""

import argparse
import csv
import datetime
import difflib
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOSS_COLS = ('l_total', 'l_region', 'l_bce', 'l_affinity')
POS_WEIGHT_REF = 11.0014
POS_WEIGHT_TOL = 5e-4
MIOU36_REF, MIOU36_TOL = 0.6540486, 0.0017       # §5
BUDGET_H = 10.0                                    # D5
RUN3_TRAIN_H = 3 + 22 / 60                         # Run 3 goc: 40k iter + 10 vong val trong 03h22m
N_SNAP_DEFAULT, N_SNAP_SPARSE = 35, 20

# D0 phan TINH: Tools/phanD_d0_static.py kiem tu dong (AST + tinh loss that) tung muc tren CHINH code se chay.
D0_SCOPE = """Pham vi phan tinh (theo yeu cau D0): (1) lich LR + optimizer + AMP; (2) thu tu khoi tao model / RNG;
(3) sampler / worker seed; (4) nguon pos_weight va lambda; (5) lich validation / chon checkpoint.
Moi dong la mot phep kiem tu dong tren AST cua 2 file (bo docstring/log/I-O), tru 2 phep tinh that:
module loss khong co tham so/buffer, va l_total + 3 so hang trung bit tren cung dau vao."""


def read_loss_log(path):
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return {int(r['iter']): {k: float(r[k]) for k in LOSS_COLS} for r in rows}, rows


def max_abs_diff(a, b):
    its = sorted(set(a) & set(b))
    if not its or set(a) != set(b):
        return float('inf'), its
    m = 0.0
    for t in its:
        for k in LOSS_COLS:
            m = max(m, abs(a[t][k] - b[t][k]))
    return m, its


def pos_weight_from_log(path):
    """Dong '--stop-at-iter: dung sau iter N (pos_weight=X)' do train script ghi (6 chu so)."""
    if not path or not os.path.exists(path):
        return None
    val = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            m = re.search(r'--stop-at-iter: .*pos_weight=([0-9.]+)', ln)
            if m:
                val = float(m.group(1))
    return val


def write_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def now():
    return datetime.datetime.now().isoformat(timespec='seconds')


# ═══════════════════════════════════ subcommand ═══════════════════════════════

def cmd_script_diff(args):
    from Tools.phanD_d0_static import run_static_checks, to_markdown
    static = run_static_checks(args.orig, args.weighted)
    static.update({'gate': 'D0-static', 'created_at': now()})
    write_json(static, args.static_json)
    a = open(args.orig, encoding='utf-8').read().splitlines()
    b = open(args.weighted, encoding='utf-8').read().splitlines()
    diff = '\n'.join(difflib.unified_diff(a, b, fromfile=args.orig, tofile=args.weighted, lineterm=''))
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('# D0 — Khac biet giua 2 script train (Phan D)\n\n')
        f.write('## 1. Phan tich tinh theo muc\n\n' + D0_SCOPE + '\n\n')
        f.write(to_markdown(static) + '\n')
        f.write('## 2. Kiem thuc nghiem 500 iter (seed 19, 2 GPU)\n\n'
                'Dieu kien: (a) loss iter 1 cua W-a va O-a TRUNG BIT (4 so hang); '
                '(b) max|W-a - O-a| <= max|W-b - W-a| tren 500 iter (W-a == W-b thi yeu cau W-a == O-a); '
                '(c) pos_weight = 11.0014 +- 5e-4. Ket qua: `gates_dryrun.json` -> cong `D0`.\n\n'
                'D0 PASS khi phan tinh PASS **va** (a), (b), (c) deu dat.\n\n')
        f.write('## 3. `diff -u` day du (ban dang chay, gom ca cac co Phan D mac dinh TAT)\n\n```diff\n')
        f.write(diff + '\n```\n')
    print(f"  D0 phan tinh: {static['status']} "
          f"({', '.join(i['key'] + '=' + i['status'] for i in static['items'])}) -> {args.out}")


def cmd_compare(args):
    A, _ = read_loss_log(args.a)
    B, _ = read_loss_log(args.b)
    C, _ = read_loss_log(args.c)
    noise, its_ab = max_abs_diff(A, B)
    dev, its_ac = max_abs_diff(A, C)
    identical_ab = noise == 0.0
    if identical_ab:
        ok = dev == 0.0
        rule = 'A == B (trung bit) -> yeu cau C == A'
    else:
        ok = dev <= noise
        rule = 'max|C - A| <= max|B - A|'
    res = {'gate': args.gate, 'name': args.name, 'status': 'PASS' if ok else 'FAIL', 'rule': rule,
           'value': {'max_abs_B_minus_A': noise, 'max_abs_C_minus_A': dev, 'n_iter': len(its_ac),
                     'A': args.a, 'B': args.b, 'C': args.c, 'cols': list(LOSS_COLS)},
           'created_at': now()}
    if args.iter1_exact:
        a1, b1, c1 = A.get(1), B.get(1), C.get(1)
        iter1_ok = a1 is not None and c1 is not None and all(a1[k] == c1[k] for k in LOSS_COLS)
        res['value']['iter1'] = {'A': a1, 'B': b1, 'C': c1, 'A_equals_C': iter1_ok,
                                 'A_equals_B': (a1 is not None and b1 is not None and
                                                all(a1[k] == b1[k] for k in LOSS_COLS))}
        res['rule'] += '; loss iter 1 cua A va C trung bit (4 so hang)'
        if not iter1_ok:
            res['status'] = 'FAIL'
    if args.static_json:
        st = json.load(open(args.static_json, encoding='utf-8'))
        res['value']['static'] = {'status': st['status'],
                                  'items': {i['key']: i['status'] for i in st['items']}}
        res['rule'] += '; phan tinh (Tools/phanD_d0_static.py) PASS'
        if st['status'] != 'PASS':
            res['status'] = 'FAIL'
    # D0 la cong CHON script (FAIL -> chuyen sang script goc), khong chan; pos_weight sai thi luon chan.
    res['blocking'] = not args.selection
    if args.pos_weight_log:
        pw = pos_weight_from_log(args.pos_weight_log)
        pw_ok = pw is not None and abs(pw - POS_WEIGHT_REF) <= POS_WEIGHT_TOL
        res['value'].update({'pos_weight': pw, 'pos_weight_ref': POS_WEIGHT_REF, 'pos_weight_tol': POS_WEIGHT_TOL,
                             'pos_weight_ok': pw_ok, 'pos_weight_log': args.pos_weight_log})
        if not pw_ok:
            res['status'] = 'FAIL'
            res['blocking'] = True
            res['note'] = f'pos_weight {pw} lech {POS_WEIGHT_REF} qua {POS_WEIGHT_TOL} — nghi splits/du lieu da doi'
    write_json(res, args.out)
    print(f"  cong {args.gate}: {res['status']}  max|B-A|={noise:.3g}  max|C-A|={dev:.3g}"
          + (f"  pos_weight={res['value'].get('pos_weight')}" if args.pos_weight_log else ''))
    sys.exit(0 if res['status'] == 'PASS' else (3 if not res['blocking'] else 2))


def cmd_timing(args):
    _A, rows = read_loss_log(args.loss_log)
    w = [float(r['wall_s']) for r in rows]
    # bo 50 iter dau (khoi dong cuDNN/DataLoader)
    s_iter = (w[-1] - w[50]) / (len(w) - 51) if len(w) > 60 else (w[-1] / max(len(w), 1))
    d3 = json.load(open(args.d3_json, encoding='utf-8'))
    s_snap = float(d3['value']['s_per_snapshot'])
    n_gpu = max(args.n_gpu, 1)
    val_h = max(RUN3_TRAIN_H - 40000 * s_iter / 3600, 0.0)    # phan validation suy tu Run 3
    p1 = 40000 * s_iter / 3600 + val_h
    p2 = N_SNAP_DEFAULT * s_snap / 3600 / n_gpu
    sparse = p1 > BUDGET_H or p2 > BUDGET_H
    p2_sparse = N_SNAP_SPARSE * s_snap / 3600 / n_gpu
    res = {'gate': 'D5', 'name': 'thoi gian', 'status': 'PASS' if not sparse or p2_sparse <= BUDGET_H else 'FAIL',
           'value': {'s_per_iter': s_iter, 's_per_snapshot': s_snap, 'n_gpu_measure': n_gpu,
                     'val_hours_assumed_from_run3': val_h, 'phien1_hours': p1, 'phien2_hours': p2,
                     'schedule': 'sparse' if sparse else 'default',
                     'phien2_hours_if_sparse': p2_sparse},
           'threshold': f'phien 1 <= {BUDGET_H}h; phien 2 <= {BUDGET_H}h; vuot -> gian lich theo D5',
           'note': 'GIAN lich theo D5 (ghi lai)' if sparse else 'giu lich 35 moc', 'created_at': now()}
    write_json(res, args.out)
    print(f"  D5: {s_iter:.3f} s/iter, {s_snap:.0f} s/snapshot -> phien1 ~{p1:.2f}h, phien2 ~{p2:.2f}h "
          f"({n_gpu} GPU) -> lich {res['value']['schedule']}")
    with open(args.schedule_out, 'w') as f:
        f.write(res['value']['schedule'] + '\n')
    sys.exit(0 if res['status'] == 'PASS' else 2)


def _csv_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _miou_col(row):
    for k in ('miou9', 'mIoU'):
        if k in row and row[k] not in ('', None):
            return float(row[k])
    return None


def cmd_post(args):
    rows = _csv_rows(args.bench_csv)
    ext = _csv_rows(args.ext_csv) if args.ext_csv and os.path.exists(args.ext_csv) else []
    by_iter = {int(r['iter']): r for r in rows}
    ext_by = {int(r['iter']): r for r in ext}
    r36 = by_iter.get(36000)
    m36 = _miou_col(r36) if r36 else None
    ok5 = m36 is not None and abs(m36 - MIOU36_REF) <= MIOU36_TOL
    sanity = {'status': 'PASS' if ok5 else 'FAIL', 'miou9_36k': m36, 'ref': MIOU36_REF, 'tol': MIOU36_TOL,
              'abs_diff': (abs(m36 - MIOU36_REF) if m36 is not None else None),
              'note': 'Truot -> KHONG dien giai quy dao (S2 §5); script van do theo lua chon nguoi dung.'}
    r40 = by_iter.get(40000)
    bf = {}
    src36 = ext_by.get(36000) or r36 or {}
    for k in ('bf_score', 'bf_precision', 'bf_recall'):
        if src36.get(k) not in (None, ''):
            bf[k] = float(src36[k])
    d6 = json.load(open(args.d6_json, encoding='utf-8')) if args.d6_json and os.path.exists(args.d6_json) else \
        {'status': 'MISSING'}
    interp = ok5 and d6.get('status') == 'PASS'
    post = {'sanity5': sanity, 'd6': {'status': d6.get('status'), 'detail': d6.get('value')},
            'miou9_40k': _miou_col(r40) if r40 else None, 'bf_36k': bf,
            'bf_36k_note': 'so voi run goc Static s19 @36k (summary_static_s19.txt / Buoc 1)',
            'interpretable': bool(interp), 'created_at': now()}
    write_json(post, args.out)
    print(f"  Sanity §5: {sanity['status']} (mIoU-9@36k={m36}); D6: {d6.get('status')}; "
          f"dien giai duoc: {'CO' if interp else 'KHONG (van do tiep, gan co)'}")


def cmd_assemble(args):
    meta = json.load(open(args.meta, encoding='utf-8')) if args.meta and os.path.exists(args.meta) else {}
    gates = []
    for p in sorted(glob.glob(os.path.join(args.gate_dir, '*.json'))):
        try:
            g = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        if isinstance(g, dict) and 'gate' in g:
            g['_file'] = os.path.basename(p)
            gates.append(g)
    order = ['R0', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5']
    gates.sort(key=lambda g: order.index(g['gate']) if g['gate'] in order else 99)
    post = json.load(open(args.post, encoding='utf-8')) if args.post and os.path.exists(args.post) else None
    need = [g for g in order if g not in {x['gate'] for x in gates}]
    blocking_fail = [g['gate'] for g in gates if g['status'] != 'PASS' and g.get('blocking', True)]
    overall = 'PASS' if not blocking_fail and not need else 'FAIL'
    out = {'spec': 'docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md §4', 'overall_dryrun': overall,
           'blocking_fail': blocking_fail, 'missing_gates': need, 'meta': meta, 'gates': gates,
           'created_at': now()}
    if post is not None:
        out['post'] = post
    write_json(out, args.out)
    for g in gates:
        print(f"  {g['gate']:3s} {g['status']:5s} {g.get('name', '')}")
    if need:
        print(f"  thieu cong: {need}")
    print(f"  DRY-RUN TONG: {overall} -> {args.out}")


def cmd_regress(args):
    """R0: moi CSV cua Phan A do bang code cu (commit Buoc 1) va code moi phai trung (rtol 1e-6)."""
    import numpy as np
    import pandas as pd
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.old, '*.csv')))
    detail, ok = {}, bool(names)
    for nm in names:
        pn = os.path.join(args.new, nm)
        if not os.path.exists(pn):
            detail[nm] = 'THIEU o ban moi'
            ok = False
            continue
        a, b = pd.read_csv(os.path.join(args.old, nm)), pd.read_csv(pn)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            detail[nm] = f'khac cot/so hang ({a.shape} vs {b.shape})'
            ok = False
            continue
        worst = 0.0
        for c in a.columns:
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
                # sai so chia cho bien do cua CA COT (nhu cong 5: |a-b|max / |a|max) — tranh phong dai o gia tri ~0
                x, y = a[c].to_numpy(np.float64), b[c].to_numpy(np.float64)
                if (np.isnan(x) != np.isnan(y)).any():
                    worst = float('inf')
                    continue
                m = ~np.isnan(x)
                if m.any():
                    scale = max(float(np.abs(x[m]).max()), 1e-300)
                    worst = max(worst, float(np.abs(x[m] - y[m]).max()) / scale)
            elif not (a[c].astype(str) == b[c].astype(str)).all():
                worst = float('inf')
        detail[nm] = worst
        ok = ok and worst <= args.rtol
    res = {'gate': 'R0', 'name': 'hoi quy probe: che do cu khong doi', 'status': 'PASS' if ok else 'FAIL',
           'threshold': f'|cu - moi|max / |cu|max <= {args.rtol} theo tung cot so; cot chu trung tuyet doi',
           'value': {'old': args.old, 'new': args.new, 'max_rel_by_file': detail}, 'created_at': now()}
    write_json(res, args.out)
    print(f"  R0: {res['status']} ({len(names)} file CSV) -> {args.out}")
    sys.exit(0 if ok else 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('regress')
    p.add_argument('--old', required=True)
    p.add_argument('--new', required=True)
    p.add_argument('--rtol', type=float, default=1e-6)
    p.add_argument('--out', required=True)
    p = sub.add_parser('script-diff')
    p.add_argument('--orig', default='src/train_static_boundary.py')
    p.add_argument('--weighted', default='src/train_static_boundary_weighted.py')
    p.add_argument('--out', required=True)
    p.add_argument('--static-json', required=True)
    p = sub.add_parser('compare')
    p.add_argument('--gate', required=True)
    p.add_argument('--name', default='')
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.add_argument('--c', required=True)
    p.add_argument('--pos-weight-log', default=None, help='train_log.txt cua run C de kiem pos_weight = 11.0014')
    p.add_argument('--selection', action='store_true', help='Cong chon (D0): FAIL thoat ma 3, khong chan')
    p.add_argument('--iter1-exact', action='store_true', help='D0: loss iter 1 cua A va C phai trung bit')
    p.add_argument('--static-json', default=None, help='D0: ket qua phan tinh (script-diff --static-json)')
    p.add_argument('--out', required=True)
    p = sub.add_parser('timing')
    p.add_argument('--loss-log', required=True)
    p.add_argument('--d3-json', required=True)
    p.add_argument('--n-gpu', type=int, default=2)
    p.add_argument('--out', required=True)
    p.add_argument('--schedule-out', required=True)
    p = sub.add_parser('post')
    p.add_argument('--bench-csv', required=True)
    p.add_argument('--ext-csv', default=None)
    p.add_argument('--d6-json', default=None)
    p.add_argument('--out', required=True)
    p = sub.add_parser('assemble')
    p.add_argument('--gate-dir', required=True)
    p.add_argument('--meta', default=None)
    p.add_argument('--post', default=None)
    p.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    {'script-diff': cmd_script_diff, 'compare': cmd_compare, 'timing': cmd_timing, 'post': cmd_post,
     'assemble': cmd_assemble, 'regress': cmd_regress}[args.cmd](args)


if __name__ == '__main__':
    main()
