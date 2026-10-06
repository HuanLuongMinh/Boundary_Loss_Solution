"""Tools/link_holes_conflict.py — Phan C cua
docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md (mo ta, chay khi Phan B
khong ra "khong do loss").

Tren Static s19@36k: voi cac thanh phan GT NHO (8-lien thong theo lop, < 4096 px
full-res), so cosW (BCE, Aff) va (Region, Aff) BEN TRONG thanh phan co lo gia
vs thanh phan khong co lo gia CUNG LOP. Bootstrap theo thanh phan (10.000 lan,
rng 19, lay mau lai trong tung o lop x nhom). Neu ben co lo gia am hon ro (CI
chenh lech loai tru 0, chenh lech < 0) -> ung ho "loi 5 va xung dot cung mot goc".

Dau vao (khong tinh lai gi):
  <grad-dir>/per_component_small_<label>.csv   (Tools/grad_conflict_probe.py measure)
  <holes-dir>/holes_<label>.csv                (Tools/spurious_holes.py count)
  <holes-dir>/pairs_bootstrap.json             (Tools/spurious_holes.py compare — cong chay)
comp_id o hai file la CUNG dinh danh (Tools/grad_conflict_common.gt_components tren GT 1024).

Dai luong chinh: chenh lech cosW (co lo gia − khong co lo gia) TRONG tung lop, gop
cac lop bang trung binh co trong so so thanh phan co lo gia (chi lop co ca hai nhom).

Output: <holes-dir>/part_c_link.csv, <holes-dir>/part_c_link.md, va muc "Phan C"
chen vao cuoi <holes-dir>/pairs_bootstrap.md (thay the neu da co).

Chay: python Tools/link_holes_conflict.py --grad-dir output/grad_conflict \\
          --holes-dir output/spurious_holes
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import SEED, read_json, write_json, get_git_commit

PAIRS = ('bce_aff', 'region_aff')
MARK_BEGIN = '<!-- PHAN C BEGIN -->'
MARK_END = '<!-- PHAN C END -->'


def cosw(dot, n1, n2):
    den = np.sqrt(n1 * n2)
    return np.where(den > 0, dot / np.where(den > 0, den, 1.0), np.nan)


def boot_cell_sums(rng, arr, n_boot, chunk=1000):
    """arr (k,3) [dot,n1,n2] -> (n_boot,3) tong sau khi lay mau lai co hoan lai k thanh phan."""
    k = len(arr)
    out = np.empty((n_boot, 3))
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        idx = rng.integers(0, k, size=(b, k))
        out[s:s + b] = arr[idx].sum(axis=1)
    return out


def analyse(comp, spurious_keys, n_boot):
    """comp: DataFrame per_component_small (1 pair). Tra ve list row dict."""
    comp = comp[comp['n_pos'] > 0].copy()
    comp['has_hole'] = [(i, c) in spurious_keys for i, c in zip(comp['image_id'], comp['comp_id'])]
    rng = np.random.default_rng(SEED)
    rows = []
    per_class = {}
    for cls in sorted(comp['class'].unique()):
        sub = comp[comp['class'] == cls]
        yes = sub[sub['has_hole']][['sum_dot', 'sum_n1', 'sum_n2']].to_numpy(dtype=np.float64)
        no = sub[~sub['has_hole']][['sum_dot', 'sum_n1', 'sum_n2']].to_numpy(dtype=np.float64)
        row = {'class': cls, 'n_yes': len(yes), 'n_no': len(no)}
        if len(yes) == 0 or len(no) == 0:
            row.update(cosw_yes=np.nan, cosw_no=np.nan, diff=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                       ci_excludes_0=False, note='thieu mot nhom')
            rows.append(row)
            continue
        cy = float(cosw(*yes.sum(0)))
        cn = float(cosw(*no.sum(0)))
        by = boot_cell_sums(rng, yes, n_boot)
        bn = boot_cell_sums(rng, no, n_boot)
        d = cosw(by[:, 0], by[:, 1], by[:, 2]) - cosw(bn[:, 0], bn[:, 1], bn[:, 2])
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        row.update(cosw_yes=cy, cosw_no=cn, diff=cy - cn, ci_lo=float(lo), ci_hi=float(hi),
                   ci_excludes_0=bool(lo > 0 or hi < 0), note='')
        rows.append(row)
        per_class[cls] = (len(yes), cy - cn, d)
    if per_class:
        w = np.array([v[0] for v in per_class.values()], dtype=np.float64)
        w = w / w.sum()
        obs = float(sum(wi * v[1] for wi, v in zip(w, per_class.values())))
        boot = sum(wi * v[2] for wi, v in zip(w, per_class.values()))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append({'class': 'gop (trong so n_yes)', 'n_yes': int(sum(v[0] for v in per_class.values())),
                     'n_no': int(sum(r['n_no'] for r in rows if r['class'] in per_class)),
                     'cosw_yes': np.nan, 'cosw_no': np.nan, 'diff': obs, 'ci_lo': float(lo), 'ci_hi': float(hi),
                     'ci_excludes_0': bool(lo > 0 or hi < 0), 'note': f'{len(per_class)} lop co du hai nhom'})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grad-dir', default='output/grad_conflict')
    ap.add_argument('--holes-dir', default='output/spurious_holes')
    ap.add_argument('--label', default='static_s19_36k')
    ap.add_argument('--a-min', type=int, default=64)
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--force', action='store_true', help='Chay du Phan B ket luan "khong do loss"')
    args = ap.parse_args(argv)

    pb = os.path.join(args.holes_dir, 'pairs_bootstrap.json')
    if os.path.exists(pb):
        s = read_json(pb)
        if not s.get('run_part_c') and not args.force:
            print(f"Phan B: '{s['verdict_54']['text']}' -> Phan C KHONG chay (muc 6). Dung --force neu van muon chay.")
            return
    else:
        print(f"CANH BAO: chua co {pb} (chua chay 'spurious_holes.py compare') — van tiep tuc.")

    comp_path = os.path.join(args.grad_dir, f'per_component_small_{args.label}.csv')
    holes_path = os.path.join(args.holes_dir, f'holes_{args.label}.csv')
    for p in (comp_path, holes_path):
        if not os.path.exists(p):
            raise SystemExit(f"Thieu {p}")
    comp = pd.read_csv(comp_path)
    holes = pd.read_csv(holes_path)
    sp = holes[(holes['a_min'] == args.a_min) & (holes['is_spurious'] == 1) & (holes['gt_comp_id'] > 0)]
    keys = set(zip(sp['image_id'].astype(str), sp['gt_comp_id'].astype(int)))
    comp['image_id'] = comp['image_id'].astype(str)

    out = []
    for pn in PAIRS:
        sub = comp[comp['pair'] == pn]
        if sub.empty:
            continue
        for r in analyse(sub, keys, args.n_boot):
            out.append({'pair': pn, **r})
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(args.holes_dir, 'part_c_link.csv'), index=False)

    verdict = {}
    for pn in PAIRS:
        g = df[(df['pair'] == pn) & (df['class'].astype(str).str.startswith('gop'))]
        if g.empty:
            verdict[pn] = 'khong du du lieu'
        else:
            r = g.iloc[0]
            verdict[pn] = ('ben co lo gia AM HON ro' if (r['ci_excludes_0'] and r['diff'] < 0) else
                           'ben co lo gia DUONG HON ro' if r['ci_excludes_0'] else 'khong khac biet ro')
    support = any(v == 'ben co lo gia AM HON ro' for v in verdict.values())
    concl = ('Ung ho "loi 5 va xung dot cung mot goc" -> gop loi 5 vao truc chinh cua Huong B'
             if support else 'Khong ung ho "cung mot goc"')

    L = [MARK_BEGIN, '## Phan C — Noi lo gia voi xung dot gradient (mo ta)', '',
         f"Checkpoint {args.label}, A_min = {args.a_min}, thanh phan GT nho (< 4096 px) co >= 1 vi tri stride-4. "
         f"Chenh lech = cosW(co lo gia) − cosW(khong co lo gia), cung lop; gop = trung binh co trong so so thanh phan "
         f"co lo gia. Bootstrap theo thanh phan {args.n_boot} lan, rng {SEED}.", '',
         f"**{concl}** — " + '; '.join(f'{k}: {v}' for k, v in verdict.items()), '',
         '| Cap | Lop | n co lo gia | n khong | cosW co lo gia | cosW khong | Chenh lech | CI 95% | CI loai 0 |',
         '|---|---|---|---|---|---|---|---|---|']

    def f(x):
        return '—' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.4f}'

    for _, r in df.iterrows():
        L.append(f"| {r['pair']} | {r['class']} | {r['n_yes']} | {r['n_no']} | {f(r['cosw_yes'])} | "
                 f"{f(r['cosw_no'])} | {f(r['diff'])} | [{f(r['ci_lo'])}, {f(r['ci_hi'])}] | "
                 f"{'co' if r['ci_excludes_0'] else 'khong'} |")
    L += ['', MARK_END]
    section = '\n'.join(L)
    with open(os.path.join(args.holes_dir, 'part_c_link.md'), 'w', encoding='utf-8') as fh:
        fh.write(section.replace(MARK_BEGIN + '\n', '').replace('\n' + MARK_END, '') + '\n')
    md = os.path.join(args.holes_dir, 'pairs_bootstrap.md')
    if os.path.exists(md):
        txt = open(md, encoding='utf-8').read()
        if MARK_BEGIN in txt:
            txt = txt[:txt.index(MARK_BEGIN)] + txt[txt.index(MARK_END) + len(MARK_END):]
        with open(md, 'w', encoding='utf-8') as fh:
            fh.write(txt.rstrip('\n') + '\n\n' + section + '\n')
    write_json({'label': args.label, 'a_min': args.a_min, 'verdict_by_pair': verdict, 'conclusion': concl,
                'supports_common_root': support,
                'created_at': datetime.datetime.now().isoformat(timespec='seconds'), 'git_commit': get_git_commit()},
               os.path.join(args.holes_dir, 'part_c_link.json'))
    print(concl)
    print(f"Da ghi part_c_link.csv/.md/.json (+ muc Phan C trong pairs_bootstrap.md) vao {args.holes_dir}")


if __name__ == '__main__':
    main()
