"""Tools/phanD_traj.py — Phan D, tong hop quy dao (CPU, khong can GPU):
docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md §2.3 (thong ke), §6 (luat doc — CHOT TRUOC), §7 (output).

Dau vao (do `grad_conflict_probe.py phanD-measure` sinh): <od>/gram/iter_<t>.npz, <od>/feat/iter_<t>_*.csv.
Dau ra:
  traj_param.csv          t x cap: cos_all, ci_lo, ci_hi, null_lo, null_hi, ratio, cos_batch_1..4, tieu chi 6.1
  traj_feat_strata.csv    t x cap x truc x tang: sum_dot, sum_n1, sum_n2, n_pos, n_support, n_neg, cosW, CI, null
  windows.csv             moi cua so xung dot (>= 3 snapshot lien tiep, khong tinh t=0)
  summary_phanD.md/.json  sanity §5, bang 6.3, mo ta 6.4
  figures/traj_cos_param.png, figures/traj_ratio.png

Chay: python Tools/phanD_traj.py --output-dir output/phanD [--gates-json output/phanD/gates_dryrun.json]
"""

import argparse
import datetime
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import ALL_PAIRS, SEED, pair_name, write_json, read_json

LABEL = 'phanD_s19'
TAU_PARAM = 0.05
WINDOW_MIN = 3                 # §6.2
T_LATE = 36000                 # §6.3: cua so "ket thuc truoc 36000" / "keo toi >= 36000"
LAST5 = (32000, 34000, 36000, 38000, 40000)   # §6.4 do on dinh
STEP1_RATIOS = {'region_aff': 0.021, 'region_bce': 0.093, 'bce_aff': 0.239}   # Buoc 1 @36k (val)
RATIO_NAMES = {'region_aff': '‖G_aff‖/‖G_region‖', 'region_bce': '‖G_bce‖/‖G_region‖',
               'bce_aff': '‖G_aff‖/‖G_bce‖'}


# ═════════════════════════════ muc tham so (Gram) ═════════════════════════════

def gram_pair_blocks(G, terms, k1, k2):
    nt = len(terms)
    n = G.shape[0] // nt
    i1 = np.arange(n) * nt + list(terms).index(k1)
    i2 = np.arange(n) * nt + list(terms).index(k2)
    return G[np.ix_(i1, i1)], G[np.ix_(i2, i2)], G[np.ix_(i1, i2)]


def _quad(W, A, V=None):
    """Hang r: W[r] A V[r] (V mac dinh = W)."""
    V = W if V is None else V
    return np.einsum('ri,ij,rj->r', W, A, V)


def weighted_cos(W, G11, G22, G12, V=None):
    """cos(sum_i W_i a1_i, sum_j V_j a2_j) cho moi hang trong so."""
    V = W if V is None else V
    num = _quad(W, G12, V)
    den = np.sqrt(np.clip(_quad(W, G11), 0, None) * np.clip(_quad(V, G22), 0, None))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def boot_weights(n, n_boot, seed=SEED):
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    W = np.zeros((n_boot, n))
    np.add.at(W, (np.repeat(np.arange(n_boot), n), idx.ravel()), 1.0)
    return W


def half_splits(n, n_null, seed=SEED):
    """1.000 lan chia ngau nhien n anh thanh 2 nua roi nhau H1, H2 (rng 19)."""
    rng = np.random.default_rng(seed)
    U = np.zeros((n_null, n))
    V = np.zeros((n_null, n))
    for r in range(n_null):
        perm = rng.permutation(n)
        U[r, perm[:n // 2]] = 1.0
        V[r, perm[n // 2:]] = 1.0
    return U, V


def param_stats(G, terms, batches, k1, k2, W_boot, U, V):
    G11, G22, G12 = gram_pair_blocks(G, terms, k1, k2)
    n = G11.shape[0]
    one = np.ones((1, n))
    cos_all = float(weighted_cos(one, G11, G22, G12)[0])
    boot = weighted_cos(W_boot, G11, G22, G12)
    boot = boot[~np.isnan(boot)]
    null = weighted_cos(U, G11, G22, G12, V)
    null = null[~np.isnan(null)]
    s11, s22 = float(_quad(one, G11)[0]), float(_quad(one, G22)[0])
    ratio = math.sqrt(s22 / s11) if s11 > 0 else float('nan')
    B = np.zeros((len(batches), n))
    for b, members in enumerate(batches):
        B[b, list(members)] = 1.0
    cos_b = weighted_cos(B, G11, G22, G12)
    ci = np.percentile(boot, [2.5, 97.5]) if len(boot) else (np.nan, np.nan)
    nl = np.percentile(null, [2.5, 97.5]) if len(null) else (np.nan, np.nan)
    return {'cos_all': cos_all, 'ci_lo': float(ci[0]), 'ci_hi': float(ci[1]), 'null_lo': float(nl[0]),
            'null_hi': float(nl[1]), 'ratio': ratio, 'cos_batch': [float(c) for c in cos_b],
            'norm1_all': math.sqrt(max(s11, 0)), 'norm2_all': math.sqrt(max(s22, 0))}


# ═════════════════════════════ tieu chi / cua so / quyet dinh ═════════════════

def criteria(level, pair, axis, stratum, value, ci_lo, ci_hi, null_lo, null_hi, frac_neg, mag, cov, n_units, tau,
             n_pos=None):
    from Tools.grad_conflict_probe import _criteria
    return _criteria(LABEL, level, pair, axis, stratum, value, ci_lo, ci_hi, null_lo, null_hi, frac_neg, mag, cov,
                     n_units, tau, n_pos=n_pos)


def find_windows(df, min_len=WINDOW_MIN):
    """df: cot t, level, pair, axis, stratum, conflict (bool). Cua so = >= min_len snapshot LIEN TIEP
    (theo thu tu cac snapshot da do, t=0 bi loai) cung thoa 6.1 cho cung (muc, cap, truc, tang)."""
    ts = sorted(t for t in df['t'].unique() if t != 0)
    pos = {t: j for j, t in enumerate(ts)}
    out = []
    for key, sub in df[df['t'] != 0].groupby(['level', 'pair', 'axis', 'stratum'], sort=False):
        hit = sorted(pos[t] for t, c in zip(sub['t'], sub['conflict']) if bool(c))
        run = []
        for j in hit + [None]:
            if run and (j is None or j != run[-1] + 1):
                if len(run) >= min_len:
                    out.append({'level': key[0], 'pair': key[1], 'axis': key[2], 'stratum': key[3],
                                't_start': ts[run[0]], 't_end': ts[run[-1]], 'n_snapshots': len(run),
                                'snapshots': '|'.join(str(ts[x]) for x in run)})
                run = []
            if j is not None:
                run.append(j)
    return pd.DataFrame(out, columns=['level', 'pair', 'axis', 'stratum', 't_start', 't_end', 'n_snapshots',
                                      'snapshots'])


def decide(windows):
    """Bang 6.3 NGUYEN VAN. Thu tu D-G > D-RA > D-L; D-N khi khong co cua so nao.
    D-dai-dang (co cua so keo toi >= 36000) duoc bao rieng va GHI DE verdict ("dung dien giai") —
    cach doc cua code nay, ghi ro trong bao cao."""
    w = windows
    early = w[w['t_end'] < T_LATE] if len(w) else w
    late = w[w['t_end'] >= T_LATE] if len(w) else w

    def has(df, level, pair, axes=None):
        if not len(df):
            return False
        m = (df['level'] == level) & (df['pair'] == pair)
        if axes is not None:
            m &= df['axis'].isin(axes)
        return bool(m.any())

    lit = {
        'D-G': has(early, 'param', 'bce_aff'),
        'D-RA': has(early, 'param', 'region_aff'),
        'D-L': has(early, 'feat', 'bce_aff', ['dist', 'size']),
        'D-dai-dang': bool(len(late)),
        'D-N': not len(w),
    }
    literal = next((c for c in ('D-G', 'D-RA', 'D-L') if lit[c]), 'D-N' if lit['D-N'] else None)
    verdict = 'D-dai-dang' if lit['D-dai-dang'] else literal
    text = {
        'D-G': 'D-G — Xung dot (BCE, Aff) muc tham so co that nhung bi thoa hiep truoc 36000. Huong B song '
               '(khung "xung dot theo thoi gian"); ung vien PCGrad chi trong cua so. BAT BUOC lap quy dao seed 86 '
               'truoc khi viet spec phuong phap.',
        'D-RA': 'D-RA — Affinity can nhiem vu chinh o mot giai doan (Region, Aff muc tham so). Ung vien cong '
                'gradient kieu Du et al. trong cua so; lap seed 86 truoc.',
        'D-L': 'D-L — Xung dot (BCE, Aff) khu tru theo khong gian va thoi gian (muc dac trung, tang khoang cach/'
               'kich thuoc). Nhanh L gioi han trong cua so; lap seed 86 truoc.',
        'D-dai-dang': 'D-dai-dang — Co cua so keo toi >= 36000, mau thuan Buoc 1. DUNG DIEN GIAI: kiem khac biet '
                      'train vs val (tap do) truoc moi ket luan.',
        'D-N': 'D-N — Khong co cua so xung dot nao trong qua trinh train. Dung Huong B phan phuong phap (kich '
               'ban S3); quy dao vao phan phan tich bai 2.',
        None: 'Khong ma nao cua bang 6.3 dat dung dieu kien, nhung CO cua so (vd chi o cap region_bce hoac truc '
              'lop) — xem bang cua so.',
    }[verdict]
    return {'verdict': verdict, 'literal_verdict': literal, 'literal': lit, 'text': text,
            'note': 'Mot seed => moi ket qua duong chi la UNG VIEN; khong viet claim truoc khi co seed 86.'}


# ═════════════════════════════ mo ta 6.4 ══════════════════════════════════════

def first_ci_positive(tp, pair):
    sub = tp[(tp['pair'] == pair) & (tp['t'] != 0)].sort_values('t')
    hit = sub[sub['ci_lo'] > 0]
    return (int(hit['t'].iloc[0]) if len(hit) else None), sub['t'].tolist()


def describe(tp):
    d = {}
    t_bce, ts = first_ci_positive(tp, 'region_bce')
    t_aff, _ = first_ci_positive(tp, 'region_aff')
    d['first_ci_pos_region_bce'] = t_bce
    d['first_ci_pos_region_aff'] = t_aff
    if t_bce is not None and (t_aff is None or ts.index(t_aff) - ts.index(t_bce) >= 2):
        d['hypothesis_11_1'] = 'phu hop voi gia thuyet §11.1 (BCE som hon ro, >= 2 snapshot)'
    elif t_bce is None:
        d['hypothesis_11_1'] = 'khong danh gia duoc (cos(Region, BCE) khong co CI > 0 o snapshot nao)'
    else:
        d['hypothesis_11_1'] = 'khong phu hop (BCE khong som hon >= 2 snapshot)'
    last = tp[(tp['pair'] == 'bce_aff') & (tp['t'].isin(LAST5))].sort_values('t')
    d['std_cos_bce_aff_last5'] = float(last['cos_all'].std(ddof=1)) if len(last) >= 2 else None
    d['last5_t'] = last['t'].tolist()
    d['last5_cos'] = last['cos_all'].tolist()
    r36 = tp[tp['t'] == 36000].set_index('pair')['ratio'].to_dict()
    d['ratios_36k'] = {pn: r36.get(pn) for pn in RATIO_NAMES}
    return d


# ═════════════════════════════ hinh ═══════════════════════════════════════════

def plot(tp, fig_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    pairs = [pair_name(p) for p in ALL_PAIRS]
    fig, axes = plt.subplots(len(pairs), 1, figsize=(11, 3.2 * len(pairs)), sharex=True)
    for ax, pn in zip(axes, pairs):
        s = tp[tp['pair'] == pn].sort_values('t')
        x = s['t'].to_numpy()
        ax.fill_between(x, s['null_lo'], s['null_hi'], color='0.85', label='null 2.5–97.5%')
        ax.fill_between(x, s['ci_lo'], s['ci_hi'], color='tab:blue', alpha=0.25, label='CI 95% bootstrap')
        ax.plot(x, s['cos_all'], 'o-', color='tab:blue', ms=3, label='cos_all (32 anh)')
        conf = s[s['conflict']]
        ax.plot(conf['t'], conf['cos_all'], 'x', color='tab:red', ms=8, label='thoa 6.1')
        ax.axhline(0, color='k', lw=0.6)
        for y in (-TAU_PARAM, TAU_PARAM):
            ax.axhline(y, color='k', lw=0.5, ls=':')
        ax.axvline(T_LATE, color='tab:orange', lw=0.8, ls='--')
        ax.set_ylabel(f'cos ({pn})')
        ax.grid(alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=8, ncol=2)
    axes[-1].set_xlabel('iter (snapshot)')
    fig.suptitle('Phan D — cos muc tham so theo iter (Static s19, 32 anh train; t=0 khong dung de quyet dinh)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'traj_cos_param.png'), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for pn, lab in RATIO_NAMES.items():
        s = tp[tp['pair'] == pn].sort_values('t')
        ax.plot(s['t'], s['ratio'], 'o-', ms=3, label=lab)
        ax.axhline(STEP1_RATIOS[pn], ls=':', lw=0.8, color=ax.lines[-1].get_color())
    ax.set_yscale('log')
    ax.axhspan(0.1, 10, color='tab:green', alpha=0.07, label='[0.1, 10] (tieu chi 4)')
    ax.set_xlabel('iter (snapshot)')
    ax.set_ylabel('ti le do lon (log)')
    ax.grid(alpha=0.3, which='both')
    ax.legend(fontsize=8)
    ax.set_title('Phan D — ti le do lon gradient theo iter (cham: Buoc 1 @36k tren val)')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'traj_ratio.png'), dpi=120)
    plt.close(fig)


# ═════════════════════════════ main ═══════════════════════════════════════════

def collect(od, n_boot, n_null):
    from Tools.grad_conflict_probe import feat_summary_rows, TAU_FEAT
    files = sorted(glob.glob(os.path.join(od, 'gram', 'iter_*.npz')),
                   key=lambda p: int(os.path.basename(p)[5:-4]))
    if not files:
        raise SystemExit(f"Khong co {od}/gram/iter_*.npz — chay phanD-measure truoc.")
    prow, frow = [], []
    W = U = V = None
    for f in files:
        z = np.load(f)
        t = int(z['iter'])
        terms = [str(x) for x in z['terms']]
        G = z['G']
        n = G.shape[0] // len(terms)
        if W is None:
            W = boot_weights(n, n_boot)
            U, V = half_splits(n, n_null)
        batches = [list(b) for b in z['batches']]
        for k1, k2 in ALL_PAIRS:
            pn = pair_name((k1, k2))
            s = param_stats(G, terms, batches, k1, k2, W, U, V)
            frac_neg = float(np.mean(np.array(s['cos_batch']) < 0))
            c = criteria('param', pn, 'all', 'all', s['cos_all'], s['ci_lo'], s['ci_hi'], s['null_lo'],
                         s['null_hi'], frac_neg, s['ratio'], 1.0, n, TAU_PARAM)
            row = {'t': t, 'pair': pn, 'cos_all': s['cos_all'], 'ci_lo': s['ci_lo'], 'ci_hi': s['ci_hi'],
                   'null_lo': s['null_lo'], 'null_hi': s['null_hi'], 'ratio': s['ratio'],
                   **{f'cos_batch_{b + 1}': v for b, v in enumerate(s['cos_batch'])},
                   'norm1_all': s['norm1_all'], 'norm2_all': s['norm2_all'],
                   **{k: c[k] for k in ('crit1_ci_below_0', 'crit2_below_null', 'crit3_abs_ge_tau',
                                        'crit4_mag_in_range')},
                   'conflict': c['confirmed_single_ckpt'], 'decision_eligible': t != 0}
            prow.append(row)
        sp = os.path.join(od, 'feat', f'iter_{t}_strata.csv')
        np_ = os.path.join(od, 'feat', f'iter_{t}_null.csv')
        if os.path.exists(sp) and os.path.exists(np_):
            st = pd.read_csv(sp)
            nf = pd.read_csv(np_)
            sums = st.groupby(['pair', 'axis', 'stratum'], sort=False)[
                ['sum_dot', 'sum_n1', 'sum_n2', 'n_pos', 'n_support', 'n_neg']].sum()
            for r in feat_summary_rows(LABEL, st, nf, n_boot):
                key = (r['pair'], r['axis'], r['stratum'])
                agg = sums.loc[key]
                frow.append({'t': t, 'pair': r['pair'], 'axis': r['axis'], 'stratum': r['stratum'],
                             'sum_dot': agg['sum_dot'], 'sum_n1': agg['sum_n1'], 'sum_n2': agg['sum_n2'],
                             'n_pos': int(agg['n_pos']), 'n_support': int(agg['n_support']),
                             'n_neg': int(agg['n_neg']), 'cosW': r['value'], 'ci_lo': r['ci_lo'],
                             'ci_hi': r['ci_hi'], 'null_lo': r['null_lo'], 'null_hi': r['null_hi'],
                             'mag_ratio': r['mag_ratio'], 'support_cov': r['support_cov'],
                             'flag_insufficient': r['flag_insufficient'], 'tau': TAU_FEAT,
                             **{k: r[k] for k in ('crit1_ci_below_0', 'crit2_below_null', 'crit3_abs_ge_tau',
                                                  'crit4_mag_in_range')},
                             'conflict': r['confirmed_single_ckpt'], 'decision_eligible': t != 0})
    return pd.DataFrame(prow), pd.DataFrame(frow)


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return '—'
    return f'{x:.{nd}f}' if isinstance(x, (float, np.floating)) else str(x)


def write_md(od, tp, tf, win, dec, desc, gates, n_boot, n_null):
    meta = (gates or {}).get('meta', {})
    post = (gates or {}).get('post', {})
    interp = post.get('interpretable', None)
    L = []
    L.append(f"**Config:** `{meta.get('config', 'configs/unet_former_resnet18_static_boundary/static_boundary.yaml')}` "
             f"(md5 `{meta.get('config_md5', '?')}`) · **seed** 19 · **script train:** `{meta.get('train_script', '?')}` · "
             f"**che do do:** model.eval() (BN running stats), fp32, 1 anh/backward, w_region=1 w_bce=0.4 w_aff=0.4 · "
             f"**dien giai duoc:** {'CO' if interp else ('KHONG — sanity §5 / D6 FAIL' if interp is False else '?')}")
    L.append('')
    L.append('# Phan D — Quy dao xung dot gradient (Static s19)')
    L.append('')
    L.append(f"Tao luc {datetime.datetime.now().isoformat(timespec='seconds')}. "
             f"{tp['t'].nunique()} snapshot do; bootstrap {n_boot} lan, null {n_null} lan chia nua (rng 19).")
    L.append('')
    if interp is False:
        L.append('> **CANH BAO: sanity §5 hoac D6 FAIL — run nay KHONG duoc dien giai quy dao (S2 §5). '
                 'So lieu duoi day chi de luu tru.**')
        L.append('')
    L.append('## Sanity §5 + D6')
    L.append('')
    if post:
        for k in ('sanity5', 'd6', 'miou9_40k', 'bf_36k'):
            if k in post:
                L.append(f"- **{k}:** `{json.dumps(post[k], ensure_ascii=False, default=str)}`")
    else:
        L.append('- (khong co muc `post` trong gates_dryrun.json)')
    L.append('')
    L.append('## Bang 6.3 — verdict')
    L.append('')
    L.append(f"**{dec['text']}**")
    L.append('')
    L.append(f"- Ma theo dung thu tu D-G > D-RA > D-L (khong xet D-dai-dang): `{dec['literal_verdict']}`")
    L.append(f"- Dieu kien tung ma: `{json.dumps(dec['literal'])}`")
    L.append('- Cach doc cua code: D-dai-dang (co cua so keo toi >= 36000) ghi de verdict vi bang 6.3 yeu cau '
             '"dung dien giai"; cac cua so cua D-G/D-RA/D-L chi tinh khi ket thuc truoc 36000.')
    L.append(f"- {dec['note']}")
    L.append('')
    L.append('## Cua so xung dot (§6.2)')
    L.append('')
    if len(win):
        L.append('| muc | cap | truc | tang | t dau | t cuoi | so snapshot |')
        L.append('|---|---|---|---|---|---|---|')
        for _, r in win.iterrows():
            L.append(f"| {r['level']} | {r['pair']} | {r['axis']} | {r['stratum']} | {r['t_start']} | {r['t_end']} | "
                     f"{r['n_snapshots']} |")
    else:
        L.append('Khong co cua so nao.')
    L.append('')
    L.append('## Muc tham so theo iter')
    L.append('')
    L.append('| t | cap | cos_all | CI 95% | null 2.5–97.5% | ti le | thoa 6.1 |')
    L.append('|---|---|---|---|---|---|---|')
    for _, r in tp.sort_values(['t', 'pair']).iterrows():
        L.append(f"| {r['t']} | {r['pair']} | {_f(r['cos_all'])} | [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] | "
                 f"[{_f(r['null_lo'])}, {_f(r['null_hi'])}] | {_f(r['ratio'], 4)} | "
                 f"{'**CO**' if r['conflict'] else ''}{' (t=0, khong quyet dinh)' if r['t'] == 0 else ''} |")
    L.append('')
    L.append('## Mo ta 6.4 (khong dung de quyet dinh)')
    L.append('')
    r36 = desc['ratios_36k']
    L.append('- Ti le do lon @36k (tap do train) vs Buoc 1 @36k (val): ' + '; '.join(
        f"{RATIO_NAMES[k]} = {_f(r36.get(k), 4)} (Buoc 1: {STEP1_RATIOS[k]})" for k in RATIO_NAMES) +
        ' — chi so dau va bac do lon (S2 §5).')
    L.append(f"- Iter dau tien CI cos(Region, BCE) > 0: **{desc['first_ci_pos_region_bce']}**; "
             f"cos(Region, Aff) > 0: **{desc['first_ci_pos_region_aff']}** → {desc['hypothesis_11_1']}.")
    L.append(f"- Do on dinh: SD(ddof=1) cos_all (BCE, Aff) qua {desc['last5_t']} = "
             f"**{_f(desc['std_cos_bce_aff_last5'], 4)}** (gia tri: {[round(x, 3) for x in desc['last5_cos']]}); "
             f"Buoc 1 s86 nhay 0.477 → 0.012.")
    if len(tf):
        n_feat = int(tf[(tf['t'] != 0) & tf['conflict']].shape[0])
        L.append(f"- Muc dac trung: {n_feat} (t, cap, truc, tang) thoa 6.1 (tang du ho tro, t != 0) — chi tiet "
                 f"`traj_feat_strata.csv`.")
    L.append('')
    L.append('## Hinh')
    L.append('')
    L.append('- `figures/traj_cos_param.png` — 3 cap, CI + dai null theo iter')
    L.append('- `figures/traj_ratio.png` — 3 ti le do lon (thang log)')
    with open(os.path.join(od, 'summary_phanD.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default='output/phanD')
    ap.add_argument('--gates-json', default=None, help='gates_dryrun.json (meta + post) de ghi vao bao cao')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--n-null', type=int, default=1000)
    args = ap.parse_args(argv)
    od = args.output_dir
    tp, tf = collect(od, args.n_boot, args.n_null)
    tp = tp.sort_values(['t', 'pair']).reset_index(drop=True)
    tp.to_csv(os.path.join(od, 'traj_param.csv'), index=False)
    if len(tf):
        tf = tf.sort_values(['t', 'pair', 'axis', 'stratum']).reset_index(drop=True)
        tf.to_csv(os.path.join(od, 'traj_feat_strata.csv'), index=False)
    keys = ['t', 'level', 'pair', 'axis', 'stratum', 'conflict']
    cand = tp.assign(level='param', axis='all', stratum='all')[keys]
    if len(tf):
        cand = pd.concat([cand, tf.assign(level='feat')[keys]], ignore_index=True)
    win = find_windows(cand)
    win.to_csv(os.path.join(od, 'windows.csv'), index=False)
    dec = decide(win)
    desc = describe(tp)
    gates = read_json(args.gates_json) if args.gates_json and os.path.exists(args.gates_json) else None
    plot(tp, os.path.join(od, 'figures'))
    write_md(od, tp, tf, win, dec, desc, gates, args.n_boot, args.n_null)
    write_json({'decision': dec, 'describe': desc, 'n_snapshots': int(tp['t'].nunique()),
                'windows': win.to_dict('records'),
                'interpretable': (gates or {}).get('post', {}).get('interpretable')},
               os.path.join(od, 'summary_phanD.json'))
    print(f"VERDICT: {dec['text']}")
    return dec


if __name__ == '__main__':
    main()
