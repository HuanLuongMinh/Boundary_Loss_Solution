"""Tools/spurious_holes.py — Phan B (va B2) cua
docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md: dem lo gia (loi loai 5 —
vat dac bi du doan thanh vanh khuyen) tren mask du doan DA DUMP.

CPU, khong can checkpoint. Chi doc PNG mask (output/dump/<run>/masks/<id>.png,
1024x1024, gia tri 0..8 — dinh dang Tools/per_image_dump.save_pred_masks) va GT
val (qua get_val_transforms, Resize nearest -> 1024).

Subcommand:
  selftest   Cong 6 (muc 4): dia dac -> 0 lo; vanh khuyen -> 1 lo dung dien tich;
             2 dia roi -> 0 lo; lo 63 px bi loai o A_min=64. Ghi gates_holes.json.
  count      Dem lo theo tung checkpoint -> per_image_<ckpt>.csv, holes_<ckpt>.csv.
             LUON can --masks baseline_40k=... (de tinh "lo gia moi" so voi Baseline).
  compare    Gop per_image_*.csv da co -> bootstrap ghep cap H1..H5, N1, N2,
             verdict bang 5.4, cong tan suat 3.3 -> pairs_bootstrap.{csv,md,json}.
  fill-test  Phan B2: lap lo <= A_fill (khong dung GT), do lai mIoU-9, BF P/R/F,
             ASD 2 chieu, bootstrap truoc/sau -> fill_test.{csv,md}.

Dinh nghia van hanh (muc 3.1, KHONG doi):
  1. M_c = mask du doan lop c; thanh phan du doan 8-lien thong.
  2. Lo = binary_fill_holes(M_c) - M_c, tach thanh phan lo 4-lien thong (doi ngau 8/4).
  3. Giu lo co dien tich >= A_min (chinh 64; phu 16, 256).
  4. Lo gia: >= 50% pixel cua lo co GT = c.
  5. Vanh khuyen: thanh phan du doan co tong dien tich lo gia >= 10% dien tich sau khi lap.
  6. Lo gia moi: lo gia cua X khong chong (IoU < 0.1) voi BAT KY lo gia nao cua
     Baseline cung anh, cung lop (cung A_min).

Xem docs/huong-dan-chay-grad-conflict-va-lo-gia.md de biet lenh day du.
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, label as cc_label

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (
    CHECKPOINTS, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, SEED, STRUCT_4, STRUCT_8, CFG_STATIC,
    gt_components, parse_label_values, check_labels_known, get_git_commit, write_json, read_json,
    append_log, load_gt_cached,
)

A_MIN_MAIN = 64
A_MIN_EXTRA = (16, 256)
SPURIOUS_FRAC = 0.5
ANNULUS_FRAC = 0.10
NEW_IOU = 0.1
FREQ_GATE = 0.1          # muc 3.3: < 0.1 lo gia moi/anh o Static s19@36k, A_min=64 -> "hiem"
BASELINE_LABEL = 'baseline_40k'
METRICS = ('n_lo_gia', 'dien_tich_lo_gia', 'n_vanh_khuyen', 'n_lo_gia_moi', 'n_holes_total')
VERDICT_METRIC = 'n_lo_gia'   # dai luong dung cho bang 5.4 (xem pairs_bootstrap.md, muc "Quy uoc")


# ═════════════════════════════ Phat hien lo ═══════════════════════════════════

def _fill_area(mask):
    return int(binary_fill_holes(mask).sum())


def detect_holes_class(M, min_area):
    """M (H,W) bool — mask du doan 1 lop. Tra ve (holes, fg_lab).

    holes: list dict {hole_id, area, sl (slice tuple), hmask (bool trong sl),
           comp_id (thanh phan du doan bao quanh, 8-lien thong), comp_filled_area}
    Chi giu lo co area >= min_area.
    """
    holes = []
    if not M.any():
        return holes, None
    fg_lab, _nfg = cc_label(M, structure=STRUCT_8)
    filled = binary_fill_holes(M)            # structure mac dinh = chu thap -> nen 4-lien thong
    hole_px = filled & ~M
    if not hole_px.any():
        return holes, fg_lab
    hl, nh = cc_label(hole_px, structure=STRUCT_4)
    areas = np.bincount(hl.ravel(), minlength=nh + 1)
    objs = find_objects(hl)
    H, W = M.shape
    fill_cache = {}
    comp_objs = None
    for j in range(1, nh + 1):
        if areas[j] < min_area:
            continue
        s0, s1 = objs[j - 1]
        sl = (slice(max(s0.start - 1, 0), min(s0.stop + 1, H)),
              slice(max(s1.start - 1, 0), min(s1.stop + 1, W)))
        hmask = hl[sl] == j
        ring = binary_dilation(hmask, structure=STRUCT_4) & ~hmask
        cands = np.unique(fg_lab[sl][ring])
        cands = cands[cands > 0]
        comp_id = int(cands[0]) if len(cands) else 0
        if len(cands) > 1:
            # Nhieu thanh phan cham lo (vd dao nho nam trong lo) -> chon thanh phan
            # ma ban lap cua no chua tron lo.
            if comp_objs is None:
                comp_objs = find_objects(fg_lab)
            ys, xs = np.nonzero(hmask)
            ys = ys + sl[0].start
            xs = xs + sl[1].start
            for cid in cands:
                cs = comp_objs[cid - 1]
                if not (cs[0].start <= ys.min() and ys.max() < cs[0].stop and
                        cs[1].start <= xs.min() and xs.max() < cs[1].stop):
                    continue
                cf = binary_fill_holes(fg_lab[cs] == cid)
                if cf[ys - cs[0].start, xs - cs[1].start].all():
                    comp_id = int(cid)
                    break
        if comp_id not in fill_cache:
            if comp_objs is None:
                comp_objs = find_objects(fg_lab)
            cs = comp_objs[comp_id - 1] if comp_id > 0 else None
            fill_cache[comp_id] = _fill_area(fg_lab[cs] == comp_id) if cs is not None else 0
        holes.append({'hole_id': j, 'area': int(areas[j]), 'sl': sl, 'hmask': hmask,
                      'comp_id': comp_id, 'comp_filled_area': fill_cache[comp_id]})
    return holes, fg_lab


def holes_for_image(pred, gt, min_area_store=min(A_MIN_EXTRA + (A_MIN_MAIN,))):
    """Moi lop c: lo (area >= min_area_store) + frac GT=c + thanh phan GT chua lo.
    Tra ve list dict (1 dict / lo)."""
    gt_comp_map, gt_table = gt_components(gt)
    out = []
    for c in range(NUM_CLASSES):
        holes, _ = detect_holes_class(pred == c, min_area_store)
        for h in holes:
            sl, hm = h['sl'], h['hmask']
            g = gt[sl][hm]
            is_c = g == c
            frac = float(is_c.mean()) if g.size else 0.0
            gcomp = gt_comp_map[sl][hm][is_c]
            if gcomp.size:
                vals, cnts = np.unique(gcomp, return_counts=True)
                gid = int(vals[np.argmax(cnts)])
            else:
                gid = 0
            out.append({
                'class_id': c, 'hole_id': h['hole_id'], 'area': h['area'], 'frac_gt_c': frac,
                'is_spurious': frac >= SPURIOUS_FRAC, 'comp_id': h['comp_id'],
                'comp_filled_area': h['comp_filled_area'],
                'gt_comp_id': gid, 'gt_comp_area': int(gt_table[gid][1]) if gid else 0,
                'bbox': (sl[0].start, sl[1].start, sl[0].stop, sl[1].stop),
                'sl': sl, 'hmask': hm,
            })
    return out


def _hole_pixels_flat(h, W):
    ys, xs = np.nonzero(h['hmask'])
    return (ys + h['sl'][0].start) * W + (xs + h['sl'][1].start)


def max_iou_vs_reference(holes, ref_holes, shape, a_min):
    """Voi moi lo gia trong `holes` (area >= a_min): IoU lon nhat voi lo gia cua
    ref_holes (cung lop, area >= a_min). Tra ve dict index -> max_iou."""
    H, W = shape
    out = {}
    for c in range(NUM_CLASSES):
        refs = [r for r in ref_holes if r['class_id'] == c and r['is_spurious'] and r['area'] >= a_min]
        mine = [(i, h) for i, h in enumerate(holes)
                if h['class_id'] == c and h['is_spurious'] and h['area'] >= a_min]
        if not mine:
            continue
        if not refs:
            for i, _h in mine:
                out[i] = 0.0
            continue
        ref_map = np.zeros(H * W, dtype=np.int32)
        ref_area = {}
        for k, r in enumerate(refs, start=1):
            ref_map[_hole_pixels_flat(r, W)] = k
            ref_area[k] = r['area']
        for i, h in mine:
            ids, inter = np.unique(ref_map[_hole_pixels_flat(h, W)], return_counts=True)
            best = 0.0
            for k, n in zip(ids, inter):
                if k == 0:
                    continue
                iou = n / (h['area'] + ref_area[k] - n)
                best = max(best, float(iou))
            out[i] = best
    return out


def per_image_rows(image_id, holes, new_iou_by_amin, a_mins):
    """Hang per-image (class 'all' + tung lop) cho moi A_min."""
    rows = []
    for a_min in a_mins:
        new_iou = new_iou_by_amin[a_min]
        for cls in ['all'] + CLASS_NAMES:
            cid = None if cls == 'all' else CLASS_NAMES.index(cls)
            sel = [(i, h) for i, h in enumerate(holes)
                   if h['area'] >= a_min and (cid is None or h['class_id'] == cid)]
            spur = [(i, h) for i, h in sel if h['is_spurious']]
            comp_spur_area = {}
            for _i, h in spur:
                key = (h['class_id'], h['comp_id'])
                comp_spur_area[key] = comp_spur_area.get(key, 0) + h['area']
            comp_filled = {(h['class_id'], h['comp_id']): h['comp_filled_area'] for _i, h in spur}
            n_annulus = sum(1 for k, a in comp_spur_area.items()
                            if comp_filled[k] > 0 and a >= ANNULUS_FRAC * comp_filled[k])
            n_new = sum(1 for i, _h in spur if new_iou.get(i, 0.0) < NEW_IOU)
            rows.append({
                'image_id': image_id, 'a_min': a_min, 'class': cls,
                'n_lo_gia': len(spur),
                'dien_tich_lo_gia': int(sum(h['area'] for _i, h in spur)),
                'n_vanh_khuyen': int(n_annulus),
                'n_lo_gia_moi': int(n_new),
                'n_holes_total': len(sel),
            })
    return rows


def hole_table_rows(image_id, holes, new_iou_by_amin, a_mins):
    rows = []
    for a_min in a_mins:
        new_iou = new_iou_by_amin[a_min]
        for i, h in enumerate(holes):
            if h['area'] < a_min:
                continue
            miou = new_iou.get(i, float('nan')) if h['is_spurious'] else float('nan')
            rows.append({
                'image_id': image_id, 'class': CLASS_NAMES[h['class_id']], 'a_min': a_min,
                'hole_id': h['hole_id'], 'area': h['area'], 'frac_gt_c': round(h['frac_gt_c'], 6),
                'is_spurious': int(h['is_spurious']),
                'is_new': int(h['is_spurious'] and miou < NEW_IOU),
                'max_iou_baseline': miou,
                'comp_id': h['comp_id'], 'comp_filled_area': h['comp_filled_area'],
                'gt_comp_id': h['gt_comp_id'], 'gt_comp_area': h['gt_comp_area'],
                'bbox': '{}:{}:{}:{}'.format(*h['bbox']),
            })
    return rows


def load_pred_png(mask_dir, image_id):
    path = os.path.join(mask_dir, image_id + '.png')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Thieu mask du doan: {path}")
    return np.array(Image.open(path)).astype(np.int64)


# ═════════════════════════════ selftest (cong 6) ══════════════════════════════

def _disk(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def run_selftest():
    """Cong 6 — tra ve list (ten, pass, chi tiet)."""
    res = []
    shape = (128, 128)

    m = _disk(shape, 64, 64, 30)
    holes, _ = detect_holes_class(m, A_MIN_MAIN)
    res.append(('dia_dac_0_lo', len(holes) == 0, f'so lo = {len(holes)}'))

    inner = _disk(shape, 64, 64, 10)
    ann = _disk(shape, 64, 64, 30) & ~inner
    holes, _ = detect_holes_class(ann, A_MIN_MAIN)
    ok = len(holes) == 1 and holes[0]['area'] == int(inner.sum())
    res.append(('vanh_khuyen_1_lo_dung_dien_tich', ok,
                f"so lo = {len(holes)}, dien tich = {[h['area'] for h in holes]}, ky vong {int(inner.sum())}"))

    two = _disk(shape, 30, 30, 15) | _disk(shape, 95, 95, 15)
    holes, _ = detect_holes_class(two, A_MIN_MAIN)
    res.append(('hai_dia_roi_0_lo', len(holes) == 0, f'so lo = {len(holes)}'))

    sq = np.zeros(shape, dtype=bool)
    sq[20:60, 20:60] = True
    sq[30:37, 30:39] = False          # 7 x 9 = 63 px
    holes64, _ = detect_holes_class(sq, 64)
    holes16, _ = detect_holes_class(sq, 16)
    ok = len(holes64) == 0 and len(holes16) == 1 and holes16[0]['area'] == 63
    res.append(('lo_63px_bi_loai_o_A_min_64', ok,
                f'A_min=64: {len(holes64)} lo; A_min=16: {[h["area"] for h in holes16]}'))
    return res


def cmd_selftest(args):
    res = run_selftest()
    all_ok = all(ok for _n, ok, _d in res)
    for name, ok, detail in res:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    write_json({'gate': 6, 'name': 'unit test lo (Phan B)', 'status': 'PASS' if all_ok else 'FAIL',
                'cases': [{'case': n, 'pass': ok, 'detail': d} for n, ok, d in res],
                'created_at': datetime.datetime.now().isoformat(timespec='seconds')},
               os.path.join(args.output_dir, 'gates_holes.json'))
    print(f"Cong 6: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        sys.exit(1)


# ═════════════════════════════════ count ══════════════════════════════════════

def _gate7_image_list(label, mask_dir, image_ids, per_image_stats_csv):
    """Cong 7 cho Phan B: mask dir co DUNG tap anh val; per_image_stats.csv (neu co)
    co dung danh sach + thu tu."""
    pngs = sorted(os.path.splitext(f)[0] for f in os.listdir(mask_dir) if f.lower().endswith('.png'))
    if pngs != sorted(image_ids):
        missing = sorted(set(image_ids) - set(pngs))[:5]
        extra = sorted(set(pngs) - set(image_ids))[:5]
        raise AssertionError(f"[{label}] CONG 7 FAIL: tap mask PNG khac tap anh val "
                             f"(thieu {missing}..., thua {extra}...)")
    if per_image_stats_csv:
        imgs = pd.read_csv(per_image_stats_csv, usecols=['image'])['image'].astype(str).tolist()
        if imgs != list(image_ids):
            raise AssertionError(f"[{label}] CONG 7 FAIL: per_image_stats.csv khac danh sach/thu tu anh val.")
    return True


def cmd_count(args):
    masks = parse_label_values(args.masks, '--masks')
    check_labels_known(masks, '--masks')
    stats_csv = parse_label_values(args.per_image_stats, '--per-image-stats')
    if BASELINE_LABEL not in masks:
        raise SystemExit(f"Can --masks {BASELINE_LABEL}=<dir> (lo gia moi duoc tinh so voi Baseline, muc 3.1 buoc 6).")
    a_mins = sorted({args.a_min, *[int(x) for x in args.a_min_extra.split(',') if x]})
    os.makedirs(args.output_dir, exist_ok=True)

    from Tools.oracle_boundary_ceiling import load_cfg, build_val_dataset, image_ids_of
    cfg = load_cfg(args.config, args.data_root)
    val_ds = build_val_dataset(cfg)
    image_ids = image_ids_of(val_ds)
    n = len(image_ids) if args.max_images is None else min(args.max_images, len(image_ids))

    todo = []
    for label in masks:
        done_path = os.path.join(args.output_dir, f'done_holes_{label}.json')
        if os.path.exists(done_path) and not args.force:
            d = read_json(done_path)
            if d.get('n_images') == n and d.get('a_mins') == a_mins:
                print(f"[{label}] da co ket qua (done_holes_{label}.json) — bo qua (--force de chay lai).")
                continue
        if args.max_images is None:
            _gate7_image_list(label, masks[label], image_ids, stats_csv.get(label))
            print(f"[{label}] Cong 7 (danh sach anh) PASS.")
        todo.append(label)
    if not todo:
        return

    log_path = os.path.join(args.output_dir, 'run_log.txt')
    append_log(log_path, f"[{datetime.datetime.now().isoformat(timespec='seconds')}] count "
                         f"labels={todo} a_mins={a_mins} n={n} git={get_git_commit()} argv={' '.join(sys.argv)}")
    per_img = {lb: [] for lb in todo}
    per_hole = {lb: [] for lb in todo}
    t0 = time.time()
    for i in range(n):
        name, gt = load_gt_cached(val_ds, i, args.gt_cache or None)
        assert name == image_ids[i]
        gt = np.asarray(gt).astype(np.int64)
        base_pred = load_pred_png(masks[BASELINE_LABEL], name)
        assert base_pred.shape == gt.shape, f"{name}: mask {base_pred.shape} != GT {gt.shape}"
        base_holes = holes_for_image(base_pred, gt, min(a_mins))
        for label in todo:
            if label == BASELINE_LABEL:
                holes = base_holes
            else:
                pred = load_pred_png(masks[label], name)
                assert pred.shape == gt.shape, f"[{label}] {name}: mask {pred.shape} != GT {gt.shape}"
                holes = holes_for_image(pred, gt, min(a_mins))
            new_iou = {a: max_iou_vs_reference(holes, base_holes, gt.shape, a) for a in a_mins}
            per_img[label].extend(per_image_rows(name, holes, new_iou, a_mins))
            per_hole[label].extend(hole_table_rows(name, holes, new_iou, a_mins))
        el = time.time() - t0
        print(f"  ... {i + 1}/{n} anh ({el:.0f}s)", end='\r')
    print()
    for label in todo:
        pd.DataFrame(per_img[label]).to_csv(os.path.join(args.output_dir, f'per_image_{label}.csv'), index=False)
        pd.DataFrame(per_hole[label], columns=[
            'image_id', 'class', 'a_min', 'hole_id', 'area', 'frac_gt_c', 'is_spurious', 'is_new',
            'max_iou_baseline', 'comp_id', 'comp_filled_area', 'gt_comp_id', 'gt_comp_area', 'bbox',
        ]).to_csv(os.path.join(args.output_dir, f'holes_{label}.csv'), index=False)
        write_json({'label': label, 'mask_dir': masks[label], 'n_images': n, 'a_mins': a_mins,
                    'wall_time_s_all_labels': time.time() - t0,
                    'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                    'git_commit': get_git_commit()},
                   os.path.join(args.output_dir, f'done_holes_{label}.json'))
        print(f"[{label}] Da ghi per_image_{label}.csv, holes_{label}.csv")


# ═════════════════════════════════ compare ════════════════════════════════════

PAIRS = [
    # id, A, B, mo ta, bien do lien-seed ap dung
    ('H1', 'bce04_40k', 'baseline_40k', 'BCE lambda=0.4 - Baseline (BCE mot minh)', 'max(N1,N2)'),
    ('H2', 'affonly_s19_40k', 'baseline_40k', 'AffOnly s19 - Baseline (affinity mot minh)', 'N2'),
    ("H2'", 'affonly_s86_40k', 'baseline_40k', 'AffOnly s86 - Baseline', 'N2'),
    ('H3', 'static_s19_36k', 'baseline_40k', 'Static s19 - Baseline (loss ghep)', 'N1'),
    ("H3'", 'static_s86_36k', 'baseline_40k', 'Static s86@36k - Baseline', 'N1'),
    ('H4', 'static_s19_36k', 'bce04_40k', 'Static s19 - BCE lambda=0.4 (them affinity vao BCE)', 'N1'),
    ('N1', 'static_s86_36k', 'static_s19_36k', 'Bien do lien-seed Static (s86 - s19, @36k)', None),
    ('N2', 'affonly_s86_40k', 'affonly_s19_40k', 'Bien do lien-seed AffOnly (s86 - s19)', None),
    # mo ta, khong vao bang 5.4
    ('X1', 'static_s86_40k', 'baseline_40k', '[mo ta] Static s86@40k - Baseline', 'N1'),
    ('X2', 'bce02_40k', 'baseline_40k', '[mo ta] BCE lambda=0.2 - Baseline (ckpt bo sung)', 'max(N1,N2)'),
    ('X3', 'static_s86_40k', 'static_s86_36k', '[mo ta] Static s86: 40k - 36k (on dinh theo moc)', None),
]
CONTRASTS = [
    ('H5', ('static_s19_36k', 'bce04_40k', 'affonly_s19_40k', 'baseline_40k'),
     '(Static s19 - BCE) - (AffOnly s19 - Baseline)', 'max(N1,N2)'),
    ("H5'", ('static_s86_36k', 'bce04_40k', 'affonly_s86_40k', 'baseline_40k'),
     '(Static s86 - BCE) - (AffOnly s86 - Baseline)', 'max(N1,N2)'),
]


def _boot_idx(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def _ci(vals):
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def load_per_image_matrix(output_dir, labels):
    """dict label -> DataFrame per-image (da sap theo image_id goc trong file)."""
    out = {}
    for lb in labels:
        p = os.path.join(output_dir, f'per_image_{lb}.csv')
        if os.path.exists(p):
            out[lb] = pd.read_csv(p)
    return out


def _vector(df, a_min, cls, metric, image_order):
    sub = df[(df['a_min'] == a_min) & (df['class'] == cls)].set_index('image_id')
    return sub.loc[image_order, metric].to_numpy(dtype=np.float64)


def cmd_compare(args):
    labels = list(CHECKPOINTS)
    dfs = load_per_image_matrix(args.output_dir, labels)
    if not dfs:
        raise SystemExit(f"Khong co per_image_*.csv nao trong {args.output_dir} — chay 'count' truoc.")
    ref_label = next(iter(dfs))
    image_order = dfs[ref_label][dfs[ref_label]['class'] == 'all']['image_id'].drop_duplicates().tolist()
    for lb, df in dfs.items():
        ids = df[df['class'] == 'all']['image_id'].drop_duplicates().tolist()
        if ids != image_order:
            raise AssertionError(f"CONG 7 FAIL: per_image_{lb}.csv khac danh sach/thu tu anh voi {ref_label}.")
    n = len(image_order)
    a_mins = sorted(dfs[ref_label]['a_min'].unique().tolist())
    idx = _boot_idx(n, args.n_boot, SEED)
    counts = np.zeros((args.n_boot, n))
    np.add.at(counts, (np.repeat(np.arange(args.n_boot), n), idx.ravel()), 1.0)

    rows = []
    for a_min in a_mins:
        for cls in ['all'] + CLASS_NAMES:
            for metric in METRICS:
                vec = {lb: _vector(df, a_min, cls, metric, image_order) for lb, df in dfs.items()}
                boot_mean = {lb: counts @ v / n for lb, v in vec.items()}
                obs_mean = {lb: float(v.mean()) for lb, v in vec.items()}
                amp = {}
                for pid, a, b, desc, _amp in PAIRS:
                    if pid in ('N1', 'N2') and a in vec and b in vec:
                        amp[pid] = abs(obs_mean[a] - obs_mean[b])
                for pid, a, b, desc, amp_rule in PAIRS:
                    if a not in vec or b not in vec:
                        rows.append(_missing_row(pid, desc, a_min, cls, metric, [a, b]))
                        continue
                    d = boot_mean[a] - boot_mean[b]
                    rows.append(_pair_row(pid, desc, a_min, cls, metric, obs_mean[a] - obs_mean[b], d,
                                          amp_rule, amp, obs_mean[a], obs_mean[b]))
                for pid, (s, bce, aff, base), desc, amp_rule in CONTRASTS:
                    need = [s, bce, aff, base]
                    if not all(x in vec for x in need):
                        rows.append(_missing_row(pid, desc, a_min, cls, metric, need))
                        continue
                    d = (boot_mean[s] - boot_mean[bce]) - (boot_mean[aff] - boot_mean[base])
                    obs = (obs_mean[s] - obs_mean[bce]) - (obs_mean[aff] - obs_mean[base])
                    rows.append(_pair_row(pid, desc, a_min, cls, metric, obs, d, amp_rule, amp, None, None))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.output_dir, 'pairs_bootstrap.csv'), index=False)

    # Tan suat (3.3)
    freq = {}
    for lb in dfs:
        v = _vector(dfs[lb], A_MIN_MAIN, 'all', 'n_lo_gia_moi', image_order)
        freq[lb] = {'mean_new_per_image': float(v.mean()), 'total_new': int(v.sum())}
    verdict = decide_54(res)
    s19 = freq.get('static_s19_36k')
    if s19 is None:
        freq_status = 'CHUA XAC DINH (thieu static_s19_36k)'
        rare = None
    else:
        rare = s19['mean_new_per_image'] < FREQ_GATE
        freq_status = 'loi loai 5 hiem' if rare else 'khong hiem'
    summary = {
        'n_images': n, 'n_boot': args.n_boot, 'seed': SEED, 'a_mins': a_mins, 'a_min_main': A_MIN_MAIN,
        'verdict_metric': VERDICT_METRIC, 'verdict_54': verdict,
        'frequency_gate': {'threshold_new_per_image': FREQ_GATE, 'status': freq_status, 'rare': rare,
                           'per_checkpoint': freq},
        'run_part_c': verdict['key'] not in ('none', 'unavailable'),
        'run_part_b2': (rare is False),
        'labels_present': list(dfs), 'labels_missing': [lb for lb in labels if lb not in dfs],
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_commit': get_git_commit(),
    }
    write_json(summary, os.path.join(args.output_dir, 'pairs_bootstrap.json'))
    write_pairs_md(res, summary, os.path.join(args.output_dir, 'pairs_bootstrap.md'))
    print(f"Verdict 5.4: {verdict['text']}")
    print(f"Cong tan suat 3.3: {freq_status}")
    print(f"Da ghi pairs_bootstrap.csv/.md/.json trong {args.output_dir}")


def _missing_row(pid, desc, a_min, cls, metric, need):
    return {'pair_id': pid, 'description': desc, 'a_min': a_min, 'class': cls, 'metric': metric,
            'observed': np.nan, 'mean_a': np.nan, 'mean_b': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
            'p_lt0': np.nan, 'interseed_amp_rule': '', 'interseed_amp': np.nan,
            'ci_excludes_0': False, 'confirmed': False, 'status': f'thieu {need}'}


def _pair_row(pid, desc, a_min, cls, metric, obs, boot, amp_rule, amp, mean_a, mean_b):
    lo, hi = _ci(boot)
    excl = (lo > 0) or (hi < 0)
    if amp_rule == 'max(N1,N2)':
        av = [amp[k] for k in ('N1', 'N2') if k in amp]
        amp_v = max(av) if av else np.nan
    elif amp_rule in ('N1', 'N2'):
        amp_v = amp.get(amp_rule, np.nan)
    else:
        amp_v = np.nan
    confirmed = bool(excl and (not np.isnan(amp_v)) and abs(obs) > amp_v) if amp_rule else False
    return {'pair_id': pid, 'description': desc, 'a_min': a_min, 'class': cls, 'metric': metric,
            'observed': obs, 'mean_a': mean_a, 'mean_b': mean_b, 'ci_lo': lo, 'ci_hi': hi,
            'p_lt0': float(np.mean(boot < 0)), 'interseed_amp_rule': amp_rule or '',
            'interseed_amp': amp_v, 'ci_excludes_0': bool(excl), 'confirmed': confirmed,
            'status': 'ok' if (amp_rule is None or not np.isnan(amp_v)) else 'thieu bien do lien-seed'}


def decide_54(res):
    """Bang 5.4 NGUYEN VAN, tren VERDICT_METRIC, A_min chinh, class 'all'."""
    sub = res[(res['a_min'] == A_MIN_MAIN) & (res['class'] == 'all') & (res['metric'] == VERDICT_METRIC)]
    get = {r['pair_id']: r for _, r in sub.iterrows()}
    need = ['H1', 'H2', "H2'", 'H3', "H3'", 'H5', "H5'"]
    missing = [p for p in need if p not in get or get[p]['status'] != 'ok']
    if missing:
        return {'key': 'unavailable', 'text': f'CHUA AP DUOC bang 5.4 — thieu/khong du cap: {missing}',
                'pairs': {}}

    def up(p):     # xac nhan TANG (quy tac kep + dau duong)
        r = get[p]
        return bool(r['confirmed'] and r['observed'] > 0)

    def conf(p):
        return bool(get[p]['confirmed'])

    h5_both = bool(get['H5']['ci_excludes_0'] and get["H5'"]['ci_excludes_0'])
    flags = {p: {'confirmed': conf(p), 'confirmed_increase': up(p)} for p in ['H1', 'H2', "H2'", 'H3', "H3'", 'H4']}
    flags['H5_ci_excludes_0_both_seeds'] = h5_both
    matched = []
    if up('H2') and up("H2'") and up('H3') and up("H3'") and not conf('H1'):
        matched.append(('affinity', 'Loi 5 do affinity, doc lap voi BCE'))
    if up('H1') and not conf('H2') and not conf("H2'"):
        matched.append(('bce', 'Loi 5 do BCE'))
    only_h3 = (conf('H3') and conf("H3'") and not conf('H1') and not conf('H2') and not conf("H2'"))
    if only_h3 and h5_both:
        matched.append(('interaction', 'Loi 5 la hieu ung tuong tac — cung ho voi Nhanh R'))
    any_conf = any(conf(p) for p in ['H1', 'H2', "H2'", 'H3', "H3'", 'H4']) or h5_both
    if not any_conf:
        matched.append(('none', 'Loi 5 khong do loss; vi du bon chua la truong hop rieng le hoac co o ca Baseline'))
    if not matched:
        return {'key': 'unmatched', 'text': 'Khong khop dong nao cua bang 5.4 — xem bang cap de doc mau ket qua',
                'pairs': flags}
    return {'key': matched[0][0], 'text': '; '.join(t for _k, t in matched),
            'all_matched': [k for k, _t in matched], 'pairs': flags}


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '—'
    return f'{x:.{nd}f}'


def write_pairs_md(res, summary, path):
    L = []
    L.append('# Phan B — Lo gia (loi loai 5): bootstrap ghep cap')
    L.append('')
    L.append(f"A_min chinh = {A_MIN_MAIN} px (phu: {[a for a in summary['a_mins'] if a != A_MIN_MAIN]}) · "
             f"bootstrap ghep cap per-image {summary['n_boot']} lan, rng {SEED} · {summary['n_images']} anh · "
             f"tao {summary['created_at']} · git {summary['git_commit'][:10]}")
    L.append('')
    L.append('## Verdict bang 5.4')
    L.append('')
    L.append(f"**{summary['verdict_54']['text']}**")
    L.append('')
    L.append(f"## Cong tan suat 3.3")
    L.append('')
    fg = summary['frequency_gate']
    L.append(f"Nguong: < {FREQ_GATE} lo gia moi/anh o Static s19@36k, A_min={A_MIN_MAIN} → **{fg['status']}**")
    L.append('')
    L.append('| Checkpoint | Lo gia moi / anh | Tong lo gia moi |')
    L.append('|---|---|---|')
    for lb, v in fg['per_checkpoint'].items():
        L.append(f"| {lb} | {v['mean_new_per_image']:.4f} | {v['total_new']} |")
    L.append('')
    L.append(f"Chay Phan C: **{'co' if summary['run_part_c'] else 'khong'}** · "
             f"Chay Phan B2: **{'co' if summary['run_part_b2'] else 'khong'}**")
    L.append('')
    L.append('## Quy uoc')
    L.append('')
    L.append(f"- Dai luong cua bang 5.4: `{VERDICT_METRIC}` (so lo gia / anh, lop gop 'all'). Ly do: "
             "`n_lo_gia_moi` duoc dinh nghia so voi Baseline nen Baseline luon = 0 — chi dung cho cong tan suat.")
    L.append('- "Xac nhan" = CI 95% loai tru 0 VA |Δ| > bien do lien-seed (N1 cho cap Static, N2 cho cap AffOnly; '
             'H1 va H5 khong co cap lap seed rieng → dung max(N1, N2)). Cot H5 trong bang 5.4 chi doi CI loai tru 0.')
    L.append('- Δ = trung binh per-image (A) − (B).')
    L.append('')
    for a_min in summary['a_mins']:
        L.append(f"## Bang cap — A_min = {a_min}, lop 'all'")
        L.append('')
        L.append('| Cap | Mo ta | Dai luong | Δ quan sat | CI 95% | P(Δ<0) | Bien do | Xac nhan |')
        L.append('|---|---|---|---|---|---|---|---|')
        sub = res[(res['a_min'] == a_min) & (res['class'] == 'all')]
        for _, r in sub.iterrows():
            if r['status'].startswith('thieu ') and np.isnan(r['observed']):
                L.append(f"| {r['pair_id']} | {r['description']} | {r['metric']} | — | — | — | — | {r['status']} |")
                continue
            ci = f"[{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
            amp = f"{_fmt(r['interseed_amp'])} ({r['interseed_amp_rule']})" if r['interseed_amp_rule'] else '—'
            L.append(f"| {r['pair_id']} | {r['description']} | {r['metric']} | {_fmt(r['observed'])} | {ci} | "
                     f"{_fmt(r['p_lt0'], 3)} | {amp} | {'**CO**' if r['confirmed'] else 'khong'} |")
        L.append('')
    L.append('Bang theo tung lop: xem `pairs_bootstrap.csv` (cot `class`).')
    L.append('')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# ═════════════════════════════════ fill-test (B2) ═════════════════════════════

def fill_small_holes(pred, a_fill):
    """Lap MOI lo (moi lop) co dien tich <= a_fill bang nhan lop bao quanh; khong dung GT.
    Lo lon to truoc, lo nho to sau (lo long nhau: lo trong cung thang)."""
    out = pred.copy()
    jobs = []
    for c in range(NUM_CLASSES):
        holes, _ = detect_holes_class(pred == c, 1)
        for h in holes:
            if h['area'] <= a_fill:
                jobs.append((h['area'], c, h))
    jobs.sort(key=lambda t: -t[0])
    for _a, c, h in jobs:
        region = out[h['sl']]
        region[h['hmask']] = c
    return out


def cmd_fill_test(args):
    import torch
    from Tools.per_image_dump import compute_per_image_stats
    from Tools.oracle_boundary_ceiling import load_cfg, build_val_dataset, image_ids_of, labels_to_pseudo_logits
    from Tools.bootstrap_boundary_ci import aggregate_point_estimate, run_bootstrap, summarize_pair

    pb = os.path.join(args.output_dir, 'pairs_bootstrap.json')
    if os.path.exists(pb) and not args.force:
        s = read_json(pb)
        if s.get('run_part_b2') is False:
            print(f"pairs_bootstrap.json: cong tan suat = '{s['frequency_gate']['status']}' -> Phan B2 KHONG chay "
                  f"(muc 3.4). Dung --force neu van muon chay.")
            return
    masks = parse_label_values(args.masks, '--masks')
    check_labels_known(masks, '--masks')
    a_fills = [int(x) for x in args.a_fill.split(',')]
    cfg = load_cfg(args.config, args.data_root)
    val_ds = build_val_dataset(cfg)
    image_ids = image_ids_of(val_ds)
    n = len(image_ids) if args.max_images is None else min(args.max_images, len(image_ids))

    rows_out = []
    for label, mdir in masks.items():
        variants = {'before': []}
        variants.update({f'fill{a}': [] for a in a_fills})
        t0 = time.time()
        for i in range(n):
            name, gt = load_gt_cached(val_ds, i, args.gt_cache or None)
            gt = np.asarray(gt).astype(np.int64)
            pred = load_pred_png(mdir, name)
            tgt = torch.from_numpy(gt)[None]
            for key in variants:
                p = pred if key == 'before' else fill_small_holes(pred, int(key[4:]))
                r, _ = compute_per_image_stats(labels_to_pseudo_logits(p, NUM_CLASSES), tgt, [name], CLASS_NAMES,
                                               IGNORE_INDEX, 4, 0, (1, 2, 4), 2)
                variants[key].extend(r)
            print(f"  [{label}] {i + 1}/{n} ({time.time() - t0:.0f}s)", end='\r')
        print()
        dfs = {k: pd.DataFrame(v) for k, v in variants.items()}
        obs_before = aggregate_point_estimate(dfs['before'], 'micro', 'skip')
        q = ['mIoU_9', 'mIoU_8', 'bf_precision', 'bf_recall', 'bf_score', 'asd_pred_to_gt', 'asd_gt_to_pred']
        for a in a_fills:
            key = f'fill{a}'
            obs_after = aggregate_point_estimate(dfs[key], 'micro', 'skip')
            va, vb = run_bootstrap(dfs[key], dfs['before'], 'micro', 'skip', args.n_boot, SEED)
            for r in summarize_pair(va, vb, obs_after, obs_before, quantities=q):
                rows_out.append({'label': label, 'a_fill': a, 'quantity': r['quantity'],
                                 'before': obs_before[r['quantity']], 'after': obs_after[r['quantity']],
                                 'delta': r['observed_delta'], 'ci_lo': r['ci_lo'], 'ci_hi': r['ci_hi'],
                                 'p_delta_lt_0': r['p_delta_lt_0'],
                                 'ci_excludes_0': bool(r['ci_lo'] > 0 or r['ci_hi'] < 0)})
    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(args.output_dir, 'fill_test.csv'), index=False)
    L = ['# Phan B2 — Thu hau xu ly lap lo (khong dung GT)', '',
         f"Lap moi lo co dien tich <= A_fill ∈ {a_fills}; do lai bang Tools/per_image_dump.compute_per_image_stats "
         f"+ Tools/bootstrap_boundary_ci (quy uoc micro). Bootstrap ghep cap {args.n_boot} lan, rng {SEED}. "
         f"precision/recall = BF precision/recall (theta=2px). Δ = sau − truoc.", '']
    for label in masks:
        L.append(f'## {label}')
        L.append('')
        L.append('| A_fill | Dai luong | Truoc | Sau | Δ | CI 95% | CI loai 0 |')
        L.append('|---|---|---|---|---|---|---|')
        for _, r in df[df['label'] == label].iterrows():
            L.append(f"| {r['a_fill']} | {r['quantity']} | {_fmt(r['before'])} | {_fmt(r['after'])} | "
                     f"{_fmt(r['delta'], 5)} | [{_fmt(r['ci_lo'], 5)}, {_fmt(r['ci_hi'], 5)}] | "
                     f"{'co' if r['ci_excludes_0'] else 'khong'} |")
        L.append('')
    with open(os.path.join(args.output_dir, 'fill_test.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f"Da ghi fill_test.csv/.md trong {args.output_dir}")


# ═════════════════════════════════ CLI ════════════════════════════════════════

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, need_data=True):
        p.add_argument('--output-dir', default='output/spurious_holes')
        if need_data:
            p.add_argument('--config', default=CFG_STATIC, help='YAML de doc DATASET (val split dung chung)')
            p.add_argument('--data-root', default=None)
            p.add_argument('--gt-cache', default='output/_cache/gt1024', help="cache GT .npy ('' = tat)")
            p.add_argument('--max-images', type=int, default=None, help='dry-run: chi N anh dau')
            p.add_argument('--force', action='store_true')

    p = sub.add_parser('selftest', help='Cong 6')
    common(p, need_data=False)

    p = sub.add_parser('count', help='Dem lo theo checkpoint')
    common(p)
    p.add_argument('--masks', action='append', required=True, metavar='LABEL=DIR',
                   help='Thu muc mask PNG da dump; lap lai. BAT BUOC co baseline_40k.')
    p.add_argument('--per-image-stats', action='append', default=[], metavar='LABEL=CSV',
                   help='(tuy chon) per_image_stats.csv cua dump — cong 7 kiem danh sach + thu tu anh')
    p.add_argument('--a-min', type=int, default=A_MIN_MAIN)
    p.add_argument('--a-min-extra', default=','.join(str(a) for a in A_MIN_EXTRA))

    p = sub.add_parser('compare', help='Bootstrap H1..H5, N1, N2 + verdict 5.4')
    common(p, need_data=False)
    p.add_argument('--n-boot', type=int, default=10000)

    p = sub.add_parser('fill-test', help='Phan B2')
    common(p)
    p.add_argument('--masks', action='append', required=True, metavar='LABEL=DIR')
    p.add_argument('--a-fill', default='64,256,1024')
    p.add_argument('--n-boot', type=int, default=10000)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)
    {'selftest': cmd_selftest, 'count': cmd_count, 'compare': cmd_compare,
     'fill-test': cmd_fill_test}[args.cmd](args)


if __name__ == '__main__':
    main()
