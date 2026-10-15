"""Tools/hole_audit_sample.py — K2-b cua docs/K2-ke-hoach-kiem-lo-gia-va-quyet-dinh-huong-giai-phap.md:
chon mau lo gia de KIEM BANG MAT + sinh crop + file cham trong.

CPU, khong can checkpoint. Doc mask du doan DA DUMP cua Static s19@36k + GT val + anh RGB val
(cung get_val_transforms -> 1024x1024, nen RGB, GT, mask trung hinh hoc).

Quan the (§2.1): lo gia (is_spurious = 1, A_min = 64) cua Static s19@36k — dinh nghia y het
Tools/spurious_holes.holes_for_image (Buoc 1). Cong K-b0: dem theo lop phai KHOP tuyet doi
con so cua Buoc 1 (8.836 lo; Rangeland 3.458, Developed 2.767, Tree 1.081, Building 938,
Agriculture 320, Khac 272) — va khop holes_static_s19_36k.csv neu truyen --ref-holes.

Phan tang: 6 nhom lop x 3 nhom kich thuoc (64–149 · 150–699 · >=700 px) = 18 o; 7 lo/o (o it hon
thi lay het); them 20 muc lap an (rut tu chinh mau); xao thu tu; danh so H001... Tat ca bang
np.random.default_rng(19), theo thu tu co dinh:
  1. quan the sap theo (image_id, class_id, hole_id); o duyet theo thu tu GROUPS x SIZE_BINS;
     moi o: rng.choice(n_o, min(7, n_o), replace=False) (chi so trong o, giu thu tu tang).
  2. lap an: rng.choice(n_mau, 20, replace=False).
  3. xao: rng.permutation(n_mau + 20) -> H001...

Thu tu ghi (§2.1 "chot truoc khi ai xem crop"): audit_population.csv + audit_sample.csv TRUOC,
roi moi sinh audit_crops/H###.png. Neu audit_sample.csv da co -> KHONG rut lai (chi sinh lai crop
con thieu); --force-resample moi rut lai. audit_sample_rating.csv KHONG BAO GIO bi ghi de.

Crop (§2.2): cua so vuong = bbox lo no x3 (toi thieu 128 px full-res), phong len 400 px; 2 o:
RGB goc | RGB + vien do quanh lo. Tieu de CHI co ma muc va ten lop c — khong GT, mask, model,
frac_gt_c, kich thuoc.

Output (mac dinh output/hole_oracle/): audit_population.csv, audit_sample.csv (NGUOI CHAM KHONG XEM),
audit_crops/H###.png + audit_crops/HUONG_DAN_CHAM.md, audit_sample_rating.csv, gates_audit.json.
Huong dan Kaggle: docs/huong-dan-kaggle-K2.md
"""

import argparse
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (
    CLASS_NAMES, SEED, CFG_STATIC, get_git_commit, write_json, append_log, load_gt_cached,
)

LABEL = 'static_s19_36k'
DUMPNAME = 'run3_static_s19_iter36000'
A_MIN = 64
GROUPS = [   # (ten nhom, lop)
    ('Rangeland', ('Rangeland',)),
    ('Developed', ('Developed',)),
    ('Tree', ('Tree',)),
    ('Building', ('Building',)),
    ('Agriculture', ('Agriculture',)),
    ('Khac', ('Road', 'Bareland', 'Water', 'Background')),
]
SIZE_BINS = [('64-149', 64, 149), ('150-699', 150, 699), ('>=700', 700, None)]
PER_CELL = 7
N_DUP = 20
EXPECTED_POP = {'Rangeland': 3458, 'Developed': 2767, 'Tree': 1081, 'Building': 938, 'Agriculture': 320, 'Khac': 272}
WINDOW_MULT = 3
WINDOW_MIN = 128
PANEL = 400
OUTLINE_PX = 2
RATING_COLS = ['item_id', 'nhan', 'vat', 'ghi_chu']


def class_group(cname):
    for g, members in GROUPS:
        if cname in members:
            return g
    raise ValueError(cname)


def size_bin(area):
    for name, lo, hi in SIZE_BINS:
        if area >= lo and (hi is None or area <= hi):
            return name
    raise ValueError(area)


# ═════════════════════════════ Quan the ═══════════════════════════════════════

def spurious_holes_image(task):
    """task = (mask_dir, name, gt_path) -> list dict lo gia (A_min = 64) cua anh."""
    from Tools.spurious_holes import holes_for_image, load_pred_png
    mask_dir, name, gt_path = task
    gt = np.load(gt_path).astype(np.int64)
    pred = load_pred_png(mask_dir, name)
    assert pred.shape == gt.shape, f"{name}: mask {pred.shape} != GT {gt.shape}"
    out = []
    for h in holes_for_image(pred, gt, A_MIN):
        if not h['is_spurious'] or h['area'] < A_MIN:
            continue
        ys, xs = np.nonzero(h['hmask'])
        y0, x0 = int(ys.min() + h['sl'][0].start), int(xs.min() + h['sl'][1].start)
        y1, x1 = int(ys.max() + h['sl'][0].start + 1), int(xs.max() + h['sl'][1].start + 1)
        cname = CLASS_NAMES[h['class_id']]
        out.append({'image_id': name, 'class': cname, 'class_id': h['class_id'], 'hole_id': h['hole_id'],
                    'area': h['area'], 'frac_gt_c': round(h['frac_gt_c'], 6),
                    'bbox': f'{y0}:{x0}:{y1}:{x1}',
                    'class_group': class_group(cname), 'size_bin': size_bin(h['area'])})
    return out


def build_population(mask_dir, image_ids, gt_paths, workers):
    tasks = [(mask_dir, n, p) for n, p in zip(image_ids, gt_paths)]
    rows, t0 = [], time.time()
    if workers > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(workers) as pool:
            for k, r in enumerate(pool.imap(spurious_holes_image, tasks, chunksize=2)):
                rows += r
                print(f"  quan the: {k + 1}/{len(tasks)} anh ({time.time() - t0:.0f}s)", end='\r')
    else:
        for k, t in enumerate(tasks):
            rows += spurious_holes_image(t)
            print(f"  quan the: {k + 1}/{len(tasks)} anh ({time.time() - t0:.0f}s)", end='\r')
    print()
    pop = pd.DataFrame(rows, columns=['image_id', 'class', 'class_id', 'hole_id', 'area', 'frac_gt_c', 'bbox',
                                      'class_group', 'size_bin'])
    return pop.sort_values(['image_id', 'class_id', 'hole_id'], kind='stable').reset_index(drop=True)


def gate_population(pop, ref_holes, full):
    """K-b0: so lo gia theo nhom lop khop Buoc 1."""
    got = {g: int((pop['class_group'] == g).sum()) for g, _ in GROUPS}
    checks = [{'group': g, 'expected': EXPECTED_POP[g], 'got': got[g], 'pass': got[g] == EXPECTED_POP[g]}
              for g, _ in GROUPS]
    detail = f"{len(pop)} lo gia (ky vong {sum(EXPECTED_POP.values())})"
    if ref_holes:
        ref = pd.read_csv(ref_holes)
        ref = ref[(ref['a_min'] == A_MIN) & (ref['is_spurious'] == 1)]
        a = set(zip(pop['image_id'].astype(str), pop['class'], pop['hole_id'].astype(int), pop['area'].astype(int)))
        b = set(zip(ref['image_id'].astype(str), ref['class'], ref['hole_id'].astype(int), ref['area'].astype(int)))
        checks.append({'group': f'tap lo == {os.path.basename(ref_holes)}', 'expected': len(b), 'got': len(a),
                       'pass': a == b, 'only_here': len(a - b), 'only_ref': len(b - a)})
        detail += f"; doi chieu {os.path.basename(ref_holes)}: {'trung' if a == b else 'KHAC'}"
    if not full:
        return {'gate': 'K-b0', 'status': 'SKIP_DRYRUN', 'checks': checks, 'detail': detail + ' (dry-run)'}
    ok = all(c['pass'] for c in checks)
    return {'gate': 'K-b0', 'status': 'PASS' if ok else 'FAIL', 'checks': checks, 'detail': detail}


# ═════════════════════════════ Chon mau ═══════════════════════════════════════

def population_table(pop):
    rows = []
    for g, _ in GROUPS:
        for b, _lo, _hi in SIZE_BINS:
            n = int(((pop['class_group'] == g) & (pop['size_bin'] == b)).sum())
            rows.append({'stratum': f'{g}|{b}', 'class_group': g, 'size_bin': b, 'N_pop': n,
                         'n_sample': min(PER_CELL, n)})
    return pd.DataFrame(rows)


def draw_sample(pop, seed=SEED):
    """Tra ve DataFrame audit_sample (thu tu = thu tu cham, item_id H001...)."""
    rng = np.random.default_rng(seed)
    picked = []
    for g, _ in GROUPS:
        for b, _lo, _hi in SIZE_BINS:
            cell = pop[(pop['class_group'] == g) & (pop['size_bin'] == b)]
            if len(cell) == 0:
                continue
            idx = np.sort(rng.choice(len(cell), min(PER_CELL, len(cell)), replace=False))
            for i in idx:
                r = cell.iloc[int(i)].to_dict()
                r['stratum'] = f'{g}|{b}'
                picked.append(r)
    n_main = len(picked)
    dup_src = rng.choice(n_main, min(N_DUP, n_main), replace=False)
    items = [dict(r, _src=k, _dup=None) for k, r in enumerate(picked)]
    items += [dict(picked[int(k)], _src=int(k), _dup=int(k)) for k in dup_src]
    order = rng.permutation(len(items))
    items = [items[int(i)] for i in order]
    width = max(3, len(str(len(items))))
    first_id = {}
    for pos, it in enumerate(items):
        it['item_id'] = f'H{pos + 1:0{width}d}'
    for it in items:
        if it['_dup'] is None:
            first_id[it['_src']] = it['item_id']
    rows = []
    for it in items:
        rows.append({'item_id': it['item_id'], 'image_id': it['image_id'], 'class': it['class'],
                     'hole_id': it['hole_id'], 'area': it['area'], 'bbox': it['bbox'], 'frac_gt_c': it['frac_gt_c'],
                     'stratum': it['stratum'], 'class_group': it['class_group'], 'size_bin': it['size_bin'],
                     'duplicate_of': first_id[it['_src']] if it['_dup'] is not None else ''})
    return pd.DataFrame(rows)


# ═════════════════════════════════ Crop ═══════════════════════════════════════

def crop_window(bbox, shape, mult=WINDOW_MULT, min_side=WINDOW_MIN):
    """Cua so vuong, tam = tam bbox, canh = max(mult * canh dai bbox, min_side), dich vao trong anh."""
    y0, x0, y1, x1 = bbox
    H, W = shape
    side = int(min(max(mult * max(y1 - y0, x1 - x0), min_side), H, W))
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    top = int(round(cy - side / 2.0))
    left = int(round(cx - side / 2.0))
    top = min(max(top, 0), H - side)
    left = min(max(left, 0), W - side)
    return top, left, side


def hole_mask_full(pred, class_id, hole_id):
    """Lay lai mask lo (toan anh) tu mask du doan — dung detect_holes_class nhu luc dem."""
    from Tools.spurious_holes import detect_holes_class
    holes, _ = detect_holes_class(pred == class_id, A_MIN)
    for h in holes:
        if h['hole_id'] == hole_id:
            m = np.zeros(pred.shape, dtype=bool)
            m[h['sl']] = h['hmask']
            return m
    raise KeyError(f'khong tim lai duoc lo class={class_id} hole_id={hole_id}')


def render_crop(rgb, hole_mask, bbox, title, panel=PANEL):
    from PIL import Image, ImageDraw, ImageFont
    from scipy.ndimage import binary_dilation
    top, left, side = crop_window(bbox, hole_mask.shape)
    sub = rgb[top:top + side, left:left + side]
    m = hole_mask[top:top + side, left:left + side]
    rgb_up = np.array(Image.fromarray(sub).resize((panel, panel), Image.LANCZOS))
    m_up = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((panel, panel), Image.NEAREST)) > 0
    ring = binary_dilation(m_up, iterations=OUTLINE_PX) & ~m_up
    over = rgb_up.copy()
    over[ring] = (255, 0, 0)
    gap, head = 12, 34
    canvas = Image.new('RGB', (2 * panel + gap, panel + head), (255, 255, 255))
    canvas.paste(Image.fromarray(rgb_up), (0, head))
    canvas.paste(Image.fromarray(over), (panel + gap, head))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 20)
    except OSError:
        try:
            font = ImageFont.load_default(size=20)
        except TypeError:
            font = ImageFont.load_default()
    draw.text((8, 6), title, fill=(0, 0, 0), font=font)
    return canvas


RATING_GUIDE = """# Huong dan cham (K2-b) — doc truoc khi mo crop

Moi file `H###.png` co 2 o: **trai** = anh that; **phai** = cung anh, vien DO bao quanh mot vung.
Tieu de chi ghi ma muc va **lop c** (lop bao quanh vung do).

**Cau hoi:** Ben trong vien do, **tren anh thuc**, co vat hoac lop phu **khac lop c** chiem phan lon vung khong?

| Nhan | Nghia | Vi du |
|---|---|---|
| **A** — mo hinh sai | Vung trong vien dong nhat voi lop c | Co dong nhat, ruong dong nhat; **bong do tren nen cung lop van la A** |
| **B** — nhan tho | Co vat that khac lop ma nhan khong ve | Cay don le, mai nha nho, vung nuoc, mang dat trong |
| **C** — khong phan dinh | Anh khong du de quyet | Qua mo, qua nho so voi do phan giai, may/nhieu |

Ghi vao `audit_sample_rating.csv`, moi dong 1 muc:
- `nhan`: A, B hoac C
- `vat` (chi khi B): cay / nha / nuoc / dat trong / duong / khac
- `ghi_chu`: tu do (co the de trong)

KHONG mo `audit_sample.csv` (khoa mau) cho toi khi cham xong. Mot so muc co the trong giong nhau — cu cham doc lap tung muc.
"""


def write_rating_template(sample, path):
    if os.path.exists(path):
        print(f"  {os.path.basename(path)} da co — KHONG ghi de.")
        return False
    df = pd.DataFrame({'item_id': sample['item_id'], 'nhan': '', 'vat': '', 'ghi_chu': ''}, columns=RATING_COLS)
    df.to_csv(path, index=False, encoding='utf-8')
    return True


def make_crops(sample, val_ds, image_ids, mask_dir, crop_dir, force=False):
    from Tools.spurious_holes import load_pred_png
    from src.utils.visualizer import denormalize
    os.makedirs(crop_dir, exist_ok=True)
    with open(os.path.join(crop_dir, 'HUONG_DAN_CHAM.md'), 'w', encoding='utf-8') as f:
        f.write(RATING_GUIDE)
    idx_of = {n: i for i, n in enumerate(image_ids)}
    made, t0 = 0, time.time()
    todo = sample if force else sample[[not os.path.exists(os.path.join(crop_dir, f'{i}.png')) for i in sample['item_id']]]
    for k, (img, grp) in enumerate(todo.groupby('image_id', sort=True)):
        image_t, _mask = val_ds[idx_of[img]]
        rgb = denormalize(image_t.permute(1, 2, 0).numpy())
        pred = load_pred_png(mask_dir, img)
        assert pred.shape == rgb.shape[:2], f"{img}: mask {pred.shape} != RGB {rgb.shape}"
        for _, r in grp.iterrows():
            cid = CLASS_NAMES.index(r['class'])
            hm = hole_mask_full(pred, cid, int(r['hole_id']))
            assert int(hm.sum()) == int(r['area']), f"{r['item_id']}: dien tich lo khac audit_sample.csv"
            bbox = tuple(int(x) for x in str(r['bbox']).split(':'))
            render_crop(rgb, hm, bbox, f"{r['item_id']}   |   lop c = {r['class']}").save(
                os.path.join(crop_dir, f"{r['item_id']}.png"))
            made += 1
        print(f"  crop: {made}/{len(todo)} ({time.time() - t0:.0f}s)", end='\r')
    print()
    return made


# ═════════════════════════════════ main ═══════════════════════════════════════

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', default='output/hole_oracle')
    ap.add_argument('--dump-root', default=None, help=f'thu muc dump/ (chua {DUMPNAME}/masks)')
    ap.add_argument('--masks', default=None, help='ghi de thu muc mask PNG cua Static s19@36k')
    ap.add_argument('--config', default=CFG_STATIC)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--gt-cache', default='output/_cache/gt1024')
    ap.add_argument('--ref-holes', default=None, help='(tuy chon) holes_static_s19_36k.csv cua Buoc 1 -> K-b0 doi chieu tap lo')
    ap.add_argument('--workers', type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument('--max-images', type=int, default=None, help='dry-run: chi N anh dau (K-b0 -> SKIP)')
    ap.add_argument('--allow-count-mismatch', action='store_true', help='van rut mau khi K-b0 FAIL (KHONG khuyen)')
    ap.add_argument('--force-resample', action='store_true', help='rut LAI mau du audit_sample.csv da co')
    ap.add_argument('--force-crops', action='store_true', help='ve lai moi crop')
    args = ap.parse_args(argv)

    from Tools.oracle_boundary_ceiling import load_cfg, build_val_dataset, image_ids_of
    from Tools.spurious_holes import _gate7_image_list
    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    mask_dir = args.masks or (os.path.join(args.dump_root, DUMPNAME, 'masks') if args.dump_root else None)
    if not mask_dir or not os.path.isdir(mask_dir):
        raise SystemExit(f"Khong thay thu muc mask Static s19@36k: {mask_dir} — truyen --dump-root hoac --masks")
    cfg = load_cfg(args.config, args.data_root)
    val_ds = build_val_dataset(cfg)
    image_ids = image_ids_of(val_ds)
    full = args.max_images is None
    n = len(image_ids) if full else min(args.max_images, len(image_ids))
    if full:
        _gate7_image_list(LABEL, mask_dir, image_ids, None)
    append_log(os.path.join(out, 'run_log.txt'),
               f"[{datetime.datetime.now().isoformat(timespec='seconds')}] audit-sample n={n} mask={mask_dir} "
               f"git={get_git_commit()} argv={' '.join(sys.argv)}")

    sample_path = os.path.join(out, 'audit_sample.csv')
    gate = None
    if os.path.exists(sample_path) and not args.force_resample:
        sample = pd.read_csv(sample_path, keep_default_na=False)
        print(f"  audit_sample.csv da co ({len(sample)} muc) — giu nguyen mau da chot, chi sinh crop con thieu.")
    else:
        t0 = time.time()
        gt_paths = []
        for i in range(n):
            name, _gt = load_gt_cached(val_ds, i, args.gt_cache)
            gt_paths.append(os.path.join(args.gt_cache, f'{name}.npy'))
            print(f"  GT cache {i + 1}/{n} ({time.time() - t0:.0f}s)", end='\r')
        print()
        pop = build_population(mask_dir, image_ids[:n], gt_paths, args.workers)
        gate = gate_population(pop, args.ref_holes, full)
        for c in gate['checks']:
            print(f"    {'ok ' if c['pass'] else 'LECH'} {c['group']:<28s} ky vong {c['expected']:>6} — co {c['got']:>6}")
        print(f"  [{gate['status']}] K-b0: {gate['detail']}")
        write_json({'gate_kb0': gate, 'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
                    'git_commit': get_git_commit(), 'n_images': n, 'mask_dir': mask_dir},
                   os.path.join(out, 'gates_audit.json'))
        if gate['status'] == 'FAIL' and not args.allow_count_mismatch:
            raise SystemExit("K-b0 FAIL: quan the lo gia khac Buoc 1 -> KHONG rut mau (sai dump / sai dataset?). "
                             "Xem gates_audit.json; --allow-count-mismatch neu chac chan muon tiep.")
        ptab = population_table(pop)
        ptab.to_csv(os.path.join(out, 'audit_population.csv'), index=False)
        sample = draw_sample(pop)
        sample.to_csv(sample_path, index=False)          # chot mau TRUOC khi sinh crop
        print(f"  Da chot mau: {len(sample)} muc ({(sample['duplicate_of'] != '').sum()} lap an) -> {sample_path}")

    made = make_crops(sample, val_ds, image_ids, mask_dir, os.path.join(out, 'audit_crops'), args.force_crops)
    write_rating_template(sample, os.path.join(out, 'audit_sample_rating.csv'))
    n_png = len([f for f in os.listdir(os.path.join(out, 'audit_crops')) if f.endswith('.png')])
    print(f"Xong K2-b: {len(sample)} muc, crop moi ve {made}, tong {n_png} PNG trong audit_crops/")
    if n_png < len(sample):
        sys.exit(1)


if __name__ == '__main__':
    main()
