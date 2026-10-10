"""Tools/dump_archive.py — Dong goi thu muc dump (mask du doan + per_image_stats.csv) vao repo git
de clone tren Kaggle, KHONG can upload dump thanh Kaggle Dataset rieng.

GitHub tu choi file > 100 MB, nen dump.zip (~175 MB) duoc CHIA thanh cac phan nho (mac dinh 45 MB)
commit binh thuong trong data/dump_parts/, kem manifest.json (kich thuoc + sha256 tung phan va ca file).

  split   (chay tren may ban, TRUOC khi git push)
          python Tools/dump_archive.py split --zip data/dump.zip --out-dir data/dump_parts
  join    (chay tren Kaggle, tu dong boi scripts/run_grad_conflict_kaggle.sh)
          python Tools/dump_archive.py join --parts-dir data/dump_parts --dest /tmp/dump_repo
          -> ghep, kiem sha256, giai nen, kiem 7 dump can dung; dong cuoi in: DUMP_ROOT=<thu muc>
          Chay lai: neu da giai nen dung ban (cung sha256) thi bo qua.
  check   kiem 7 dump can dung trong 1 thu muc da giai nen: python Tools/dump_archive.py check --root <dir>
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile

NEEDED = {
    'static_s19_36k': 'run3_static_s19_iter36000',
    'static_s86_36k': 'run5_static_s86_best_iter36000',
    'static_s86_40k': 'run5_static_s86_final_iter40000',
    'bce04_40k': 'run2b_bce04_iter40000',
    'affonly_s19_40k': 'run7_affinity_only_seed19_best',
    'affonly_s86_40k': 'run7_affinity_only_seed86_best',
    'baseline_40k': 'run1_baseline_iter40000',
}
ANCHOR = 'run3_static_s19_iter36000'
CHUNK = 1 << 20


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(CHUNK), b''):
            h.update(b)
    return h.hexdigest()


def cmd_split(args):
    src = args.zip
    if not os.path.exists(src):
        sys.exit(f"Khong thay {src}")
    with zipfile.ZipFile(src) as z:
        bad = z.testzip()
        if bad:
            sys.exit(f"File zip hong tai: {bad}")
    os.makedirs(args.out_dir, exist_ok=True)
    for f in os.listdir(args.out_dir):
        if f.startswith('dump.zip.part') or f == 'manifest.json':
            os.remove(os.path.join(args.out_dir, f))
    size = args.part_mb * (1 << 20)
    parts = []
    with open(src, 'rb') as f:
        k = 0
        while True:
            data = f.read(size)
            if not data:
                break
            name = f'dump.zip.part{k:03d}'
            with open(os.path.join(args.out_dir, name), 'wb') as g:
                g.write(data)
            parts.append({'name': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            k += 1
    man = {'source_name': os.path.basename(src), 'total_bytes': os.path.getsize(src),
           'sha256': sha256_file(src), 'part_mb': args.part_mb, 'parts': parts}
    with open(os.path.join(args.out_dir, 'manifest.json'), 'w', encoding='utf-8') as g:
        json.dump(man, g, indent=2)
    print(f"Da chia {src} ({man['total_bytes'] / 2**20:.1f} MB) thanh {len(parts)} phan trong {args.out_dir}")
    for p in parts:
        print(f"  {p['name']}  {p['bytes'] / 2**20:.1f} MB")


def find_root(base):
    """Thu muc CHA cua cac run..._iter... (zip co the co them cap 'dump/')."""
    for dirpath, dirnames, _files in os.walk(base):
        if ANCHOR in dirnames:
            return dirpath
    return None


def check_root(root, n_expected=384):
    ok = True
    ref = None
    for lb, name in NEEDED.items():
        d = os.path.join(root, name)
        m = os.path.join(d, 'masks')
        csv = os.path.join(d, 'per_image_stats.csv')
        if not (os.path.isdir(m) and os.path.exists(csv)):
            print(f"  [THIEU] {lb:16s} {d}")
            ok = False
            continue
        pngs = sorted(f[:-4] for f in os.listdir(m) if f.endswith('.png'))
        with open(csv, encoding='utf-8') as f:
            header = f.readline().rstrip('\n').split(',')
            col = header.index('image')
            imgs = [line.rstrip('\n').split(',')[col] for line in f if line.strip()]
        good = len(pngs) == n_expected and sorted(imgs) == pngs and (ref is None or imgs == ref)
        ref = ref or imgs
        ok &= good
        print(f"  [{'OK' if good else 'LOI'}]    {lb:16s} {name:34s} png={len(pngs)} csv={len(imgs)}")
    print("  (bce02_40k khong co trong dump — phien GPU tao bang --make-missing-dumps)")
    return ok


def cmd_join(args):
    man_path = os.path.join(args.parts_dir, 'manifest.json')
    if not os.path.exists(man_path):
        sys.exit(f"Khong thay {man_path} — da chay 'split' va commit data/dump_parts chua?")
    with open(man_path, encoding='utf-8') as f:
        man = json.load(f)
    stamp = os.path.join(args.dest, '.extracted_sha256')
    if os.path.exists(stamp) and open(stamp).read().strip() == man['sha256'] and not args.force:
        print(f"Da giai nen truoc do (sha256 khop) -> {args.dest}")
    else:
        if os.path.isdir(args.dest):
            shutil.rmtree(args.dest)
        os.makedirs(args.dest)
        tmpzip = os.path.join(args.dest, man['source_name'])
        h = hashlib.sha256()
        with open(tmpzip, 'wb') as out:
            for p in man['parts']:
                pp = os.path.join(args.parts_dir, p['name'])
                if not os.path.exists(pp):
                    sys.exit(f"Thieu phan {pp}")
                data = open(pp, 'rb').read()
                if len(data) != p['bytes'] or hashlib.sha256(data).hexdigest() != p['sha256']:
                    sys.exit(f"Phan {p['name']} sai kich thuoc/sha256 — clone lai repo "
                             f"(co the la file con tro Git LFS hoac tai do dang).")
                out.write(data)
                h.update(data)
        if h.hexdigest() != man['sha256']:
            sys.exit("sha256 cua dump.zip sau khi ghep KHONG khop manifest — dung.")
        print(f"Ghep {len(man['parts'])} phan -> {tmpzip} ({man['total_bytes'] / 2**20:.1f} MB), sha256 OK")
        with zipfile.ZipFile(tmpzip) as z:
            z.extractall(args.dest)
        os.remove(tmpzip)
        with open(stamp, 'w') as f:
            f.write(man['sha256'])
        print(f"Da giai nen vao {args.dest}")
    root = find_root(args.dest)
    if root is None:
        sys.exit(f"Khong tim thay thu muc {ANCHOR} trong {args.dest}")
    if not args.skip_check and not check_root(root):
        sys.exit("Kiem tra dump THAT BAI — xem cac dong [THIEU]/[LOI] o tren.")
    print(f"DUMP_ROOT={root}")


def cmd_check(args):
    root = find_root(args.root) or args.root
    ok = check_root(root)
    print(f"DUMP_ROOT={root}")
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('split')
    p.add_argument('--zip', default='data/dump.zip')
    p.add_argument('--out-dir', default='data/dump_parts')
    p.add_argument('--part-mb', type=int, default=45)
    p = sub.add_parser('join')
    p.add_argument('--parts-dir', default='data/dump_parts')
    p.add_argument('--dest', default='/tmp/dump_repo')
    p.add_argument('--force', action='store_true')
    p.add_argument('--skip-check', action='store_true')
    p = sub.add_parser('check')
    p.add_argument('--root', required=True)
    args = ap.parse_args(argv)
    {'split': cmd_split, 'join': cmd_join, 'check': cmd_check}[args.cmd](args)


if __name__ == '__main__':
    main()
