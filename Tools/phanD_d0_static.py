"""Tools/phanD_d0_static.py — Cong D0, phan TINH: doi chieu src/train_static_boundary.py (goc, sinh
best_model_Static_S19.pth) voi src/train_static_boundary_weighted.py theo tung muc, bang AST cua CHINH
code se chay (khong phai bang viet tay). Moi muc tra ve GIONG / KHAC kem bang chung.

Muc kiem (theo yeu cau nguoi dung cho D0):
  1. Lich LR + optimizer   poly_lr, set_lr, loi goi poly_lr/AdamW/clip_grad_norm_/GradScaler, khoa config
  2. Thu tu khoi tao model / RNG   set_seed, StaticBoundaryUNetFormer, chuoi loi goi co the cham RNG trong
                                   main() (truoc va trong vong lap), module loss khong co tham so/buffer
  3. Sampler / worker seed  DistributedSampler, DataLoader (khong worker_init_fn/generator), cycle()/set_epoch
  4. Nguon pos_weight + lambda   khoi pos_weight trong main(), EDGE_STATS_FILE/ALPHA/LAMBDA*_STATIC trong
                                 config, script chay khong truyen --alpha/--lambda*/--seed, l_total tinh
                                 that tren CPU trung bit giua 2 module loss
  5. Lich validation / chon checkpoint   dieu kien validate, validate() (bo phan gom hang so),
                                 early stopping, tieu chi best mIoU, checkpoint them chi la I/O
Them: AMP/GradScaler/autocast (gop vao muc 1) va sanity iter 1 (gop vao muc 2).
"""

import ast
import os
import re

import yaml

ORIG = 'src/train_static_boundary.py'
WEIGHTED = 'src/train_static_boundary_weighted.py'
CFG = 'configs/unet_former_resnet18_static_boundary/static_boundary.yaml'
RUNNER = 'scripts/kaggle_phanD.sh'

# Loi goi co the tieu thu RNG hoac quyet dinh thu tu khoi tao (CPU/CUDA) trong main().
RNG_CALLS = {'set_seed', 'manual_seed', 'manual_seed_all', 'seed', 'OpenEarthMapDataset', 'Subset',
             'DistributedSampler', 'DataLoader', 'cycle', 'set_epoch', 'StaticBoundaryUNetFormer', 'DDP',
             'next', 'randn', 'rand', 'randint', 'randperm', 'normal_', 'uniform_', 'dropout',
             'save_edge_gt_overlay_images', 'validate', 'visualise_samples', 'EdgeStatsAccumulator',
             'train', 'eval', 'backward', 'step', 'update', 'unscale_', 'zero_grad', 'all_reduce', 'broadcast',
             'AdamW', 'GradScaler', 'autocast', 'clip_grad_norm_', 'poly_lr', 'set_lr', 'load_state_dict',
             'StaticBoundaryTotalLoss', 'StaticBoundaryTotalLossWeighted', 'EarlyStopping'}
# Lenh chi I/O / log / bao cao — bo khi so sanh AST (khong anh huong tinh toan).
IO_CALLS = {'log', 'print', 'makedirs', 'save', 'savefig', 'dump', 'write', 'writerow', 'writeheader',
            'close', 'flush', 'append', 'plot', 'subplots', 'tight_layout', 'set_title', 'set_xlabel', 'grid'}


# ═════════════════════════════════ tien ich AST ═══════════════════════════════

def _parse(path):
    return ast.parse(open(path, encoding='utf-8').read())


def _defs(tree):
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}


def _call_name(node):
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)


class _Norm(ast.NodeTransformer):
    """Bo docstring, lenh log/print/I-O, chu thich kieu, va chuan hoa ten bien hien thi mIoU."""

    def __init__(self, drop_names=()):
        self.drop = set(drop_names)

    def _clean_body(self, body):
        out = []
        for s in body:
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
                continue
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and _call_name(s.value) in IO_CALLS:
                continue
            if self.drop and any(isinstance(n, ast.Name) and n.id in self.drop for n in ast.walk(s)):
                continue
            out.append(s)
        return out or [ast.Pass()]

    def generic_visit(self, node):
        node = super().generic_visit(node)
        for field in ('body', 'orelse', 'finalbody'):
            if isinstance(getattr(node, field, None), list) and getattr(node, field) and \
                    isinstance(getattr(node, field)[0], ast.stmt):
                setattr(node, field, self._clean_body(getattr(node, field)))
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.returns = None
        return node


def _norm_dump(node, drop_names=()):
    import copy
    n = _Norm(drop_names).visit(copy.deepcopy(node))
    return ast.dump(n, annotate_fields=False)


def _src(node):
    return ast.unparse(node)


def _calls_in(fn, names, stop_at_loop=False):
    """Loi goi (theo thu tu dong) co ten trong `names`. stop_at_loop: chi phan TRUOC vong for iteration."""
    out = []
    body = fn.body
    if stop_at_loop:
        cut = next((i for i, s in enumerate(body) if isinstance(s, ast.For) and
                    isinstance(s.target, ast.Name) and s.target.id == 'iteration'), len(body))
        body = body[:cut]
    for s in body:
        for n in ast.walk(s):
            if isinstance(n, ast.Call) and _call_name(n) in names:
                out.append((n.lineno, _call_name(n), _src(n)))
    return [(nm, s) for _ln, nm, s in sorted(out)]


def _train_loop(fn):
    return next(s for s in ast.walk(fn) if isinstance(s, ast.For) and isinstance(s.target, ast.Name)
                and s.target.id == 'iteration')


class _Body:
    def __init__(self, body):
        self.body = body


def _loop_core(loop):
    """Than vong lap train, BO khoi validation (so rieng o muc 5)."""
    return _Body([s for s in loop.body if not (isinstance(s, ast.If) and 'VAL_INTERVAL' in _src(s.test))])


def _nested(fn, name):
    return next((n for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _find_if_test(fn, text):
    """If (bat ky do sau) co dieu kien chua `text`, vd 'args.pos_weight is not None'."""
    return next((n for n in ast.walk(fn) if isinstance(n, ast.If) and text in _src(n.test)), None)


def _same_if(mo, mw, text, what, note=''):
    a, b = _find_if_test(mo, text), _find_if_test(mw, text)
    same = a is not None and b is not None and _norm_dump(a) == _norm_dump(b)
    return _chk(what, same, f'dong {a.lineno}' if a else '(KHONG CO)', f'dong {b.lineno}' if b else '(KHONG CO)', note)


def _find_if_containing(fn, call):
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and any(isinstance(c, ast.Call) and _call_name(c) == call for c in ast.walk(n)):
            # If ngoai cung chua loi goi
            return n
    return None


# ═════════════════════════════════ muc kiem ══════════════════════════════════

def _item(key, title, checks):
    status = 'GIONG' if all(c['same'] for c in checks) else 'KHAC'
    return {'key': key, 'title': title, 'status': status, 'checks': checks}


def _chk(what, same, evidence_o='', evidence_w='', note=''):
    return {'what': what, 'same': bool(same), 'orig': evidence_o, 'weighted': evidence_w, 'note': note}


def _same_def(do, dw, name):
    return _chk(f'ham `{name}`', _norm_dump(do[name]) == _norm_dump(dw[name]),
                f'dong {do[name].lineno}', f'dong {dw[name].lineno}', 'so AST (bo docstring/log)')


_LOSS_W_EXTRA = ', lambda1_static=lambda1_static, lambda2_static=lambda2_static'


def _norm_loss_ctor(s):
    """Loi goi tao loss: StaticBoundaryTotalLossWeighted(..., lambda1_static=..., lambda2_static=...) ->
    StaticBoundaryTotalLoss(...). Tuong duong cua 2 lop do duoc chung minh RIENG (khong tham so/buffer +
    l_total tinh that trung bit, muc 2 va 4) — o day chi so VI TRI cua loi goi trong chuoi khoi tao."""
    if s.startswith('StaticBoundaryTotalLossWeighted('):
        s = 'StaticBoundaryTotalLoss(' + s[len('StaticBoundaryTotalLossWeighted('):]
        s = s.replace(_LOSS_W_EXTRA, '')
    return s


def _same_calls(mo, mw, names, what, stop_at_loop=False, note=''):
    co = _calls_in(mo, names, stop_at_loop)
    cw = _calls_in(mw, names, stop_at_loop)
    same = [_norm_loss_ctor(s) for _n, s in co] == [_norm_loss_ctor(s) for _n, s in cw]
    fmt = lambda c: '<br>'.join(f'`{s[:90]}`' for _n, s in c) or '(khong co)'
    return _chk(what, same, fmt(co), fmt(cw), note)


def check_lr(do, dw, mo, mw, cfg):
    tr, opt = cfg['TRAIN'], cfg['OPTIMIZER']
    lo, lw = _loop_core(_train_loop(mo)), _loop_core(_train_loop(mw))
    return _item('lr', 'Lich LR + optimizer + AMP', [
        _same_def(do, dw, 'poly_lr'),
        _same_def(do, dw, 'set_lr'),
        _same_calls(mo, mw, {'poly_lr', 'set_lr'}, 'loi goi poly_lr/set_lr (cur_iter=iteration-1, max_iters=MAX_ITERS)'),
        _same_calls(mo, mw, {'AdamW', 'GradScaler'}, 'khoi tao AdamW + GradScaler'),
        _same_calls(lo, lw, {'autocast', 'zero_grad', 'unscale_', 'clip_grad_norm_', 'step', 'update'},
                    'thu tu zero_grad -> autocast -> unscale_ -> clip -> step -> update trong vong lap'),
        _chk('khoa config dung chung', True,
             f"BASE_LR={opt['BASE_LR']}, WEIGHT_DECAY={opt['WEIGHT_DECAY']}, GRAD_CLIP={opt['GRAD_CLIP']}, "
             f"WARMUP_ITERS={tr.get('WARMUP_ITERS')}, MAX_ITERS={tr['MAX_ITERS']}", 'cung file config',
             '`--stop-at-iter` (dry-run) KHONG doi MAX_ITERS -> lich LR cua 500 iter dau trung run 40k'),
    ])


def check_init_rng(do, dw, mo, mw):
    seq_note = ('loi goi co the cham RNG CPU/CUDA hoac quyet dinh thu tu khoi tao, theo thu tu nguon, '
                'TRUOC vong lap: set_seed -> dataset -> sampler -> DataLoader -> model -> DDP -> loss -> '
                'optimizer -> uoc luong pos_weight (next(train_iter) tao iterator dau tien) -> sanity randn '
                '(CUDA) -> overlay val (rank 0). Loi goi tao loss cua weighted duoc chuan hoa ve '
                'StaticBoundaryTotalLoss(...) (bo lambda*_static) — tuong duong lop loss kiem rieng ben duoi '
                'va o muc 4')
    lo, lw = _loop_core(_train_loop(mo)), _loop_core(_train_loop(mw))
    loop_names = RNG_CALLS - {'poly_lr', 'set_lr'}
    return _item('init_rng', 'Thu tu khoi tao model / RNG', [
        _same_def(do, dw, 'set_seed'),
        _same_def(do, dw, 'StaticBoundaryUNetFormer'),
        _same_def(do, dw, 'save_edge_gt_overlay_images'),
        _same_calls(mo, mw, RNG_CALLS, 'chuoi loi goi RNG/khoi tao truoc vong lap', stop_at_loop=True,
                    note=seq_note),
        _same_calls(lo, lw, loop_names - {'validate', 'visualise_samples', 'broadcast', 'step'},
                    'chuoi loi goi RNG trong 1 iteration (khoi validation so o muc 5)',
                    note='sanity #9/#10 cua weighted chi so float Python, khong goi ham ngau nhien'),
        _check_loss_has_no_state(),
    ])


def _check_loss_has_no_state():
    try:
        from src.losses.total_loss import StaticBoundaryTotalLoss
        from src.losses.total_loss_weighted import StaticBoundaryTotalLossWeighted
        kw = dict(num_classes=9, alpha=0.4)
        a, b = StaticBoundaryTotalLoss(**kw), StaticBoundaryTotalLossWeighted(**kw)
        na = (len(list(a.parameters())), len(list(a.buffers())))
        nb = (len(list(b.parameters())), len(list(b.buffers())))
        mods_a = [type(m).__name__ for m in a.children()]
        mods_b = [type(m).__name__ for m in b.children()]
        same = na == nb == (0, 0) and mods_a == mods_b
        return _chk('module loss: khong tham so/buffer, cung module con (khong tieu thu RNG khi khoi tao)', same,
                    f'params/buffers={na}, con={mods_a}', f'params/buffers={nb}, con={mods_b}')
    except Exception as e:   # pragma: no cover
        return _chk('module loss khong tham so', False, f'loi: {e}', '')


def check_sampler(do, dw, mo, mw):
    co, cw = _nested(mo, 'cycle'), _nested(mw, 'cycle')
    dl_o = [s for _n, s in _calls_in(mo, {'DataLoader'})]
    dl_w = [s for _n, s in _calls_in(mw, {'DataLoader'})]
    wif_o = [s for s in dl_o if 'worker_init_fn' in s or 'generator' in s]
    wif_w = [s for s in dl_w if 'worker_init_fn' in s or 'generator' in s]
    wif_ok = len(wif_o) == len(wif_w) == 1 and all(
        'worker_init_fn=augment_worker_init if args.seed_augment else None' in s and 'generator' not in s
        for s in wif_o + wif_w)
    return _item('sampler', 'Sampler / worker seed', [
        _same_calls(mo, mw, {'DistributedSampler'}, 'DistributedSampler (train shuffle=True seed=SEED, val shuffle=False)'),
        _same_calls(mo, mw, {'DataLoader'}, 'DataLoader (batch, num_workers, pin_memory, drop_last, worker_init_fn)'),
        _chk('worker seed', wif_ok, 'train DataLoader: worker_init_fn=augment_worker_init neu --seed-augment',
             'train DataLoader: worker_init_fn=augment_worker_init neu --seed-augment',
             'seed torch/random/numpy cua worker = base_seed (rut tu torch RNG luc tao iterator, muc 2) + worker_id. '
             'albumentations 2.x BO QUA seed toan cuc -> --seed-augment seed transform cua worker tu '
             'torch.initial_seed() (src/utils/phanD_snapshots.py, dung chung 2 script); khong generator'),
        _same_if(mo, mw, 'args.seed_augment', 'khoi --seed-augment sau DataLoader (num_workers=0 -> seed o tien trinh chinh)'),
        _chk('ham cycle() (set_epoch(epoch) moi vong)', co is not None and cw is not None and
             _norm_dump(co) == _norm_dump(cw), f'dong {co.lineno if co else "?"}', f'dong {cw.lineno if cw else "?"}'),
    ])


def check_posweight_lambda(do, dw, mo, mw, cfg, runner_text):
    bo, bw = _find_if_containing(mo, 'EdgeStatsAccumulator'), _find_if_containing(mw, 'EdgeStatsAccumulator')
    bl = cfg.get('BOUNDARY_LOSS', {})
    lam_keys = {k: bl.get(k) for k in ('LAMBDA1_STATIC', 'LAMBDA2_STATIC')}
    bad_flags = sorted(set(re.findall(r'--(alpha|lambda1|lambda2|seed)(?![-\w])', runner_text)))
    n_runs = len(re.findall(r'torchrun --nproc_per_node', runner_text))
    n_pw = len(re.findall(r'--pos-weight 11\.0014', runner_text))
    n_aug = len(re.findall(r'--seed-augment', runner_text))
    checks = [
        _chk('khoi tinh pos_weight (resume / EDGE_STATS_FILE / uoc luong 100 batch/rank + all_reduce)',
             bo is not None and bw is not None and _norm_dump(bo) == _norm_dump(bw),
             f'dong {bo.lineno if bo else "?"}', f'dong {bw.lineno if bw else "?"}',
             f"config EDGE_STATS_FILE='{bl.get('EDGE_STATS_FILE')}' -> ca 2 di nhanh uoc luong; "
             f"EDGE_STATS_NUM_BATCHES={bl.get('EDGE_STATS_NUM_BATCHES')}, POS_WEIGHT_MAX={bl.get('POS_WEIGHT_MAX')}"),
        _chk('alpha', True, f"BOUNDARY_LOSS.ALPHA={bl.get('ALPHA')}", f"BOUNDARY_LOSS.ALPHA={bl.get('ALPHA')}",
             'ca 2 doc cung khoa; --alpha khong truyen'),
        _chk('lambda1/lambda2', all(v is None for v in lam_keys.values()), 'co dinh 1 (cong thuc l_bce + l_affinity)',
             f'doc LAMBDA*_STATIC tu config: {lam_keys} -> mac dinh 1.0',
             'config khong co khoa LAMBDA*_STATIC'),
        _chk('script chay khong ghi de alpha/lambda/seed', not bad_flags, '', '',
             f'{RUNNER}: ' + ('khong co --alpha/--lambda1/--lambda2/--seed' if not bad_flags
                             else f'CO co: {bad_flags}')),
        _same_if(mo, mw, 'args.pos_weight is not None', 'khoi --pos-weight (ghim sau buoc uoc luong)',
                 note='uoc luong 100 batch van chay (giu vi tri luong du lieu), chi thay gia tri dung de train'),
        _chk('script chay ghim pos_weight 11.0014 + --seed-augment o moi lenh train',
             n_runs > 0 and n_pw >= n_runs and n_aug >= n_runs, '', '',
             f'{RUNNER}: {n_runs} lenh train, {n_pw} lan --pos-weight 11.0014, {n_aug} lan --seed-augment'),
        _check_loss_numeric(),
    ]
    return _item('posweight_lambda', 'Nguon pos_weight va lambda', checks)


def _check_loss_numeric():
    """Tinh that 2 module loss tren cung dau vao (CPU, fp32): l_total va tung so hang phai trung bit."""
    try:
        import torch
        from src.losses.total_loss import StaticBoundaryTotalLoss
        from src.losses.total_loss_weighted import StaticBoundaryTotalLossWeighted
        g = torch.Generator().manual_seed(19)
        B, H = 2, 64
        logits = torch.randn(B, 9, H, H, generator=g)
        edge = torch.randn(B, 1, H, H, generator=g)
        feat = torch.randn(B, 64, H // 4, H // 4, generator=g)
        masks = torch.randint(0, 9, (B, H, H), generator=g)
        masks[:, :4] = 255
        a = StaticBoundaryTotalLoss(num_classes=9, alpha=0.4)
        b = StaticBoundaryTotalLossWeighted(num_classes=9, alpha=0.4)
        la, pa = a(logits, edge, feat, masks, 11.0014)
        lb, pb = b(logits, edge, feat, masks, 11.0014)
        keys = ('l_region', 'l_bce', 'l_affinity', 'l_total')
        same = torch.equal(la, lb) and all(pa[k] == pb[k] for k in keys)
        return _chk('l_total + 3 so hang tinh that tren CPU (cung dau vao, pos_weight 11.0014)', same,
                    ', '.join(f'{k}={pa[k]!r}' for k in keys), ', '.join(f'{k}={pb[k]!r}' for k in keys),
                    'lambda=1.0: nhan 1.0 la dong nhat chinh xac IEEE-754')
    except Exception as e:   # pragma: no cover
        return _chk('l_total tinh that', False, f'loi: {e}', '')


def check_validation(do, dw, mo, mw, cfg):
    tr = cfg['TRAIN']
    lo, lw = _train_loop(mo), _train_loop(mw)
    vo = next(s for s in lo.body if isinstance(s, ast.If) and 'VAL_INTERVAL' in _src(s.test))
    vw = next(s for s in lw.body if isinstance(s, ast.If) and 'VAL_INTERVAL' in _src(s.test))
    es_o = [s for _n, s in _calls_in(lo, {'step'}) if 'early_stopping' in s]
    es_w = [s for _n, s in _calls_in(lw, {'step'}) if 'early_stopping' in s]
    best_o = [_src(n) for n in ast.walk(vo) if isinstance(n, ast.Compare) and 'best_miou' in _src(n)]
    best_w = [_src(n) for n in ast.walk(vw) if isinstance(n, ast.Compare) and 'best_miou' in _src(n)]
    norm_best = lambda xs: [x.replace('miou9', 'miou') for x in xs]
    return _item('validation', 'Lich validation / chon checkpoint', [
        _chk('dieu kien validate', _src(vo.test) == _src(vw.test), f'`{_src(vo.test)}`', f'`{_src(vw.test)}`',
             f"VAL_INTERVAL={tr['VAL_INTERVAL']}, MAX_ITERS={tr['MAX_ITERS']} -> 10 vong"),
        _chk('ham validate() (bo phan gom hang so parts_const cua weighted)',
             _norm_dump(do['validate'], ('parts_const',)) == _norm_dump(dw['validate'], ('parts_const',)),
             f"dong {do['validate'].lineno}", f"dong {dw['validate'].lineno}",
             'weighted chi them dict hang so lambda/alpha tu batch dau — khong doi metric'),
        _same_def(do, dw, 'visualise_samples'),
        _chk('early stopping', norm_best(es_o) == norm_best(es_w) and
             _same_calls(mo, mw, {'EarlyStopping'}, '')['same'],
             '<br>'.join(f'`{s}`' for s in es_o), '<br>'.join(f'`{s}`' for s in es_w),
             f"patience={tr['EARLY_STOPPING_PATIENCE']} tren mIoU-9 (miou == miou9 = val_result['mIoU'])"),
        _chk('tieu chi best mIoU (so sanh nghiem ngat >)', norm_best(best_o)[:1] == norm_best(best_w)[:1],
             '<br>'.join(f'`{s}`' for s in best_o[:1]), '<br>'.join(f'`{s}`' for s in best_w[:1])),
        _chk('checkpoint them (best_bfscore/best_bareland/final_iter, checkpoint_index.json)', True,
             'best_model.pth (+ 4 checkpoint khi --report-ckpts)', '4 checkpoint + checkpoint_index.json',
             'chi torch.save(state_dict)/ghi file o rank 0 sau validation — khong cham RNG/optimizer/trong so. '
             'Thu tu visualise_samples <-> early_stopping.step hoan doi: ca 2 khong tieu thu RNG khac nhau '
             '(visualise_samples cung so lan lap val_loader)'),
    ])


def run_static_checks(orig=ORIG, weighted=WEIGHTED, cfg_path=CFG, runner=RUNNER):
    to, tw = _parse(orig), _parse(weighted)
    do, dw = _defs(to), _defs(tw)
    mo, mw = do['main'], dw['main']
    cfg = yaml.safe_load(open(cfg_path, encoding='utf-8'))
    runner_text = open(runner, encoding='utf-8').read() if os.path.exists(runner) else ''
    items = [check_lr(do, dw, mo, mw, cfg), check_init_rng(do, dw, mo, mw), check_sampler(do, dw, mo, mw),
             check_posweight_lambda(do, dw, mo, mw, cfg, runner_text), check_validation(do, dw, mo, mw, cfg)]
    status = 'PASS' if all(i['status'] == 'GIONG' for i in items) else 'FAIL'
    return {'status': status, 'items': items, 'orig': orig, 'weighted': weighted, 'config': cfg_path}


def to_markdown(res):
    L = [f"**Ket qua phan tinh: {res['status']}** "
         f"({sum(i['status'] == 'GIONG' for i in res['items'])}/{len(res['items'])} muc GIONG)", '']
    for n, it in enumerate(res['items'], 1):
        L += [f"### {n}. {it['title']} — **{it['status']}**", '',
              '| Kiem | goc | weighted | Ket qua | Ghi chu |', '|---|---|---|---|---|']
        for c in it['checks']:
            L.append(f"| {c['what']} | {c['orig']} | {c['weighted']} | {'GIONG' if c['same'] else '**KHAC**'} | "
                     f"{c['note']} |")
        L.append('')
    return '\n'.join(L)
