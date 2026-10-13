"""Tools/affinity_support.py — Phan D §3 (docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md):
kiem vung ho tro cua L_aff tren Fused Feature F. MO TA, khong phai cong.

Tren `best_model_Static_S19.pth` @36k va 2 anh val dau tien trong figure_images.json:
  - ban do ‖∂L_aff/∂F(p)‖ tren luoi stride 4; khoang cach lon nhat (px full-res) toi bien GT co chuan > 1e-12;
    do phu ho tro theo tang khoang cach (doi chieu Buoc 1: tang (4,8] chi 1.3%);
  - ti le cap co hinge hoat dong: cap khac lop co D < margin; cap cung lop co gradient khac 0 (D > 1e-6);
  - tach dong gop gradient vao TAM vs LANG GIENG (2 ban sao F: mot cho tam, mot cho unfold).

Chi DOC src/losses/affinity.py: ban tai hien (_affinity_pairs) duoc doi chieu voi AffinityLoss that (sai so
loss < 1e-6, tong gradient tam + lang gieng = gradient that) truoc khi bao cao.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F_

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tools.grad_conflict_common import (DIST_EDGES, DIST_STRATA, CFG_STATIC, gt_edge_np, downsample_nearest,
                                        read_json, write_json)

SUPPORT_EPS = 1e-12


def _affinity_pairs(loss_mod, f_center, f_neigh, mask):
    """Tai hien AffinityLoss.forward (src/losses/affinity.py) voi F tam va F lang gieng tach rieng.
    Tra ve (loss, D, A, pair_valid) — loss(F, F) phai trung AffinityLoss(F)."""
    from src.losses.boundary_bce import extract_edge_gt
    B, C, Hf, Wf = f_center.shape
    K, pad = loss_mod.K, loss_mod.pad
    mask_ds = F_.interpolate(mask.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()
    edge_gt, _ = extract_edge_gt(mask_ds, ignore_index=loss_mod.ignore_index,
                                 connectivity=loss_mod.connectivity, dilation_radius=loss_mod.dilation_radius)
    sample_mask = edge_gt.squeeze(1).bool()
    valid = mask_ds != loss_mod.ignore_index
    feat_unf = F_.unfold(f_neigh, kernel_size=K, padding=pad).view(B, C, K * K, Hf, Wf)
    label_unf = F_.unfold(mask_ds.float().unsqueeze(1), kernel_size=K, padding=pad).view(B, K * K, Hf, Wf).long()
    valid_unf = F_.unfold(valid.float().unsqueeze(1), kernel_size=K, padding=pad).view(B, K * K, Hf, Wf).bool()
    c = F_.normalize(f_center.unsqueeze(2), dim=1, eps=loss_mod.eps)
    n = F_.normalize(feat_unf, dim=1, eps=loss_mod.eps)
    D = 1.0 - (c * n).sum(dim=1)
    A = (label_unf == mask_ds.unsqueeze(1)).float()
    pv = valid_unf & valid.unsqueeze(1)
    pv = pv.clone()
    pv[:, loss_mod.center_idx] = False
    pv = pv & sample_mask.unsqueeze(1)
    pvf = pv.float()
    per = (A * D + (1.0 - A) * torch.clamp(loss_mod.margin - D, min=0.0)) * pvf
    return per.sum() / pvf.sum().clamp_min(1.0), D, A, pv


def analyse_image(probe, x, m, gt):
    loss_mod = probe.loss_mod.affinity
    with torch.no_grad():
        _l, _e, Fm = probe.forward(x)
    Fm = Fm.detach()
    # gradient that cua AffinityLoss (module repo)
    Fr = Fm.clone().requires_grad_(True)
    L_ref = loss_mod(Fr, m)
    g_ref, = torch.autograd.grad(L_ref, Fr)
    # tach tam / lang gieng
    Fc = Fm.clone().requires_grad_(True)
    Fn = Fm.clone().requires_grad_(True)
    L_mine, D, A, pv = _affinity_pairs(loss_mod, Fc, Fn, m)
    gc, gn = torch.autograd.grad(L_mine, [Fc, Fn])
    check = {'abs_err_loss_vs_module': abs(float(L_mine - L_ref)),
             'rel_err_grad_center_plus_neigh_vs_module': float((gc + gn - g_ref).norm() / g_ref.norm().clamp_min(1e-30))}

    norm = g_ref[0].double().pow(2).sum(0).sqrt().cpu().numpy()          # (h, w)
    nc = gc[0].double().pow(2).sum(0).sqrt().cpu().numpy()
    nn_ = gn[0].double().pow(2).sum(0).sqrt().cpu().numpy()
    h, w = norm.shape
    from scipy.ndimage import distance_transform_edt
    edge = gt_edge_np(gt)
    dist_full = distance_transform_edt(~edge) if edge.any() else np.full(gt.shape, np.inf)
    dist = downsample_nearest(dist_full.astype(np.float64), (h, w))
    valid = downsample_nearest((gt != 255).astype(np.int64), (h, w)).astype(bool)
    sup = (norm > SUPPORT_EPS) & valid
    strata = np.digitize(dist, DIST_EDGES, right=True)
    cov = {}
    for s, name in enumerate(DIST_STRATA):
        sel = valid & (strata == s)
        cov[name] = {'n_pos': int(sel.sum()), 'n_support': int((sup & sel).sum()),
                     'coverage': float((sup & sel).sum() / sel.sum()) if sel.sum() else None,
                     'sum_sq_grad': float((norm[sel] ** 2).sum()),
                     'sum_sq_grad_center': float((nc[sel] ** 2).sum()),
                     'sum_sq_grad_neigh': float((nn_[sel] ** 2).sum())}

    pvb = pv[0]
    diff = pvb & (A[0] == 0)
    same = pvb & (A[0] == 1)
    margin = loss_mod.margin
    hinge = {'n_pairs_valid': int(pvb.sum()), 'n_pairs_diff_class': int(diff.sum()),
             'n_pairs_same_class': int(same.sum()),
             'frac_diff_class_hinge_active_D_lt_margin': float(((D[0] < margin) & diff).sum() / diff.sum())
             if diff.sum() else None,
             'frac_same_class_grad_nonzero_D_gt_1e-6': float(((D[0] > 1e-6) & same).sum() / same.sum())
             if same.sum() else None,
             'n_center_pixels_sampled': int(pvb.any(0).sum())}
    tot = float((gc.double() ** 2).sum() + (gn.double() ** 2).sum())
    split = {'norm_grad_center': float(gc.double().norm()), 'norm_grad_neigh': float(gn.double().norm()),
             'frac_energy_center': float((gc.double() ** 2).sum()) / tot if tot else None,
             'cos_center_neigh': float((gc.double() * gn.double()).sum() /
                                       (gc.double().norm() * gn.double().norm()).clamp_min(1e-300))}
    return {'L_aff': float(L_ref), 'grid_hw': [h, w], 'check': check,
            'max_dist_px_fullres_with_support': float(dist[sup].max()) if sup.any() else None,
            'n_support_pixels': int(sup.sum()), 'n_valid_pixels': int(valid.sum()),
            'coverage_by_dist': cov, 'hinge': hinge, 'center_vs_neigh': split}, norm


def write_md(path, res):
    L = ['# Phan D §3 — Vung ho tro cua L_aff (mo ta, khong la cong)', '',
         f"Checkpoint: `{res['checkpoint']}` · anh: {', '.join(r['image_id'] for r in res['images'])} · "
         f"F stride 4, cua so K={res['window_K']}, margin={res['margin']}.", '',
         'Tom tat kich thuoc: tam chi lay tren pixel bien GT (stride 4); lang gieng trong cua so KxK => '
         'ho tro ly thuyet toi da ~ (K//2)*4 px full-res quanh bien, cong sai so lam tron NEAREST.', '']
    for r in res['images']:
        L += [f"## {r['image_id']}", '',
              f"- L_aff = {r['L_aff']:.6f}; doi chieu module: |Δloss| = {r['check']['abs_err_loss_vs_module']:.2e}, "
              f"‖g_tam + g_lang_gieng − g‖/‖g‖ = {r['check']['rel_err_grad_center_plus_neigh_vs_module']:.2e}",
              f"- Khoang cach lon nhat (px full-res) toi bien GT co ‖∂L_aff/∂F‖ > 1e-12: "
              f"**{r['max_dist_px_fullres_with_support']}**",
              f"- Pixel co ho tro: {r['n_support_pixels']} / {r['n_valid_pixels']} pixel hop le (luoi {r['grid_hw']})",
              '', '| tang khoang cach | n_pos | n_support | do phu | Σ‖g‖² | Σ‖g_tam‖² | Σ‖g_lang_gieng‖² |',
              '|---|---|---|---|---|---|---|']
        for k, v in r['coverage_by_dist'].items():
            cov = '—' if v['coverage'] is None else f"{v['coverage']:.4f}"
            L.append(f"| {k} | {v['n_pos']} | {v['n_support']} | {cov} | {v['sum_sq_grad']:.3e} | "
                     f"{v['sum_sq_grad_center']:.3e} | {v['sum_sq_grad_neigh']:.3e} |")
        hg, sp = r['hinge'], r['center_vs_neigh']
        L += ['', f"- Cap hop le: {hg['n_pairs_valid']} (khac lop {hg['n_pairs_diff_class']}, cung lop "
                  f"{hg['n_pairs_same_class']}); pixel tam duoc lay mau: {hg['n_center_pixels_sampled']}",
              f"- Cap khac lop co hinge hoat dong (D < margin): **{hg['frac_diff_class_hinge_active_D_lt_margin']}**",
              f"- Cap cung lop co gradient khac 0 (D > 1e-6): **{hg['frac_same_class_grad_nonzero_D_gt_1e-6']}**",
              f"- Tam vs lang gieng: ‖g_tam‖ = {sp['norm_grad_center']:.3e}, ‖g_lang_gieng‖ = "
              f"{sp['norm_grad_neigh']:.3e}, ti phan nang luong tam = {sp['frac_energy_center']}, "
              f"cos(g_tam, g_lang_gieng) = {sp['cos_center_neigh']:.3f}", '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='best_model_Static_S19.pth (@36k)')
    ap.add_argument('--figures-json', required=True)
    ap.add_argument('--config', default=CFG_STATIC)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--pos-weight', type=float, default=11.0014)
    ap.add_argument('--n-images', type=int, default=2)
    ap.add_argument('--device', default=None)
    ap.add_argument('--output-dir', default='output/phanD')
    args = ap.parse_args(argv)
    from Tools.grad_conflict_probe import Probe, resolve_device_det
    from Tools.oracle_boundary_ceiling import load_cfg, build_val_dataset, image_ids_of
    cfg = load_cfg(args.config, args.data_root)
    device = resolve_device_det(args.device)
    probe = Probe('static_s19_36k', args.ckpt, cfg, device, args.pos_weight)
    val_ds = build_val_dataset(cfg)
    ids = image_ids_of(val_ds)
    wanted = [d['image_id'] for d in read_json(args.figures_json)['images']][:args.n_images]
    res = {'checkpoint': args.ckpt, 'window_K': probe.loss_mod.affinity.K,
           'margin': probe.loss_mod.affinity.margin, 'images': []}
    maps = {}
    for name in wanted:
        i = ids.index(name)
        x, m = val_ds[i]
        gt = m.numpy().astype(np.int64)
        r, norm = analyse_image(probe, x[None].to(device), m[None].to(device), gt)
        r['image_id'] = name
        res['images'].append(r)
        maps[name] = norm
        print(f"  [{name}] max dist ho tro = {r['max_dist_px_fullres_with_support']} px; "
              f"kiem doi chieu: {r['check']}")
    os.makedirs(args.output_dir, exist_ok=True)
    write_json(res, os.path.join(args.output_dir, 'affinity_support.json'))
    np.savez_compressed(os.path.join(args.output_dir, 'affinity_support_gradnorm.npz'), **maps)
    write_md(os.path.join(args.output_dir, 'affinity_support.md'), res)
    print(f"  -> {os.path.join(args.output_dir, 'affinity_support.md')}")


if __name__ == '__main__':
    main()
