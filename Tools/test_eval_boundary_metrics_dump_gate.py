"""Tools/test_eval_boundary_metrics_dump_gate.py — Gate 4.0 + co so gate 5.6-
5.8 (docs/spec-per-image-bootstrap-eval-v2-ban-giao-claude-code.md) cho Phan A.

Dung du lieu SYNTHETIC (3 anh .tif nho tu dung, KHONG can dataset/checkpoint
that tren Kaggle) de dung THAT CLI `python Tools/eval_boundary_metrics.py`
qua subprocess — kiem:

  Gate 4.0: chay KHONG co 3 co dump moi (--dump-preds/--dump-per-image-stats/
    --dump-only) phai cho JSON output GIONG HET (tru truong 'elapsed_sec' —
    wall time, tu nhien khac giua 2 lan chay) ban script TRUOC KHI sua (lay
    tu `git show HEAD:...`) va ban DA SUA trong phien nay. Day la bang chung
    truc tiep rang 3 co moi khong lam thay doi duong tinh mac dinh.

  Gate 5.8 (co so): tu PNG mask + tu label .tif goc, tinh lai inter_<c>/
    union_<c> cho ca 3 anh, phai KHOP tuyet doi voi cot CSV da dump.

  Kiem tra noi bo them (khong thuoc muc 5 cua spec, nhung re va huu ich):
    chay --dump-only + --output cung luc, JSON --output phai KHOP HET (tru
    elapsed_sec) voi JSON cua duong khong-dump — chung minh nhanh dump khong
    lam sai lech seg_result/boundary_result du dung 2 accumulator gioi han.

Chay: `python Tools/test_eval_boundary_metrics_dump_gate.py` tu repo root
(can torch/rasterio/pandas — da co san trong requirements.txt cua repo).
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import rasterio
import torch
import yaml
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.data.transforms import get_val_transforms
from src.models.unet_former_resnet18 import build_model


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_CLASSES = OpenEarthMapDataset.NUM_CLASSES
IGNORE = OpenEarthMapDataset.IGNORE_INDEX
IMG_SIZE = 48  # nho, get_val_transforms() se resize len 1024 nhu that


def write_tif(path, array, dtype):
    array = np.asarray(array)
    count = 1 if array.ndim == 2 else array.shape[0]
    h, w = array.shape[-2:]
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(
        path, 'w', driver='GTiff', height=h, width=w, count=count,
        dtype=dtype, transform=transform,
    ) as dst:
        if array.ndim == 2:
            dst.write(array.astype(dtype), 1)
        else:
            for i in range(count):
                dst.write(array[i].astype(dtype), i + 1)


def build_synthetic_dataset(root):
    """3 anh 48x48, RGB uint8 + label uint8 (0..8, mot vai pixel 255 ignore va
    mot vai vung dong lien de co bien that su cho Boundary IoU/BF/ASD)."""
    img_dir = os.path.join(root, 'images', 'val')
    mask_dir = os.path.join(root, 'labels', 'val')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    rng = np.random.default_rng(19)
    names = []
    for i in range(3):
        name = f'synth_{i:02d}'
        names.append(name)
        image = rng.integers(0, 255, size=(3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        label[:, IMG_SIZE // 2:] = (1 + i) % NUM_CLASSES
        label[5:12, 5:12] = (3 + i) % NUM_CLASSES
        label[0:3, 0:3] = IGNORE  # goc anh, vung ignore nho

        write_tif(os.path.join(img_dir, name + '.tif'), image, 'uint8')
        write_tif(os.path.join(mask_dir, name + '.tif'), label, 'uint8')

    return names, img_dir, mask_dir


def build_synthetic_config(root):
    return {
        'DATASET': {
            'ROOT_DIR': root, 'TRAIN_IMG_DIR': 'images/train', 'TRAIN_MASK_DIR': 'labels/train',
            'VAL_ROOT_DIR': root, 'VAL_IMG_DIR': 'images/val', 'VAL_MASK_DIR': 'labels/val',
        },
        'MODEL': {
            'ARCH': 'UnetFormer', 'ENCODER': 'resnet18.fb_swsl_ig1b_ft_in1k',
            'PRETRAINED': False, 'DECODE_CHANNELS': 64, 'WINDOW_SIZE': 8,
            'NUM_HEADS': 8, 'MLP_RATIO': 4.0, 'DROP_PATH_RATE': 0.0, 'DROPOUT_RATIO': 0.0,
        },
        'TRAIN': {'NUM_CLASSES': NUM_CLASSES},
    }


def run_script(script_path, config_path, checkpoint_path, output_path, extra_args=()):
    cmd = [sys.executable, script_path,
           '--config', config_path, '--checkpoint', checkpoint_path,
           '--model-type', 'baseline', '--device', 'cpu',
           '--batch-size', '2', '--num-workers', '0', '--output', output_path,
           *extra_args]
    # PYTHONPATH=REPO_ROOT: ban script goc trich ra nam THANG trong tmp_dir
    # (khong phai tmp_dir/Tools/...), nen sys.path.insert() dua vao __file__
    # cua no se KHONG tro dung repo root — them PYTHONPATH de `from src...`
    # van resolve dung du script vat ly nam o dau.
    env = dict(os.environ, PYTHONPATH=REPO_ROOT, PYTHONIOENCODING='utf-8')
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                             encoding='utf-8', errors='replace', env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Lenh that bai (exit {result.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
    return result


def load_json_ignore_timing(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    data.pop('elapsed_sec', None)
    return data


def main():
    tmp_dir = tempfile.mkdtemp(prefix='eval_boundary_dump_gate_')
    try:
        data_root = os.path.join(tmp_dir, 'data')
        names, img_dir, mask_dir = build_synthetic_dataset(data_root)

        cfg = build_synthetic_config(data_root)
        config_path = os.path.join(tmp_dir, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f)

        # 1 checkpoint co dinh, dung chung cho ca 2 ban script — moi lech du
        # doan chi co the den tu logic dump, khong the den tu trong so model.
        model = build_model(cfg)
        model.eval()
        checkpoint_path = os.path.join(tmp_dir, 'checkpoint.pth')
        torch.save(model.state_dict(), checkpoint_path)

        # Ban script TRUOC KHI sua (git HEAD) — trich ra file rieng.
        original_script = subprocess.run(
            ['git', 'show', 'HEAD:Tools/eval_boundary_metrics.py'],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
            encoding='utf-8', errors='replace').stdout
        original_path = os.path.join(tmp_dir, 'eval_boundary_metrics_ORIGINAL.py')
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(original_script)

        edited_path = os.path.join(REPO_ROOT, 'Tools', 'eval_boundary_metrics.py')

        print("=== Gate 4.0: chay ban GOC (HEAD) khong co, ban DA SUA khong co ===")
        out_before = os.path.join(tmp_dir, 'out_before.json')
        out_after_noflags = os.path.join(tmp_dir, 'out_after_noflags.json')
        run_script(original_path, config_path, checkpoint_path, out_before)
        run_script(edited_path, config_path, checkpoint_path, out_after_noflags)

        before = load_json_ignore_timing(out_before)
        after = load_json_ignore_timing(out_after_noflags)
        assert before == after, (
            f"GATE 4.0 FAIL — JSON khac nhau (tru elapsed_sec):\nBEFORE={before}\nAFTER={after}")
        print("PASS — JSON output (tru elapsed_sec) GIONG HET giua ban goc va ban da them 3 co.\n")

        print("=== Kiem tra noi bo: --dump-only + --output phai khop JSON voi duong khong-dump ===")
        dump_dir = os.path.join(tmp_dir, 'dump')
        masks_dir = os.path.join(dump_dir, 'masks')
        stats_csv = os.path.join(dump_dir, 'per_image_stats.csv')
        out_after_dump = os.path.join(tmp_dir, 'out_after_dump.json')
        run_script(edited_path, config_path, checkpoint_path, out_after_dump, extra_args=[
            '--dump-preds', masks_dir, '--dump-per-image-stats', stats_csv,
            '--dump-only', '--checkpoint-iter', '12345', '--run-name', 'synthetic_test',
        ])
        after_dump = load_json_ignore_timing(out_after_dump)
        assert after_dump == after, (
            f"JSON --output khi dump-only phai KHOP duong khong-dump (tru elapsed_sec):\n"
            f"NO_DUMP={after}\nDUMP={after_dump}")
        print("PASS — seg_result/boundary_result khong doi du chay them nhanh dump.\n")

        print("=== Kiem CSV/meta da duoc ghi dung ===")
        assert os.path.isfile(stats_csv), "Khong thay per_image_stats.csv"
        meta_path = os.path.splitext(stats_csv)[0] + '_meta.json'
        assert os.path.isfile(meta_path), "Khong thay <stem>_meta.json"
        with open(meta_path, encoding='utf-8') as f:
            meta = json.load(f)
        assert meta['checkpoint_iter'] == 12345
        assert meta['run_name'] == 'synthetic_test'
        assert meta['n_images'] == 3
        assert meta['class_names'] == list(OpenEarthMapDataset.CLASSES)
        assert meta['connectivity'] == 4
        assert meta['ignore_index'] == IGNORE
        assert meta['empty_image_policy_observed'] is None
        print("PASS — meta.json co du truong, dung gia tri da truyen qua CLI.\n")

        df = pd.read_csv(stats_csv)
        assert len(df) == 3, f"CSV phai co dung 3 hang (3 anh), duoc {len(df)}"
        assert list(df['image']) == names, (
            f"Thu tu/ten anh trong CSV phai khop dung val_ds.samples: {list(df['image'])} vs {names}")
        print(f"PASS — CSV co {len(df)} hang, dung thu tu ten anh {names}.\n")

        print("=== Gate 5.8 (co so): inter/union tu PNG mask phai khop CSV cho ca 3 anh ===")
        from PIL import Image
        # GT phai lay SAU KHI qua dung transform (get_val_transforms() resize
        # ve 1024x1024) — day chinh la mask pipeline that su dua vao khi eval,
        # KHONG phai .tif goc 48x48 (mask PNG du doan cung o resolution 1024).
        val_ds_check = OpenEarthMapDataset(
            root_dir=data_root, img_dir='images/val', mask_dir='labels/val',
            transform=get_val_transforms())
        gt_by_name = {}
        for i in range(len(val_ds_check)):
            img_name = os.path.splitext(os.path.basename(val_ds_check.samples[i][0]))[0]
            _img, mask_t = val_ds_check[i]
            gt_by_name[img_name] = mask_t.numpy().astype(np.int64)

        for name in names:
            mask_png = np.array(Image.open(os.path.join(masks_dir, name + '.png')))
            gt = gt_by_name[name]
            assert gt.shape == mask_png.shape, (
                f"[{name}] shape GT {gt.shape} != shape mask PNG {mask_png.shape}")
            valid = gt != IGNORE
            row = df[df['image'] == name].iloc[0]
            for c, cname in enumerate(OpenEarthMapDataset.CLASSES):
                gt_c = (gt == c) & valid
                pred_c = (mask_png == c) & valid
                inter = int(np.logical_and(gt_c, pred_c).sum())
                union = int(np.logical_or(gt_c, pred_c).sum())
                assert inter == row[f'inter_{cname}'], (
                    f"[{name}] inter_{cname} tu PNG={inter} != CSV={row[f'inter_{cname}']}")
                assert union == row[f'union_{cname}'], (
                    f"[{name}] union_{cname} tu PNG={union} != CSV={row[f'union_{cname}']}")
        print("PASS — inter/union tinh lai tu PNG mask khop CHINH XAC voi CSV, ca 3 anh, 9 lop.\n")

        print("=== Doi chieu aggregate tu CSV (quy uoc micro) voi JSON --output ===")
        with open(out_after_dump, encoding='utf-8') as f:
            output_json = json.load(f)
        seg = output_json['segmentation']
        boundary = output_json['boundary']

        per_class_iou = []
        for cname in OpenEarthMapDataset.CLASSES:
            inter_sum = df[f'inter_{cname}'].sum()
            union_sum = df[f'union_{cname}'].sum()
            per_class_iou.append(inter_sum / (union_sum + 1e-10) if union_sum > 0 else 0.0)
        valid_classes = [df[f'union_{c}'].sum() > 0 for c in OpenEarthMapDataset.CLASSES]
        recomputed_miou = float(np.mean([v for v, ok in zip(per_class_iou, valid_classes) if ok]))
        assert abs(recomputed_miou - seg['mIoU']) < 1e-9, (
            f"mIoU tai lap tu CSV (micro) lech: {recomputed_miou} vs {seg['mIoU']}")
        print(f"PASS — mIoU tai lap tu CSV (micro) = {recomputed_miou:.9f}, "
              f"khop JSON = {seg['mIoU']:.9f}.\n")

        bf_tp_pred, bf_n_pred = df['bf_tp_pred'].sum(), df['bf_n_pred'].sum()
        bf_tp_gt, bf_n_gt = df['bf_tp_gt'].sum(), df['bf_n_gt'].sum()
        precision = bf_tp_pred / bf_n_pred if bf_n_pred > 0 else 0.0
        recall = bf_tp_gt / bf_n_gt if bf_n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        assert abs(precision - boundary['bf_precision']) < 1e-9
        assert abs(recall - boundary['bf_recall']) < 1e-9
        assert abs(f1 - boundary['bf_score']) < 1e-9
        print(f"PASS — BF precision/recall/score tai lap tu CSV khop JSON "
              f"({precision:.6f}/{recall:.6f}/{f1:.6f}).\n")

        print("TAT CA GATE/KIEM TRA CUA test_eval_boundary_metrics_dump_gate.py PASS.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
