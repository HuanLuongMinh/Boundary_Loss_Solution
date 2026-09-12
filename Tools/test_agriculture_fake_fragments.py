"""Tools/test_agriculture_fake_fragments.py — Tester cho
Tools/agriculture_fake_fragments.py.

Du lieu SYNTHETIC — KHONG can checkpoint/model (Phan D chi doc PNG mask +
GT tu dataset, khong chay inference). Xac nhan:
  1. count_fake_fragments() dem dung tren cac truong hop biet truoc dap so
     (component trung khop hoan toan, component khong overlap gi, component
     overlap DUNG NGUONG — kiem bien tren/duoi).
  2. images_with_agriculture() loc dung anh co class_present_Agriculture=1,
     va BAT LOI khi 2 CSV (2 "checkpoint") lech nhau ve co Agriculture hay
     khong (thuoc tinh GT, phai giong het nhau).
  3. build_gt_agriculture_masks() + run() end-to-end tren 1 bo dataset .tif
     nho tu dung, voi PNG "pred" tu viet tay (khong qua model that) — kiem
     CSV ket qua khop dung so manh gia da thiet ke.

Chay: `python Tools/test_agriculture_fake_fragments.py` tu repo root.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from Tools.agriculture_fake_fragments import (
    connectivity_structure, count_fake_fragments, images_with_agriculture,
    build_gt_agriculture_masks, run, AGRICULTURE_CLASS_ID,
)


def test_connectivity_structure():
    print("=== Test 1: connectivity_structure() ===")
    s4 = connectivity_structure(4)
    assert s4.tolist() == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    s8 = connectivity_structure(8)
    assert s8.tolist() == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    try:
        connectivity_structure(6)
        raise AssertionError("Le ra phai raise voi connectivity khong hop le")
    except ValueError:
        pass
    print("PASS\n")


def test_count_fake_fragments_known_cases():
    print("=== Test 2: count_fake_fragments tren cac truong hop biet truoc dap so ===")
    H, W = 20, 20
    gt = np.zeros((H, W), dtype=bool)
    gt[2:6, 2:6] = True   # 1 component GT, dien tich 16

    pred = np.zeros((H, W), dtype=bool)
    pred[2:6, 2:6] = True     # component A: trung khop hoan toan GT (overlap=100%) -> KHONG gia
    pred[15:18, 15:18] = True  # component B: cach xa GT, overlap=0% -> GIA, dien tich 9

    result = count_fake_fragments(pred, gt, connectivity=4, overlap_threshold=0.05)
    assert result['n_components_gt'] == 1
    assert result['n_components_pred'] == 2
    assert result['n_fake_components'] == 1, f"Ky vong 1 manh gia, duoc {result['n_fake_components']}"
    assert result['fake_area_total'] == 9
    print(f"PASS (2 component pred, 1 trung GT, 1 xa GT -> {result})\n")

    # Kiem bien nguong: component overlap DUNG 5% (duoi nguong <5% -> GIA;
    # dung bang 5% -> KHONG duoi nguong -> KHONG gia, vi dieu kien la strict <).
    pred2 = np.zeros((H, W), dtype=bool)
    pred2[10:12, 10:20] = True  # component dien tich 20, dat 1 pixel de overlap = 1/20 = 5%
    gt2 = np.zeros((H, W), dtype=bool)
    gt2[10, 10] = True  # 1 pixel GT, nam trong component pred2 -> overlap=1, frac=1/20=0.05

    r_at_threshold = count_fake_fragments(pred2, gt2, connectivity=4, overlap_threshold=0.05)
    assert r_at_threshold['n_fake_components'] == 0, (
        f"overlap dung 5%% (khong < 5%%) khong duoc tinh la gia, duoc {r_at_threshold}")
    r_below_threshold = count_fake_fragments(pred2, gt2, connectivity=4, overlap_threshold=0.06)
    assert r_below_threshold['n_fake_components'] == 1, (
        f"overlap 5%% < nguong 6%% phai tinh la gia, duoc {r_below_threshold}")
    print("PASS (bien nguong: overlap dung bang nguong -> khong gia; duoi nguong -> gia)\n")

    empty_pred = np.zeros((H, W), dtype=bool)
    r_empty = count_fake_fragments(empty_pred, gt, connectivity=4, overlap_threshold=0.05)
    assert r_empty == {'n_components_pred': 0, 'n_components_gt': 1,
                        'n_fake_components': 0, 'fake_area_total': 0}
    print("PASS (pred rong -> 0 component, 0 manh gia, khong crash)\n")


def test_images_with_agriculture_consistency():
    print("=== Test 3: images_with_agriculture() loc dung + bat loi khi CSV lech nhau ===")
    tmp = tempfile.mkdtemp(prefix='agri_test_')
    try:
        df1 = pd.DataFrame({'image': ['a', 'b', 'c'], 'class_present_Agriculture': [1, 0, 1]})
        df2 = pd.DataFrame({'image': ['a', 'b', 'c'], 'class_present_Agriculture': [1, 0, 1]})
        p1, p2 = os.path.join(tmp, '1.csv'), os.path.join(tmp, '2.csv')
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)
        result = images_with_agriculture({'cfg1': p1, 'cfg2': p2})
        assert result == ['a', 'c'], f"Ky vong ['a','c'], duoc {result}"
        print(f"PASS (loc dung anh co Agriculture, nhat quan giua 2 CSV): {result}")

        df3 = pd.DataFrame({'image': ['a', 'b', 'c'], 'class_present_Agriculture': [1, 1, 1]})
        p3 = os.path.join(tmp, '3.csv')
        df3.to_csv(p3, index=False)
        try:
            images_with_agriculture({'cfg1': p1, 'cfg3': p3})
            raise AssertionError("Le ra phai raise khi 2 CSV lech nhau ve class_present_Agriculture")
        except AssertionError as e:
            assert 'GT' in str(e) or 'khac nhau' in str(e).lower()
        print("PASS (2 CSV lech nhau ve GT Agriculture -> raise)\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_tif(path, array, dtype):
    array = np.asarray(array)
    count = 1 if array.ndim == 2 else array.shape[0]
    h, w = array.shape[-2:]
    transform = from_origin(0, 0, 1, 1)
    with rasterio.open(path, 'w', driver='GTiff', height=h, width=w, count=count,
                        dtype=dtype, transform=transform) as dst:
        if array.ndim == 2:
            dst.write(array.astype(dtype), 1)
        else:
            for i in range(count):
                dst.write(array[i].astype(dtype), i + 1)


def test_end_to_end_synthetic_dataset():
    print("=== Test 4: end-to-end tren dataset .tif nho + PNG pred viet tay ===")
    tmp = tempfile.mkdtemp(prefix='agri_e2e_')
    try:
        img_dir = os.path.join(tmp, 'data', 'images', 'val')
        mask_dir = os.path.join(tmp, 'data', 'labels', 'val')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        H = W = 16  # se duoc resize len 1024x1024 boi get_val_transforms()
        # anh0: co Agriculture (2 vung tach biet trong GT). anh1: KHONG co Agriculture.
        label0 = np.zeros((H, W), dtype=np.uint8)
        label0[2:5, 2:5] = AGRICULTURE_CLASS_ID
        label0[10:13, 10:13] = AGRICULTURE_CLASS_ID
        label1 = np.zeros((H, W), dtype=np.uint8)  # toan Background, khong Agriculture

        write_tif(os.path.join(img_dir, 'img0.tif'),
                  np.random.default_rng(1).integers(0, 255, (3, H, W), dtype=np.uint8), 'uint8')
        write_tif(os.path.join(mask_dir, 'img0.tif'), label0, 'uint8')
        write_tif(os.path.join(img_dir, 'img1.tif'),
                  np.random.default_rng(2).integers(0, 255, (3, H, W), dtype=np.uint8), 'uint8')
        write_tif(os.path.join(mask_dir, 'img1.tif'), label1, 'uint8')

        cfg = {'DATASET': {
            'ROOT_DIR': os.path.join(tmp, 'data'), 'TRAIN_IMG_DIR': 'images/train',
            'TRAIN_MASK_DIR': 'labels/train', 'VAL_ROOT_DIR': os.path.join(tmp, 'data'),
            'VAL_IMG_DIR': 'images/val', 'VAL_MASK_DIR': 'labels/val',
        }}
        config_path = os.path.join(tmp, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f)

        gt_by_name = build_gt_agriculture_masks(config_path)
        assert set(gt_by_name.keys()) == {'img0', 'img1'}
        assert gt_by_name['img0'].sum() > 0, "img0 phai co pixel Agriculture sau resize"
        assert gt_by_name['img1'].sum() == 0, "img1 khong co Agriculture"
        RH, RW = gt_by_name['img0'].shape
        print(f"PASS (build_gt_agriculture_masks: 2 anh, resolution sau transform = {RH}x{RW})")

        # Stats CSV: chi img0 co Agriculture (dung dinh nghia muc 8.3).
        stats_df = pd.DataFrame({'image': ['img0', 'img1'], 'class_present_Agriculture': [1, 0]})
        stats_path = os.path.join(tmp, 'stats.csv')
        stats_df.to_csv(stats_path, index=False)

        # 2 "cau hinh": cfg_good du doan gan dung GT (0 manh gia), cfg_bad them
        # 1 manh o giua anh (xa GT) -> 1 manh gia.
        good_dir = os.path.join(tmp, 'masks_good')
        bad_dir = os.path.join(tmp, 'masks_bad')
        os.makedirs(good_dir, exist_ok=True)
        os.makedirs(bad_dir, exist_ok=True)

        pred_good = np.zeros((RH, RW), dtype=np.uint8)
        pred_good[gt_by_name['img0']] = AGRICULTURE_CLASS_ID  # trung khop hoan toan GT
        Image.fromarray(pred_good, mode='L').save(os.path.join(good_dir, 'img0.png'))
        Image.fromarray(np.zeros((RH, RW), dtype=np.uint8), mode='L').save(
            os.path.join(good_dir, 'img1.png'))

        pred_bad = pred_good.copy()
        cy, cx = RH // 2, RW // 2
        pred_bad[cy:cy + 20, cx:cx + 20] = AGRICULTURE_CLASS_ID  # manh gia moi, xa 2 vung GT
        Image.fromarray(pred_bad, mode='L').save(os.path.join(bad_dir, 'img0.png'))
        Image.fromarray(np.zeros((RH, RW), dtype=np.uint8), mode='L').save(
            os.path.join(bad_dir, 'img1.png'))

        agriculture_images = images_with_agriculture({'good': stats_path, 'bad': stats_path})
        assert agriculture_images == ['img0']

        df = run({'good': good_dir, 'bad': bad_dir}, gt_by_name, agriculture_images,
                 connectivity=4, overlap_threshold=0.05)
        assert len(df) == 2, f"Ky vong 2 hang (1 anh x 2 cau hinh), duoc {len(df)}"

        row_good = df[df['config'] == 'good'].iloc[0]
        row_bad = df[df['config'] == 'bad'].iloc[0]
        assert row_good['n_fake_components'] == 0, f"cau hinh 'good' khong duoc co manh gia: {row_good}"
        assert row_good['n_components_gt'] == 2, "GT phai co 2 component tach biet"
        assert row_bad['n_fake_components'] == 1, f"cau hinh 'bad' phai co dung 1 manh gia: {row_bad}"
        assert row_bad['fake_area_total'] == 400, f"manh gia 20x20=400px, duoc {row_bad['fake_area_total']}"
        print(f"PASS end-to-end — 'good': {dict(row_good)}, 'bad': {dict(row_bad)}\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_connectivity_structure()
    test_count_fake_fragments_known_cases()
    test_images_with_agriculture_consistency()
    test_end_to_end_synthetic_dataset()
    print("Tat ca self-test Tools/agriculture_fake_fragments.py PASS.")
