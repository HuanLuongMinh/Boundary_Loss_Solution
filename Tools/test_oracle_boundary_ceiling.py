"""Tools/test_oracle_boundary_ceiling.py — Tester cho Tools/oracle_boundary_ceiling.py.

Chay bang du lieu SYNTHETIC (khong can checkpoint/model that, khong can GPU) —
xac nhan logic cot loi (band/oracle substitution, cong 5.1/5.2/5.3, validate
so checkpoint tu CLI) dung TRUOC KHI chay that tren Kaggle voi checkpoint that.

Chay: `python Tools/test_oracle_boundary_ceiling.py` tu repo root.
Quy uoc giong self-test co san o cuoi src/utils/boundary_metrics.py — assert
+ print PASS tung phan, crash ngay neu sai (khong nuot loi).
"""

import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OpenEarthMapDataset
from src.losses.boundary_bce import extract_edge_gt
from src.utils.metrics import SegmentationMetrics
from Tools.oracle_boundary_ceiling import (
    apply_oracle_boundary, apply_oracle_interior, band_pixel_counts,
    labels_to_pseudo_logits, accumulate_variant,
    gate_5_1_check, gate_5_1_relation_check, gate_5_1_per_class_check,
    gate_5_2_check, gate_5_3_check,
    validate_run_specs, parse_int_list,
    compute_checkpoint_oracle, write_checkpoint_outputs, write_comparison_md,
)


IGNORE = OpenEarthMapDataset.IGNORE_INDEX
NUM_CLASSES = OpenEarthMapDataset.NUM_CLASSES


def make_synthetic_case():
    """32x32, bien doc o cot 16 (class 0 trai / class 1 phai). y_hat sai o 2
    vung TACH BIET: 1 khoi SAT bien (cot 15-16) va 1 khoi XA bien (goc tren
    trai, cot 0-2) — de phan biet ro oracle_boundary vs oracle_interior."""
    H = W = 32
    y = np.zeros((H, W), dtype=np.int64)
    y[:, W // 2:] = 1

    y_hat = y.copy()
    y_hat[:, 15:17] = 1 - y_hat[:, 15:17]          # sai SAT bien (cach bien <=1px)
    y_hat[0:6, 0:3] = 1 - y_hat[0:6, 0:3]          # sai XA bien (cach bien >=13px)

    mask_t = torch.from_numpy(y).unsqueeze(0)
    edge_gt, _valid = extract_edge_gt(mask_t, ignore_index=IGNORE, connectivity=4, dilation_radius=0)
    edge_np = edge_gt[0, 0].numpy().astype(bool)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(~edge_np).astype(np.float32)
    return y, y_hat, dist


def miou9_of(y_pred_array, y):
    sm = SegmentationMetrics(NUM_CLASSES, IGNORE)
    y_t = torch.from_numpy(y).unsqueeze(0)
    accumulate_variant(sm, y_pred_array, y_t, NUM_CLASSES)
    return sm.compute()['mIoU']


def test_apply_oracle_boundary_fixes_only_near_boundary():
    y, y_hat, dist = make_synthetic_case()
    y_ob0 = apply_oracle_boundary(y, y_hat, dist, d=0)
    # d=0 chi phu dung cot bien (dist==0) -> sai o cot 0-2 (xa bien) PHAI CON NGUYEN
    assert not np.array_equal(y_ob0[0:6, 0:3], y[0:6, 0:3]), "d=0 khong duoc dung toi vung xa bien"
    # sai sat bien (cot 15/16, dist<=1) co the van con sai o d=0 vi dist>0 tai 1 vai pixel — kiem d=1 phai het
    y_ob1 = apply_oracle_boundary(y, y_hat, dist, d=1)
    assert np.array_equal(y_ob1[:, 15:17], y[:, 15:17]), "d=1 phai sua het loi sat bien (cach bien <=1px)"
    assert not np.array_equal(y_ob1[0:6, 0:3], y[0:6, 0:3]), "d=1 (con nho) khong duoc dung toi vung xa bien"
    print("PASS: apply_oracle_boundary chi sua trong dai, giu nguyen ngoai dai")


def test_apply_oracle_interior_is_complement():
    y, y_hat, dist = make_synthetic_case()
    d = 3
    y_ob = apply_oracle_boundary(y, y_hat, dist, d)
    y_oi = apply_oracle_interior(y, y_hat, dist, d)
    band = dist <= d
    assert np.array_equal(y_ob[band], y[band]) and np.array_equal(y_ob[~band], y_hat[~band])
    assert np.array_equal(y_oi[band], y_hat[band]) and np.array_equal(y_oi[~band], y[~band])
    print("PASS: oracle_boundary va oracle_interior la 2 nua bu tru chinh xac cua cung 1 band")


def test_gate_5_3_all_variant_gives_miou_one():
    y, y_hat, _dist = make_synthetic_case()
    miou_all = miou9_of(y, y)  # "all" variant: du doan = chinh y
    gate_5_3_check(miou_all)  # khong duoc raise
    try:
        gate_5_3_check(0.9)
        raise AssertionError("gate_5_3_check phai raise khi mIoU != 1.0")
    except RuntimeError:
        pass
    print("PASS: cong 5.3 (variant 'all' -> mIoU=1.0) dung ca nhanh pass va nhanh fail")


def test_gate_5_2_monotonic():
    gate_5_2_check([0.0, 0.01, 0.02, 0.02, 0.05])  # khong duoc raise (khong giam)
    try:
        gate_5_2_check([0.05, 0.02, 0.01])
        raise AssertionError("gate_5_2_check phai raise khi Delta giam")
    except RuntimeError:
        pass
    print("PASS: cong 5.2 (Delta don dieu) dung ca nhanh pass va nhanh fail")


def test_gate_5_2_real_curve_is_monotonic():
    y, y_hat, dist = make_synthetic_case()
    deltas = []
    base = miou9_of(y_hat, y)
    for d in (0, 1, 2, 4, 8, 16):
        y_ob = apply_oracle_boundary(y, y_hat, dist, d)
        deltas.append(miou9_of(y_ob, y) - base)
    gate_5_2_check(deltas)  # phai KHONG raise tren du lieu that (band long nhau)
    assert deltas[-1] > deltas[0], "d lon nhat phai sua duoc nhieu hon d=0 tren case co loi ngoai d=0"
    print(f"PASS: duong cong Delta thuc te don dieu: {[round(x, 4) for x in deltas]}")


def test_gate_5_1():
    gate_5_1_check(0.6551, 0.6183, None, 1e-3)  # None -> bo qua, khong raise
    gate_5_1_check(0.6551, 0.6183, 0.6551, 1e-3)  # khop mIoU-9 -> khong raise
    try:
        gate_5_1_check(0.6566, 0.6183, 0.6183, 1e-3)  # cai bay bce_lambda04: EXPECTED bi lay nham mIoU-8
        raise AssertionError("gate_5_1_check phai raise khi lech vuot nguong")
    except RuntimeError as e:
        msg = str(e)
        assert 'mIoU-9' in msg and 'mIoU-8' in msg, (
            "thong bao loi cong 5.1 phai in ca computed mIoU-9 lan mIoU-8 co nhan (muc 3.1 spec) "
            "de lo ngay neu EXPECTED bi lay nham quy uoc")
    print("PASS: cong 5.1 (khop none voi expected-miou) dung ca 3 nhanh, "
          "thong bao loi in ca mIoU-9 va mIoU-8")


def test_gate_5_1_relation_check():
    per_class = {'Background': 0.9609492}
    miou8 = 0.6168605
    miou9_ok = (8 * miou8 + per_class['Background']) / 9
    gate_5_1_relation_check(miou9_ok, miou8, per_class)  # khong raise

    try:
        gate_5_1_relation_check(0.6183, miou8, per_class)  # co tinh sai lech
        raise AssertionError("gate_5_1_relation_check phai raise khi quan he macro khong thoa")
    except RuntimeError:
        pass
    print("PASS: cong quan he mIoU-9/mIoU-8 (muc 3.3 spec) dung ca nhanh pass va nhanh fail")


def test_gate_5_1_per_class_check():
    computed = {'Background': 0.9609492, 'Bareland': 0.3646590, 'Water': 0.7278831}

    gate_5_1_per_class_check('t', computed, None, 1e-3)  # None -> bo qua, khong raise
    gate_5_1_per_class_check('t', computed, dict(computed), 1e-3)  # khop het -> khong raise

    expected_bad = {'Background': 0.9609492, 'Bareland': 0.60, 'Water': 0.50}  # 2 lop lech
    try:
        gate_5_1_per_class_check('t', computed, expected_bad, 1e-3)
        raise AssertionError("gate_5_1_per_class_check phai raise khi co lop lech qua nguong")
    except RuntimeError as e:
        msg = str(e)
        assert 'Bareland' in msg and 'Water' in msg, (
            "thong bao loi phai liet ke TAT CA lop lech (Bareland VA Water), khong dung o lop dau")
    print("PASS: cong per-class (muc 3.2 spec) dung ca 3 nhanh, bao cao tat ca lop lech chu khong "
          "dung o lop dau")


def test_band_pixel_counts_hand_computed():
    y = np.array([[0, 0, 1, 1]], dtype=np.int64)
    y_hat = np.array([[0, 1, 1, 1]], dtype=np.int64)   # sai 1 pixel tai cot 1
    dist = np.array([[2.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    # d=1 -> band = dist<=1 = cot [1,2,3] (3 pixel valid trong band / 4 valid)
    p_num, p_den, e_num, e_den = band_pixel_counts(y, y_hat, dist, d=1, ignore_index=IGNORE)
    assert (p_num, p_den) == (3, 4), f"P_band sai: {(p_num, p_den)}"
    assert (e_num, e_den) == (1, 1), f"E_band sai: {(e_num, e_den)}"  # loi duy nhat (cot1) nam trong band
    print("PASS: band_pixel_counts khop tay tren mang nho")


def test_labels_to_pseudo_logits_roundtrip():
    y_pred = np.array([[0, 1, 8], [255, 3, 5]], dtype=np.int64)
    logits = labels_to_pseudo_logits(y_pred, NUM_CLASSES)
    recovered = logits.argmax(dim=1)[0].numpy()
    # pixel 255 (ignore) bi clip cho one-hot, khong can roundtrip dung — chi kiem
    # cac pixel nhan hop le (0..8) phai roundtrip CHINH XAC.
    valid = y_pred != 255
    assert np.array_equal(recovered[valid], y_pred[valid]), "pseudo-logits phai argmax lai dung nhan goc"
    print("PASS: labels_to_pseudo_logits roundtrip dung qua argmax (khong viet lai cong thuc mIoU)")


def test_validate_run_specs_counts_from_cli():
    args_ok = SimpleNamespace(
        checkpoint=['a.pth', 'b.pth'], config=['a.yaml', 'b.yaml'],
        model_type=['baseline', 'bce_edge'], label=['baseline', 'bce'],
        iter=[40000, 40000], expected_miou=[], expected_per_class=[],
    )
    specs = validate_run_specs(args_ok)
    assert len(specs) == 2, "N phai duoc dem tu so lan --checkpoint, khong hard-code"
    assert specs[0]['label'] == 'baseline' and specs[1]['model_type'] == 'bce_edge'
    assert specs[0]['expected_per_class'] is None, "khong truyen --expected-per-class -> None cho tat ca"

    args_5 = SimpleNamespace(
        checkpoint=[f'{i}.pth' for i in range(5)], config=[f'{i}.yaml' for i in range(5)],
        model_type=['baseline'] * 5, label=[f'l{i}' for i in range(5)],
        iter=[1000 * i for i in range(5)], expected_miou=[], expected_per_class=[],
    )
    assert len(validate_run_specs(args_5)) == 5, "phai chay dung duoc 5 checkpoint, khong chi 3/4"

    args_mismatch = SimpleNamespace(
        checkpoint=['a.pth', 'b.pth'], config=['a.yaml'],  # thieu 1
        model_type=['baseline', 'bce_edge'], label=['baseline', 'bce'],
        iter=[40000, 40000], expected_miou=[], expected_per_class=[],
    )
    try:
        validate_run_specs(args_mismatch)
        raise AssertionError("phai raise ValueError khi so luong cac co lech nhau")
    except ValueError:
        pass

    args_dup_label = SimpleNamespace(
        checkpoint=['a.pth', 'b.pth'], config=['a.yaml', 'b.yaml'],
        model_type=['baseline', 'bce_edge'], label=['same', 'same'],
        iter=[40000, 40000], expected_miou=[], expected_per_class=[],
    )
    try:
        validate_run_specs(args_dup_label)
        raise AssertionError("phai raise ValueError khi --label trung lap")
    except ValueError:
        pass

    args_skip = SimpleNamespace(
        checkpoint=['a.pth', 'b.pth'], config=['a.yaml', 'b.yaml'],
        model_type=['baseline', 'bce_edge'], label=['baseline', 'bce'],
        iter=[40000, 40000], expected_miou=['0.6551', 'skip'], expected_per_class=[],
    )
    specs_skip = validate_run_specs(args_skip)
    assert specs_skip[0]['expected_miou'] == 0.6551 and specs_skip[1]['expected_miou'] is None

    args_per_class = SimpleNamespace(
        checkpoint=['a.pth', 'b.pth'], config=['a.yaml', 'b.yaml'],
        model_type=['baseline', 'bce_edge'], label=['baseline', 'bce'],
        iter=[40000, 40000], expected_miou=[],
        expected_per_class=['{"Background": 0.9609492, "Water": 0.7278831}', 'skip'],
    )
    specs_pc = validate_run_specs(args_per_class)
    assert specs_pc[0]['expected_per_class'] == {'Background': 0.9609492, 'Water': 0.7278831}, (
        "--expected-per-class phai parse duoc JSON inline")
    assert specs_pc[1]['expected_per_class'] is None, "'skip' phai bo qua rieng 1 checkpoint"

    print("PASS: validate_run_specs dem dung N (2 va 5), bat loi lech do dai/label trung, "
          "va parse 'skip'/JSON inline tung checkpoint rieng le (ca --expected-miou va "
          "--expected-per-class)")


def test_parse_int_list():
    assert parse_int_list('0,1,2,4,8') == [0, 1, 2, 4, 8]
    assert parse_int_list('1,2,4') == [1, 2, 4]
    print("PASS: parse_int_list")


def test_output_pipeline_end_to_end():
    """Khong dung inference/model that — chi ghi thang cache GT/dist/pred roi
    goi dung pipeline tinh + xuat file (JSON/Markdown/PNG) nhu main() se lam,
    cho N=2 checkpoint gia lap, de bat loi format/khoa truoc khi len Kaggle."""
    y, y_hat, dist = make_synthetic_case()
    canonical_ids = ['img0']
    distances = [0, 1, 2, 4]
    interior_distances = [1, 2]
    class_names = OpenEarthMapDataset.CLASSES

    tmp_dir = tempfile.mkdtemp(prefix='oracle_test_')
    try:
        cache_dir = os.path.join(tmp_dir, 'cache')
        output_dir = os.path.join(tmp_dir, 'output')
        os.makedirs(os.path.join(cache_dir, '_gt'), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, '_dist'), exist_ok=True)
        np.save(os.path.join(cache_dir, '_gt', 'img0.npy'), y.astype(np.uint8))
        np.save(os.path.join(cache_dir, '_dist', 'img0.npy'), dist)

        all_results = {}
        for label, fix_far_error in (('cfgA', False), ('cfgB', True)):
            pred_dir = os.path.join(cache_dir, label, 'preds')
            os.makedirs(pred_dir, exist_ok=True)
            y_hat_variant = y_hat.copy()
            if fix_far_error:
                y_hat_variant[0:6, 0:3] = y[0:6, 0:3]  # cfgB: ruot vung tot hon cfgA
            np.save(os.path.join(pred_dir, 'img0.npy'), y_hat_variant.astype(np.uint8))

            spec = {'checkpoint': f'{label}.pth', 'config': f'{label}.yaml',
                    'model_type': 'baseline', 'label': label, 'iter': 40000,
                    'expected_miou': None}
            results = compute_checkpoint_oracle(spec, canonical_ids, cache_dir, distances,
                                                 interior_distances, NUM_CLASSES, IGNORE, class_names)
            write_checkpoint_outputs(spec, results, distances, interior_distances, class_names,
                                      output_dir, len(canonical_ids))
            all_results[label] = {'iter': spec['iter'], 'checkpoint': spec['checkpoint'],
                                   'model_type': spec['model_type'], 'results': results}

        write_comparison_md(all_results, distances, output_dir)

        for label in ('cfgA', 'cfgB'):
            out_dir = os.path.join(output_dir, label)
            assert os.path.exists(os.path.join(out_dir, 'oracle_results.json'))
            assert os.path.exists(os.path.join(out_dir, 'oracle_table.md'))
            assert os.path.exists(os.path.join(out_dir, 'oracle_curve.png'))
            with open(os.path.join(out_dir, 'oracle_results.json'), encoding='utf-8') as f:
                payload = json.load(f)
            assert payload['class_names'] == class_names, "class_names phai co trong JSON (muc 4.1)"
            assert 'oracle_boundary_d2' in payload['variants']
            assert set(payload['variants']['none']['per_class_iou'].keys()) == set(class_names), (
                "per_class_iou phai la dict co ten lop, KHONG duoc la mang thuan (bai hoc hoan doi "
                "Water<->Agriculture da neu o muc 4.1 spec)")

        comparison_path = os.path.join(output_dir, 'comparison.md')
        assert os.path.exists(comparison_path)
        with open(comparison_path, encoding='utf-8') as f:
            comparison_text = f.read()
        assert 'cfgA' in comparison_text and 'cfgB' in comparison_text, (
            "comparison.md phai generic theo N checkpoint thuc te, khong hard-code ten")

        print("PASS: pipeline xuat file (JSON/Markdown/PNG) chay dung cho N=2 checkpoint gia lap, "
              "khong crash, per-class kem ten lop, comparison.md generic theo N")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    tests = [
        test_apply_oracle_boundary_fixes_only_near_boundary,
        test_apply_oracle_interior_is_complement,
        test_gate_5_3_all_variant_gives_miou_one,
        test_gate_5_2_monotonic,
        test_gate_5_2_real_curve_is_monotonic,
        test_gate_5_1,
        test_gate_5_1_relation_check,
        test_gate_5_1_per_class_check,
        test_band_pixel_counts_hand_computed,
        test_labels_to_pseudo_logits_roundtrip,
        test_validate_run_specs_counts_from_cli,
        test_parse_int_list,
        test_output_pipeline_end_to_end,
    ]
    for t in tests:
        print(f"=== {t.__name__} ===")
        t()
        print()
    print(f"TAT CA {len(tests)} TEST PASS.")
