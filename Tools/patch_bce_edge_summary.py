"""Tools/patch_bce_edge_summary.py — Điền số thật vào 4 dòng "N/A" của 1
`summary.txt` Run 2 (BCE edge) đã sinh sẵn bởi src/train_bce_edge.py, dùng kết
quả JSON từ Tools/eval_boundary_metrics.py. File MỚI, KHÔNG import/sửa
src/train_bce_edge.py — chỉ patch text theo đúng 4 nhãn dòng CỐ ĐỊNH đã có sẵn
trong write_bce_edge_summary() (nếu 4 nhãn này đổi trong tương lai, script sẽ
raise lỗi rõ ràng thay vì âm thầm không patch được dòng nào).

4 dòng cần điền (đúng nguyên văn label, xem
src/train_bce_edge.py::write_bce_edge_summary()):
    Best validation BFScore     : N/A -- ...
    Best Boundary IoU @1        : N/A -- ...
    Best Boundary IoU @2        : N/A -- ...
    Best Boundary IoU @4        : N/A -- ...

Vì sao @1/@2/@4 (không phải @1/@3/@5 như spec tổng quát docs/workflow_2.md
mục 4): 3 nhãn dòng này đã được ghi cứng vào code khi Run 2 chạy thật — xem
docstring src/utils/boundary_metrics.py và Tools/eval_boundary_metrics.py để
biết chi tiết mâu thuẫn này. Script này chỉ đọc đúng key
`boundary_iou_d1`/`boundary_iou_d2`/`boundary_iou_d4` từ JSON — JSON đó phải
được sinh ra bởi lệnh có `--boundary-distances 1,2,4` (mặc định của
eval_boundary_metrics.py), nếu không sẽ báo lỗi thiếu key ngay.

CHỈ patch text — KHÔNG cần model/GPU/dataset. Luôn ghi ra 1 FILE MỚI, KHÔNG
BAO GIỜ ghi đè summary.txt gốc (đã là output của 1 run huấn luyện đã hoàn
tất) — mọi dòng khác ngoài 4 dòng trên được giữ NGUYÊN VĂN.

Cách dùng:

    python Tools/patch_bce_edge_summary.py \\
        --summary /path/to/work_dir/summary.txt \\
        --metrics-json docs/results/run2_lambda04_boundary_metrics.json \\
        --output /path/to/work_dir/summary_boundary_filled.txt
"""

import argparse
import json
import os


# Nhãn dòng CỐ ĐỊNH, đúng nguyên văn src/train_bce_edge.py::write_bce_edge_summary()
# — key JSON tương ứng lấy từ Tools/eval_boundary_metrics.py.
_TARGET_LINES = [
    ('Best validation BFScore', lambda m: m['boundary']['bf_score']),
    ('Best Boundary IoU @1',    lambda m: m['boundary']['boundary_iou_d1']),
    ('Best Boundary IoU @2',    lambda m: m['boundary']['boundary_iou_d2']),
    ('Best Boundary IoU @4',    lambda m: m['boundary']['boundary_iou_d4']),
]


def format_value(label: str, value: float) -> str:
    if value != value:  # NaN check (không import math chỉ cho 1 chỗ)
        return 'N/A (khong co pixel bien hop le nao trong val set)'
    return f'{value:.4f}'


def patch_summary(lines: list, metrics: dict) -> tuple:
    """Trả về (lines_moi, so_dong_da_patch)."""
    patched = list(lines)
    n_patched = 0
    for i, line in enumerate(patched):
        if ':' not in line:
            continue
        prefix, _, _rest = line.partition(':')
        label = prefix.strip()
        for target_label, getter in _TARGET_LINES:
            if label == target_label:
                value = getter(metrics)
                patched[i] = f'{prefix}: {format_value(target_label, value)}'
                n_patched += 1
                break
    return patched, n_patched


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--summary', required=True, help='Đường dẫn summary.txt gốc (Run 2, còn N/A)')
    ap.add_argument('--metrics-json', required=True,
                    help='JSON output của Tools/eval_boundary_metrics.py (--boundary-distances 1,2,4)')
    ap.add_argument('--output', required=True,
                    help='Đường dẫn FILE MỚI để ghi ra — KHÔNG được trùng --summary '
                         '(script từ chối ghi đè summary.txt gốc)')
    args = ap.parse_args()

    if os.path.abspath(args.output) == os.path.abspath(args.summary):
        raise ValueError(
            "--output không được trùng --summary — script này chỉ tạo file mới, "
            "không ghi đè summary.txt gốc. Chọn 1 đường dẫn khác cho --output.")

    with open(args.summary, encoding='utf-8') as f:
        original_lines = f.read().splitlines()

    with open(args.metrics_json, encoding='utf-8') as f:
        metrics = json.load(f)

    for target_label, getter in _TARGET_LINES:
        try:
            getter(metrics)
        except KeyError as e:
            raise KeyError(
                f"Thiếu key {e} trong {args.metrics_json} — JSON này phải được sinh ra bởi "
                f"Tools/eval_boundary_metrics.py với --boundary-distances 1,2,4 (mặc định) "
                f"để khớp đúng nhãn '{target_label}' trong summary.txt.") from e

    patched_lines, n_patched = patch_summary(original_lines, metrics)
    if n_patched != len(_TARGET_LINES):
        raise RuntimeError(
            f"Chỉ tìm thấy {n_patched}/{len(_TARGET_LINES)} dòng cần patch trong {args.summary} — "
            f"nhãn dòng trong file có thể đã khác với src/train_bce_edge.py::write_bce_edge_summary() "
            f"hiện tại. Kiểm tra lại file summary.txt đầu vào trước khi dùng script này.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write('\n'.join(patched_lines) + '\n')

    print(f"Đã patch {n_patched} dòng (BFScore + Boundary IoU @1/@2/@4) từ {args.metrics_json}")
    print(f"Đã ghi file mới: {args.output}")
    print(f"(file gốc không bị đụng tới: {args.summary})")


if __name__ == '__main__':
    main()
