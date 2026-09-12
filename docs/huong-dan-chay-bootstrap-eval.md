# Hướng dẫn chạy — Bootstrap CI + đếm mảnh giả Agriculture

Tài liệu này hướng dẫn chạy phần triển khai của
`docs/spec-per-image-bootstrap-eval-v2-ban-giao-claude-code.md`. Code mới là
**nhánh phụ thuần cộng thêm** — không sửa hành vi mặc định của
`Tools/eval_boundary_metrics.py`, không chạm `src/losses/*`, không train lại
gì. Tất cả 3 phần (A/C/D) chạy độc lập với nhau và độc lập với các script
khác trong `Tools/`.

## 0. Đã làm gì, và giới hạn của phiên làm việc này

Phiên này **không có checkpoint (`.pth`) hay dataset ảnh thật** trong máy —
hai thứ đó chỉ tồn tại trên Kaggle. Vì vậy:

- Toàn bộ code (Phần A/C/D) đã được viết và **tự-test bằng dữ liệu tổng hợp**
  (ảnh `.tif` nhỏ tự tạo, tensor ngẫu nhiên, CSV thống kê giả) — xem mục 4.
- **Chưa** chạy được gate 5.1–5.5 thật (đối chiếu số với Run 1–5 thật) vì
  không có checkpoint/JSON thật ở đây. Bạn cần tự chạy Phần B (dump 7
  checkpoint) trên Kaggle rồi chạy `Tools/bootstrap_boundary_ci.py` với
  `--reference` trỏ tới JSON đã công bố để gate đó chạy thật.
- Việc đọc kết quả cuối (mục 7 của spec, viết `claude/ket-qua-bootstrap-ci-...md`)
  vẫn cần **bạn** làm sau khi có số thật — không tự bịa số.

## 1. Tổng quan các file mới

| File | Vai trò |
|---|---|
| `Tools/per_image_dump.py` | Tính thống kê đủ per-image + lưu mask PNG (helper của Phần A) |
| `Tools/eval_boundary_metrics.py` | **Đã sửa** — thêm 3 cờ `--dump-preds/--dump-per-image-stats/--dump-only` (Phần A). Không cờ nào ⇒ y hệt trước khi sửa. |
| `Tools/bootstrap_boundary_ci.py` | Giai đoạn 2 — đọc CSV đã dump, tái lập số gộp, bootstrap ghép cặp (Phần C) |
| `Tools/agriculture_fake_fragments.py` | Đếm mảnh giả Agriculture trên mask đã dump (Phần D) |
| `Tools/test_per_image_dump.py` | Tự-test Phần A (đơn vị) |
| `Tools/test_eval_boundary_metrics_dump_gate.py` | Gate 4.0 + cơ sở gate 5.6–5.8 (tích hợp, dữ liệu giả) |
| `Tools/test_bootstrap_boundary_ci.py` | Tự-test Phần C (đơn vị + gate 5.6/5.7) |
| `Tools/test_agriculture_fake_fragments.py` | Tự-test Phần D |
| `docs/huong-dan-chay-bootstrap-eval.md` | Chính là file này |
| `.gitignore` | Thêm `output/` (kết quả sinh ra, không commit) |

Mọi kết quả sinh ra khi chạy thật đều nằm trong thư mục `output/` ở gốc repo
(đã thêm vào `.gitignore`) — không lẫn với `docs/results/` hiện có.

## 2. Chạy test (làm TRƯỚC, không cần checkpoint/dataset thật)

```bash
python Tools/test_per_image_dump.py
python Tools/test_eval_boundary_metrics_dump_gate.py
python Tools/test_bootstrap_boundary_ci.py
python Tools/test_agriculture_fake_fragments.py
```

Cả 4 file đều tự tạo dữ liệu giả, in `PASS` từng bước, dừng ngay (raise) nếu
sai — không có test nào cần mạng, GPU, hay dataset OpenEarthMap thật.
`test_eval_boundary_metrics_dump_gate.py` mất khoảng 15–30 giây (dựng model
UNetFormer/ResNet-18 thật, chạy forward pass CPU 3 lần).

## 3. Phần A — dump 7 checkpoint (chạy trên Kaggle, hoặc máy có dataset thật)

Cú pháp không đổi so với trước, chỉ thêm cờ. Ví dụ cho checkpoint #1 (Run 3
Static, seed 19, iter 36000):

```bash
python Tools/eval_boundary_metrics.py \
    --config configs/unet_former_resnet18_static_boundary/static_boundary.yaml \
    --checkpoint /path/to/run3_static_s19/best_model.pth \
    --model-type static_boundary \
    --dump-preds output/dump/run3_static_s19_iter36000/masks \
    --dump-per-image-stats output/dump/run3_static_s19_iter36000/per_image_stats.csv \
    --run-name run3_static_boundary --checkpoint-iter 36000 \
    --dump-only
```

Lặp lại cho đủ 7 checkpoint theo bảng mục 3 của spec, đổi
`--config`/`--checkpoint`/`--model-type`/`--run-name`/`--checkpoint-iter` và
đường dẫn `output/dump/<run_key>_iter<iter>/...` cho từng cái:

| `<run_key>` gợi ý | Config | model-type | iter |
|---|---|---|---|
| `run1_baseline` | `configs/unet_former_resnet18_combineLoss/baseline.yaml` | `baseline` | 40000 |
| `run2b_bce04` | `configs/unet_former_resnet18_bce_edge/bce_edge.yaml` | `bce_edge` | 40000 |
| `run3_static_s19` | `configs/unet_former_resnet18_static_boundary/static_boundary.yaml` | `static_boundary` | 36000 |
| `run3b_aff02` | `configs/unet_former_resnet18_static_boundary/static_boundary.yaml` (bản α=0.2) | `static_boundary` | 40000 |
| `run4_dynamic` | `configs/unet_former_resnet18_dynamic_boundary/*.yaml` | tuỳ script Run 4 dùng | 40000 |
| `run5_static_s86_best` | static_boundary (seed 86) | `static_boundary` | 36000 |
| `run5_static_s86_final` | static_boundary (seed 86) | `static_boundary` | 40000 |

Bỏ `--dump-only` nếu muốn vẫn thấy báo cáo gộp in ra console/`--output` JSON
như cách dùng cũ — không bắt buộc phải dùng cùng lúc.

**Trước khi tin bất kỳ số dump nào**: chạy gate 4.0 thật trên máy Kaggle nếu
muốn (không bắt buộc vì đã chứng minh bằng test synthetic ở mục 2, nhưng nếu
muốn chắc chắn tuyệt đối trên đúng môi trường Kaggle):

```bash
# Chạy 1 checkpoint KHÔNG cờ mới, lưu JSON, so với JSON cũ đã có trong docs/results/
python Tools/eval_boundary_metrics.py --config ... --checkpoint ... --model-type ... --output /tmp/check.json
diff /tmp/check.json docs/results/run1_boundary_metrics.json
```

## 4. Phần C — bootstrap CI ghép cặp

Sau khi có đủ CSV `per_image_stats.csv` cho các checkpoint cần dùng:

```bash
python Tools/bootstrap_boundary_ci.py \
  --checkpoint run1_baseline=output/dump/run1_baseline_iter40000/per_image_stats.csv \
  --checkpoint run2b_bce04=output/dump/run2b_bce04_iter40000/per_image_stats.csv \
  --checkpoint run3_static_s19=output/dump/run3_static_s19_iter36000/per_image_stats.csv \
  --checkpoint run3b_aff02=output/dump/run3b_aff02_iter40000/per_image_stats.csv \
  --checkpoint run4_dynamic=output/dump/run4_dynamic_iter40000/per_image_stats.csv \
  --checkpoint run5_static_s86_best=output/dump/run5_static_s86_best_iter36000/per_image_stats.csv \
  --reference run1_baseline=docs/results/run1_boundary_metrics.json \
  --reference run2b_bce04=docs/results/run2_lambda04_boundary_metrics.json \
  --reference run3_static_s19=docs/results/run3_static_boundary_boundary_metrics.json \
  --n-boot 10000 --seed 19 \
  --output-dir output/bootstrap
```

Ghi chú quan trọng:

- **Nhãn checkpoint dùng đúng tên** trong bảng ở mục 3 (`run1_baseline`,
  `run2b_bce04`, `run3_static_s19`, `run3b_aff02`, `run4_dynamic`,
  `run5_static_s86_best`) — script dùng đúng 6 tên này để tự ghép 6 cặp mặc
  định của mục 4.4 spec. Muốn tự định nghĩa cặp khác thì thêm
  `--pair NHAN_A=NHAN_B` (lặp lại), sẽ ghi đè toàn bộ mặc định.
- `--reference` là **tuỳ chọn nhưng nên truyền** — đây là JSON gốc do chính
  `Tools/eval_boundary_metrics.py --output` sinh ra lúc trước (không cần
  cờ dump). Có nó, script tự DÒ xem quy ước gộp là `micro` hay `macro` và
  chính sách ảnh rỗng nào tái lập đúng số đã công bố (mục 4.2 spec), in ra
  từng trường lệch bao nhiêu. **Không có** `--reference` nào, script mặc định
  dùng `micro` (đây chính là quy ước thật của `SegmentationMetrics`/
  `BoundaryMetrics` — đã đọc code xác nhận, không phải đoán).
- Nếu gate dò quy ước **không tìm được combo nào khớp** trong ngưỡng
  `--gate-tol` (mặc định `1e-6`), script sẽ dừng và báo lỗi — đúng tinh thần
  "DỪNG LẠI, báo cáo, không đoán" của mục 4.2/5.1 spec.
- **⚠ Cặp #6** (`run5_static_s86_best` − `run3_static_s19`): mục 4.4 của spec
  đặt TÊN cặp là "Static s19 − Static s86", nhưng mục 7.5 chốt **hướng tính
  thực sự dùng để báo cáo là ngược lại** (s86 − s19, giữ dấu +0.5343px quan
  sát được). Code đã cài đặt đúng theo mục 7.5 (minuend = s86, trừ = s19) —
  nếu đọc lại spec thấy khác, đây là chỗ đầu tiên cần đối chiếu.

Output: `output/bootstrap/bootstrap_ci.csv` (1 hàng/cặp×đại lượng) và
`bootstrap_ci.md` (bảng đọc được, dán thẳng vào tài liệu). **Áp dụng bảng
7.1–7.6 của spec thủ công** dựa trên các số này — script không tự viết verdict
diễn giải (cố ý, để tránh áp sai các quy tắc đọc tinh tế của mục 7.5/7.6).

## 5. Phần D — đếm mảnh giả Agriculture

Cần mask PNG đã dump ở Phần A cho đúng 4 cấu hình (Baseline, BCE λ=0.4,
Static seed19, Run 3b α=0.2 — mục 8.2 spec):

```bash
python Tools/agriculture_fake_fragments.py \
  --config configs/unet_former_resnet18_combineLoss/baseline.yaml \
  --mask-dir baseline=output/dump/run1_baseline_iter40000/masks \
  --mask-dir bce04=output/dump/run2b_bce04_iter40000/masks \
  --mask-dir static_s19=output/dump/run3_static_s19_iter36000/masks \
  --mask-dir run3b_aff02=output/dump/run3b_aff02_iter40000/masks \
  --per-image-stats baseline=output/dump/run1_baseline_iter40000/per_image_stats.csv \
  --per-image-stats bce04=output/dump/run2b_bce04_iter40000/per_image_stats.csv \
  --per-image-stats static_s19=output/dump/run3_static_s19_iter36000/per_image_stats.csv \
  --per-image-stats run3b_aff02=output/dump/run3b_aff02_iter40000/per_image_stats.csv \
  --output-dir output/agriculture
```

`--config` chỉ dùng phần `DATASET` (để dựng lại GT đúng resolution sau
resize, khớp với mask PNG đã dump) — dùng config nào trong 4 cái cũng được,
miễn cùng val split. Ngưỡng "mảnh giả" mặc định 5% diện tích component pred
(`--overlap-threshold 0.05`), đổi được qua CLI nếu muốn thử ngưỡng khác (mục
8.3 bước 4 — ghi rõ ngưỡng đã chọn khi báo cáo).

Output: `output/agriculture/agriculture_fake_fragments.csv` (per-image,
per-config) và `agriculture_fake_fragments_summary.md` (bảng gộp 4 cấu hình).

## 6. Việc còn lại sau khi có số thật (không tự động hoá)

1. Chạy gate 5.1–5.8 và ghi lại từng gate PASS/FAIL riêng (script đã in gate
   5.1–5.5/5.7 tự động khi có `--reference`; gate 5.6/5.8 đã chứng minh bằng
   test synthetic, có thể chạy lại thủ công trên dữ liệu thật nếu muốn).
2. Áp bảng 7.1–7.6 nguyên văn, viết verdict vào tài liệu mới (spec gợi ý tên
   `claude/ket-qua-bootstrap-ci-ket-luan-do-lon.md`).
3. Đối chiếu bảng biên độ liên-seed Run 5 (mục 7.6) trước khi in bất kỳ CI
   nào vào bài.
4. Đọc `agriculture_fake_fragments_summary.md` cùng bảng oracle C5b để xem
   thứ tự "ít mảnh giả nhất" có khớp thứ tự Δ Agriculture hay không (mục 8.4).
