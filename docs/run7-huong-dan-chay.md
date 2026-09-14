# Hướng dẫn chạy — Run 7 (Affinity-only, seed nạp lúc chạy)

Tài liệu này hướng dẫn chạy phần triển khai của
`docs/spec-run7-affinity-only-ban-giao-claude-code.md`. Toàn bộ code là file
**MỚI** — không sửa `src/losses/total_loss.py`/`total_loss_weighted.py`,
không sửa bất kỳ `src/train_static_boundary*.py`/`train_bce_edge.py`, không
sửa `Tools/eval_boundary_metrics.py`/`Tools/bootstrap_boundary_ci.py`. Run 7
và mọi run cũ chạy lại độc lập bất kỳ lúc nào, không ảnh hưởng nhau.

## 0. Vì sao kiến trúc Run 7 không cần BoundaryHead

`src/losses/affinity.py::AffinityLoss` không có tham số học được (không
`nn.Parameter`/`Conv`/`Linear`) — nó là hàm loss chạy trực tiếp trên Fused
Feature vốn đã được decoder sinh ra sẵn. Vì vậy Run 7 dùng `UNetFormer` TRẦN
(giống hệt Run 1/baseline), không có head phụ nào — khác Run 3/3b cần thêm
`BoundaryHead` cho nhánh BCE. Hệ quả: checkpoint Run 7 có state_dict giống
hệt checkpoint baseline, nên `Tools/eval_boundary_metrics.py` dùng thẳng
`--model-type baseline` mà không cần sửa gì.

## 1. File mới của Run 7

| File | Vai trò |
|---|---|
| `configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml` | Config chính, liều α=0.4, `SEED` không đặt (bắt buộc `--seed`) |
| `configs/unet_former_resnet18_affinity_only/run7b_affinity_only_lambda02.yaml` | Liều phụ α=0.2 |
| `src/losses/total_loss_affinity_only.py` | `AffinityOnlyTotalLoss` — không có số hạng BCE nào |
| `src/train_affinity_only.py` | Script train, `--seed` bắt buộc, `{SEED}` bắt buộc trong `WORK_DIR` |
| `Tools/bootstrap_boundary_ci_run7.py` | Manifest tự sinh + cặp 7/8/9/10/11 + bảng liên-seed, output tách biệt |
| `Tools/test_bootstrap_boundary_ci_run7.py` | Tự-test (dữ liệu tổng hợp) |
| `docs/run7-huong-dan-chay.md` | Chính là file này |

## 2. Tự-test (làm TRƯỚC, không cần checkpoint/dataset thật)

```bash
python -m src.losses.total_loss_affinity_only
python Tools/test_bootstrap_boundary_ci_run7.py
```

File đầu tự-test công thức loss + gradient (dùng model UNetFormer thật, CPU,
seed cố định). File sau tự-test manifest scan, cặp #9 (contrast cộng tuyến —
đối chiếu với `aggregate_point_estimate()` thật, không phải công thức viết
tay lại), và bảng biên độ liên-seed — tất cả trên dữ liệu tổng hợp.

## 3. Dry-run (mỗi seed định chạy)

```bash
torchrun --nproc_per_node=2 src/train_affinity_only.py \
    --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \
    --seed 19 --dry-run
```

Nếu quên `--seed`, script dừng ngay với lỗi rõ ràng TRƯỚC khi khởi tạo
`torch.distributed` — không tự chọn seed mặc định (mục 1.3.2 spec). Kiểm tra
trong log: seed in đúng dòng đầu; hash danh sách file val (so thủ công với
Run 3/Run 5 — **chưa có giá trị hash tham chiếu cứng trong repo này để assert
tự động**, đây là việc cần làm thủ công); `alpha_bce_effective` luôn `0.0`;
`alpha_affinity_effective` luôn `0.4`; đồng nhất thức loss PASS; gradient
chảy tới decoder qua affinity PASS; `nvidia-smi` ra 2×T4 (kiểm tay, script
không tự assert).

## 4. Train thật

```bash
torchrun --nproc_per_node=2 src/train_affinity_only.py \
    --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \
    --seed 19
```

Seed thứ 2 (khuyến nghị 86, đúng vai trò Run 5 đã tạo cho Static — để có
biên độ liên-seed thật của chính Run 7 thay vì mượn tạm của Static):

```bash
torchrun --nproc_per_node=2 src/train_affinity_only.py \
    --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \
    --seed 86
```

Output: `work_dirs/phase1/run7_affinity_only_seed<N>/` (đường dẫn thật theo
`OUTPUT.WORK_DIR` trong config, Kaggle-absolute) — 4 checkpoint +
`checkpoint_index.json` + `benchmark_results.csv` + `summary.txt`.

## 5. Dump preds/per-image-stats (mỗi seed, cả 2 mốc)

`--model-type baseline` (không phải lỗi — xem mục 0):

```bash
python Tools/eval_boundary_metrics.py \
    --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \
    --checkpoint <work_dir>/best_miou.pth \
    --model-type baseline \
    --dump-preds output/dump/run7_affinity_only_seed19_best/masks \
    --dump-per-image-stats output/dump/run7_affinity_only_seed19_best/per_image_stats.csv \
    --run-name run7_affinity_only_seed19_best --checkpoint-iter <iter_that_hit_best_miou> \
    --dump-only

python Tools/eval_boundary_metrics.py \
    --config configs/unet_former_resnet18_affinity_only/run7_affinity_only.yaml \
    --checkpoint <work_dir>/final_iter40000.pth \
    --model-type baseline \
    --dump-preds output/dump/run7_affinity_only_seed19_final40000/masks \
    --dump-per-image-stats output/dump/run7_affinity_only_seed19_final40000/per_image_stats.csv \
    --run-name run7_affinity_only_seed19_final40000 --checkpoint-iter 40000 \
    --dump-only
```

**Trước khi tin `--model-type baseline` load đúng**: in vài key đầu của
state_dict checkpoint (`python -c "import torch; sd=torch.load('<ckpt>.pth', map_location='cpu'); print(list(sd.keys())[:5])"`)
và xác nhận KHÔNG có tiền tố `base.`/`boundary_head.` — tức đúng là
`UNetFormer` trần.

`--run-name` và `--checkpoint-iter` **bắt buộc phải truyền** (không có mặc
định hợp lý) — `Tools/bootstrap_boundary_ci_run7.py` đọc `checkpoint_iter` từ
`_meta.json` để dựng manifest; thiếu trường này, checkpoint sẽ bị BỎ QUA kèm
cảnh báo khi quét manifest (không đoán iteration).

**Lưu ý về seed trong `_meta.json`**: `Tools/eval_boundary_metrics.py`
(không bị sửa ở đây) hiện KHÔNG ghi trường `seed` riêng vào `_meta.json`.
`Tools/bootstrap_boundary_ci_run7.py` suy ra seed bằng regex `seed(\d+)` trên
`checkpoint_path` (ví dụ `.../run7_affinity_only_seed19/best_miou.pth` →
seed=19) — vì vậy đường dẫn checkpoint gốc PHẢI còn chứa `seed<N>` (đúng như
`{SEED}` trong `OUTPUT.WORK_DIR` đã đảm bảo).

## 6. Bootstrap CI cho Run 7 (cặp 7-11, tách biệt hoàn toàn `output/bootstrap/` cũ)

Cần sẵn `per_image_stats.csv` của 3 checkpoint cũ (baseline@40k, BCE λ=0.4@40k,
Static s19@36k — đã dump từ trước theo `docs/huong-dan-chay-bootstrap-eval.md`):

```bash
python Tools/bootstrap_boundary_ci_run7.py \
  --checkpoint run1_baseline=output/dump/run1_baseline_iter40000/per_image_stats.csv \
  --checkpoint run2b_bce04=output/dump/run2b_bce04_iter40000/per_image_stats.csv \
  --checkpoint run3_static_s19=output/dump/run3_static_s19_iter36000/per_image_stats.csv \
  --dump-dir output/dump \
  --n-boot 10000 --seed 19 \
  --output-dir output/bootstrap_run7
```

Script tự quét `output/dump/` tìm mọi checkpoint Run 7/7b, tự sinh
`output/dump/checkpoints_manifest_run7.json`, rồi tính cho **MỌI seed tìm
thấy** (không chỉ 1 seed "chính" — đúng nguyên tắc "không bỏ sót seed", mục
6.3 spec):

- Cặp #7 (`AffOnly@40k − Baseline@40k`), #8 (`AffOnly@40k − BCE04@40k`) —
  cho từng seed.
- Cặp #9 (contrast cộng tuyến 4 số hạng, `(AffOnly − Baseline) − (Static −
  BCE04)`) — chỉ tính khi cả `run1_baseline`/`run2b_bce04`/`run3_static_s19`
  đều được truyền qua `--checkpoint` VÀ seed đó có checkpoint `@40000`.
- Cặp #10 (`AffOnly(0.2) − AffOnly(0.4)`) — chỉ khi `run7b` cùng seed, cùng
  iteration 40000, có trong manifest.
- Cặp #11 (liên-seed) — chỉ khi ≥2 seed Run 7 có trong manifest.

Output: `output/bootstrap_run7/bootstrap_ci_run7.csv` + `.md`, và
`output/bootstrap_run7/run7_interseed_amplitude.md` (chỉ sinh khi ≥2 seed).
**Không đụng** `output/bootstrap/bootstrap_ci.csv` cũ — đây là lệch có chủ ý
so với mục 4.4 của spec gốc (spec muốn thêm hàng vào file cũ), ưu tiên theo
yêu cầu "file cũ/mới độc lập tuyệt đối" đã thống nhất khi triển khai.

## 7. Áp bảng quyết định 6.1/6.2 của spec

Đọc `output/bootstrap_run7/bootstrap_ci_run7.csv`/`.md`, áp nguyên văn bảng
6.1 (câu hỏi A) và 6.2 (câu hỏi B, P/Q/R) của
`docs/spec-run7-affinity-only-ban-giao-claude-code.md`, ghi verdict vào tài
liệu mới. **Việc còn thiếu, chưa làm được trong phiên triển khai này:**

- `khe-danh-gia-run7-affinity-only-chot-truoc-khi-co-so.md` — spec nói "đã
  chuẩn bị sẵn" nhưng file này KHÔNG có trong repo hiện tại. Nếu file đó tồn
  tại ở nơi khác (máy khác, tài liệu ngoài repo), đưa vào `docs/` trước khi
  điền số; nếu chưa có, cần tạo mới dựa đúng khung bảng 6.1/6.2.
- Chỉ dùng bảng biên độ liên-seed CHÍNH THỨC của Run 7
  (`run7_interseed_amplitude.md`) khi đã có ≥2 seed. Với 1 seed, mọi verdict
  ở mục 6.2 là **tạm thời** — dùng tạm bảng biên độ của Static làm proxy
  thận trọng (mục 4.3 spec), ghi rõ đây là proxy khi báo cáo.
