# Hướng dẫn chạy — Chẩn đoán xung đột gradient (Phần A) + lỗ giả (Phần B, B2) + nối (Phần C)

**Spec:** `docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md`
**Không train, không sửa `src/losses/`.** Chỉ đọc checkpoint đã có và mask đã dump.

---

## 0. Phạm vi và giới hạn

- Code + test đã viết và chạy trên **dữ liệu tổng hợp** (CPU, local). **Chưa chạy trên checkpoint/dataset thật** — phần đó bạn chạy trên Kaggle theo hướng dẫn này.
- Phần D (train lại có callback) **không** nằm trong lần này. Quyết định có chạy D hay không đọc từ `summary_grad_conflict.md` (nhánh N ⇒ chạy D, xem mục 7 spec).
- Script **tự áp** bảng 5.1/5.2 (Phần A) và 5.4 + cổng tần suất 3.3 (Phần B) **nguyên văn** và ghi verdict vào file. Ngưỡng nằm trong code dưới dạng hằng số, không có cờ dòng lệnh để đổi.
- Trên máy Windows của bạn, DLL `rasterio` bị Application Control chặn. Vì vậy test local giả lập dataset trong bộ nhớ. Trên Kaggle không bị ảnh hưởng.

## 1. File mới

| File | Vai trò |
|---|---|
| `Tools/grad_conflict_common.py` | Bảng checkpoint cố định (config, model-type, iter, seed, trọng số, cặp hợp lệ, mIoU-9 kỳ vọng). Thành phần GT (dùng chung A/B/C). Các tầng khoảng cách/lớp/kích thước. |
| `Tools/select_figure_images.py` | Chốt 6 ảnh minh hoạ **từ GT, trước khi tính** (mục 2.4). Không ghi đè. |
| `Tools/grad_conflict_probe.py` | Phần A: `gates` (cổng 1,2,3,4,5,7), `measure`, `summarize` (bảng 5.1/5.2/5.3). |
| `Tools/spurious_holes.py` | Phần B: `selftest` (cổng 6), `count`, `compare` (bảng 5.4, cổng 3.3), `fill-test` (B2). |
| `Tools/link_holes_conflict.py` | Phần C. |
| `Tools/test_grad_conflict_probe.py`, `Tools/test_spurious_holes.py` | Test (dữ liệu tổng hợp). |
| `scripts/run_grad_conflict.sh` | Chạy một lệnh, hoặc từng stage / từng checkpoint. |
| `scripts/run_grad_conflict_kaggle.sh` | **Lệnh gọn nhất (khuyến nghị):** chỉ cần truyền thư mục checkpoint + thư mục dump. Script tự ghép từng file với nhãn, config và dump, rồi gọi `run_grad_conflict.sh`. Xem mục 3A. |

## 2. Chuẩn bị trên Kaggle

1. Notebook có GPU **T4** (T4 x2 nếu muốn chạy song song, xem 5.3). Internet không cần.
2. **Input:**
   - Repo này (Kaggle Dataset hoặc `git clone`).
   - Dataset OpenEarthMap như lúc train.
   - Một Kaggle Dataset chứa **8 checkpoint**.
   - Các thư mục **dump cũ**. Mỗi dump có `masks/*.png` và `per_image_stats.csv`, do `Tools/eval_boundary_metrics.py --dump-preds/--dump-per-image-stats` tạo ra.
3. **pos_weight** của từng run có BCE: lấy dòng `pos_weight used` trong `summary.txt`, hoặc dòng `pos_weight ước lượng ... pos_weight=` trong log train.
   - Nếu không tìm được, truyền `auto`: script ước lượng lại trên 200 batch × 4 crop, seed 19. Giá trị này **xấp xỉ** và có ghi nguồn.
   - pos_weight **ảnh hưởng hướng gradient BCE**, nên ưu tiên giá trị trong log.

### 2.1. Bảng nhãn checkpoint (cố định trong code)

| Nhãn (`LABEL`) | File | `model_type` | Config | iter / seed | Trọng số hiệu dụng | Cặp đo | Cổng 1: mIoU-9 kỳ vọng | Dump gợi ý | Cần pos_weight |
|---|---|---|---|---|---|---|---|---|---|
| `static_s19_36k` | `best_model_Static_S19.pth` | static_boundary | `static_boundary.yaml` | 36000 / 19 | bce 0.4, aff 0.4 | 3 cặp | 0.6540486 | `run3_static_s19_iter36000` | có |
| `static_s86_36k` | `best_miou_Static_s86.pth` | static_boundary | `static_boundary.yaml` | 36000 / 86 | 0.4 / 0.4 | 3 cặp | 0.6557 | `run5_static_s86_best_iter36000` | có |
| `static_s86_40k` | `final_iter40000_static_s86.pth` | static_boundary | `static_boundary.yaml` | 40000 / 86 | 0.4 / 0.4 | 3 cặp | từ dump (*) | `run5_static_s86_final_iter40000` | có |
| `bce04_40k` | `best_model_BCE_04.pth` | bce_edge | `bce_edge.yaml` | 40000 / 19 | bce 0.4, aff-probe 0.4 | 3 cặp (probe) | 0.6566 | `run2b_bce04_iter40000` | có |
| `bce02_40k` (bổ sung) | `best_model_BCE_02.pth` | bce_edge | `bce_edge.yaml` | 40000 / 19 | bce 0.2, aff-probe 0.4 | 3 cặp (probe) | 0.6490 | (xem 2.2) | có |
| `affonly_s19_40k` | `best_miou_aff_s19.pth` | baseline | `run7_affinity_only.yaml` | 40000 / 19 | aff 0.4 | chỉ (Region, Aff) | từ dump (*) | `run7_affinity_only_seed19_best` | không |
| `affonly_s86_40k` | `best_miou_aff_s86.pth` | baseline | `run7_affinity_only.yaml` | 40000 / 86 | aff 0.4 | chỉ (Region, Aff) | từ dump (*) | `run7_affinity_only_seed86_best` | không |
| `baseline_40k` | `best_model_baseline.pth` | baseline | `baseline.yaml` | 40000 / 19 | aff-probe 0.4 | (Region, Aff-probe) | 0.6550926 | `run1_baseline_iter40000` | không |

(*) Spec không có hằng số cho checkpoint này. Cổng 1 khi đó so với mIoU-9 tính lại từ `per_image_stats.csv` của dump (quy ước micro, dung sai 0.001), và ghi rõ nguồn vào `gates_<ckpt>.json`.

Run 3b và Run 4 không có checkpoint. Báo cáo ghi "không có". `bce02_40k` là checkpoint bổ sung: được báo cáo mô tả, không tham gia bảng 5.2/5.4.

### 2.2. Nếu thiếu dump của một checkpoint (vd BCE λ=0.2)

Cổng 7 bắt buộc có dump. Tạo dump bằng tool sẵn có, mất khoảng 3 phút GPU:

```bash
python Tools/eval_boundary_metrics.py \
  --config configs/unet_former_resnet18_bce_edge/bce_edge.yaml \
  --checkpoint /kaggle/input/<ckpts>/best_model_BCE_02.pth --model-type bce_edge \
  --dump-preds output/dump/run2_bce02_iter40000/masks \
  --dump-per-image-stats output/dump/run2_bce02_iter40000/per_image_stats.csv \
  --run-name run2_bce02 --checkpoint-iter 40000 --dump-only
```

Dump tạo mới chỉ dùng được cho cổng 7 (danh sách ảnh). Không dùng nó làm tham chiếu cổng 1, vì nó sinh ra từ chính checkpoint đang kiểm. Cổng 1 của `bce02_40k` vẫn so với hằng số 0.6490.

## 3. Chạy test trước (local hoặc Kaggle, không cần checkpoint)

```bash
python Tools/test_spurious_holes.py            # ~30 s: cổng 6, lỗ giả, vành khuyên, lỗ mới, B2, Phần C, bảng 5.4
python Tools/test_grad_conflict_probe.py       # ~10 s: tầng, thống kê đặc trưng, null, phân batch, bảng 5.2
python Tools/test_grad_conflict_probe.py --e2e # chậm: gates -> measure -> summarize thật trên dữ liệu giả lập
```

## 3A. Cách nhanh nhất — `scripts/run_grad_conflict_kaggle.sh`

### Bước 1. Chuẩn bị input trên Kaggle

- **Một thư mục chứa đủ 8 checkpoint**, giữ nguyên tên như trong `docs/Checkpoint_summary.txt`. Script nhận diện checkpoint theo tên file:

  ```
  /kaggle/input/my-ckpts/
    best_model_Static_S19.pth        -> static_s19_36k
    best_miou_Static_s86.pth         -> static_s86_36k
    final_iter40000_static_s86.pth   -> static_s86_40k
    best_model_BCE_04.pth            -> bce04_40k
    best_model_BCE_02.pth            -> bce02_40k
    best_miou_aff_s19.pth            -> affonly_s19_40k
    best_miou_aff_s86.pth            -> affonly_s86_40k
    best_model_baseline.pth          -> baseline_40k
  ```

  Nếu một file mang tên khác, truyền riêng nó: `--ckpt static_s19_36k=/duong/dan/file_khac.pth`.

- **Một thư mục chứa các dump cũ**. Mỗi dump là một thư mục con có `masks/` và `per_image_stats.csv`. Tên thư mục con mặc định:

  ```
  /kaggle/input/my-dumps/
    run3_static_s19_iter36000/  run5_static_s86_best_iter36000/  run5_static_s86_final_iter40000/
    run2b_bce04_iter40000/      run2_bce02_iter40000/            run7_affinity_only_seed19_best/
    run7_affinity_only_seed86_best/  run1_baseline_iter40000/
  ```

  Nếu tên khác, truyền riêng: `--dump bce04_40k=/duong/dan/dump_khac`. Dump còn thiếu (thường là BCE λ=0.2) thì thêm `--make-missing-dumps`; script tự tạo vào `output/dump/<tên>/`.

- **pos_weight cho 5 checkpoint có BCE**: lấy dòng `pos_weight used` trong `summary.txt` của từng run. Không tìm được thì truyền `=auto`.

### Bước 2. Mở terminal hoặc cell notebook tại thư mục repo

```bash
cd /kaggle/working/BoundaryLossSolution     # (hoặc nơi bạn clone/copy repo)
python Tools/test_spurious_holes.py && python Tools/test_grad_conflict_probe.py   # kiểm nhanh môi trường
```

### Bước 3. Đặt biến cho gọn

```bash
CK=/kaggle/input/my-ckpts
DP=/kaggle/input/my-dumps
DATA=/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap
PW="--pos-weight static_s19_36k=<log> --pos-weight static_s86_36k=<log> --pos-weight static_s86_40k=<log> \
    --pos-weight bce04_40k=<log> --pos-weight bce02_40k=<log>"
```

### Bước 4. Kiểm lệnh trước khi chạy (không tốn GPU)

```bash
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir $CK --dump-root $DP --data-root $DATA $PW --make-missing-dumps --print-only
```

Lệnh này in bảng ghép `nhãn → ckpt → dump`. Hãy xem có dòng nào `(KHONG CO)` không trước khi chạy thật.

### Bước 5. Dry-run 4 ảnh (khoảng 15–30 phút)

```bash
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir $CK --dump-root $DP --data-root $DATA $PW --make-missing-dumps --dry-run 4
```

Kết quả nằm ở `output/grad_conflict_dryrun/` và `output/spurious_holes_dryrun/`. Mở `run_log.txt` / `done_*.json` để xem thời gian mỗi ảnh. Số liệu của dry-run không dùng để kết luận.

### Bước 6. Chạy thật

Chọn một trong hai cách:

```bash
# 1 GPU (T4): chạy tuần tự 8 checkpoint
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir $CK --dump-root $DP --data-root $DATA $PW --make-missing-dumps

# 2 GPU (T4 x2): chia checkpoint cho cuda:0 / cuda:1, sau đó tự gộp (summarize, B, C, B2)
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir $CK --dump-root $DP --data-root $DATA $PW --make-missing-dumps --parallel
```

Với `--parallel`, log từng GPU nằm ở `output/parallel_cuda0.log` và `output/parallel_cuda1.log`. Nếu một GPU có cổng FAIL, script dừng và in 20 dòng cuối của log.

### Bước 7. Đọc kết quả

- `output/grad_conflict/summary_grad_conflict.md`: Phần A, nhánh theo bảng 5.2.
- `output/spurious_holes/pairs_bootstrap.md`: Phần B, verdict theo bảng 5.4 và cổng tần suất. Phần C được chèn ở cuối file này nếu đã chạy.
- `output/spurious_holes/fill_test.md`: Phần B2, chỉ có nếu đã chạy.
- Mô tả từng file và từng cột: mục 6.

### Tuỳ chọn khác của script gọn

| Tuỳ chọn | Tác dụng |
|---|---|
| `--ckpts "static_s19_36k static_s86_36k"` | Chỉ chạy các nhãn này (vd ưu tiên hai checkpoint bắt buộc trước) |
| `--stages "gates A"` | Chỉ chạy các stage này (`figures gates A summarize B C B2`). Khi dùng `--parallel`, tuỳ chọn này chỉ áp cho bước gộp cuối. |
| `--device cuda:1` | Chọn GPU khi không dùng `--parallel` |
| `--out /kaggle/working/out` | Thư mục output gốc (mặc định `output`) |
| `--force` | Chạy lại dù đã có kết quả. Mặc định, checkpoint đã xong sẽ được bỏ qua, nên khi phiên Kaggle bị ngắt chỉ cần chạy lại đúng lệnh cũ. |

Nếu chia việc qua nhiều phiên Kaggle: cuối mỗi phiên, lưu thư mục `output/` (vd thành Kaggle Dataset); đầu phiên sau, copy nó về đúng chỗ rồi chạy lại cùng lệnh.

## 4. Cách 1 — một lệnh chạy hết (script gốc, truyền từng file)

Ví dụ dưới đặt `CK=/kaggle/input/<ckpts>`, `DP=/kaggle/input/<dumps>`. Thay giá trị pos_weight bằng số trong log.

```bash
CK=/kaggle/input/my-ckpts; DP=/kaggle/input/my-dumps
bash scripts/run_grad_conflict.sh \
  --data-root /kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap \
  --ckpt static_s19_36k=$CK/best_model_Static_S19.pth   --dump static_s19_36k=$DP/run3_static_s19_iter36000        --pos-weight static_s19_36k=<log> \
  --ckpt static_s86_36k=$CK/best_miou_Static_s86.pth    --dump static_s86_36k=$DP/run5_static_s86_best_iter36000   --pos-weight static_s86_36k=<log> \
  --ckpt static_s86_40k=$CK/final_iter40000_static_s86.pth --dump static_s86_40k=$DP/run5_static_s86_final_iter40000 --pos-weight static_s86_40k=<log> \
  --ckpt bce04_40k=$CK/best_model_BCE_04.pth            --dump bce04_40k=$DP/run2b_bce04_iter40000             --pos-weight bce04_40k=<log> \
  --ckpt bce02_40k=$CK/best_model_BCE_02.pth            --dump bce02_40k=output/dump/run2_bce02_iter40000      --pos-weight bce02_40k=<log> \
  --ckpt affonly_s19_40k=$CK/best_miou_aff_s19.pth      --dump affonly_s19_40k=$DP/run7_affinity_only_seed19_best \
  --ckpt affonly_s86_40k=$CK/best_miou_aff_s86.pth      --dump affonly_s86_40k=$DP/run7_affinity_only_seed86_best \
  --ckpt baseline_40k=$CK/best_model_baseline.pth       --dump baseline_40k=$DP/run1_baseline_iter40000
```

Thứ tự stage: `figures → gates → A → summarize → B → C → B2`.

- Nếu cổng của checkpoint nào FAIL, script **dừng** (mục 4 spec).
- C tự bỏ qua nếu Phần B kết luận "không do loss".
- B2 tự bỏ qua nếu cổng tần suất là "hiếm".

**Nên dry-run trước** (4 ảnh, output ra `output/*_dryrun/`, không lẫn với kết quả thật). Mục đích là đo thời gian/ảnh và bộ nhớ GPU:

```bash
bash scripts/run_grad_conflict.sh --dry-run 4 <cùng các --ckpt/--dump/--pos-weight như trên>
```

Ở dry-run:
- Cổng 1 so với `per_image_stats.csv` của dump trên đúng 4 ảnh đó.
- Cổng 7 kiểm danh sách đầy đủ.
- Kết quả dry-run **không** dùng để kết luận.

## 5. Cách 2 — tách phần, tách checkpoint

Mọi kết quả ghi theo từng checkpoint, kèm file `done_*.json` (có dấu vân tay file ckpt + tham số). Chạy lại sẽ **tự bỏ qua** checkpoint đã xong, trừ khi thêm `--force`. Vì vậy có thể chia việc qua nhiều phiên Kaggle, rồi gộp ở bước cuối.

### 5.1. Dùng script với `--stages` / `--ckpts`

```bash
# Phiên 1: chỉ hai checkpoint bắt buộc của bảng quyết định
bash scripts/run_grad_conflict.sh --stages "figures gates A" --ckpts "static_s19_36k static_s86_36k" <các --ckpt/--dump/--pos-weight>
# Phiên 2: các checkpoint còn lại
bash scripts/run_grad_conflict.sh --stages "gates A" --ckpts "bce04_40k bce02_40k affonly_s19_40k affonly_s86_40k baseline_40k static_s86_40k" <...>
# Cuối cùng: gộp (CPU), Phần B/C/B2
bash scripts/run_grad_conflict.sh --stages "summarize B C B2" <các --dump của mọi checkpoint>
```

Khi chia phiên, phải mang theo thư mục `output/grad_conflict/` (vd lưu thành Kaggle Dataset rồi copy vào `output/`). `summarize` chỉ gộp những gì có trong thư mục đó. Checkpoint chưa đo sẽ hiện `CHUA DO` trong báo cáo.

### 5.2. Gọi trực tiếp từng tool

```bash
# 0) Chốt ảnh minh hoạ (1 lần, trước Phần A)
python Tools/select_figure_images.py --data-root $DATA --output output/grad_conflict/figure_images.json

# 1) Cổng kiểm (từng hoặc nhiều checkpoint; --ckpt/--dump-dir/--pos-weight lặp lại theo LABEL=...)
python Tools/grad_conflict_probe.py gates --data-root $DATA \
  --ckpt static_s19_36k=$CK/best_model_Static_S19.pth --dump-dir static_s19_36k=$DP/run3_static_s19_iter36000 \
  --pos-weight static_s19_36k=<log>

# 2) Đo (từ chối chạy nếu gates_<ckpt>.json chưa PASS cho đúng file ckpt + pos_weight)
python Tools/grad_conflict_probe.py measure --data-root $DATA \
  --ckpt static_s19_36k=$CK/best_model_Static_S19.pth --pos-weight static_s19_36k=<log>

# 3) Gộp + bảng 5.2
python Tools/grad_conflict_probe.py summarize --output-dir output/grad_conflict --holes-dir output/spurious_holes

# 4) Phần B (CPU) — luôn kèm baseline_40k
python Tools/spurious_holes.py selftest
python Tools/spurious_holes.py count --data-root $DATA \
  --masks baseline_40k=$DP/run1_baseline_iter40000/masks \
  --masks static_s19_36k=$DP/run3_static_s19_iter36000/masks \
  --per-image-stats static_s19_36k=$DP/run3_static_s19_iter36000/per_image_stats.csv   # (tuỳ chọn, cổng 7)
python Tools/spurious_holes.py compare

# 5) Phần C, Phần B2
python Tools/link_holes_conflict.py
python Tools/spurious_holes.py fill-test --data-root $DATA \
  --masks static_s19_36k=$DP/run3_static_s19_iter36000/masks --masks static_s86_36k=$DP/run5_static_s86_best_iter36000/masks
```

Tuỳ chọn hay dùng:
- `--max-images N`: dry-run.
- `--device cuda:1`: chọn GPU.
- `--n-perm 20`: số hoán vị null.
- `--batch-size 8`: batch ở mức tham số.
- `--scratch-dir`: nơi ghi tạm gradient tham số, xem mục 8.
- `--keep-scratch`: giữ lại file tạm đó.
- `--continue-on-fail` (gates): không dừng ở checkpoint FAIL đầu tiên.
- `--skip-gate-check` (measure): **không khuyến nghị**; có ghi cảnh báo vào log.

### 5.3. Chạy song song trên T4 x2

Hai tiến trình, mỗi tiến trình một tập checkpoint và một GPU, cùng `--out`:

```bash
bash scripts/run_grad_conflict.sh --stages "gates A" --device cuda:0 --ckpts "static_s19_36k bce04_40k affonly_s19_40k baseline_40k" <...> &
bash scripts/run_grad_conflict.sh --stages "gates A" --device cuda:1 --ckpts "static_s86_36k static_s86_40k bce02_40k affonly_s86_40k" <...> &
wait
bash scripts/run_grad_conflict.sh --stages "summarize B C B2" <các --dump>
```

Chạy stage `figures` **trước** hai tiến trình trên. Không chạy hai tiến trình `figures` cùng lúc.

## 6. Danh mục file kết quả

### 6.1. `output/grad_conflict/` (Phần A)

| File | Nội dung | Cột / khoá |
|---|---|---|
| `figure_images.json` | 6 ảnh chốt từ GT trước khi tính | `images[]`: `image_id, criterion, score_key, score, rank_in_criterion`; `created_at, git_commit, n_images_scanned` |
| `gates_<ckpt>.json` | Cổng 1,2,3,4,5,7 | `overall` (PASS/FAIL), `checkpoint_fingerprint`, `pos_weight`, `dry_run`, `gates[]`: `gate, name, status, value, threshold, note` |
| `per_image_strata_<ckpt>.csv` | Thống kê đủ mức đặc trưng, theo ảnh × cặp × tầng | `image_id, pair, axis, stratum, sum_dot, sum_n1, sum_n2, n_pos, n_support, n_neg` |
| `per_batch_param_<ckpt>.csv` | Mức tham số, 48 batch × cặp | `batch_id, image_ids, n_images, pair, dot, norm1, norm2, cos, mag_ratio` |
| `null_param_<ckpt>.csv` | Null mức tham số: mọi cặp batch lệch | `pair, b, b_prime, cos` |
| `null_feat_<ckpt>.csv` | Null mức đặc trưng: 20 hoán vị/ảnh/tầng | `pair, axis, stratum, perm_id, image_id, sum_dot_perm` |
| `per_component_small_<ckpt>.csv` | Chỉ Static: thống kê trong từng thành phần GT nhỏ | `image_id, pair, comp_id, class, area, sum_dot, sum_n1, sum_n2, n_pos, n_support` |
| `per_image_loss_<ckpt>.csv` | Giá trị từng số hạng loss (chưa nhân trọng số) | `image_id, image_idx, batch_id, L_region, L_aff, [L_bce]` |
| `done_<ckpt>.json` | Đánh dấu đo xong + cấu hình chạy | `iter, seed, mode, pos_weight(+source), n_images, n_batches, batch_size, n_shared_params, weights, pairs, device, gpu, torch, wall_time_s, git_commit` |
| `summary_<ckpt>.csv` | Một dòng cho mỗi (mức, cặp, trục, tầng) | `level, pair, axis, stratum, value, value_kind, ci_lo, ci_hi, null_lo, null_hi, frac_neg, mag_ratio, support_cov, n_pos, flag_insufficient, n_units, tau, crit1_ci_below_0, crit2_below_null, crit3_abs_ge_tau, crit4_mag_in_range, confirmed_single_ckpt` |
| `summary_grad_conflict.md` | **Báo cáo chính Phần A** | Xem cấu trúc bên dưới bảng này |
| `summary_grad_conflict.json` | Bản máy đọc của quyết định | `decision_52`: `branch, text, literal, raw, ablation_variants, confirmed_both, L_detail, C_detail`; `present, missing, missing_runs` |
| `figures/<image_id>.png`, `figures/<image_id>_cosmaps.npz` | 6 hình (Static s19@36k): RGB · GT · cos_p(BCE,Aff) · cos_p(Region,Aff), kèm mảng cos gốc | — |
| `run_log.txt` | Nhật ký: lệnh, git commit, torch, GPU, pos_weight + nguồn, thời gian | — |
| `pos_weight_auto.json` | Chỉ khi dùng `auto` | `pos_weight, source, stats` |

Cấu trúc `summary_grad_conflict.md`:
1. Dòng đầu: checkpoint, iter, seed, chế độ `eval()`, batch size mức tham số, A_min chính.
2. Bảng checkpoint (đã đo / chưa đo / không có), có thời gian thật.
3. Bảng cổng 1–7.
4. Nhánh bảng 5.2, kèm điều kiện "theo bảng" và "thô", biến thể ablation, danh sách xung đột xác nhận ở cả hai seed.
5. Mô tả 5.3.
6. Bảng chi tiết từng checkpoint.
7. Ghi chú quy ước.

Giải nghĩa cột và giá trị:
- **`pair`**: `bce_aff` = (BCE, Aff), `region_aff` = (Region, Aff), `region_bce` = (Region, BCE). Tỉ lệ độ lớn = ‖g của số hạng sau‖ / ‖g của số hạng trước‖.
- **`axis` / `stratum`**:
  - `all/all`: toàn ảnh.
  - `dist`: `d0-4, d4-8, d8-16, d16-32, d>32`, khoảng cách px full-res tới biên GT.
  - `class`: 9 lớp.
  - `size`: `small, medium, large`, kích thước thành phần GT.
- **`level = param`**:
  - `value` là trung vị cos theo batch, CI bootstrap trên batch.
  - Null là cos giữa `G_k1(b)` và `G_k2(b')` với b ≠ b'.
- **`level = feat`**:
  - `value` là `cosW` gộp micro, CI bootstrap per-image.
  - Null là 20 hoán vị vị trí trong cùng tầng, cùng ảnh.
  - `support_cov` = n_support / n_pos. Nếu < 1% thì đặt `flag_insufficient = True`.

### 6.2. `output/spurious_holes/` (Phần B, B2, C)

| File | Nội dung | Cột / khoá |
|---|---|---|
| `gates_holes.json` | Cổng 6 | `status`, `cases[]`: `case, pass, detail` |
| `per_image_<ckpt>.csv` | Số đếm per-image theo `a_min` ∈ {16, 64, 256} và lớp (`all` + 9 lớp) | `image_id, a_min, class, n_lo_gia, dien_tich_lo_gia, n_vanh_khuyen, n_lo_gia_moi, n_holes_total` |
| `holes_<ckpt>.csv` | Mỗi lỗ (diện tích ≥ a_min) một dòng | `image_id, class, a_min, hole_id, area, frac_gt_c, is_spurious, is_new, max_iou_baseline, comp_id, comp_filled_area, gt_comp_id, gt_comp_area, bbox (y0:x0:y1:x1)` |
| `done_holes_<ckpt>.json` | Đánh dấu đếm xong | `mask_dir, n_images, a_mins, wall_time_s_all_labels` |
| `pairs_bootstrap.csv` | Mọi cặp × `a_min` × lớp × đại lượng | `pair_id, description, a_min, class, metric, observed, mean_a, mean_b, ci_lo, ci_hi, p_lt0, interseed_amp_rule, interseed_amp, ci_excludes_0, confirmed, status` |
| `pairs_bootstrap.md` | **Báo cáo chính Phần B** | Verdict 5.4, cổng tần suất 3.3, có chạy C/B2 không, quy ước, bảng cặp cho từng A_min. Mục Phần C được chèn ở cuối nếu chạy. |
| `pairs_bootstrap.json` | Bản máy đọc | `verdict_54` (`key` ∈ affinity/bce/interaction/none/unmatched/unavailable, `text`, `pairs`), `frequency_gate`, `run_part_c`, `run_part_b2`, `labels_present/missing` |
| `part_c_link.csv/.md/.json` | Phần C | `pair, class, n_yes, n_no, cosw_yes, cosw_no, diff, ci_lo, ci_hi, ci_excludes_0, note`; JSON: `verdict_by_pair, conclusion, supports_common_root` |
| `fill_test.csv/.md` | Phần B2 (chỉ khi không "hiếm") | `label, a_fill, quantity (mIoU_9, mIoU_8, bf_precision, bf_recall, bf_score, asd_pred_to_gt, asd_gt_to_pred), before, after, delta, ci_lo, ci_hi, p_delta_lt_0, ci_excludes_0` |

Các cặp trong `pairs_bootstrap.*`:

| Nhóm | Cặp | Ghi chú |
|---|---|---|
| Theo spec | H1, H2, H2', H3, H3', H4 | |
| Contrast | H5, H5' | (Static − BCE) − (AffOnly − Baseline), mỗi seed |
| Biên độ liên-seed | N1, N2 | |
| Mô tả, không vào bảng 5.4 | X1 | Static s86@40k − Baseline |
| | X2 | BCE λ=0.2 − Baseline |
| | X3 | Static s86 40k − 36k |

## 7. Đọc cổng FAIL — xử lý

| Cổng | FAIL nghĩa là | Làm gì |
|---|---|---|
| 1 | mIoU-9 lệch > 0.001 so với kỳ vọng | Kiểm đúng file / đúng nhãn / đúng config. Nếu `computed mIoU-8` trùng số kỳ vọng thì bạn đang nhầm quy ước 8/9 lớp. **DỪNG**, không đo. |
| 2 | L_total của module repo ≠ tổng số hạng | Sai trọng số trong bảng checkpoint, hoặc pos_weight không khớp. |
| 3 | Tách gradient sai (kiểm ở float64) | Lỗi code. Báo lại, đừng nới ngưỡng. Số float32 trong JSON chỉ để tham khảo. |
| 4 | Checkpoint có BCE mà BoundaryHead như mới khởi tạo, hoặc ngược lại | Nhầm file / nhầm nhãn. |
| 5 | Hai lần chạy lệch > 1e-6 | Kernel CUDA không tất định (vd backward của nội suy bilinear). Ghi nhận, **không nới ngưỡng**. Gửi lại `max_rel_diff`. |
| 6 | Unit test lỗ sai | Lỗi code Phần B. |
| 7 | Danh sách ảnh khác dump | Sai val split hoặc sai thư mục dump. Nếu `pixel_agreement` < 0.99, kiểm lại checkpoint ↔ dump. |

## 8. Thời gian, tài nguyên

- **Phần A, mỗi checkpoint** (ước tính, đo lại bằng dry-run):
  - Cổng 1: ~2–3 phút.
  - Cổng 3 ở float64: chậm trên T4, ~1 phút/ảnh × 4 ảnh.
  - `measure`: ~1–2 s/ảnh × 384 ảnh ≈ 10–15 phút.
  - Tổng 8 checkpoint ≈ 2.5–3.5 giờ trên 1 T4, hoặc khoảng một nửa nếu chạy T4 x2.
  - Thời gian thật được ghi vào `done_<ckpt>.json` và bảng checkpoint trong báo cáo.
- **Đĩa tạm:** gradient tham số theo batch được ghi ra memmap (48 batch × 3 số hạng × ~11.65 triệu tham số × 4 byte ≈ **6.7 GB**) ở `--scratch-dir` (mặc định `$TMPDIR` hoặc `/tmp`). File này tự xoá sau khi tính ma trận Gram. Trên Kaggle, `/tmp` đủ chỗ. **Không** đặt nó trong `/kaggle/working` (giới hạn 20 GB output).
- **Phần B:** CPU, khoảng 1–2 s/ảnh/checkpoint. B2 chậm hơn, vì đo lại BF/ASD cho mỗi A_fill.
- **Bộ nhớ GPU:** 1 ảnh 1024² fp32, giữ đồ thị qua 3 lần backward. Lần chạy cổng 3 ở float64 cần nhiều hơn. Dry-run sẽ cho biết con số thật.

## 9. Quy ước cần biết khi đọc số (đã ghi cả trong báo cáo)

1. **Chế độ tính:** fp32, không autocast (lúc train dùng AMP). `model.eval()`: BN dùng running stats, dropout/drop-path tắt. Mỗi lần backward 1 ảnh, `torch.manual_seed(19 + idx)`. AffinityLoss của repo vốn tất định.
2. **L_region tính trên từng ảnh** (Dice gộp trong 1 ảnh), khác Dice gộp batch 4 lúc train. `G_k(b)` = tổng gradient theo ảnh trong batch 8 (= 4 × 2 GPU), phân hoạch bằng `default_rng(19).permutation(384)`.
3. **F** lấy bằng forward hook trên `frh`. `shared_params` = mọi tham số trừ `classifier.*` và `boundary_head.*` (~11.65 triệu).
4. **Cổng 3 quyết định ở float64**, để tránh sai số làm tròn khi cộng hơn 11 triệu phần tử. Ở fp32, sai số tương đối quan sát được khoảng 2–5e-5 dù code đúng. Ngưỡng 1e-5 giữ nguyên.
5. **Tiêu chí 5.1 ở mức tham số:** giá trị so sánh = **trung vị** cos theo batch. CI là CI của trung vị. Null là phân phối cos của các cặp batch lệch.
6. **Tiêu chí 5 (hai seed):** một (mức, cặp, trục, tầng) chỉ được tính khi **mỗi** checkpoint Static s19@36k và s86@36k đều đạt tiêu chí 1–4, và cùng dấu.
7. **Bảng 5.2:**
   - **L** = xác nhận ở ≥ 1 tầng `dist`/`size` đủ hỗ trợ, nhưng **không phải mọi** tầng đủ hỗ trợ.
   - **C** = 1–3 lớp.
   - **RA** = (BCE, Aff) không xác nhận ở đâu, và (Region, Aff) xác nhận ở mức tham số hoặc ở tầng `d8-16`, `d16-32`, `d>32`.
   - Báo cáo có cả cột "thô" (không áp loại trừ "không đạt G…") để thấy biến thể ablation.
8. **Bảng 5.4:**
   - Đại lượng dùng là `n_lo_gia` (số lỗ giả/ảnh, lớp gộp, A_min 64). `n_lo_gia_moi` được định nghĩa so với Baseline (Baseline luôn = 0), nên chỉ dùng cho cổng tần suất.
   - "Xác nhận" = CI loại 0 **và** |Δ| > biên độ liên-seed: N1 cho cặp Static, N2 cho cặp AffOnly. H1 và H5 không có cặp lặp seed riêng, nên dùng max(N1, N2). Riêng điều kiện H5 trong bảng 5.4 chỉ đòi CI loại 0, đúng chữ spec.
9. **Lỗ:**
   - Thành phần dự đoán 8-liên thông, lỗ 4-liên thông (`binary_fill_holes`).
   - Một lỗ thuộc thành phần bao quanh nó; diện tích "sau khi lấp" là của chính thành phần đó.
   - `gt_comp_id` = thành phần GT cùng lớp chứa phần lớn pixel lỗ. Đây là khoá nối với Phần C.
10. **B2:** "precision/recall" = BF precision/recall (θ = 2 px), đo bằng `Tools/per_image_dump.compute_per_image_stats` và gộp micro bằng `Tools/bootstrap_boundary_ci`. Lỗ lồng nhau: lỗ trong cùng được lấp bằng lớp bao quanh trực tiếp.
11. **Phần C:** chỉ xét thành phần GT nhỏ có ≥ 1 vị trí stride 4. So sánh trong từng lớp, gộp bằng trung bình có trọng số theo số thành phần có lỗ giả.
12. **File null tách đôi:** `null_param_*` và `null_feat_*` (spec ghi chung một tên `null_<ckpt>.csv`).

## 10. Đối chiếu checklist bàn giao (mục 10 spec)

| Mục | Trạng thái |
|---|---|
| `tools/grad_conflict_probe.py` + `tools/spurious_holes.py` | Có, đặt trong `Tools/` (chữ T hoa, đúng quy ước import của repo). Thêm `grad_conflict_common.py`, `select_figure_images.py`, `link_holes_conflict.py`. |
| Cổng kiểm 7/7 | Cổng 1,2,3,4,5,7 nằm trong `gates_<ckpt>.json`; cổng 6 nằm trong `gates_holes.json`; bảng tổng hợp có trong `summary_grad_conflict.md`. Chạy thật trên Kaggle. |
| Phần A trên A1 + A2, ghi thời gian thật | Có trong `done_<ckpt>.json` và bảng checkpoint. |
| Phần B + cổng tần suất | `pairs_bootstrap.md/.json`. |
| Áp bảng 5.2 và 5.4 nguyên văn | Tự động, trong `summary_grad_conflict.md` và `pairs_bootstrap.md`. |
| Phần C / B2 có điều kiện | Tự động bỏ qua theo `pairs_bootstrap.json`. |
| Quyết định Phần D | Đọc nhánh trong `summary_grad_conflict.md` (N ⇒ chạy D). Code Phần D chưa viết. |
| Gửi `summary_grad_conflict.md` + `pairs_bootstrap.md` | Hai file đó. |
