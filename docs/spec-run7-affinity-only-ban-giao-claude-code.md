# Spec bàn giao Claude Code — Run 7: Affinity-only (α_bce = 0, chỉ bật affinity)

**Ngày:** 14/9/2026
**Repo:** `Boundary_Loss_Solution`
**Chi phí:** ~2.5–3.5h GPU (2×T4) mỗi seed — ước tính giữa mốc Run 2 (~2h08m, không affinity) và Run 3 (~3h22m, có cả BCE+affinity); đo lại thời gian thật ở dry-run/run đầu và cập nhật vào tài liệu tham chiếu.
**Không train lại, không sửa bất kỳ run nào đã có.** Đây là một arm **mới**, độc lập tài nguyên với mọi việc khác đang chạy.

---

## 0. Vì sao chạy run này

Toàn bộ phát biểu hiện có về vai trò của affinity ("affinity nâng recall, hạ precision, cải thiện `gt→pred`") đều được suy ra bằng **phép trừ hai cấu hình ghép**: `Static (BCE.4+Aff.4) − BCE λ=0.4`. Điều đó ngầm giả định hai số hạng loss **cộng tuyến** khi cùng tối ưu chung một Fused Feature — chưa có gì kiểm chứng giả định đó.

Nhìn vào bảng cấu hình đã có: BCE-edge được đo **độc lập với baseline** ở hai liều (λ=0.2, λ=0.4) — cho một "hiệu ứng BCE thuần". Affinity thì **chưa bao giờ** được huấn luyện một mình; nó chỉ xuất hiện khi cộng vào BCE λ=0.4 cố định. Run này lấp đúng ô còn thiếu trong bảng 2×2 của thiết kế (có/không BCE) × (có/không affinity):

| | không affinity | có affinity |
|---|---|---|
| **không BCE** | Run 1 — Baseline ✅ | **Run 7 — Affinity-only (run này)** ❌ chưa có |
| **có BCE (α=0.4)** | Run 2b — BCE λ=0.4 ✅ | Run 3/3b — Static/Aff.2 ✅ |

Hai việc run này cho phép làm, mà cách suy luận bằng trừ hiện tại không làm được:

1. **Đo trực tiếp hiệu ứng của affinity so với baseline** — đối xứng với cách BCE đã được đo — thay vì suy luận gián tiếp qua Static − BCE.
2. **Kiểm tra tính cộng tuyến (additivity)**: so `(Affinity-only − Baseline)` với `(Static − BCE λ=0.4)`. Nếu hai con số này gần nhau, giả định cộng tuyến đang dùng xuyên suốt dự án được xác nhận thực nghiệm (không chỉ là giả định). Nếu lệch nhiều, đó là bằng chứng về **tương tác** giữa hai loss — bản thân nó là một phát hiện đáng viết cho khung "phân rã chức năng từng thành phần" của bài.

**Lưu ý quan trọng đã thống nhất trước khi viết spec này:** nút thắt hiện tại của dự án không phải "thiếu cấu hình" mà là **thiếu seed để tách nhiễu huấn luyện khỏi hiệu ứng thật** (xem `khung-dien-giai-va-trich-dan-cho-paper.md` bản 5, mục 0). Vì vậy run này **không** được coi là "xong" sau một seed — spec này được viết để hỗ trợ **nhiều seed, nạp tại thời điểm chạy**, và để pipeline bootstrap/tổng hợp kết quả chạy được ngay khi có seed đầu tiên, rồi tự mở rộng khi có thêm seed sau.

---

## 1. Cấu hình

### 1.1. Cấu hình chính (bắt buộc) — liều đối xứng với Static (α_affinity = 0.4)

Kế thừa **nguyên** config Run 3 (`run3_static_boundary.yaml`), tắt hẳn nhánh BCE, và dùng đúng cơ chế đã dùng cho "BCE λ=0.2/0.4" (α một mình quyết định liều khi chỉ một nhánh bật):

```yaml
# configs/research_boundary/phase1_fixed_split/run7_affinity_only.yaml
# Ke thua NGUYEN config Run 3 (static_boundary), tat BCE, doi ALPHA lam lieu affinity don.

TRAIN:
  SEED: null                    # <── KHÔNG hardcode — nạp bắt buộc qua --seed lúc chạy (xem mục 1.3)
  MAX_ITERS: 40000              # GIU NGUYEN

BOUNDARY_LOSS:
  USE_BCE: false                 # <── BIẾN CHÍNH — tắt hẳn nhánh BCE, đối xứng với cách Run 2 tắt affinity
  USE_AFFINITY: true             # GIU NGUYEN (so voi Run 3)
  DYNAMIC_WEIGHTS: false         # GIU NGUYEN (tĩnh)
  ALPHA: 0.4                     # <── liều affinity đơn — đúng cơ chế đã dùng cho "BCE λ=0.4" (ALPHA một mình quyết định liều khi chỉ một nhánh bật, LAMBDA*_STATIC giữ 1.0)
  LAMBDA1_STATIC: 1.0            # GIU NGUYEN — không có tác dụng vì USE_BCE=false, giữ để log/so sánh nhất quán
  LAMBDA2_STATIC: 1.0            # GIU NGUYEN — ALPHA một mình quyết định liều, không đổi field này
  WINDOW_K: 5                    # GIU NGUYEN
  DISTANCE: cosine               # GIU NGUYEN
  MARGIN: 1.0                    # GIU NGUYEN
  CONNECTIVITY: 4                # GIU NGUYEN
  DILATION_RADIUS: 0             # GIU NGUYEN — nghi vấn nearest-downsample stride-4 vẫn còn mở, KHÔNG sửa ở run này
  POS_WEIGHT_MAX: 20.0           # GIU NGUYEN — trường này bất hoạt (chỉ dùng cho nhánh BCE), giữ lại cho config đồng dạng, đừng xoá

OUTPUT:
  WORK_DIR: work_dirs/phase1/run7_affinity_only_seed{SEED}   # <── placeholder {SEED}, xem mục 1.3
```

Với cấu hình này: `alpha_bce_effective = 0` ở **mọi** vòng (vì `use_bce=False` ⇒ `l_bce` không được tính vào tổng, không phải chỉ nhân trọng số 0), `alpha_affinity_effective = ALPHA × LAMBDA2_STATIC = 0.4 × 1.0 = 0.4` — đúng liều của Static, để hai phép so sánh chính (mục 4) không lẫn thêm biến liều.

### 1.2. Cấu hình phụ (tuỳ chọn, ưu tiên thấp hơn) — liều 0.2, đối xứng với BCE λ=0.2 và Run 3b

Chỉ chạy nếu ngân sách GPU cho phép, **sau khi** đã có kết quả liều 0.4 ở ít nhất 1 seed. Mục đích: hoàn thiện đường liều đối xứng (BCE đã có 2 liều; affinity-khi-ghép đã có 2 liều qua Run3b/Static; affinity-đơn nên có ít nhất điểm đối chiếu để không neo kết luận vào một liều duy nhất).

```yaml
# configs/research_boundary/phase1_fixed_split/run7b_affinity_only_lambda02.yaml
# Giống HỆT run7_affinity_only.yaml, CHỈ đổi ALPHA
BOUNDARY_LOSS:
  ALPHA: 0.2                     # <── DUY NHẤT khác run7 chính
OUTPUT:
  WORK_DIR: work_dirs/phase1/run7b_affinity_only_lambda02_seed{SEED}
```

### 1.3. Seed nạp lúc chạy — thay đổi bắt buộc trong `train_boundary_research.py`

**Không hardcode seed trong YAML.** Người dùng sẽ chạy run này nhiều lần với các seed khác nhau (tối thiểu 19; khuyến nghị thêm 86 để tái dùng đúng vai trò mà Run 5 đã tạo cho Static — một biên độ liên-seed cho chính arm này). Vì vậy:

1. Thêm CLI arg bắt buộc cho **riêng file config này** (không ảnh hưởng script cho các config khác đã có `SEED` cố định):
   ```bash
   torchrun --nproc_per_node=2 src/train_boundary_research.py \
       --config configs/research_boundary/phase1_fixed_split/run7_affinity_only.yaml \
       --seed 19 \
       --dry-run   # bỏ cờ này khi chạy thật
   ```
2. Script: nếu `cfg.TRAIN.SEED is None` **và** `--seed` không được truyền ⇒ **dừng ngay, báo lỗi rõ ràng** ("SEED bắt buộc cho run7_affinity_only, truyền qua --seed"). Không được tự chọn seed mặc định.
3. Nếu `--seed` được truyền: override `cfg.TRAIN.SEED = args.seed`, rồi **format lại `cfg.OUTPUT.WORK_DIR`** bằng cách thay thế chuỗi `{SEED}` bằng giá trị seed thật (ví dụ `work_dirs/phase1/run7_affinity_only_seed19`). Đây là cơ chế duy nhất để nhiều seed không ghi đè lên nhau — áp dụng y hệt cho `run7b_...`.
4. `summary.txt` và `checkpoint_index.json` phải in seed thật ở dòng đầu, không được ẩn trong path.
5. **Không** áp dụng cơ chế `{SEED}` này cho bất kỳ config nào khác — mọi run đã có (`run1_baseline.yaml`, `run3_static_boundary.yaml`, …) giữ nguyên `SEED` cố định trong YAML như hiện tại.

### 1.4. Splits — vẫn là điểm dễ sai nhất

`SEED` (dù là 19, 86, hay bất kỳ giá trị nào người dùng nạp) chỉ được phép ảnh hưởng: khởi tạo trọng số, thứ tự lấy batch, lấy mẫu augmentation. **Không** ảnh hưởng việc chia dữ liệu. Train split và val split (384 ảnh) phải **giống hệt** mọi run khác — sinh sẵn với `--seed 19` ở `Tools/create_splits.py`/`build_clean_val_split.py`, **không sinh lại dù seed huấn luyện là gì**.

Kiểm tra bắt buộc ở dry-run, cho **mọi giá trị seed** được nạp: hash danh sách file val phải khớp hash của Run 3/Run 5.

---

## 2. Checkpoint & log — áp dụng đầy đủ đặc tả (bài học từ Run 1/2/3, không được rút gọn)

- **4 checkpoint** + `checkpoint_index.json`: `best_miou.pth`, `best_bfscore.pth`, `best_bareland.pth`, **`final_iter40000.pth`**.
- **`benchmark_results.csv` 10 hàng đủ cột**: per-class IoU kèm tên lớp (`CLASS_NAMES` chuẩn của dự án), ASD **hai chiều**, `bf_precision`/`bf_recall`, per-class Boundary IoU 3 ngưỡng, cột `lambda1, lambda2, alpha, alpha_bce_effective, alpha_affinity_effective` (để không ai phải đoán lại liều khi đọc file này độc lập).
- **Boundary metrics đầy đủ cho tối thiểu 4 vòng cuối** (round 7–10); nếu một vòng validate < ~5 phút, tính đủ cả 10 vòng. Ghi rõ lựa chọn nào vào `summary.txt` — không được im lặng bỏ qua (đây chính là lỗi đã làm Run 3 mất phân rã ở 9/10 vòng).
- `summary.txt`: khối 4-checkpoint + khối mean±std 4 vòng cuối (**ddof=1**) + khối seed (mục 1.3) + khối liều (`ALPHA`, `alpha_bce_effective ≡ 0.0000 (không đổi)`, `alpha_affinity_effective ≡ 0.4000` hoặc `0.2000`).
- Sau khi train xong, chạy `eval_boundary_metrics.py --dump-preds --dump-per-image-stats` (đã có sẵn từ `spec-per-image-bootstrap-eval-v2`, không cần code mới ở bước này) trên **cả hai** mốc:
  - checkpoint `best_miou.pth` (bất kể rơi vào iter nào)
  - `final_iter40000.pth`

  **Vì sao cả hai:** Static chỉ tồn tại tại 36000; BCE λ=0.4/Baseline chỉ tồn tại tại 40000. Có cả hai mốc cho Affinity-only cho phép so **đúng cùng iteration** với cả hai phía, thay vì buộc phải chấp nhận lệch mốc như Static đang phải chịu.

  Đặt dump vào:
  ```
  dump/run7_affinity_only_seed<SEED>_best/       (best_miou, iter thật ghi trong meta.json)
  dump/run7_affinity_only_seed<SEED>_final40000/
  ```
  (và tương tự `run7b_affinity_only_lambda02_seed<SEED>_...` nếu chạy liều phụ).

---

## 3. Cổng dry-run — `MAX_ITERS=100`

Áp dụng cho **mỗi lần** người dùng chạy (mỗi seed):

1. Log in ra đúng seed đã truyền qua `--seed`; nếu không truyền ⇒ script phải dừng với lỗi rõ ràng (mục 1.3.2).
2. **Hash danh sách file val khớp Run 3/Run 5.** Lệch ⇒ **DỪNG**.
3. `alpha_bce_effective == 0.0` ở **mọi** vòng (không phải gần 0 — bằng 0 tuyệt đối, vì `use_bce=False`).
4. `alpha_affinity_effective == 0.4` (hoặc `0.2` cho `run7b`) ở **mọi** vòng — tĩnh, không lịch trình.
5. Đồng nhất thức `l_total = l_region + alpha*(lam1*l_bce + lam2*l_affinity)` với `l_bce ≡ 0` — sai số < 1e-5, mọi vòng. (Vì `use_bce=False`, `l_bce` không được cộng vào; kiểm tra `l_total == l_region + alpha*lam2*l_affinity`.)
6. `l_bce` **không được log** hoặc log là `null`/`0` tường minh (không phải giá trị rác) trong CSV — vì `BoundaryHead` không nhận gradient ở run này. Ghi rõ trong `summary.txt` rằng đây là hành vi **kỳ vọng**, không phải lỗi, để tránh bị hiểu nhầm là bug khi review log.
7. `pos_weight` — không áp dụng cho run này (nhánh BCE tắt); nếu script vẫn tính (do dùng chung hàm ước lượng), giá trị này **không được dùng để gate** như các run có BCE — bỏ qua cổng `pos_weight ∈ [10.8, 11.1]` cho riêng run này.
8. 4 checkpoint + `checkpoint_index.json` ghi ra đúng, load lại được, seed thật xuất hiện trong `checkpoint_index.json`.
9. CSV đủ cột theo mục 2; thời gian một vòng validate được đo và ghi log để quyết boundary metrics 4 vòng hay 10 vòng.
10. `nvidia-smi` ra **2×T4**, không phải P100.

**Không đổi ngưỡng ở mục này sau khi thấy số.**

---

## 4. Tích hợp vào pipeline bootstrap — bắt buộc để "chạy xong là có ngay kết quả trong bảng"

Đây là phần quan trọng nhất để đáp ứng yêu cầu "khi chạy sẽ có luôn tất cả kết quả thực nghiệm cho vào bảng kết quả". Không lặp lại thao tác thủ công mỗi lần có seed mới — sửa `tools/bootstrap_boundary_ci.py` (đã có từ `spec-per-image-bootstrap-eval-v2`) để **tự mở rộng** khi có checkpoint mới, thay vì bảng cứng trong code.

### 4.1. Chuyển danh sách checkpoint từ bảng cứng sang manifest tự sinh

Hiện script đang cầm một bảng cứng 7 checkpoint (viết tay trong code, theo `spec-per-image-bootstrap-eval-v2` mục 3). Vì Run 7 có thể có 1, 2, hay nhiều seed tuỳ người dùng chạy bao nhiêu lần, **không được viết cứng seed vào code**. Thay bằng:

1. Script quét thư mục `dump/` theo pattern `run7_affinity_only_seed*_*` và `run7b_affinity_only_lambda02_seed*_*`, tự nhận diện mỗi checkpoint đã dump (dựa vào có đủ 3 file: `per_image_stats.csv`, thư mục mask PNG, `_meta.json`).
2. Ghi lại danh sách đã quét được vào `dump/checkpoints_manifest.json` (tự sinh, KHÔNG sửa tay), mỗi dòng gồm `key, run_name, iter, seed, path` — lấy `iter`/`seed`/`run_name` trực tiếp từ `_meta.json` của mỗi checkpoint (đã có sẵn theo đặc tả `spec-per-image-bootstrap-eval-v2` mục 2.3), không đoán từ tên thư mục.
3. 7 checkpoint cũ (Run 1, 2b, 3, 3b, 4, Run5×2) **giữ nguyên trong manifest** — không xoá, không tính lại (idempotent: chạy lại script không đổi giá trị cũ nếu file dump không đổi).
4. Mọi phần dưới đây (4.2–4.4) đọc checkpoint từ manifest này, **không** import bảng cứng.

### 4.2. Bảng cặp mới cần tính — nối tiếp 6 cặp đã có (đánh số 7 trở đi)

| # | Cặp | Vì sao | Ghi chú mốc |
|---|---|---|---|
| **7** | Affinity-only@40k (seed chính) − Baseline@40k | Hiệu ứng **trực tiếp** của affinity so với baseline — đối xứng với cặp "BCE λ=0.4 − Baseline" đã có, thay cho suy luận gián tiếp qua Static−BCE | cùng 40000 |
| **8** | Affinity-only@40k (seed chính) − BCE λ=0.4@40k | So sánh trực diện hai nhánh đơn — nhánh nào một mình mạnh hơn ở đâu | cùng 40000 |
| **9 ⭐** | **Contrast cộng tuyến (additivity):** `(Affinity-only@40k − Baseline@40k) − (Static s19@36k − BCE λ=0.4@40k)` | **Phép kiểm chính của run này.** Nếu affinity cộng tuyến với BCE, hai cách đo hiệu ứng affinity (trực tiếp vs. suy luận bằng trừ) phải cho cùng một con số → contrast này phải gần 0 | Static neo ở 36k trong khi 3 checkpoint còn lại neo ở 40k — **đây là đúng kiểu lệch mốc đã chấp nhận từ trước cho cặp #1** (Static−BCE), không phải tiêu chuẩn lỏng hơn mới đặt ra riêng cho run này. Ghi rõ mốc từng thành phần khi báo cáo |
| **10** | Affinity-only(0.2) − Affinity-only(0.4) *(chỉ nếu chạy `run7b`)* | Đường liều cho nhánh affinity-đơn, đối xứng đường liều đã có của BCE-đơn và của affinity-khi-ghép | cùng iteration nếu cả hai đều 40k |
| **11 🔁 (điều kiện)** | Affinity-only seed A − Affinity-only seed B *(chỉ khi ≥ 2 seed đã dump)* | Đúng vai trò cặp #6 (Static s19−s86) nhưng cho arm mới — cho arm này **biên độ liên-seed của chính nó**, thay vì phải mượn tạm biên độ của Static | cùng iteration (ưu tiên 40k nếu cả hai seed đều có `final_iter40000`) |

**Công thức tính contrast #9 (4 checkpoint, cùng bộ chỉ số resample cho cả bốn):**

```python
rng = np.random.default_rng(19)                    # giữ nguyên seed bootstrap của toàn dự án
idx = rng.integers(0, n_images, size=(B, n_images))  # B = 10000

for b in range(B):
    agg_affonly   = aggregate(stats_affonly,   idx[b])   # theo đúng quy ước micro đã xác nhận
    agg_baseline  = aggregate(stats_baseline,  idx[b])
    agg_static    = aggregate(stats_static,    idx[b])
    agg_bce04     = aggregate(stats_bce04,     idx[b])
    contrast[b] = (agg_affonly - agg_baseline) - (agg_static - agg_bce04)
    #            = agg_affonly - agg_baseline - agg_static + agg_bce04
```

Báo cáo: trung vị, CI 95% (percentile 2.5/97.5), `P(contrast < 0)`, cho cùng bộ đại lượng đã dùng ở mục 4.5 của spec bootstrap v2 (`asd_pred_to_gt`, `asd_gt_to_pred`, `asd`, `bf_precision`, `bf_recall`, `bf_score`, `mIoU-9`, per-class IoU của Bareland/Water/Agriculture).

### 4.3. Nếu có ≥ 2 seed — tự sinh bảng biên độ liên-seed cho Affinity-only

Đúng vai trò Run 5 đã làm cho Static (`khung-dien-giai...` mục 0.2), nhưng **tự động, không hardcode giá trị hai seed cụ thể**: script tính `|AffOnly_seedA − AffOnly_seedB|` tại iteration chung gần nhất có ở cả hai, cho toàn bộ danh sách đại lượng ở mục 4.5 (spec v2). Ghi ra `run7_interseed_amplitude.md`, cùng định dạng bảng đã dùng trong `khung-dien-giai...` mục 0.2, để có thể dán thẳng vào tài liệu diễn giải.

**Cho tới khi có bảng này** (tức là khi mới chạy 1 seed): dùng tạm bảng biên độ liên-seed của **Static** (mục 0.2, `khung-dien-giai...`) làm ngưỡng thận trọng cho mọi phát biểu độ lớn về Affinity-only — ghi rõ đây là **proxy tạm**, không phải biên độ đo trực tiếp của chính arm này, và phải thay bằng số thật ngay khi có seed thứ hai.

### 4.4. Output

- `bootstrap_ci.csv` / `bootstrap_ci.md` — **thêm hàng** cho cặp 7–11 vào đúng file đã có (không tạo file riêng, để bảng tổng hợp không bị phân mảnh).
- `run7_interseed_amplitude.md` — chỉ sinh khi ≥2 seed.
- Không sửa giá trị của 6 cặp cũ.

---

## 5. Cổng kiểm bootstrap — áp dụng nguyên mục 5 của spec v2, cộng thêm

- Cổng tái lập (dung sai 1e-6 nếu có JSON hậu kỳ ngay từ đầu — run này **có** vì dump chạy ngay sau train, không phải hồi cứu từ CSV cũ, nên **không cần nới dung sai** như Run 4/3b/5 — áp dụng dung sai chặt 1e-6 cho Run 7 ngay từ lần đầu).
- Danh sách `image` của checkpoint Affinity-only phải khớp tuyệt đối (nội dung + thứ tự) với 7 checkpoint cũ.
- Assert `alpha_bce_effective == 0` được ghi đúng trong `_meta.json`/log của mọi checkpoint Affinity-only — nếu không, dump từ config sai, dừng lại.

---

## 6. Quy tắc đọc kết quả — CHỐT TRƯỚC KHI CÓ SỐ

### 6.1. Câu hỏi A — Affinity-only có hiệu ứng đã xác nhận so với Baseline không? (cặp #7)

Áp dụng đúng quy tắc kép đã dùng cho mọi cặp khác trong dự án (spec v2 mục 7.6): một hiệu số chỉ "đã xác nhận độ lớn" nếu **đồng thời**:
(i) CI bootstrap 95% loại trừ 0, **và**
(ii) |Δ quan sát| lớn hơn biên độ liên-seed tương ứng (dùng bảng của chính Affinity-only nếu đã có ≥2 seed — mục 4.3; nếu chưa, dùng tạm bảng của Static làm proxy thận trọng, mục 4.3).

Không đạt cả hai ⇒ ghi "không phân biệt được", không đưa vào Abstract, đúng văn phong đã áp dụng cho mọi con số khác trong dự án.

### 6.2. Câu hỏi B — Affinity có cộng tuyến với BCE không? (contrast #9) — bảng quyết định chính của run này

| Nhánh | Điều kiện trên CI 95% của contrast #9 (đại lượng chính: `asd_gt_to_pred`) | Đọc ra | Hệ quả cho cách viết bài |
|---|---|---|---|
| **P — Cộng tuyến** | CI trùm 0 | Không phát hiện được tương tác ở độ phân giải hiện tại — cách suy luận "Static − BCE" đang dùng xuyên suốt dự án được **xác nhận thực nghiệm** thêm một bậc, không còn chỉ là giả định | Giữ nguyên mọi phát biểu cơ chế đã có (`khung-dien-giai...` mục 2). Thêm một câu trong Method: "kiểm tra cộng tuyến bằng run affinity-đơn không phát hiện tương tác đáng kể (CI trùm 0)" — một câu robustness-check ngắn, không đổi claim chính |
| **Q — Tương tác nhỏ nhưng có thật** | CI loại trừ 0 nhưng \|trung vị\| < biên độ liên-seed tương ứng | Có tín hiệu tương tác nhưng chưa vượt sàn nhiễu huấn luyện — giống hệt cách đọc mọi hiệu số ASD/recall khác trong dự án | Ghi nhận là quan sát sơ bộ, chưa đưa vào Abstract; cần seed thứ 2/3 của Affinity-only để xác nhận trước khi viết thành claim |
| **R — Tương tác lớn, đã xác nhận** | CI loại trừ 0 **và** \|trung vị\| ≥ biên độ liên-seed | Hai số hạng loss **không** hoạt động độc lập — vai trò của affinity phụ thuộc vào có BCE hay không | **Đây là một phát hiện mới, tự nó đáng một mục trong Results/Discussion** (không phải điều cần giấu). Phải viết lại phần "phạm vi đóng góp" (`khung-dien-giai...` mục 4) để nói rõ: kết luận về affinity chỉ có giá trị **trong bối cảnh** BCE=0.4 đang bật, không phải một thuộc tính độc lập của affinity |

**Không đổi ngưỡng ở bảng này sau khi thấy số.** Nếu ngưỡng có vẻ sai, ghi nhận xét bên cạnh, giữ nguyên verdict — đúng nguyên tắc đã áp dụng cho mọi bảng quyết định khác của dự án.

### 6.3. Không được làm

- Không so BIoU/BFScore giữa Affinity-only@36k-ish và BCE/Baseline@40k rồi kết luận theo dấu — các chỉ số ngưỡng-cứng đã biết là đảo dấu tuỳ mốc chọn checkpoint (bài học Run 2→Run 3). Chỉ ASD/precision/recall được phép so lệch mốc, và phải luôn ghi kèm iteration.
- Không dùng kết quả 1 seed để tuyên bố contrast #9 đã "đóng" — nhánh P/Q/R ở mục 6.2 chỉ đáng tin khi đã có bảng biên độ liên-seed thật của Affinity-only (mục 4.3); trước đó mọi verdict ở mục 6.2 là **tạm thời**.
- Không chạy nhiều seed rồi chọn seed "đẹp" để báo cáo — mọi seed đã chạy phải được đưa vào manifest và bảng kết quả, không được bỏ sót seed cho kết quả không như ý (đúng nguyên tắc đã nêu ở spec Run 5 mục 1 về việc chọn seed 86 "tuỳ ý, trước khi có số").

---

## 7. KHÔNG ĐƯỢC ĐỔI

- Splits (train + val 384 ảnh), sinh với `--seed 19` ở `Tools/create_splits.py`/`build_clean_val_split.py` — **không sinh lại**, bất kể seed huấn luyện của run này là gì
- `MAX_ITERS = 40000`, batch size, optimizer, LR schedule, augmentation policy
- `WINDOW_K = 5`, `DISTANCE = cosine`, `MARGIN = 1.0`, `CONNECTIVITY = 4`, `DILATION_RADIUS = 0`
- Cách downsample nhãn cho affinity: vẫn **nearest** về stride-4 — nghi vấn này là một thí nghiệm riêng (Run 6, đã huỷ), không gộp vào đây
- `src/losses/affinity.py`, `src/losses/boundary_bce.py`, `src/losses/total_loss.py` (phần logic loss) — **không sửa hành vi**, chỉ thêm cơ chế `--seed`/`{SEED}` ở tầng script train (mục 1.3) và cơ chế manifest ở tầng bootstrap (mục 4.1)
- Cách gộp mặc định (không cờ) của `eval_boundary_metrics.py` — vẫn phải byte-identical như trước, không đổi gì ở Phần A
- Mọi giá trị đã công bố của 6 cặp bootstrap cũ, và mọi file Track A-D / ResNet18-baseline

---

## 8. Checklist bàn giao

- [ ] Config `run7_affinity_only.yaml` theo mục 1.1 (và `run7b_affinity_only_lambda02.yaml` nếu làm liều phụ)
- [ ] `train_boundary_research.py`: thêm `--seed` bắt buộc cho config này, cơ chế format `{SEED}` vào `WORK_DIR`, dừng có lỗi rõ nếu thiếu seed
- [ ] Xác nhận **không** sinh lại splits; hash val list khớp Run 3/Run 5, cho **mỗi** seed chạy
- [ ] Dry-run: **10/10 cổng PASS** (mục 3), báo cáo từng cổng, cho seed đầu tiên chạy (khuyến nghị 19)
- [ ] Train ~2.5–3.5h/seed; đo và ghi lại thời gian thật/iter
- [ ] 4 checkpoint + `checkpoint_index.json`; CSV 10 hàng đủ cột; `summary.txt` ddof=1 + khối seed + khối liều
- [ ] Dump `--dump-preds --dump-per-image-stats` trên `best_miou` và `final_iter40000`, cho mỗi seed đã chạy
- [ ] `tools/bootstrap_boundary_ci.py`: chuyển sang đọc manifest tự sinh (mục 4.1), thêm cặp 7–11 (mục 4.2), thêm bảng biên độ liên-seed tự động khi ≥2 seed (mục 4.3)
- [ ] Cổng kiểm bootstrap (mục 5) PASS toàn bộ, dung sai 1e-6
- [ ] Áp dụng bảng quyết định 6.1/6.2 **nguyên văn**, ghi verdict (kèm rõ "tạm thời" nếu mới có 1 seed) vào tài liệu mới trong project
- [ ] Khi kết quả về: điền vào `khe-danh-gia-run7-affinity-only-chot-truoc-khi-co-so.md` (đã chuẩn bị sẵn) thay vì viết tài liệu đánh giá từ đầu
