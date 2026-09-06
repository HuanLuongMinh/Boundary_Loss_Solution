# Spec bàn giao Claude Code — Run 4: Dynamic Weighting (lịch trình đã sửa lại)

**Ngày:** 6/9/2026
**Repo:** `Boundary_Loss_Solution`
**Chi phí:** ~3.5h GPU (2×T4), vừa một phiên Kaggle.
**Quan hệ với Oracle-d:** hai thí nghiệm **độc lập**, chạy song song được. Nhưng nên chạy Oracle-d **trước** (0 GPU, ~1h) vì nó chốt cách đọc chỉ số mIoU của Run 4 **trước khi** số về — tránh đúng cái bẫy diễn giải hậu nghiệm đã mắc ở Run 3b.

---

## 0. Vì sao chạy run này, và vì sao lịch trình cũ phải bỏ

Run 4 là thành phần **phương pháp** của bài báo — tiêu đề đang là *"**Dynamic** Boundary-Aware Loss"*. Ba run tĩnh đã dựng xong phần phân tích; run này là thứ biến phân tích thành phương pháp.

Nó cũng có động cơ khoa học thật, không còn là một mục trong đề cương: phân rã chỉ số đã xác nhận **BCE lo độ đặc hiệu biên, affinity lo độ nhạy biên** — hai vai trò khác nhau, nên câu hỏi "áp lực nào, vào lúc nào của quá trình huấn luyện" là câu hỏi có nội dung.

### 0.1. Lịch trình gốc đi ngược bằng chứng — KHÔNG dùng

`src/losses/dynamic_weighting.py` hiện có:
```python
lambda1: 1.0 → 0.3     # giảm dần
lambda2: 0.0 → 1.0     # tăng tuyến tính suốt run
```

Hai lỗi, cả hai đều đã được dữ liệu chỉ ra:

1. **λ₁ giảm về 0.3** làm yếu dần nhánh BCE — nhánh duy nhất nâng precision và BIoU. Không có bằng chứng nào ủng hộ việc này.
2. **λ₂ ramp tuyến tính suốt 40k iter** khiến mô hình dành **phần lớn thời gian huấn luyện ở vùng liều thấp**. Run 3b đã đo chính xác vùng đó: ở α_aff = 0.2, `pred→gt` **xấu đi +0.41 px** và Agriculture **không được lợi gì**. Lịch trình cũ tối đa hoá thời gian ở đúng vùng có hại.

### 0.2. Đính chính một chỉ dẫn cũ đã lỗi thời

Các tài liệu viết ngày 4/9 (`workflow.md` 3.3, `huong-thuc-nghiem-tiep-theo-sau-run3` mục 6) ghi: *"λ₂ ramp 0 → λ₂_end = giá trị chốt từ Run 3b, **không** lên 1.0"*. Chỉ dẫn đó được viết khi còn kỳ vọng một liều **thấp** sẽ thắng.

Run 3b đã bác bỏ kỳ vọng đó: α_aff = 0.2 **bị chi phối hoàn toàn** — thua liều 0.4 ở `gt→pred`, recall, BFScore, `pred→gt`, ASD **và** mIoU. Liều tốt đã biết là **α_aff = 0.4**, mà với `ALPHA = 0.4` thì điều đó có nghĩa là **λ₂_end = 1.0**.

> **Chốt: `LAMBDA2_END = 1.0`.** Chỉ dẫn "không lên 1.0" đã hết hiệu lực.

---

## 1. Lịch trình mới: ramp nhanh → hold

```python
def lambda_schedule_ramp_hold(cur_iter, max_iters,
                              lambda1_const=1.0,
                              lambda2_end=1.0,
                              warmup_frac=0.10,
                              ramp_end_frac=0.30):
    """λ₁ phẳng; λ₂ giữ 0 trong warmup, ramp tuyến tính tới lambda2_end,
    rồi giữ nguyên tới hết. Vượt nhanh qua vùng liều thấp đã đo là có hại."""
    p = cur_iter / max(max_iters, 1)
    lam1 = lambda1_const
    if p < warmup_frac:
        lam2 = 0.0
    elif p < ramp_end_frac:
        lam2 = lambda2_end * (p - warmup_frac) / (ramp_end_frac - warmup_frac)
    else:
        lam2 = lambda2_end
    return lam1, lam2
```

Với `MAX_ITERS = 40000`:

| Giai đoạn | Iter | λ₁ | λ₂ | α_aff hiệu dụng |
|---|---|---|---|---|
| Warmup — chỉ BCE | 0 → 4 000 | 1.0 | 0.0 | 0.0 |
| Ramp | 4 000 → 12 000 | 1.0 | 0 → 1.0 | 0 → 0.4 |
| Hold | 12 000 → 40 000 | 1.0 | 1.0 | 0.4 |

**Ba lựa chọn thiết kế, mỗi cái có căn cứ:**

- **λ₁ phẳng 1.0** — giữ BCE ở đúng cường độ đã validate ở Run 2/3/3b. Cũng đảm bảo Run 4 chỉ đổi **một biến** so với Run 3.
- **Warmup 10%** — giữ từ thiết kế gốc. Để feature ổn định trước khi áp lực contrastive vào.
- **Ramp kết thúc ở 30%** — mô hình chỉ ở vùng liều thấp (λ₂ < 0.5, tức α_aff < 0.2) trong khoảng **4 000 iter = 10%** thời gian, thay vì ~50% như lịch trình cũ. Giai đoạn hội tụ và checkpoint cuối đều nằm trọn trong vùng liều 0.4 đã xác nhận.

**Biến thể decay (ramp → hold → decay, tắt affinity ở ~15% cuối) KHÔNG chạy ở run này.** Nó là biến thứ hai. Để dành thành Run 4b, chỉ chạy nếu Run 4 rơi vào nhánh 1 hoặc 2 của bảng quyết định.

---

## 2. Số đối chiếu — kiểm kê TRƯỚC khi chạy

### 2.1. Kiểm kê số đối chiếu — ĐÃ GIẢI QUYẾT (6/9/2026)

Đã đối chiếu trực tiếp `benchmark_results.csv` và `run3_static_boundary_boundary_metrics.json` của Run 3. **Nhánh dự phòng được kích hoạt.** Claude Code không cần kiểm tra lại, chỉ cần tuân thủ kết luận dưới đây.

**Cột thực có trong CSV Run 3 (10 hàng, đủ mọi vòng):**
```
iter, val_round, mIoU, val_loss,
l_region, l_bce, l_affinity, l_total,
bf_score, boundary_iou_d1, boundary_iou_d2, boundary_iou_d4, asd,
iou_Background ... iou_Building   (9 cột, CÓ kèm tên lớp),
is_best
```

**Ba thứ KHÔNG có trong CSV Run 3:** `asd_pred_to_gt` / `asd_gt_to_pred`, `bf_precision` / `bf_recall`, và per-class boundary IoU. Chỉ có giá trị **gộp**.

**Không tồn tại checkpoint tại iter 40000.** `is_best` chuyển True lần cuối ở vòng 9, nên `best_model.pth` = **iter 36000**, và JSON hậu kỳ được tính trên đúng checkpoint đó (mIoU JSON 0.65404855 khớp CSV vòng 9 = 0.65404774, lệch 8·10⁻⁷).

⇒ **Phân rã đầy đủ của Run 3 chỉ tồn tại tại một điểm duy nhất: iter 36000.**

**Hệ quả bắt buộc cho việc đọc Run 4:**

| Phép so | Mốc | Đại lượng dùng được |
|---|---|---|
| **Chính** — Run 4 vs Static | **cùng iter 36000** | Toàn bộ đại lượng phân rã (`pred→gt`, `gt→pred`, precision, recall) — Static lấy từ JSON |
| **Phụ** — Run 4 vs BCE λ=0.4 | cùng iter 40000 | Toàn bộ đại lượng phân rã — BCE λ=0.4 có JSON tại 40000 |
| Tham chiếu — Run 4 vs Static | cùng iter 40000 | **Chỉ** mIoU, BFScore gộp, BIoU gộp, ASD gộp |

**Vì vậy Run 4 PHẢI ghi đầy đủ boundary metrics tại vòng 9 (iter 36000)**, không chỉ 4 vòng cuối theo quy tắc tiết kiệm ở mục 4 — nếu không sẽ mất luôn cặp so sánh chính. Vòng 9 nằm trong nhóm 4 vòng cuối (round 7–10) nên điều này tự thoả, nhưng phải kiểm tra rõ ràng ở dry-run.

**Không được** so Run 4 @40000 với Static @36000 rồi kết luận — đó đúng là artifact chọn checkpoint đã làm đảo dấu BIoU giữa Run 2 và Run 3.

**Bài học lặp lại lần thứ hai:** Run 3 mất phân rã ở 9/10 vòng vì CSV không có cột. Đặc tả cột ở mục 4 của spec này là **bắt buộc tuyệt đối**, không phải khuyến nghị.

### 2.2. Bảng mốc

| | `pred→gt` ↓ | `gt→pred` ↓ | precision ↑ | recall ↑ | BF ↑ | mIoU-9 ↑ |
|---|---|---|---|---|---|---|
| Baseline @40k | 5.6797 | 4.2668 | 0.6188 | 0.5888 | 0.6034 | 0.6551 |
| BCE λ=0.4 @40k | 5.4366 | 4.3891 | **0.6299** | 0.5957 | 0.6123 | **0.6566** |
| Static (α_aff .4) @36k | **5.2158** | **4.0321** | 0.6241 | **0.6051** | **0.6144** | 0.6540 |
| Run 3b (α_aff .2) @40k | 5.8468 | 4.2393 | 0.6255 | 0.6036 | 0.6143 | 0.6442 |
| **Run 4 — cần đo** | ? | ? | ? | ? | ? | ? |

**Số bổ sung của Static, tính trực tiếp từ JSON/CSV Run 3 (6/9/2026) — dùng làm mốc, không tính lại:**

| Đại lượng | Static @36000 |
|---|---|
| mIoU-9 | 0.6540 |
| **mIoU-8** (bỏ Background) | **0.6151** — thua BCE λ=0.4 (0.6183) **0.32đ**, rộng hơn khoảng cách trên mIoU-9 (0.26đ) |
| **mIoU loại Bareland** (8 lớp) | **0.6931** |
| **mIoU-7** (bỏ cả Background và Bareland) | **0.6542** |
| BIoU macro-8 @d1 / d2 / d4 (×100) | 5.270 / 10.428 / 19.651 |
| BIoU Background @d1 / d2 / d4 (×100) | 10.519 / 21.892 / 40.713 |

**Sàn nhiễu công bố — ĐÃ SỬA (6/9/2026).** Giao thức báo cáo dùng **mean±std 4 vòng cuối (round 7–10, ddof=1)**, nên σ phải lấy theo đúng cửa sổ đó. Số cũ ±0.081 px được tính trên **3** vòng cuối và **không** dùng cho bảng quyết định:

| Đại lượng | σ (3 vòng cuối) | **σ (4 vòng cuối) — DÙNG CÁI NÀY** |
|---|---|---|
| ASD gộp | 0.081 | **0.149** |
| mIoU-9 | 0.0062 | **0.0052** |
| BFScore | 0.0028 | 0.0034 |
| Bareland | 0.023 | 0.028 |

Chênh lệch ở ASD là do vòng 7 (4.9812) lệch xa. Dùng σ = 0.149 làm ngưỡng **khắt khe hơn** — đó là lựa chọn đúng khi phải chọn. Mọi Δ nhỏ hơn σ tương ứng được ghi là **không phân biệt được**.

**Ghi chú độ lệch giữa hai pipeline eval:** tại iter 36000, mọi chỉ số của CSV (trong lúc train) khớp JSON (hậu kỳ) tới hết chữ số có sẵn — **trừ ASD**: CSV 4.6251 vs JSON 4.6240, lệch **0.0011 px**. Nhỏ hơn hai bậc so với các hiệu ứng đang đo (0.2–0.36 px) nên không đổi kết luận nào, nhưng phát biểu "hai pipeline cho cùng kết quả" là hơi mạnh. Khi so ASD giữa các run, **giữ nhất quán một nguồn** (JSON hậu kỳ, hoặc CSV) — đừng trộn.

---

## 3. Thay đổi code (3 file, tối thiểu)

### 3.1. `src/losses/dynamic_weighting.py`

**Thêm hàm mới, KHÔNG sửa `lambda_schedule` cũ** (giữ để tái lập được mọi thứ đã có). Thêm bộ chọn theo tên:

```python
SCHEDULES = {
    'linear':    lambda_schedule,            # bản gốc, giữ nguyên
    'ramp_hold': lambda_schedule_ramp_hold,  # MỚI — dùng cho Run 4
}
```

### 3.2. `src/losses/total_loss.py`

Nhánh `dynamic_weights=True` đọc tên lịch trình và tham số từ config thay vì gọi cứng `lambda_schedule`:

```python
if self.dynamic_weights:
    lam1, lam2 = SCHEDULES[self.schedule_name](
        cur_iter, max_iters, **self.schedule_kwargs)
else:
    lam1, lam2 = self.lambda1_static, self.lambda2_static
```

Mặc định `schedule_name = 'linear'` để config cũ không đổi hành vi. Công thức tổng **không đổi một ký tự**:
```
L_total = L_region + α · (λ₁(t) · L_BCE_edge + λ₂(t) · L_Affinity)
```
Dict trả về giữ nguyên các khoá của Run 3b, gồm `lambda1`, `lambda2`, `alpha_bce_effective`, `alpha_affinity_effective`.

**Trong warmup (λ₂ = 0): vẫn TÍNH và LOG `l_affinity` thô**, chỉ nhân trọng số 0. Bỏ tính sẽ tiết kiệm ~9 phút nhưng mất đường cong `l_affinity` toàn cục — mà đó là dữ liệu cho hình lịch trình của bài báo.

### 3.3. Config `run4_dynamic_ramp_hold.yaml`

Kế thừa **nguyên** config Run 3, đổi đúng khối này:

```yaml
BOUNDARY_LOSS:
  USE_BCE: true
  USE_AFFINITY: true
  DYNAMIC_WEIGHTS: true         # <── BIẾN DUY NHẤT ĐỔI so với Run 3
  SCHEDULE: ramp_hold           # <── MỚI
  ALPHA: 0.4                    # GIỮ NGUYÊN
  LAMBDA1_CONST: 1.0            # <── MỚI (λ₁ phẳng)
  LAMBDA2_END: 1.0              # <── MỚI (⇒ α_aff hiệu dụng đạt 0.4)
  WARMUP_FRAC: 0.10             # <── MỚI
  RAMP_END_FRAC: 0.30           # <── MỚI
  WINDOW_K: 5                   # GIỮ NGUYÊN
  DISTANCE: cosine              # GIỮ NGUYÊN
  MARGIN: 1.0                   # GIỮ NGUYÊN
  CONNECTIVITY: 4               # GIỮ NGUYÊN
  DILATION_RADIUS: 0            # GIỮ NGUYÊN
  POS_WEIGHT_MAX: 20.0          # GIỮ NGUYÊN

TRAIN:
  SEED: 19                      # GIỮ NGUYÊN
  MAX_ITERS: 40000              # GIỮ NGUYÊN

OUTPUT:
  WORK_DIR: work_dirs/phase1/run4_dynamic_ramp_hold
```

---

## 4. Checkpoint & log — áp dụng nguyên đặc tả Run 3b

**4 checkpoint** + `checkpoint_index.json`: `best_miou.pth`, `best_bfscore.pth`, `best_bareland.pth`, **`final_iter40000.pth`** (bắt buộc). Định dạng raw state dict: `torch.save(model.module.state_dict(), path)`.

**`benchmark_results.csv`**: 10 hàng, đủ bộ cột của spec Run 3b mục 4 — mọi per-class **kèm tên lớp**, ASD **cả hai chiều**, và các cột `lambda1, lambda2, alpha, alpha_bce_effective, alpha_affinity_effective`.

```python
CLASS_NAMES = ["Background","Bareland","Rangeland","Developed",
               "Road","Tree","Water","Agriculture","Building"]
```

**Mới cho run này — `lambda_schedule_log.csv`**: ghi `iter, lambda1, lambda2, alpha_bce_effective, alpha_affinity_effective, l_region, l_bce, l_affinity, l_total` mỗi **200 iter**. Đây là dữ liệu cho **hình lịch trình** của bài báo (trục thời gian huấn luyện × trọng số × giá trị từng số hạng loss). Rẻ, và không lấy lại được nếu quên.

**`summary.txt`** — thêm khối:
```
Schedule                   : ramp_hold
  lambda1 (const)          : 1.0000
  lambda2_end              : 1.0000
  warmup_frac              : 0.10   (iter 0 - 4000,  lambda2 = 0)
  ramp_end_frac            : 0.30   (iter 4000 - 12000, lambda2 0 -> 1.0)
  hold                     :        (iter 12000 - 40000, lambda2 = 1.0)
alpha                      : 0.4000
alpha_bce (effective)      : 0.4000  (khong doi suot run)
alpha_affinity (effective) : 0.0000 -> 0.4000
```
Giữ nguyên khối 4-checkpoint và khối mean±std 4 vòng cuối của Run 3b.

**Chi phí boundary metrics:** áp dụng nguyên mục 4.3 của spec Run 3b — nếu một vòng validate vượt ~5 phút, tính đầy đủ boundary metrics cho **4 vòng cuối** (round 7–10), các vòng 1–6 chỉ mIoU + per-class IoU, và **ghi rõ vào `summary.txt`**. Không được im lặng bỏ qua.

---

## 5. Cổng dry-run — `MAX_ITERS=100`, validate mỗi 25 iter

Bảy kiểm tra, tất cả phải PASS.

**5.1. Lịch trình đúng dạng đóng** — kiểm trực tiếp với `max_iters=40000`:

| cur_iter | λ₁ kỳ vọng | λ₂ kỳ vọng |
|---|---|---|
| 0 | 1.0 | 0.0 |
| 3 999 | 1.0 | 0.0 |
| 4 000 | 1.0 | 0.0 |
| 8 000 | 1.0 | **0.5** |
| 12 000 | 1.0 | **1.0** |
| 40 000 | 1.0 | 1.0 |

```python
assert abs(lam2 - expected) < 1e-6
```

**5.2. Đồng nhất công thức loss** — mọi vòng:
```python
expected = l_region + alpha * (lam1 * l_bce + lam2 * l_affinity)
assert abs(l_total - expected) < 1e-5
```

**5.3. Tương thích ngược.** Chạy lại config Run 3 cũ (`DYNAMIC_WEIGHTS: false`) và tái lập đúng đồng nhất thức đã biết. Đã xác minh trên **cả 10 hàng** CSV Run 3 (6/9/2026), dùng 3 hàng này làm mốc:
```
iter  4000:  1.4205 + 0.4*(0.9253 + 0.4042) = 1.9523   ✓
iter 36000:  1.0658 + 0.4*(0.8486 + 0.3649) = 1.5512   ✓
iter 40000:  1.0661 + 0.4*(0.8491 + 0.3653) = 1.5518   ✓
```
Cũng đã xác nhận `val_loss ≡ l_total` ở mọi hàng. Lệch ⇒ thay đổi đã phá hành vi cũ, **dừng lại**.

**5.4. Warmup thực sự vô hiệu hoá affinity.** Ở `cur_iter < warmup`, `alpha_affinity_effective == 0.0`, và `l_affinity` vẫn được **tính và log** với giá trị hữu hạn dương.

**5.5. Bốn checkpoint + `checkpoint_index.json`** ghi ra đúng, load lại được.

**5.6. CSV** đủ cột mục 4, mọi per-class kèm tên lớp, `lambda_schedule_log.csv` có hàng đúng mỗi 200 iter.

**5.7.** `grep "SEED: 19"` trên config mới trả kết quả. `pos_weight` in ra nằm trong 10.8–11.1 (khớp 4 run trước: 11.0392 / 10.8799 / 11.0014).

---

## 6. Chạy trên Kaggle

```bash
# 0. Kiem tra GPU TRUOC khi submit — training can dung 2x T4.
nvidia-smi --query-gpu=name --format=csv
# Neu ra "Tesla P100" (sm_60): KHONG chay training, doi phien khac.

ROOT=/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap
CFG=configs/research_boundary/phase1_fixed_split/run4_dynamic_ramp_hold.yaml

# 1. Dry-run
torchrun --nproc_per_node=2 src/train_boundary_research.py --config $CFG --dry-run

# 2. Run that (~3.5h; Run 3 co affinity mat 03h22m) — vua 1 phien Kaggle 9h
torchrun --nproc_per_node=2 src/train_boundary_research.py --config $CFG
```

Splits đã sinh sẵn với `--seed 19`; **không sinh lại**. Dataset đúng mirror `aletbm/...`, layout `images/{train,val}` + `label/{train,val}` (thư mục `label` **số ít** — giữ nguyên cơ chế auto-detect, không hard-code).

---

## 7. Quy tắc đọc kết quả — CHỐT TRƯỚC KHI CÓ SỐ

### 7.1. Chỉ số CHÍNH là các đại lượng đã phân rã

Đây là bài học rút ra từ Run 3b: bảng quyết định của run đó viết trên **ASD gộp**, đúng cái chỉ số mà nghiên cứu này đã kết luận là không đủ, nên nó cho một verdict đúng về hình thức nhưng nghèo thông tin.

**Chỉ số chính của Run 4: `asd_pred_to_gt` và `asd_gt_to_pred` riêng biệt, kèm `bf_precision` / `bf_recall`.**
mIoU là **chỉ số phụ**, đọc theo mục 7.3.

Mọi so sánh dùng **mean ± std (ddof=1) của 4 vòng cuối** *và* dòng **final@40000** — không dùng một checkpoint best đơn lẻ.

### 7.2. Bảng quyết định

Đối chiếu với **Static** (cùng liều cuối α_aff = 0.4, khác nhau đúng ở chỗ tĩnh hay động):

| Nhánh | Điều kiện | Diễn giải | Bước tiếp |
|---|---|---|---|
| **1 — Lịch trình thắng** | `gt→pred` **và** `pred→gt` đều ≤ Static, ít nhất một chiều tốt hơn > 1σ (**0.149**) **và** mIoU-9 ≥ **0.6514** (= BCE λ=0.4 − 1σ = 0.6566 − 0.0052) | Lịch trình mang lại thứ cấu hình tĩnh không có. Đây là kết quả tốt nhất có thể | **Run 4 là cấu hình đề xuất của bài.** Chạy Run 4b (thêm decay cuối) để dò tiếp. Nếu mIoU ≥ +0.4đ so baseline, kích hoạt 4 run đa-seed |
| **2 — Tránh được cái giá** | Hai chiều ASD ≈ Static (trong 1σ) **nhưng** mIoU-9 ≥ 0.6566 | Lịch trình giữ nguyên lợi ích biên của affinity mà **không phải trả −0.26đ mIoU**. Vẫn là một claim phương pháp thật | Run 4 là cấu hình đề xuất. Ghi rõ cơ chế: warmup + ramp tránh được giai đoạn feature bị affinity làm méo sớm |
| **3 — Vô hiệu** | Mọi đại lượng nằm trong 1σ của Static (ASD 0.149; mIoU 0.0052) | Lịch trình không thêm gì. **Đây là kết quả âm hợp lệ và phải báo cáo**, không phải thất bại | Cấu hình đề xuất = **Static**. Bài viết: "định thời không cải thiện được so với ghép tĩnh ở cùng liều cuối" — một phát hiện có ích cho người sau |
| **4 — Có hại** | Cả hai chiều ASD xấu hơn Static > 1σ (0.149) | Giai đoạn liều thấp có hại ngay cả khi ngắn, hoặc warmup làm hỏng khởi động | Chạy Run 4b': **bỏ ramp**, đặt λ₂ = 1.0 ngay sau warmup (bậc thang). Nếu vẫn xấu ⇒ định thời không phải hướng, chốt bài theo Static |

**Không đổi ngưỡng sau khi đã thấy số.** Nếu ngưỡng có vẻ sai, ghi nhận xét bên cạnh và giữ nguyên verdict.

### 7.3. mIoU — đọc theo giao thức giảm nhiễu, không đọc theo dấu

1. Chỉ so tại `final@40000`, kèm mean±std 4 vòng cuối.
2. Báo cáo **mIoU-9, mIoU-8, và mIoU loại Bareland** — cả ba, mọi hàng, kể cả baseline. Lý do loại Bareland là khách quan và độc lập với kết quả: lớp này gần như không được học ở **mọi** cấu hình (IoU 0.06–0.36, σ = 0.0735 lớn hơn mọi Δ giữa các cấu hình), nên nó hoạt động như một bộ phát nhiễu chung.
3. **Δ mIoU nhỏ hơn 0.52đ ⇒ ghi là "không phân biệt được"**, không diễn giải theo dấu, không đưa vào Abstract. (Sàn này là σ 4 vòng cuối của Run 3 = 0.0052; Run 3b cho 0.0064 — dùng giá trị của cấu hình đang so, và nói rõ dùng cái nào.)
4. Ngưỡng để mIoU trở thành claim: Δ ≥ +0.4đ so baseline **và** kết quả Oracle-d cho phép (G₂ ≥ 1.5đ theo bảng quyết định của spec oracle). Khi đó mới chi 4 run đa-seed.

### 7.4. Hình cho bài báo

Từ `lambda_schedule_log.csv`: một hình hai trục — λ₁/λ₂ theo iteration ở trên, ba số hạng `l_region` / `l_bce` / `l_affinity` ở dưới. Nó cho thấy trực quan giai đoạn nào của huấn luyện chịu áp lực nào, và là cách rẻ nhất để phần "dynamic" của tiêu đề có một hình đi kèm.

---

## 8. KHÔNG ĐƯỢC ĐỔI

Run này chỉ có giá trị khi đúng **một** biến thay đổi so với Run 3 (`DYNAMIC_WEIGHTS: false → true`). Giữ nguyên tuyệt đối:

- `SEED: 19`, `MAX_ITERS: 40000`, batch size, optimizer, LR schedule, augmentation
- `ALPHA: 0.4`; λ₁ phẳng 1.0 (**không** cho λ₁ biến thiên ở run này)
- `WINDOW_K: 5`, `DISTANCE: cosine`, `MARGIN: 1.0`, `CONNECTIVITY: 4`, `DILATION_RADIUS: 0`
- Cách downsample nhãn cho affinity: vẫn **nearest** về stride-4. Có nghi vấn về cách này, nhưng nó là **một thí nghiệm riêng**, không gộp vào đây
- Train/val split; `pos_weight` ước lượng như cũ
- `src/losses/affinity.py`, `src/losses/boundary_bce.py` — **không đụng vào**
- `lambda_schedule` gốc — thêm hàm mới bên cạnh, không sửa
- Mọi file của Track A-D và Track ResNet18-baseline

**Không gộp** projection head, sửa vùng lấy mẫu, hay decay cuối vào run này. Gộp mà tốt thì không biết công của cái nào; gộp mà xấu thì không biết lỗi của cái nào.

---

## 9. Checklist bàn giao

- [ ] Kiểm kê cột của Run 3 tại iter 40000 (mục 2.1) — báo lại trước khi chạy
- [ ] `dynamic_weighting.py`: thêm `lambda_schedule_ramp_hold` + registry `SCHEDULES`, không sửa hàm cũ
- [ ] `total_loss.py`: nhánh dynamic đọc `SCHEDULE` + kwargs từ config; warmup vẫn tính và log `l_affinity`
- [ ] Config `run4_dynamic_ramp_hold.yaml` theo mục 3.3
- [ ] Train script: 4 checkpoint + `checkpoint_index.json`; CSV 10 hàng đủ cột; `lambda_schedule_log.csv` mỗi 200 iter
- [ ] `summary.txt`: khối schedule + khối 4-checkpoint + khối mean±std 4 vòng cuối
- [ ] Dry-run: **7/7 PASS**, đo thời gian 1 vòng validate để quyết theo mục 4
- [ ] Xác nhận GPU là 2×T4 (không phải P100) rồi mới submit
- [ ] Chạy thật ~3.5h
- [ ] Ghi kết quả vào tài liệu mới trong project, áp dụng bảng 7.2 **nguyên văn**, đối chiếu đủ 5 cấu hình trên các đại lượng đã phân rã
