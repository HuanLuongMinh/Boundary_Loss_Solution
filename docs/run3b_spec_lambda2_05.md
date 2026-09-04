# Spec bàn giao Claude Code — Run 3b: Static Boundary với λ₂ = 0.5

**Ngày:** 4/9/2026
**Repo:** `Boundary_Loss_Solution`
**Mục tiêu:** chạy điểm giữa của đường quét hệ số affinity, với **đúng một biến thay đổi** so với Run 3.

---

## 0. Bối cảnh — vì sao chỉ cần 1 run này

Đường quét hệ số affinity cần 3 điểm. Hai điểm đã có sẵn, không phải chạy lại:

| α_affinity hiệu dụng | Cấu hình tương đương | Trạng thái |
|---|---|---|
| **0** | Run 2, BCE λ_edge = 0.4 (không có nhánh affinity) | ✅ đã chạy |
| **0.2** | **Run 3b — chính là run này** | 🆕 cần chạy |
| **0.4** | Run 3, `static_boundary`, λ₁ = λ₂ = 1, α = 0.4 | ✅ đã chạy |

Sau run này sẽ vẽ được đường phản ứng 3 điểm cách đều — đủ để biết affinity có "vùng ngọt" hay chỉ là một núm đánh đổi tuyến tính.

---

## 1. Nguyên tắc cài đặt — KHÔNG thêm hệ số mới

Công thức Run 3 đang chạy đã có sẵn chỗ cho hệ số này:

```
L_total = L_region + α · (λ₁ · L_BCE_edge + λ₂ · L_Affinity)
```

Hiện tại nhánh Static đang **hard-code** `λ₁ = λ₂ = 1.0`. Việc cần làm chỉ là **cho phép cấu hình 2 giá trị này**, rồi đặt `λ₂ = 0.5`:

```
L_total = L_region + 0.4 · (1.0 · L_BCE_edge + 0.5 · L_Affinity)
        = L_region + 0.4 · L_BCE_edge + 0.2 · L_Affinity
```

**Hệ số hiệu dụng:** `α_bce = 0.4` (giữ nguyên hệt Run 2/Run 3), `α_affinity = 0.2` (giảm một nửa so với Run 3).

**Không** tạo tham số mới kiểu `AFFINITY_SCALE`, **không** refactor `alpha` thành hai biến `alpha_bce`/`alpha_affinity`. Ba lý do:

1. λ₁/λ₂ đã là một phần thiết kế gốc (`workflow.md` mục 3.3/3.4) — dùng lại đúng ô đã có, không đẻ thêm khái niệm.
2. Công thức trong bài báo **không đổi một ký tự nào** — vẫn là `L_region + α(λ₁L_BCE + λ₂L_Aff)`, chỉ báo cáo λ₂ = 0.5 thay vì 1.0.
3. Cùng cái núm này chạy thẳng sang Run 4 (Dynamic), nơi `λ₂(t)` ramp từ 0 lên `λ₂_end` — giá trị tìm được ở đây trở thành `λ₂_end`, không phải chuyển đổi qua lại giữa hai hệ toạ độ tham số.

---

## 2. Thay đổi code (tối thiểu — 2 file)

### 2.1. `src/losses/total_loss.py`

Thêm 2 tham số vào `__init__`, **mặc định 1.0** để mọi config cũ tái lập y hệt:

```python
class DynamicBoundaryTotalLoss(nn.Module):
    def __init__(self, num_classes, alpha,
                 use_bce=True, use_affinity=True, dynamic_weights=True,
                 lambda1_static=1.0,      # <── MỚI, default 1.0 = hành vi Run 3
                 lambda2_static=1.0,      # <── MỚI, default 1.0 = hành vi Run 3
                 ignore_index=255, **kwargs):
        ...
        self.lambda1_static = float(lambda1_static)
        self.lambda2_static = float(lambda2_static)
```

Sửa đúng 1 nhánh trong `forward`:

```python
        if self.dynamic_weights:
            lam1, lam2 = lambda_schedule(cur_iter, max_iters)
        else:
            # TRƯỚC:  lam1, lam2 = 1.0, 1.0
            lam1, lam2 = self.lambda1_static, self.lambda2_static
```

Bổ sung vào dict trả về — **bắt buộc**, để summary/CSV không bao giờ mơ hồ về hệ số thực sự đã dùng:

```python
        return l_total, {
            'l_region': ..., 'l_bce': ..., 'l_affinity': ...,
            'lambda1': lam1, 'lambda2': lam2,
            'alpha': self.alpha,
            'alpha_bce_effective': self.alpha * lam1,        # <── MỚI
            'alpha_affinity_effective': self.alpha * lam2,   # <── MỚI
        }
```

### 2.2. Config mới

Tạo file mới **kế thừa nguyên config Run 3 hiện có** (giữ nguyên thư mục config mà Run 3 đang dùng, không tự đổi cấu trúc thư mục). Đặt tên gợi ý `run3b_static_lambda2_05.yaml`. Chỉ đổi 3 dòng:

```yaml
BOUNDARY_LOSS:
  USE_BCE: true
  USE_AFFINITY: true
  DYNAMIC_WEIGHTS: false
  ALPHA: 0.4                # GIỮ NGUYÊN
  LAMBDA1_STATIC: 1.0       # <── MỚI (giữ BCE ở đúng cường độ Run 2/Run 3)
  LAMBDA2_STATIC: 0.5       # <── MỚI (biến DUY NHẤT thay đổi)
  WINDOW_K: 5               # GIỮ NGUYÊN
  DISTANCE: cosine          # GIỮ NGUYÊN
  MARGIN: 1.0               # GIỮ NGUYÊN
  CONNECTIVITY: 4           # GIỮ NGUYÊN
  DILATION_RADIUS: 0        # GIỮ NGUYÊN
  POS_WEIGHT_MAX: 20.0      # GIỮ NGUYÊN

TRAIN:
  SEED: 19                  # GIỮ NGUYÊN

OUTPUT:
  WORK_DIR: work_dirs/phase1/run3b_static_lambda2_05
```

Script train đọc `LAMBDA1_STATIC`/`LAMBDA2_STATIC` từ YAML và truyền vào `DynamicBoundaryTotalLoss`. Nếu YAML không có 2 khoá này → dùng default 1.0 (config Run 3 cũ vẫn chạy y hệt như trước).

---

## 3. Lưu checkpoint — 4 file thay vì 1

Đây là thay đổi quan trọng thứ hai của run này. Lý do: Run 3 báo cáo tại iter 36000 trong khi Run 1/Run 2 báo cáo tại iter 40000, nên các so sánh boundary-metric giữa chúng bị lệch mốc và **đảo dấu** tuỳ checkpoint. Lưu đủ 4 checkpoint làm biến mất hoàn toàn vấn đề này, chi phí ~200MB.

Trong `work_dir`, lưu:

| File | Tiêu chí | Ghi chú |
|---|---|---|
| `best_miou.pth` | mIoU-9 cao nhất | tương thích ngược với quy ước hiện tại |
| `best_bfscore.pth` | BFScore cao nhất | |
| `best_bareland.pth` | IoU lớp Bareland cao nhất | lớp yếu nhất, biến động mạnh nhất |
| `final_iter40000.pth` | **luôn lưu vòng cuối cùng**, bất kể chỉ số | **quan trọng nhất** — cho so sánh cùng iteration giữa mọi run |

Giữ đúng định dạng đang dùng: `torch.save(model.module.state_dict(), path)` — raw state dict, không bọc thêm key `'model'`/`'epoch'` (các tool eval hiện có phụ thuộc vào điều này).

Kèm `checkpoint_index.json` trong cùng `work_dir`:

```json
{
  "best_miou":       {"file": "best_miou.pth",       "iter": 36000, "round": 9,  "metrics": { ...toàn bộ hàng CSV của vòng đó... }},
  "best_bfscore":    {"file": "best_bfscore.pth",    "iter": 40000, "round": 10, "metrics": { ... }},
  "best_bareland":   {"file": "best_bareland.pth",   "iter": 28000, "round": 7,  "metrics": { ... }},
  "final":           {"file": "final_iter40000.pth", "iter": 40000, "round": 10, "metrics": { ... }}
}
```

Không có file này thì sau vài tuần sẽ không ai biết `best_miou.pth` là của iteration nào — đúng lỗ hổng đã gây ra so sánh lệch mốc ở Run 3.

---

## 4. Log per-round — CSV đủ 10 vòng

`benchmark_results.csv` phải có **một hàng cho mỗi vòng validate** (10 hàng), không chỉ hàng best. Đây là thứ cho phép tính mean±std hậu kỳ mà không tốn thêm run nào.

Cột bắt buộc:

```
round, iter,
miou9, miou8,
iou_Background, iou_Bareland, iou_Rangeland, iou_Developed, iou_Road,
iou_Tree, iou_Water, iou_Agriculture, iou_Building,
bf_score, bf_precision, bf_recall,
biou_d1, biou_d2, biou_d4,
biou_d1_Background ... biou_d1_Building,      (9 cột)
biou_d2_Background ... biou_d2_Building,      (9 cột)
biou_d4_Background ... biou_d4_Building,      (9 cột)
asd, asd_pred_to_gt, asd_gt_to_pred,
l_region, l_bce, l_affinity, l_total,
lambda1, lambda2, alpha, alpha_bce_effective, alpha_affinity_effective,
is_best_miou, is_best_bfscore, is_best_bareland, is_final
```

### 4.1. BẮT BUỘC: mọi output per-class phải kèm TÊN LỚP

Không bao giờ xuất mảng thuần không nhãn. Thứ tự chuẩn, dùng thống nhất mọi nơi:

```python
CLASS_NAMES = ["Background", "Bareland", "Rangeland", "Developed",
               "Road", "Tree", "Water", "Agriculture", "Building"]
```

Lý do cụ thể, không phải quy tắc hình thức: file `run1_baseline_summary_boundary_filled.txt` trả `per_class_iou` dưới dạng mảng thuần, và khi lập bảng đã bị **hoán đổi Water ↔ Agriculture** (idx6 = Water = 0.72788, idx7 = Agriculture = 0.72636) trong một tài liệu tổng hợp. Lỗi rơi đúng vào hàng baseline — hàng mà mọi Δ khác đo dựa vào.

### 4.2. ASD phải in cả hai chiều

`asd_pred_to_gt` và `asd_gt_to_pred` riêng, ngoài giá trị gộp. Baseline có `5.6797` vs `4.2668` — bất đối xứng mạnh, nghĩa là biên dự đoán nằm xa GT hơn chiều ngược lại (sinh biên thừa/đứt gãy). ASD gộp che mất thông tin này, mà đây chính là thứ cho biết boundary loss đang sửa chiều nào.

### 4.3. Cảnh báo hiệu năng

Boundary metrics dùng `scipy.ndimage.distance_transform_edt` (CPU-bound). Post-hoc eval Run 1 mất **26.6 phút** cho 384 ảnh. Tính đủ cho 10 vòng có thể cộng thêm ~2-4h vào run 3.5h.

**Xử lý:** đo thời gian 1 vòng validate ở bước dry-run. Nếu vượt ~5 phút/vòng:
- Tính **đầy đủ** boundary metrics cho **4 vòng cuối** (round 7-10) — đủ cho mean±std.
- Các vòng 1-6 chỉ tính mIoU + per-class IoU (rẻ), để trống các cột boundary.
- Ghi rõ điều này vào `summary.txt`.

Không được im lặng bỏ qua boundary metrics ở mọi vòng rồi chỉ tính ở checkpoint best — như vậy quay lại đúng vấn đề cũ.

---

## 5. `summary.txt` — bổ sung

Giữ nguyên toàn bộ định dạng hiện có (nó đã rất tốt), thêm:

```
Total loss                 : L_total = L_region + alpha*(lambda1*L_BCE_edge + lambda2*L_Affinity)
alpha                      : 0.4000
lambda1 (static)           : 1.0000
lambda2 (static)           : 0.5000
alpha_bce  (effective)     : 0.4000      <-- alpha * lambda1
alpha_affinity (effective) : 0.2000      <-- alpha * lambda2
```

Và một mục mới thay cho khối "FINAL EVALUATION — BEST CHECKPOINT" đơn lẻ:

```
================================================================
 EVALUATION -- 4 CHECKPOINTS
================================================================
Checkpoint        Iter    mIoU-9   mIoU-8   BFScore  BIoU@2   BIoU@4   ASD     Bareland
best_miou         .....
best_bfscore      .....
best_bareland     .....
final             40000

================================================================
 MEAN +/- STD -- 4 VONG CUOI (round 7-10)
================================================================
mIoU-9      : 0.xxxx +/- 0.xxxx
mIoU-8      : ...
BFScore     : ...
BIoU@2      : ...
BIoU@4      : ...
ASD         : ...
Bareland    : ...
(+ mean±std cho toàn bộ 9 per-class IoU, kèm tên lớp)
```

---

## 6. Cổng dry-run — chạy TRƯỚC khi submit run thật

Chạy với `MAX_ITERS=100`, validate mỗi 25 iter. Sáu kiểm tra, tất cả phải PASS:

**6.1. Đồng nhất công thức loss** (quan trọng nhất — bắt lỗi hệ số không tới được loss):
```python
expected = l_region + alpha * (lam1 * l_bce + lam2 * l_affinity)
assert abs(l_total - expected) < 1e-5
```

**6.2. λ₂ thực sự có hiệu lực.** Log `alpha_affinity_effective` và khẳng định `== 0.2`. Nếu in ra 0.4 thì config chưa được đọc.

**6.3. Tương thích ngược.** Chạy lại config Run 3 cũ (không có 2 khoá mới) và verify tái lập đúng đồng nhất thức đã biết:
```
1.0661 + 0.4*(0.8491 + 0.3653) = 1.5518   ✓ (số thật từ summary.txt của Run 3)
```
Nếu con số này lệch, thay đổi đã phá hành vi cũ — dừng lại.

**6.4. Bốn file checkpoint + `checkpoint_index.json`** được ghi ra đúng, load lại được bằng `load_state_dict` với wrapper đúng.

**6.5. CSV** có đủ mọi cột ở mục 4, mọi cột per-class **kèm tên lớp**, không có cột nào toàn NaN ngoài các cột boundary đã cố ý bỏ trống theo 4.3.

**6.6.** `grep "SEED: 19"` trên config mới trả về kết quả. `pos_weight` in ra nằm trong khoảng 10.8–11.1 (khớp 3 run trước: 11.0392 / 10.8799 / 11.0014).

---

## 7. Việc phụ — chạy trên CPU song song, không tốn GPU

**Cập nhật 4/9/2026: lỗ hổng ASD đã được lấp, và kết quả đổi hẳn cách đọc run này.** Số đầy đủ từ `ablation.xlsx`:

| Method | ASD (px) | Δ vs baseline | Δ vs BCE λ=0.4 |
|---|---|---|---|
| Baseline | 4.9732 | — | |
| BCE λ=0.2 | 4.8347 | −0.139 (−2.8%) | |
| BCE λ=0.4 | 4.9128 | −0.060 (−1.2%) | — |
| Boundary_Static (α_aff=0.4) | 4.6251 | −0.348 (−7.0%) | **−0.288 (−5.9%)** |

**Phân bổ công:** trong tổng mức giảm 0.348 px từ baseline đến Static, BCE đóng góp 0.060 (17%), **affinity đóng góp 0.288 (83%)**. BCE gần như không tác động lên ASD (và còn phi đơn điệu: λ=0.2 tốt hơn λ=0.4). Đây là **chỉ số duy nhất affinity tạo ra hiệu ứng thật, ngoài nhiễu**:

- Kiểm tra cùng iteration: Static @40000 = 4.731 vs BCE λ=0.4 @40000 = 4.9128 → vẫn −0.182 px (−3.7%). Hiệu ứng co lại nhưng **không đảo dấu** (khác hẳn BIoU).
- Kiểm tra nhiễu: Static ASD 3 vòng cuối = 4.713 ± 0.081 → khoảng cách tới BCE λ=0.4 là 2.5 σ. Ngoài nhiễu.
- Chênh lệch ASD giữa BCE λ=0.2 và λ=0.4 (0.078) nằm trong 1 σ → xác nhận BCE không điều khiển ASD một cách hệ thống.

### 7.1. Giả thuyết cơ chế (giải thích được toàn bộ bảng số)

ASD nhạy với **đuôi phân phối** — mảnh biên giả nằm xa mọi biên thật đóng góp rất lớn. BIoU@d và BFScore thì bão hoà ngoài ngưỡng d, nên gần như mù với loại lỗi này.

⇒ **BCE định vị biên; affinity dập biên giả ở nơi không có biên.** Hai vai trò bổ sung nhau, không chồng lấn — đúng luận điểm hai thành phần của bài báo. Affinity không làm biên sắc hơn (việc của BCE), nó **xoá phản hồi biên sai trong vùng trong**, và chỉ số bắt được đúng việc đó là ASD.

### 7.2. Hai phép đo hậu kỳ còn lại — kiểm định sắc bén giả thuyết trên

Giả thuyết 7.1 đưa ra **hai dự đoán định lượng có thể sai**. Cả hai đo được ngay trên checkpoint đã có, 0 GPU:

| # | Phép đo (cho **BCE λ=0.4** và **Static**) | Dự đoán nếu 7.1 đúng | Nếu sai |
|---|---|---|---|
| 1 | `asd_pred_to_gt` và `asd_gt_to_pred` tách riêng | Lợi ích của affinity **tập trung ở chiều pred→gt** (biên dự đoán thừa, nằm xa GT). Baseline: pred→gt 5.6797 vs gt→pred 4.2668 | Nếu lợi ích nằm ở gt→pred thì cơ chế là "bắt được biên bị bỏ sót", không phải "dập biên giả" — phải viết lại phần diễn giải |
| 2 | `bf_precision` / `bf_recall` tách riêng | Affinity **nâng precision** nhiều hơn recall. Baseline: P 0.6188 / R 0.5888 | Nếu nâng recall là chính thì mâu thuẫn với 7.1 |

Đây hiện là **hai con số có giá trị cao nhất trong toàn bộ nghiên cứu**: chúng biến một quan sát ("ASD tốt hơn") thành một cơ chế có thể phát biểu và bảo vệ trong bài báo.

### 7.3. Vẫn nên làm

**Per-class BIoU@1/2/4 cho BCE λ=0.4 và Static** (baseline đã có trong JSON). Câu hỏi: Water có BIoU tăng trong khi IoU vùng giảm 2.88đ không? Nếu có, đó là affinity làm đúng việc ở biên nhưng trả giá ở vùng trong — nhất quán với 7.1.

Tất cả in kèm tên lớp và cả hai chiều ASD, theo mục 4.1/4.2.

### 7.4. Ghi chú đính chính dữ liệu

`ablation.xlsx` ghi **đúng** `iou_Water = 0.7279`, `iou_Agriculture = 0.7264` cho baseline — khớp JSON gốc. Chỗ sai chỉ nằm ở bảng trong tài liệu markdown `phase1-run2-bce-ablation-va-quyet-dinh-huong-tiep.md` (hàng baseline hoán đổi hai lớp). **Sửa tài liệu markdown, KHÔNG sửa xlsx.**

---

## 8. Danh sách KHÔNG ĐƯỢC ĐỔI

Run này chỉ có giá trị khi đúng một biến thay đổi. Giữ nguyên tuyệt đối:

- `SEED: 19`
- `ALPHA: 0.4` và `LAMBDA1_STATIC: 1.0`
- `WINDOW_K: 5`, `DISTANCE: cosine`, `MARGIN: 1.0`
- `CONNECTIVITY: 4`, `DILATION_RADIUS: 0`
- Cách downsample nhãn cho affinity (vẫn **nearest** về stride-4 — có nghi vấn về cách này, nhưng **không sửa ở run này**, để dành thành một thí nghiệm riêng)
- Train/val split, augmentation, optimizer, LR schedule, `MAX_ITERS: 40000`, batch size
- Cách ước lượng `pos_weight` (biến động giữa các run chỉ ±0.7%, không phải nguồn nhiễu)
- `src/losses/affinity.py`, `src/losses/boundary_bce.py` — **không đụng vào**
- Mọi file của Track A-D và Track ResNet18-baseline

---

## 9. Checklist bàn giao

- [ ] `total_loss.py`: thêm `lambda1_static`/`lambda2_static` (default 1.0), sửa nhánh static, trả thêm 2 khoá effective
- [ ] Config `run3b_static_lambda2_05.yaml`: `LAMBDA1_STATIC: 1.0`, `LAMBDA2_STATIC: 0.5`, phần còn lại kế thừa Run 3
- [ ] Script train: đọc 2 khoá mới; lưu 4 checkpoint + `checkpoint_index.json`
- [ ] CSV: 10 hàng per-round, đủ cột mục 4, mọi per-class kèm tên lớp, ASD 2 chiều
- [ ] `summary.txt`: khối 4-checkpoint + khối mean±std 4 vòng cuối + dòng hệ số effective
- [ ] Dry-run: 6/6 check PASS, đo thời gian 1 vòng validate để quyết theo mục 4.3
- [ ] Submit run thật (~3.5h dự kiến, Run 3 mất 03h22m)
- [ ] Song song trên CPU: 3 lệnh eval hậu kỳ ở mục 7

---

## 10. Quy tắc đọc kết quả — chốt TRƯỚC khi có số

So sánh dùng **mean±std 4 vòng cuối** và dòng **final@40000**, không dùng một checkpoint best đơn lẻ.

### 10.1. ASD là chỉ số CHÍNH của run này

Đổi so với bản trước, do dữ liệu ở mục 7. Lý do: ASD là chỉ số duy nhất affinity tạo hiệu ứng ngoài nhiễu (2.5 σ) và không đảo dấu khi so cùng iteration. BFScore/BIoU đã chứng minh là **mù** với thứ affinity làm được, còn mIoU đo cái giá phải trả. Vậy câu hỏi thật của run này:

> **Ở λ₂ = 0.5, giữ được bao nhiêu phần lợi ích ASD, và trả lại được bao nhiêu phần chi phí mIoU?**

### 10.2. Ba mốc đối chiếu

| | ASD (px) | mIoU-9 | mIoU-8 | BFScore | BIoU@4 | Bareland |
|---|---|---|---|---|---|---|
| **BCE λ=0.4** (α_aff = 0, đầu mút trái) | 4.9128 | 0.6566 | 0.6183 | 0.6123 | 0.2186 | 0.3452 |
| **Static** (α_aff = 0.4, đầu mút phải), mean±std 3 vòng cuối | 4.713 ± 0.081 | 0.6503 ± 0.0062 | — | 0.6133 ± 0.0028 | 0.2147 ± 0.0058 | 0.320 ± 0.023 |
| **Run 3b** (α_aff = 0.2) — điểm cần đo | ? | ? | ? | ? | ? | ? |

Nội suy tuyến tính giữa hai đầu mút cho ASD ≈ **4.81** và mIoU-9 ≈ **0.6535**. Dùng làm mốc "không có gì đặc biệt".

### 10.3. Bảng quyết định

| Kết quả tại λ₂ = 0.5 | Diễn giải | Bước tiếp |
|---|---|---|
| **ASD ≤ 4.75** (giữ ~phần lớn lợi ích) **và** mIoU-9 ≥ 0.6545 (hồi về sát BCE) | **Vùng ngọt tồn tại** — quan hệ ASD–α phi tuyến, lợi ích bão hoà sớm còn chi phí thì tuyến tính. Kết quả tốt nhất có thể | Chốt λ₂\* = 0.5. Sang Run 4 (Dynamic) với `λ₂_end = 0.5`. Đây trở thành cấu hình đề xuất của bài |
| ASD ≈ 4.78–4.84 **và** mIoU ≈ 0.653–0.654 (bám sát nội suy tuyến tính) | Affinity là **núm đánh đổi tuyến tính** ASD ↔ mIoU, không có vùng ngọt | Vẫn còn giá trị: bài báo phát biểu được một đánh đổi định lượng, có kiểm soát. Chọn điểm vận hành theo mục tiêu (α_aff=0.4 nếu ưu tiên ASD), rồi sang Run 4. **Không** quét thêm α |
| ASD ≥ 4.88 (mất gần hết lợi ích dù chỉ giảm một nửa cường độ) | Hiệu ứng ASD có **ngưỡng**, cần cường độ cao | Giữ α_aff = 0.4 làm cấu hình đề xuất; chuyển ngân sách còn lại sang thí nghiệm sửa vùng lấy mẫu (dilation/max-pool) để hạ chi phí mIoU thay vì hạ α |
| ASD ≥ 4.91 (bằng hoặc tệ hơn BCE) **và** mIoU cũng không hồi | Phi đơn điệu — còn yếu tố ngoài cường độ | Dừng quét α. Quay lại 2 phép đo ở mục 7.2 trước khi chi thêm GPU |

Trong **mọi** nhánh trừ nhánh cuối, nghiên cứu vẫn có kết quả viết được — điểm khác nhau chỉ là phát biểu "có cấu hình tối ưu" hay "có một đánh đổi định lượng". Cả hai đều hợp lệ.

### 10.4. Ghi kết quả

Ghi vào một tài liệu mới trong project, đối chiếu đủ **3 điểm** của đường quét (α_aff = 0 / 0.2 / 0.4) trên **cả 6 chỉ số** ở bảng 10.2, kèm mean±std và dòng final@40000. **Không** kết luận từ một checkpoint đơn lẻ. Nếu 2 phép đo ở mục 7.2 đã có kết quả, ghép luôn vào để phát biểu cơ chế.
