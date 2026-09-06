# Spec bàn giao Claude Code — Thí nghiệm Oracle-d: đo trần cải thiện mIoU của mọi phương pháp thuần biên

**Ngày:** 6/9/2026
**Repo:** `Boundary_Loss_Solution`
**Chi phí:** 0 GPU. Toàn bộ chạy CPU, ước lượng 35–60 phút.
**Ưu tiên:** cao nhất hiện nay — kết quả của nó quyết định có nên chi 20–35h GPU cho nhánh mIoU hay không.

---

## 0. Câu hỏi nghiên cứu

> Nếu một phương pháp **chỉ tác động vào vùng gần biên** sửa được **hoàn hảo** mọi pixel trong dải rộng d quanh biên thật, thì mIoU tăng tối đa bao nhiêu điểm?

Con số đó là **trần tuyệt đối** của mọi loss thuần biên (BCE-edge, affinity, và mọi lịch trình ghép chúng) trên dataset này. Không cấu hình nào vượt qua được, vì oracle giả định sửa đúng 100%.

Hiện nghiên cứu đang tranh luận liệu trọng số động có thể mang lại +0.4–0.8 điểm mIoU hay không. Ước lượng trên giấy (`huong-mo-rong-projection-head-va-contrastive.md` mục 3.2) cho trần ~1–2 điểm, nhưng đó là phép nhân thô `r_edge × độ dịch chuyển`. Thí nghiệm này thay ước lượng bằng số đo.

**Kết quả phụ, có giá trị độc lập:** bảng oracle per-class trả lời dứt điểm câu hỏi Bareland. Nếu Bareland có mức tăng oracle nhỏ, thì **không loss biên nào có thể cứu được lớp này** — đó là bằng chứng khẳng định, mạnh hơn quan sát "chưa cấu hình nào vượt baseline" đang dùng.

---

## 1. Checkpoint dùng làm gốc

**Checkpoint chính: Run 1 — Baseline (α = 0), iter 40000.**

Lý do chọn baseline chứ không phải cấu hình tốt nhất: oracle đo **headroom của bài toán**, không phải headroom còn lại của một cấu hình. Baseline là điểm neo mà mọi Δ trong Bảng 1 đo dựa vào.

Thuận lợi về mặt so sánh: Run 1 chỉ lưu **một** checkpoint, và checkpoint đó đạt best tại **vòng 10/10 = iter 40000** (xác nhận trong `phase1-run1-baseline-ket-qua.md`). Vậy nó **trùng mốc `final@40000`** của các run sau — không dính artifact chọn checkpoint đã phát hiện ở Run 3.

**Cách định vị file:**
```
work_dirs/phase1/run1_baseline/          # hoặc thư mục work_dir thực tế của Run 1
├── checkpoint_index.json                # nếu có → lấy entry "best_miou" hoặc "final"
└── *.pth                                # nếu không có index → file .pth duy nhất
```
Nếu tìm thấy nhiều `.pth`, **DỪNG và hỏi Huan** — không tự đoán.

Định dạng checkpoint (giữ nguyên quy ước repo): raw `state_dict`, load bằng
```python
model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
```
không bọc key `'model'`/`'epoch'`.

### 1.1. Chạy trên NHIỀU checkpoint — nâng từ tuỳ chọn lên yêu cầu chính

Bản đầu của spec này chỉ coi baseline là bắt buộc. **Sửa lại: chạy trên nhiều checkpoint đáng giá hơn nhiều so với chi phí thêm**, vì nó cho một phép phân tách mà nghiên cứu hiện chưa có (mục 6.1).

Bản đồ khoảng cách `dist` **chỉ phụ thuộc GT, không phụ thuộc checkpoint** — tính một lần, cache lại, dùng chung cho mọi checkpoint. Chi phí thêm của mỗi checkpoint vì vậy chỉ là một lượt suy luận 384 ảnh (~24.5 phút CPU ở 3.82 s/ảnh, đo thực tế từ log eval Run 3), phần oracle sau đó gần như miễn phí.

| Ưu tiên | Cấu hình | Checkpoint | Iter | Vì sao |
|---|---|---|---|---|
| **1 — bắt buộc** | Baseline (Run 1) | `best_model.pth` | **40000** | Trần của bài toán. Con số gate mọi thứ |
| **2 — bắt buộc** | BCE λ=0.4 (Run 2b) | `best_model.pth` | **40000** | Cấu hình tốt nhất hiện có; headroom còn lại |
| **3 — bắt buộc** | Static α_aff=0.4 (Run 3) | `best_model.pth` | **36000** ⚠ | Cấu hình có affinity. **Xem cảnh báo 1.2** |
| — hoãn | Run 3b α_aff=0.2 | `final_iter40000.pth` | 40000 | **Không chạy ở lượt này.** Chỉ chạy nếu kích hoạt điều kiện ở mục 6.1(c) |
| — hoãn | BCE λ=0.2 (Run 2a) | `best_model.pth` | 40000 | Không chạy. Nằm trên trục liều BCE, không phải trục affinity mà thí nghiệm này quan tâm |

**Ngân sách:** đúng **3 checkpoint** ≈ **75 phút + ~10 phút EDT**. Toàn bộ CPU.

**Quyết định 6/9/2026 (Huan):** không chạy Run 3b. Lý do: α_aff = 0.2 đã bị chi phối hoàn toàn ở mọi đại lượng, và khoảng cách giữa hai vòng validate của nó nằm trong nhiễu — thêm một hàng oracle cho một cấu hình đã bị loại không trả lời câu hỏi nào, trong khi vẫn tốn một lượt suy luận đầy đủ.

Chạy theo đúng thứ tự ưu tiên, ghi kết quả sau **mỗi** checkpoint — nếu phiên Kaggle bị cắt giữa chừng thì phần đã chạy vẫn dùng được.

### 1.2. ⚠ Cảnh báo lệch iteration — phải ghi vào mọi bảng

Run 3 (Static) **không có checkpoint tại 40000**; `best_model.pth` của nó là **iter 36000** (đã xác minh 6/9/2026 từ CSV: `is_best` chuyển True lần cuối ở vòng 9). Các checkpoint còn lại đều ở 40000.

So **giá trị tuyệt đối** của oracle giữa Static và các cấu hình khác vì vậy dính đúng artifact chọn checkpoint đã làm đảo dấu BIoU giữa Run 2 và Run 3. Hai quy tắc bắt buộc:

1. **Cột `iter` phải xuất hiện trong mọi bảng**, không được lược bỏ.
2. Đại lượng chính để so giữa các cấu hình là **khoảng cách oracle** `O(c) − M(c)` — một đại lượng **nội bộ trong cùng một checkpoint**, nên ít nhạy với lệch mốc hơn hẳn so với so `O(c)` tuyệt đối. Ưu tiên đọc theo khoảng cách; đọc `O(c)` tuyệt đối thì phải kèm cảnh báo này.
3. **Mức nghiêm trọng của lệch mốc này thấp hơn nhiều so với ở BIoU** — có bằng chứng số. Trong không gian mIoU, Run 3 gần như đứng yên giữa hai vòng cuối: **0.6540 @36000 vs 0.6538 @40000**, lệch **0.02đ**. Khác hẳn BIoU@4, nơi cùng hai vòng đó lệch 0.43đ và **đảo thứ tự** so với BCE λ=0.4. Vì oracle làm việc hoàn toàn trong không gian mIoU, artifact chọn checkpoint ở đây gần như không tác dụng ở mức tổng thể.

   **Nhưng ở mức per-class thì vẫn đáng lưu ý:** giữa hai vòng đó Water đổi 0.68đ (0.6991 → 0.7059) và Bareland đổi 1.6đ (0.3415 → 0.3254). Phép đọc per-class ở mục 6.1(c) vì vậy là **chỉ báo**, không phải bằng chứng chốt.

---

## 2. Định nghĩa chính xác của phép biến đổi oracle

Với **mỗi ảnh val**, gọi `y` = nhãn GT `(H,W)` int64, `y_hat` = argmax của logits `(H,W)`:

```
Bước 1. B_gt = extract_edge_gt(y, ignore_index=255, connectivity=4, dilation_radius=0)
        → mask nhị phân (H,W), biên GT dày 1 pixel, ở ĐỘ PHÂN GIẢI ĐẦY ĐỦ.

Bước 2. dist = scipy.ndimage.distance_transform_edt(~B_gt)
        → dist[i,j] = khoảng cách Euclid từ pixel (i,j) tới pixel biên GT gần nhất.
          Pixel nằm trên biên có dist = 0.

Bước 3. band_d = (dist <= d)

Bước 4. y_oracle_d = np.where(band_d, y, y_hat)
        → trong dải: lấy nhãn đúng. Ngoài dải: giữ nguyên dự đoán của model.

Bước 5. Cộng dồn y_oracle_d vs y vào confusion matrix TOÀN CỤC.
```

**Dùng đúng hàm `extract_edge_gt` mà `src/losses/boundary_bce.py` đang dùng** — import lại, không viết lại. `connectivity=4`, `dilation_radius=0` giữ đúng cấu hình của mọi run.

### 2.1. Các mức d phải chạy

| Nhãn | Mô tả | Ghi chú |
|---|---|---|
| `none` | **không sửa gì** — `y_oracle = y_hat` | **Cổng kiểm tra bắt buộc**, xem mục 5.1 |
| `d=0` | chỉ sửa đúng đường biên 1 pixel | |
| `d=1` | | |
| `d=2` | | **mức chính để kết luận** — khớp ngưỡng dung sai của BF-Score |
| `d=4` | | |
| `d=8` | | mức nới rộng, cho thấy đường cong bão hoà ở đâu |

### 2.2. Oracle nghịch đảo — chạy cùng lúc, gần như miễn phí

```
y_interior_d = np.where(band_d, y_hat, y)      # sửa hoàn hảo NGOÀI dải, giữ nguyên trong dải
```

Cho biết bao nhiêu điểm mIoU đang bị khoá trong **ruột vùng** — phần mà loss biên theo thiết kế không chạm tới. Hai con số `oracle_boundary(d)` và `oracle_interior(d)` cộng lại phân tách toàn bộ khoảng cách từ baseline tới mIoU = 1.0 thành hai phần, và tỉ lệ giữa chúng là câu trả lời trực tiếp cho lập luận "biên chỉ chiếm phần nhỏ".

### 2.3. Hai chỉ số chẩn đoán phải xuất kèm

Cộng dồn trên toàn tập val (đếm pixel, **không** trung bình theo ảnh):

```
P_band(d)  = |band_d ∩ valid| / |valid|                     # dải chiếm bao nhiêu % ảnh
E_band(d)  = |{sai ∩ band_d}| / |{sai}|                     # bao nhiêu % LỖI nằm trong dải
```
với `sai = (y_hat != y) & (y != 255)`.

Đây là tử số và mẫu số của toàn bộ câu chuyện: nếu `E_band(2) = 45%` trong khi `P_band(2) = 30%`, lỗi có tập trung ở biên nhưng chỉ hơi tập trung — và trần oracle sẽ phản ánh đúng điều đó.

---

## 3. Ràng buộc tính đúng đắn — đọc kỹ, đây là chỗ dễ hỏng nhất

### 3.1. Confusion matrix toàn cục, KHÔNG trung bình IoU theo ảnh

mIoU phải tính bằng cách cộng dồn một confusion matrix 9×9 trên **cả 384 ảnh**, rồi mới tính IoU từng lớp và lấy trung bình. Nếu tính IoU riêng từng ảnh rồi trung bình, con số sẽ **không so được** với 0.6551 và toàn bộ thí nghiệm vô nghĩa.

**Bắt buộc dùng lại `src/utils/metrics.py` (`SegmentationMetrics`) y nguyên** — cùng class đã sinh ra mọi con số trong Bảng 1. Không tự cài lại mIoU.

### 3.2. Đường suy luận phải giống hệt eval hiện có

Ảnh, tiền xử lý, resize, normalize, thứ tự val, không TTA — **copy nguyên đường load + inference từ `Tools/eval_boundary_metrics.py`**, không viết lại pipeline. Mọi khác biệt nhỏ ở đây sẽ làm dòng `none` không tái lập được 0.6551, và khi đó không biết lỗi nằm ở đâu.

Val split: đúng file `val_clean_*.txt` mà Run 1 đã dùng (384 ảnh). Không dùng `val_2000_fixed.txt`.

### 3.3. Cache prediction — làm một lần, dùng cho mọi mức d

Suy luận 384 ảnh trên CPU tốn ~25 phút (đo thực tế: 3.8 s/ảnh khi Kaggle cấp P100 sm_60 không tương thích build PyTorch). Chạy lại cho 6 mức d là lãng phí 2h.

```
Bước 1: chạy inference MỘT LẦN, lưu argmax ra đĩa:
        cache/run1_baseline/preds/<image_id>.npy   (uint8, (H,W))
Bước 2: mọi biến thể oracle đọc từ cache này.
```
`distance_transform_edt` cũng chỉ chạy **một lần mỗi ảnh** (một bản đồ `dist`), rồi mọi ngưỡng d chỉ là phép so sánh — không gọi EDT lại cho mỗi d.

Nếu Kaggle cấp **T4** thì chạy inference trên GPU cho nhanh; nếu cấp **P100 (sm_60)** thì tự chuyển CPU, đúng như `eval_boundary_metrics.py` đang làm.

### 3.4. `ignore_index = 255`

Trên dataset này là **no-op** (đã xác nhận 3 lần). Vẫn giữ mask hợp lệ trong mọi phép đếm để code tổng quát, nhưng không cần xử lý đặc biệt.

---

## 4. Output

### 4.1. `oracle_results.json`

```json
{
  "checkpoint": "work_dirs/phase1/run1_baseline/best_miou.pth",
  "checkpoint_iter": 40000,
  "val_split": "dataset/val_clean_384.txt",
  "n_images": 384,
  "class_names": ["Background","Bareland","Rangeland","Developed","Road","Tree","Water","Agriculture","Building"],
  "variants": {
    "none":              {"miou9": 0.6551, "miou8": 0.6169, "per_class_iou": {"Background": 0.961, ...}},
    "oracle_boundary_d0":{"miou9": ..., "miou8": ..., "per_class_iou": {...},
                          "delta_miou9": ..., "delta_miou8": ...,
                          "P_band": ..., "E_band": ...},
    "oracle_boundary_d1": {...}, "oracle_boundary_d2": {...},
    "oracle_boundary_d4": {...}, "oracle_boundary_d8": {...},
    "oracle_interior_d1": {...}, "oracle_interior_d2": {...},
    "oracle_interior_d4": {...}
  }
}
```

**Mọi output per-class BẮT BUỘC kèm tên lớp**, dạng dict, không bao giờ là mảng thuần. Lý do cụ thể: `run1_baseline_summary_boundary_filled.txt` từng trả mảng thuần và đã gây **hoán đổi Water ↔ Agriculture** trong một bảng tổng hợp — rơi đúng vào hàng baseline, hàng mà mọi Δ khác đo dựa vào.

### 4.2. `oracle_table.md`

Bảng 1 — đường cong trần:

| Biến thể | mIoU-9 | Δ (điểm) | mIoU-8 | Δ-8 | P_band | E_band |
|---|---|---|---|---|---|---|
| none (baseline) | 0.6551 | — | 0.6169 | — | — | — |
| oracle boundary d=0 | | | | | | |
| … d=1, 2, 4, 8 | | | | | | |
| oracle interior d=2 | | | | | | |

Bảng 2 — mức tăng oracle **per-class** tại d=2, sắp xếp giảm dần. Đây là bảng trả lời câu hỏi "lớp nào có thể được cứu bởi loss biên".

### 4.3. `oracle_curve.png`

Trục x = d, trục y = Δ mIoU-9 (điểm). Hai đường: boundary-oracle và interior-oracle. Hình này đi thẳng vào bài báo nếu kết quả có ý nghĩa.

---

## 5. Cổng kiểm tra — chạy trước khi tin bất kỳ con số nào

**5.1. Cổng đồng nhất (quan trọng nhất).** Biến thể `none` phải tái lập **chính xác** mIoU-9 của Run 1:
```python
assert abs(miou9_none - 0.6551) < 1e-3
```
Lệch quá ngưỡng này ⇒ pipeline suy luận hoặc cách tính mIoU đã khác eval gốc. **DỪNG, không chạy tiếp** — mọi số oracle sẽ vô giá trị.

**5.2. Đơn điệu.** `Δ(d=0) ≤ Δ(d=1) ≤ Δ(d=2) ≤ Δ(d=4) ≤ Δ(d=8)`. Dải rộng hơn chứa dải hẹp hơn nên mức tăng phải không giảm. Vi phạm ⇒ lỗi cài đặt band.

**5.3. Bù trừ.** `oracle_boundary_d + oracle_interior_d` phải cho mIoU = **1.0** (mọi pixel đều được sửa). Kiểm tra trực tiếp bằng cách chạy thêm một biến thể `y_all = y`. Đây là kiểm tra số học rẻ, bắt được lỗi định nghĩa band ngược dấu.

**5.4. Trực quan.** Xuất 3 ảnh overlay (`band_d2` chồng lên GT) để mắt xác nhận dải nằm đúng chỗ biên, không lệch, không rỗng.

**5.5. `P_band` hợp lý.** Biên là cấu trúc 1-D chiếm `r_edge ≈ 0.083`. Dải d=2 (rộng ~5 pixel) không thể vượt `5 × 0.083 ≈ 0.41`, và do chồng lấn sẽ thấp hơn. Nếu `P_band(2) > 0.45` hoặc `< 0.10` ⇒ nghi ngờ cài đặt.

---

## 6. Quy tắc đọc kết quả — CHỐT TRƯỚC KHI CÓ SỐ

Gọi **G₂ = Δ mIoU-9 tại oracle boundary d=2**, tính bằng **điểm** (1 điểm = 0.01).

| G₂ | Diễn giải | Quyết định |
|---|---|---|
| **≥ 4.0đ** | Headroom rộng. Mục tiêu +0.4–0.8đ chỉ là chiếm 10–20% trần — hoàn toàn khả thi | Giữ mIoU làm claim phụ có trọng lượng. Chạy Run 4, và nếu đạt ≥ +0.4đ thì chi 4 run đa-seed để xác nhận |
| **1.5 – 4.0đ** | Headroom vừa. +0.4–0.8đ tương đương chiếm 10–50% một trần hoàn hảo | Chỉ được claim mIoU **sau khi** áp giao thức giảm nhiễu (mục 7) và có ≥ 2 seed. Không claim từ n=1 |
| **< 1.5đ** | Trần quá thấp. Không loss biên nào chạm tới +0.4đ một cách đáng tin | **Bỏ hẳn nhánh claim mIoU.** Tiết kiệm 20–35h GPU. Bài phát biểu trên trục chỉ số biên đã phân rã, nơi hiệu ứng đã vượt nhiễu 2.5σ. Đưa chính con số G₂ vào Discussion như **lời giải thích định lượng** cho việc vì sao mIoU không phải trục đúng — đây là một đóng góp, không phải một thất bại |

### 6.1. Phân tách biên / ruột vùng cho từng cấu hình — phần có giá trị nhất của việc chạy nhiều checkpoint

Ký hiệu, với mỗi cấu hình `c` và d = 2:

```
M(c)      = mIoU cua c, khong sua gi
O_b(c)    = mIoU sau khi sua hoan hao TRONG dai bien
Gap(c)    = O_b(c) - M(c)        <- phan mIoU dang mat vi loi TRONG dai bien
```

Vì `O_b(c)` **giữ nguyên dự đoán của `c` ở ngoài dải**, nó chính là thước đo **chất lượng ruột vùng** của cấu hình đó. Vậy mỗi cấu hình được tách thành hai con số độc lập: chất lượng ruột (`O_b`) và lỗi biên (`Gap`). Đây là phân tách mà nghiên cứu hiện chưa có, và nó trả lời được ba câu hỏi đang treo:

**(a) Loss biên có thật sự dọn lỗi trong dải biên không — đo bằng mIoU?**
`Gap(BCE λ=0.4) < Gap(baseline)` ⇒ có. Nếu ngược lại, BCE cải thiện hình học biên (precision, `pred→gt`) mà **không** chuyển hoá thành pixel đúng — một phát hiện đáng viết, và giải thích luôn vì sao mIoU đứng yên.

**(b) ⭐ Giả thuyết xung đột gradient có đúng không — phép kiểm định 0 GPU cho kế hoạch projection head.**

Chẩn đoán hiện tại (`huong-mo-rong-projection-head-va-contrastive.md` mục 1): affinity chảy thẳng vào Fused Feature mà seg head tiêu thụ, làm méo biểu diễn ⇒ mất mIoU. Nếu đúng, thiệt hại phải nằm ở **ruột vùng**.

| Quan sát | Kết luận | Hành động |
|---|---|---|
| `O_b(Static) < O_b(BCE λ=0.4)` rõ rệt | Affinity **làm hỏng ruột vùng** ⇒ xác nhận chẩn đoán | Projection head (Giai đoạn A) là lời giải đúng, chạy nó |
| `O_b(Static) ≈ O_b(BCE λ=0.4)` | Mất mát mIoU của Static nằm **trong dải biên**, không phải ở ruột ⇒ **chẩn đoán sai** | **Huỷ Giai đoạn A.** Projection head không cứu được thứ nó được thiết kế để cứu. Tiết kiệm 3.5h GPU + thời gian code |

**(c) Hai chẩn đoán cấu trúc đang cạnh tranh — oracle phân biệt được chúng, miễn phí.**

Water sụt 2.88đ ở Static là dữ kiện mà cả hai chẩn đoán đều muốn nhận. Chúng dự đoán **khác nhau**:

| Chẩn đoán | Dự đoán về Water ở Static | Ứng viên GPU tương ứng |
|---|---|---|
| Xung đột gradient (mục 3.6a của tổng kết) | mức sụt nằm ở `O_b` — **ruột vùng** hỏng | **(B)** projection head |
| Teo vùng lấy mẫu ở stride-4 (mục 3.6b) | mức sụt nằm ở `Gap` — **biên** mất supervision | **(C)** sampling dilation + max-pool |

⇒ Xuất **bảng `O_b` và `Gap` per-class** cho baseline / BCE λ=0.4 / Static. Cột Water của bảng đó có thể **quyết định luôn thứ tự ưu tiên giữa (B) và (C)** — câu hỏi hiện đang phải chờ Huan quyết mà không có dữ liệu. Nếu tín hiệu rơi rõ về một phía, ứng viên còn lại lùi xuống sau.

Ghi rõ đây là **kiểm chứng hậu nghiệm** cho giả thuyết sinh ra từ chính dữ liệu này, và nó **không thay thế** việc chạy thí nghiệm xác nhận — nó chỉ quyết định chạy cái nào trước.

**Điều kiện kích hoạt checkpoint thứ tư (Run 3b `final_iter40000.pth`, ~25 phút):** chỉ chạy nếu cột Water của bảng này ra **mơ hồ** — nghĩa là mức sụt của Water chia gần đều giữa `O_b` và `Gap`, hoặc phần chênh lệch giữa hai phía nhỏ hơn 0.68đ (đúng biên độ Water dao động giữa vòng 9 và vòng 10 của Run 3, tức không phân biệt được với lệch mốc checkpoint). Khi đó Run 3b là điểm affinity duy nhất ở đúng iter 40000 và trở thành trọng tài. Nếu tín hiệu rõ về một phía, **không chạy** — quyết định đã đủ căn cứ.

**Đọc thêm, không phụ thuộc G₂:**

- **Tỉ lệ boundary/interior tại d=2.** Nếu interior-oracle lớn hơn boundary-oracle nhiều lần, đó là bằng chứng số cho luận điểm "91.7% pixel là ruột vùng" — và là lý do cấu trúc để ưu tiên `L_SemCon` thay vì tiếp tục dò hệ số biên.
- **Bảng oracle per-class.** Lớp nào có mức tăng oracle cao là lớp mà loss biên *có thể* giúp; lớp nào thấp thì không cách nào giúp được. Nếu **Bareland ở nhóm thấp**, giả thuyết gốc của đề cương bị bác bỏ bằng **bằng chứng khẳng định**, mạnh hơn hẳn lập luận hiện tại ("chưa bao giờ vượt baseline"). Nếu Bareland ở nhóm **cao** mà thực nghiệm vẫn không cải thiện được, thì vấn đề nằm ở **cơ chế loss**, không ở trần bài toán — một kết luận khác hẳn, và cũng đáng viết.

---

## 7. Giao thức giảm nhiễu — tuyên bố tại đây, trước khi có bất kỳ số mIoU mới nào

Áp dụng cho Run 4 và mọi run sau. Ghi vào tài liệu **trước** khi chạy, để không bị coi là chọn lọc hậu nghiệm:

1. So sánh **chỉ tại `final@40000`**, không dùng checkpoint best đơn lẻ.
2. Kèm **mean ± std (ddof=1) của 4 vòng validate cuối** cho mọi chỉ số.
3. Báo cáo thêm **mIoU loại Bareland**, song song mIoU-9 và mIoU-8. Điều kiện hợp lệ: loại **cùng một lớp ở mọi hàng kể cả baseline**; lý do khách quan độc lập với kết quả — Bareland gần như **không được học ở mọi cấu hình** (IoU dao động 0.06–0.36, σ = 0.0735, lớn hơn mọi Δ giữa các cấu hình) nên nó hoạt động như một bộ phát nhiễu chung; và **vẫn báo cáo đầy đủ** mIoU-9/mIoU-8 bên cạnh.
4. Sàn nhiễu công bố: σ(mIoU-9) ≈ **0.64 điểm** giữa các vòng validate cuối. Mọi Δ nhỏ hơn mức này được ghi là **không phân biệt được**, không được diễn giải theo dấu.

---

## 8. Checklist bàn giao

- [ ] Định vị checkpoint Run 1, xác nhận iter = 40000; nếu mơ hồ → hỏi Huan
- [ ] `Tools/oracle_boundary_ceiling.py` — dùng lại `extract_edge_gt`, `SegmentationMetrics`, và đường inference của `eval_boundary_metrics.py`
- [ ] Cache prediction 1 lần; EDT 1 lần/ảnh
- [ ] Chạy đủ 6 biến thể boundary + 3 biến thể interior + `none` + `all`
- [ ] **Cổng 5.1 PASS** (`none` ≡ 0.6551) trước khi báo cáo bất kỳ số nào
- [ ] Cổng 5.2 / 5.3 / 5.5 PASS; xuất 3 overlay của 5.4
- [ ] `oracle_results.json` (per-class **kèm tên lớp**), `oracle_table.md`, `oracle_curve.png`
- [ ] Lặp lại cho **3 checkpoint bắt buộc** (baseline / BCE λ=0.4 / Static) theo mục 1.1, cache `dist` dùng chung, ghi kết quả sau mỗi checkpoint
- [ ] Bảng `O_b` và `Gap` — tổng thể **và per-class** — cho 3 cấu hình đó (mục 6.1); cột `iter` bắt buộc có
- [ ] **KHÔNG** chạy Run 3b và BCE λ=0.2 ở lượt này — chỉ chạy Run 3b nếu điều kiện kích hoạt ở mục 6.1(c) thoả
- [ ] Ghi kết quả vào một tài liệu mới trong project, áp dụng bảng quyết định mục 6 **nguyên văn**, không sửa ngưỡng sau khi thấy số

---

## 9. KHÔNG được làm

- Không sửa `extract_edge_gt`, `metrics.py`, `eval_boundary_metrics.py` — chỉ import.
- Không đổi val split, không bật TTA, không resize khác đường eval gốc.
- Không trung bình IoU theo từng ảnh.
- Không đổi ngưỡng ở bảng mục 6 sau khi đã thấy kết quả. Nếu ngưỡng có vẻ sai, ghi nhận xét bên cạnh và **giữ nguyên verdict** — đúng bài học đã rút ra từ bảng quyết định của Run 3b.
- Không suy ra kết luận về cơ chế loss từ thí nghiệm này. Oracle đo **trần của bài toán**, không đo bất cứ loss nào.
