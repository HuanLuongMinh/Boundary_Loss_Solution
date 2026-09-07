# Spec bàn giao Claude Code — Đo `r_edge` tại stride-4 per-class (cổng C1 của Run 6)

**Ngày:** 7/9/2026
**Repo:** `Boundary_Loss_Solution`
**Chi phí:** ~30 phút **CPU**. **0 GPU, 0 quota.**
**Ưu tiên:** cao nhất trong nhóm 0-GPU — **chặn Run 6** và có thể **huỷ** nó.
**Quan hệ:** đây là bản đầy đủ của mục 1 trong `spec-run6-affinity-sampling-fix-ban-giao-claude-code.md`, và là **việc #4** treo từ 6/9.

---

## 0. ⭐ KHÔNG CẦN CHECKPOINT NÀO CẢ

Đây là câu hỏi quan trọng nhất phải làm rõ trước khi bắt tay:

> **`r_edge` là thống kê của NHÃN và của ĐƯỜNG TÍNH DOWNSAMPLE, không phải của mô hình.**

Phép đo này:

- ❌ **không** load checkpoint nào
- ❌ **không** chạy inference
- ❌ **không** dùng ảnh dự đoán
- ✅ chỉ đọc **nhãn ground-truth**, chạy `extract_edge_gt` và các cách downsample khác nhau, rồi đếm

Nó trả lời: *khi hạ nhãn từ full-res xuống stride-4 cho `L_Affinity`, tỉ lệ pixel biên còn lại là bao nhiêu, và lớp nào mất nhiều nhất.* Mô hình không tham gia vào câu hỏi đó.

### 0.1. Chạy trên tập nào — TRAIN, không phải val

**Tập chính: TRAIN crops.** Lý do: việc lấy mẫu affinity xảy ra **trong lúc huấn luyện**, trên train crops. Đó mới là tập mà tín hiệu bị teo hay không bị teo.

Dùng **đúng giao thức lấy mẫu của `EDGE_STATS`** đang có, để số so được trực tiếp với `r_edge` = 0.083336 đã đo:

- 800 crop, kích thước **512 × 512**
- bounded auto-estimate: 100 batch/rank × 2 rank
- `seed = 19`
- `connectivity = 4`, `ignore_index = 255`, `dilation_radius = 0`

**Tập phụ (tuỳ chọn, ~5 phút): 384 ảnh VAL**, chỉ để kiểm chéo với `P_band` @ d=0 = **0.0810** của oracle. Nếu `r_edge` full-res đo trên val ra ≈ 0.081 thì hai pipeline hoàn toàn độc lập đang đồng ý với nhau — một xác nhận rẻ và đáng có.

---

## 1. Ba biến thể phải đo

| Biến thể | Cách dựng `edge_gt` ở stride-4 | Đây là gì |
|---|---|---|
| **V0 — hiện tại** | nhãn → **nearest**-downsample ×4 → `extract_edge_gt` | Đường đang chạy ở Run 3 / 3b / 4 |
| **V1 — max-pool** | `extract_edge_gt` ở **full-res** → **max-pool** ×4 | Phương án "chính xác hơn" đã ghi sẵn ở `workflow.md` mục 3.2 |
| **V2 — max-pool + dilation** | như V1, rồi `AFFINITY_SAMPLING_DILATION = 1` | Dải biên dày thay vì đường 1 pixel |

Khác biệt cốt lõi giữa V0 và V1: V0 **hạ nhãn trước rồi mới tìm biên** (biên lệch 1 pixel khi hạ mẫu là biến mất hoàn toàn); V1 **tìm biên ở độ phân giải gốc rồi mới hạ**, nên biên của đối tượng mảnh được giữ.

---

## 2. Phải xuất ra những gì

Cho **mỗi** biến thể (V0/V1/V2):

| Đại lượng | Ghi chú |
|---|---|
| `r_edge_stride4` toàn cục | = N_edge / N_valid, tính ở stride-4 |
| **`r_edge_stride4` per-class** (9 lớp, **kèm tên lớp**) | Đại lượng chính của cổng |
| **`n_affinity_centers` per-class** | Số pixel trung tâm hợp lệ cho affinity — **đây mới là số hạng tử thật của gradient**, quan trọng hơn cả tỉ lệ |
| `n_valid_px_stride4` | mẫu số |
| Tỉ lệ so V0 | `n_affinity_centers(Vx) / n_affinity_centers(V0)` per-class |

Định nghĩa `r_edge` per-class: pixel trung tâm hợp lệ **thuộc lớp c** (theo nhãn ở stride-4) và có ít nhất một hàng xóm 4-liên thông hợp lệ mang nhãn khác — chia cho tổng pixel hợp lệ thuộc lớp c. Ghi rõ định nghĩa đã dùng vào output; nếu chọn định nghĩa khác (vd. cặp được tính cho cả hai lớp hai bên biên) thì **ghi rõ và giữ nhất quán** giữa ba biến thể.

**Output:**
- `redge_stride4.json` — toàn bộ số, kèm khối `meta` (giao thức lấy mẫu, seed, số crop, định nghĩa per-class, git commit)
- `redge_stride4.md` — bảng đọc được: 9 lớp × 3 biến thể × 2 đại lượng, để dán thẳng vào tài liệu project

---

## 3. Cổng kiểm — PASS trước khi tin bất kỳ số stride-4 nào

**3.1. Tái lập `r_edge` FULL-RES đã biết.** Trước khi đo ở stride-4, script phải tính `r_edge` ở full-res với cùng giao thức và ra:

```
r_edge (full-res, train, 800 crop, seed 19)  =  0.0833 ± 0.0006
pos_weight = (1 - r_edge)/r_edge             =  11.00  ± 0.08
```

Bốn lần đo đã có: 0.083062 / 0.084176 / 0.083323 / 0.083336. Ra ngoài khoảng ⇒ giao thức lấy mẫu đã lệch, **DỪNG**, không đo tiếp.

**3.2. V0 phải là đúng đường đang chạy.** Xác minh V0 dùng **cùng hàm** mà `src/losses/affinity.py` gọi trong lúc train, không phải một bản cài lại. Nếu phải viết lại, so số pixel trung tâm với một batch thật lấy từ dataloader.

**3.3. Đơn điệu theo cấu trúc.** `n_affinity_centers(V2) ≥ n_affinity_centers(V1) ≥ ...` — V2 nới dải nên không được ít hơn V1. Vi phạm ⇒ lỗi cài đặt.

**3.4. Kiểm chéo val (nếu chạy tập phụ):** `r_edge` full-res trên 384 ảnh val ≈ **0.081** (`P_band` @d=0 của oracle). Lệch > 10% ⇒ hai pipeline đang dùng định nghĩa biên khác nhau, cần truy trước khi đọc bất cứ kết luận nào.

---

## 4. Dự đoán định lượng — đăng ký TRƯỚC khi có số

Biên là cấu trúc **1 chiều**: hạ mẫu ×4 làm **diện tích** giảm 16× nhưng **chiều dài biên** chỉ giảm ~4×. Nếu biên được bảo toàn thì

```
r_edge@stride4  ≈  4 × 0.0833  ≈  0.33
```

Lớp nào tụt sâu dưới 0.33 ở V0 là lớp đang **mất supervision**. Dự đoán riêng của chẩn đoán 3.6b: **Water (dạng sông/suối) và Road** tụt sâu nhất, vì chúng mảnh và dài.

---

## 5. Bảng quyết định cổng — CHỐT TRƯỚC KHI CÓ SỐ

Đọc trên **V0**, hai lớp **Water và Road**:

| Nhánh | Điều kiện trên V0 | Kết luận | Biến thể đem train |
|---|---|---|---|
| **G1** | Water **và** Road tụt dưới **0.25** (< 75% mức bảo toàn) | Chẩn đoán teo tín hiệu **được xác nhận trực tiếp** | Biến thể **đầu tiên** trong {V1, V2} đưa Water và Road về ≥ **0.30**. V1 đủ ⇒ **chạy V1** (một biến, sạch) |
| **G2** | Water và Road ≥ **0.30** | Chẩn đoán teo tín hiệu **bị bác bỏ**. Mức sụt Water 2.75đ có nguyên nhân khác | **HUỶ Run 6**, tiết kiệm 3.5h GPU. Báo cáo kết quả âm — nó loại một trong hai chẩn đoán cấu trúc và làm ứng viên (B) mạnh lên tương đối |
| **G3** | rơi vào 0.25–0.30, hoặc V1 lẫn V2 đều không đưa được lên ≥ 0.30 | Không kết luận được từ cổng | Chạy **V2**, và ghi rõ đây là can thiệp **hai thành phần** (hệ quả: `spec-run6-...` mục 5.3) |

**Vì sao để dữ liệu chọn biến thể:** đề xuất gốc (`huong-thuc-nghiem-tiep-theo-sau-run3` mục 5) gộp max-pool **và** dilation=1 vào một run và gọi đó là "cô lập đúng 1 biến so với Run 3". Thực tế đó là **hai** thay đổi cơ chế, vi phạm nguyên tắc *mỗi run đổi đúng một biến*. Cổng này sửa việc đó bằng cách chọn can thiệp **tối thiểu đủ dùng**.

---

## 6. Cổng này KHÔNG tự mình quyết Run 6

Run 6 có **hai** cổng CPU độc lập, kiểm cùng câu hỏi từ hai phía. Quyết định cuối theo **quy tắc gộp bốn ô** ở `oracle-d-ket-qua-baseline-va-danh-gia.md` mục 7:

| C1 — `r_edge` (tài liệu này) | C5b — oracle trên Static + BCE | Quyết định |
|---|---|---|
| G1 (teo) | mức sụt Water rơi vào `Gap` | **Chạy Run 6** — hai bằng chứng độc lập cùng chiều |
| G1 (teo) | rơi vào `O_b` | **Vẫn chạy Run 6**, ghi rõ oracle không ủng hộ; **giữ (B) sống** |
| G2 (không teo) | `Gap` | **Không chạy Run 6 dạng hiện tại** — cần chẩn đoán lại |
| G2 (không teo) | `O_b` | **HUỶ Run 6**, chuyển 3.5h sang **(B) projection head** |

---

## 7. Sản phẩm phụ — ghi lại kể cả khi Run 6 bị huỷ

Bảng `r_edge@stride4` per-class là **một bảng của bài báo** bất kể run có chạy hay không. Nó định lượng một hạn chế thiết kế mà rất ít bài về loss biên nói tới: *một số hạng contrastive đặt ở stride-4 chỉ nhìn thấy bao nhiêu phần trăm biên gốc, và lớp nào bị thiệt.* Ghi vào project ngay khi có, đừng chờ Run 6.

---

## 8. KHÔNG ĐƯỢC ĐỔI

- Giao thức lấy mẫu `EDGE_STATS` (800 crop, 512×512, seed 19, bounded 100 batch/rank × 2 rank)
- `connectivity = 4`, `ignore_index = 255`, `dilation_radius = 0` cho định nghĩa biên gốc
- `src/losses/affinity.py` và `boundary_bce.py` — **chỉ đọc, không sửa**. Việc sửa là Run 6, không phải cổng này
- Không bật GPU cho notebook này

---

## 9. Checklist bàn giao

- [ ] Xác nhận **không load checkpoint nào** — script chỉ đọc nhãn
- [ ] Cổng 3.1: tái lập `r_edge` full-res = 0.0833 ± 0.0006 và `pos_weight` = 11.00 ± 0.08 → **PASS trước khi đo tiếp**
- [ ] Cổng 3.2: V0 dùng đúng hàm mà training đang gọi
- [ ] Đo V0 / V1 / V2 trên **train crops**; ghi `r_edge@stride4` **và** `n_affinity_centers`, cả hai **per-class kèm tên lớp**
- [ ] Cổng 3.3 (đơn điệu) PASS
- [ ] Tuỳ chọn: đo full-res trên 384 ảnh val, kiểm chéo với 0.0810 của oracle (cổng 3.4)
- [ ] Xuất `redge_stride4.json` + `redge_stride4.md`
- [ ] Áp dụng bảng mục 5 **nguyên văn** → công bố **G1 / G2 / G3**
- [ ] Báo kết quả để hợp với oracle theo quy tắc gộp mục 6 — **chưa phóng Run 6 khi chưa có cả hai**
