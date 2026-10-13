# Kế hoạch K2 — Kiểm lỗ giả: lỗi mô hình hay lỗi nhãn? Và luật chọn hướng giải pháp

**Ngày:** 13/10/2026 — viết **trước** khi có số oracle và trước khi xem bất kỳ crop nào. Ngưỡng ở §4 và §5 là tiền đăng ký.
**Chạy song song với:** `S2-spec-phan-D-quy-dao-xung-dot-gradient.md` (GPU). K2 chỉ dùng CPU + công kiểm bằng mắt.
**Căn cứ số liệu:** `ket-qua-buoc1-xung-dot-gradient-va-lo-gia-danh-gia.md` §3; `holes_<ckpt>.csv` (Bước 1, Phần B).

---

## 0. Vì sao cần K2

**Đã biết (Bước 1, A_min = 64):**
- Baseline có 31.939 lỗ trên 384 ảnh; **27%** là "lỗ giả theo nhãn" (≥ 50% pixel của lỗ có GT = lớp c). Static s19@36k: 33.019 lỗ, 26,8% giả. Lỗ giả không do loss biên sinh ra (bảng 5.4, dòng 4).
- Phân bố tỉ lệ GT = c trong lỗ (Baseline): < 0,1: 47,7% · 0,1–0,5: 25,4% · 0,5–0,95: 7,4% · ≥ 0,95: **19,6%**.
- Lỗ giả rất nhỏ: trung vị **149 px**, p90 ≈ 700 px trên ảnh 1024×1024.
- Lấp mù mọi lỗ ≤ 64 px (B2): precision **+0,0130**, recall **−0,0163**, `gt→pred` +0,69 px (Static s19).

**Chưa biết — hai câu K2 trả lời:**
- **K2-Q1 (đòn bẩy):** Trong kết quả B2, phần tăng precision và phần mất recall đến từ lỗ **giả** hay lỗ **thật**? Nếu chỉ xoá lỗ giả, precision tăng bao nhiêu, recall có giữ được không?
- **K2-Q2 (bản chất):** Một "lỗ giả theo nhãn" là **mô hình sai**, hay **nhãn thô** (ngoài thực địa có vật khác lớp — cây đơn lẻ, mái nhà nhỏ, vũng nước — mà người gán không vẽ, tức lỗi loại 3)? Thước đo duy nhất của Bước 1 là chính nhãn, nên không tự trả lời được câu này.

Chỉ khi lỗ giả **vừa là lỗi mô hình thật, vừa là đòn bẩy precision không mất recall**, mới đáng đầu tư giải pháp nhắm vào lỗ (vd. loss topo — Hu et al., *Topology-Preserving Deep Image Segmentation*, NeurIPS 2019; Stucki et al., *Topologically Faithful Image Segmentation via Induced Matching of Persistence Barcodes*, ICML 2023).

---

## 1. Bước K2-a — Oracle lấp lỗ có chọn lọc (Claude Code, Kaggle CPU, ~30–60 phút)

### 1.1. Thiết lập

- **Checkpoint:** Static s19@36k, Static s86@36k (bắt buộc — hai seed); BCE λ0,4@40k, Baseline@40k (mô tả).
- **Mask:** PNG đã dump của Bước 1 (`/tmp/dump_repo/dump/...`, cùng đường dẫn trong `spurious_holes/run_log.txt`).
- **Định nghĩa lỗ:** giống `Tools/spurious_holes.py` (thành phần dự đoán 8-liên thông; lỗ = `binary_fill_holes(M_c) − M_c`, tách 4-liên thông). Gắn mỗi lỗ với `frac_gt_c` — **không** áp `A_min` (giống B2: lấp mọi lỗ có diện tích ≤ `A_fill`).
- **`A_fill` ∈ {64, 256}.**

### 1.2. Bốn biến thể lấp (mỗi `A_fill`)

| Mã | Lấp lỗ nào | Vai trò |
|---|---|---|
| F-all | mọi lỗ | Tái lập B2 — phải khớp `fill_test.md` (cổng) |
| F-giả | `frac_gt_c` ≥ 0,5 | **Oracle chính** — trần của việc chỉ xoá lỗ giả |
| F-giả95 | `frac_gt_c` ≥ 0,95 | Độ nhạy — chỉ lỗ mà nhãn hoàn toàn đặc |
| F-thật | `frac_gt_c` < 0,5 | Phần bù — đo cái giá của việc xoá vật nhỏ thật |

Các biến thể dùng GT để chọn lỗ ⇒ là **trần (oracle)**, không phải phương pháp.

### 1.3. Đo

Dùng chính `Tools/per_image_dump.compute_per_image_stats` + `Tools/bootstrap_boundary_ci` (quy ước micro), như B2: mIoU-9, mIoU-8, BF precision, BF recall (θ = 2 px), BF-score, ASD `pred→gt`, `gt→pred`. Δ = sau − trước; bootstrap ghép cặp per-image 10.000 lần, rng 19.

### 1.4. Cổng

- **K-a1:** F-all tái lập `fill_test.md` (Static s19, `A_fill` 64: Δprecision 0,01304, Δrecall −0,01633), sai số < 1e-4.
- **K-a2:** Cộng tính kiểm tra: tập lỗ của F-giả và F-thật rời nhau, hợp lại bằng F-all (đếm lỗ khớp tuyệt đối).

### 1.5. Output

`hole_oracle/oracle_fill.md`, `oracle_fill.csv`, `gates_oracle.json`.

---

## 2. Bước K2-b — Kiểm bằng mắt (anh Huan, ~1,5–2 giờ)

### 2.1. Chọn mẫu (Claude Code sinh, chốt trước khi ai xem crop)

- **Quần thể:** lỗ giả (`is_spurious = 1`, `A_min = 64`) của **Static s19@36k**: 8.836 lỗ.
- **Phân tầng:** 6 nhóm lớp × 3 nhóm kích thước = 18 ô.
  - Lớp: Rangeland (3.458), Developed (2.767), Tree (1.081), Building (938), Agriculture (320), Khác = Road + Bareland + Water (272).
  - Kích thước: 64–149 px · 150–699 px · ≥ 700 px.
- **Cỡ mẫu:** 7 lỗ/ô (ô ít hơn 7 thì lấy hết) ⇒ ~126 lỗ. Rút bằng `np.random.default_rng(19)`.
- **Lặp ẩn:** thêm 20 lỗ trùng (rút ngẫu nhiên từ chính mẫu), xáo vào để đo độ nhất quán của người chấm. Tổng ~146 mục, **thứ tự xáo ngẫu nhiên** (rng 19), đánh số `H001…`.
- Lưu `audit_sample.csv` (item_id, image_id, class, area, bbox, frac_gt_c, stratum, duplicate_of) **trước** khi sinh crop. Người chấm không xem file này cho tới khi chấm xong.

### 2.2. Crop (giảm thiên lệch)

Mỗi mục một PNG, 2 ô cạnh nhau, cửa sổ = bbox nở ×3 (tối thiểu 128×128 px full-res), phóng lên ~400 px:
1. RGB gốc;
2. RGB + viền đỏ quanh lỗ.

Tiêu đề chỉ ghi **mã mục và tên lớp c** (lớp bao quanh). **Không** hiện GT, mask dự đoán, model, `frac_gt_c` hay kích thước — người chấm chỉ trả lời câu hỏi về thực địa.

### 2.3. Câu hỏi và nhãn

> "Bên trong viền đỏ, **trên ảnh thực**, có vật hoặc lớp phủ **khác lớp c** chiếm phần lớn vùng không?"

| Nhãn | Nghĩa | Ví dụ |
|---|---|---|
| **A — mô hình sai** | Vùng trong viền đồng nhất với lớp c | Cỏ đồng nhất, ruộng đồng nhất; **bóng đổ trên nền cùng lớp vẫn là A** |
| **B — nhãn thô** | Có vật thật khác lớp mà GT không vẽ | Cây đơn lẻ, mái nhà nhỏ, vũng nước, mảng đất trống |
| **C — không phân định** | Ảnh không đủ để quyết | Quá mờ, quá nhỏ so với độ phân giải, mây/nhiễu |

Với B, ghi thêm "vật gì" (cây / nhà / nước / đất trống / đường / khác). Ghi vào `audit_sample_rating.csv` (item_id, nhan, vat, ghi_chu).

### 2.4. Ước lượng

- **p_B** = tỉ lệ B trên (A + B), bỏ C; **gia trọng theo kích thước quần thể từng ô** (§2.1). CI 95% bằng bootstrap phân tầng (rút lại trong từng ô, 10.000 lần, rng 19).
- **p_C** = tỉ lệ C (gia trọng).
- **Độ nhất quán:** Cohen's κ trên 20 cặp lặp ẩn.
- Báo cáo thêm p_B theo nhóm lớp và nhóm kích thước (mô tả).

---

## 3. Lịch

| Ngày | K2 | Song song (S2) |
|---|---|---|
| Thứ 4, 14/10 | Claude Code: K2-a (oracle) + sinh mẫu và crop K2-b | Spec D → cài callback → dry-run D1–D5 |
| Thứ 5, 15/10 | Anh Huan chấm ~146 mục | Run Phần D đầy đủ |
| Thứ 6, 16/10 | Tổng hợp K2-a + K2-b, áp §4 | Áp bảng 6.3 của S2 → **quyết định chung §5** |

---

## 4. Luật đọc K2 — CHỐT TRƯỚC

### 4.1. Cổng tin cậy của kiểm bằng mắt

- κ (lặp ẩn) < 0,6 **hoặc** p_C > 30% ⇒ kiểm bằng mắt **không đủ tin cậy**. Ghi nhận; nếu cần, thêm một người chấm thứ hai trên cùng mẫu trước khi dùng §4.3.

### 4.2. Đòn bẩy (K2-a), `A_fill` = 64, **cả hai seed Static**

**Đòn bẩy lỗ giả tồn tại** khi đồng thời:
1. Δprecision(F-giả): CI loại trừ 0 **và** Δ > 0,0014 (biên độ liên-seed precision của Static);
2. Δrecall(F-giả): cận dưới CI > −0,0094 (không mất quá phần cộng hưởng recall Static − BCE; [P1:L1 §3.3]).

Dự đoán ghi trước (để kiểm, không dùng quyết định): phần lớn mất recall của B2 nằm ở F-thật.

### 4.3. Bản chất (K2-b)

| Mã | Điều kiện | Đọc ra |
|---|---|---|
| **H-nhãn** | p_B ≥ 50% | Lỗ giả chủ yếu là **lỗi loại 3** (nhãn thô). Mô hình đang bị phạt vì đúng |
| **H-mô hình** | p_B < 30% (cận trên CI < 50%) | Lỗ giả chủ yếu là **lỗi mô hình thật** (loại 5 theo nghĩa rộng) |
| **H-hỗn hợp** | còn lại | Không đủ rõ để đầu tư giải pháp nhắm vào lỗ |

---

## 5. Quyết định hướng giải pháp — gộp Phần D (S2 §6.3) và K2 (§4)

| Phần D | K2 | Hướng giải pháp |
|---|---|---|
| D-G / D-RA / D-L | bất kỳ | **Tiếp Hướng B** theo nhánh của D (lặp quỹ đạo seed 86 trước khi viết spec phương pháp). Kết quả K2 thành phân tích phụ cho loại 2/3/5 |
| D-N | H-mô hình **và** đòn bẩy tồn tại | **Đề xuất đổi hướng** sang loss bảo toàn topo cho vùng nhỏ (Hu et al. 2019; Stucki et al. 2023). Đây là đổi trục bài 2 — **anh quyết**, sau đó mới viết spec với pilot 1 seed như tổng quan §10.3 |
| D-N | H-nhãn | **Không làm giải pháp cho lỗ.** Bài 2 thành bài **phân tích** (kịch bản S2 của bàn giao): (i) hai thành phần biên không xung đột mà quá yếu (2% và 9% chuẩn gradient region); (ii) đánh đổi precision/recall tái tạo được bằng thao tác topo trên vùng nhỏ; (iii) định lượng tỉ lệ "lỗi" thực chất là lỗi nhãn — hạn chế của đánh giá biên trên OpenEarthMap |
| D-N | H-hỗn hợp, hoặc đòn bẩy không tồn tại | Như dòng trên; lỗ giả chỉ là mô tả |
| D-dai-dang | bất kỳ | Dừng; kiểm khác biệt train/val trước |

**Không đổi ngưỡng ở §4 và bảng §5 sau khi thấy số.** Nếu thấy ngưỡng có vẻ sai: ghi nhận xét bên cạnh, giữ nguyên quyết định.

---

## 6. Checklist

- [ ] `Tools/hole_oracle_fill.py` (K2-a): 4 biến thể × 2 `A_fill` × 4 checkpoint; cổng K-a1, K-a2
- [ ] `Tools/hole_audit_sample.py` (K2-b): `audit_sample.csv` → crop `audit_crops/H001.png…`, file chấm trống `audit_sample_rating.csv`
- [ ] Anh Huan chấm; Claude tính p_B, p_C, κ theo §2.4
- [ ] Gộp với kết quả Phần D, áp §5, viết tài liệu kết quả vào Project
