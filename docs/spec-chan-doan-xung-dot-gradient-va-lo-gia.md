# Spec bàn giao Claude Code — Chẩn đoán xung đột gradient (Hướng B) + đếm lỗ giả (lỗi loại 5)

**Ngày:** 5/10/2026
**Repo:** `Boundary_Loss_Solution`
**Chi phí:** Phần A ~2h GPU (1×T4 đủ, chỉ forward + backward, KHÔNG train). Phần B: CPU, vài chục phút trên mask đã dump. Phần D (tuỳ chọn, có cổng): ~3.5h GPU.
**Không train lại, không sửa hành vi loss.** Chỉ đọc checkpoint đã có và mask đã dump.
**Bối cảnh:** `huong-b-xung-dot-bce-affinity-tac-dong-novelty-gap-paper.md`, `diem-chot-truoc-brainstorm-5-10-2026.md`.

---

## 0. Câu hỏi spec này trả lời

1. **(Phần A)** Trên Fused Feature (64 kênh, stride 4) — biểu diễn dùng chung — gradient của `L_BCE` và `L_aff` có **ngược hướng** nhau không? Ngược ở đâu: toàn cục (tham số dùng chung), theo khoảng cách tới biên, theo lớp, theo kích thước vật? Và **affinity có xung đột với chính `L_region`** không (ứng viên giải thích thiếu hụt ruột vùng K7)?
2. **(Phần B)** Lỗi loại 5 (vật đặc → vành khuyên) có tồn tại đo được không, có phổ biến không, và **thành phần loss nào** sinh ra nó?
3. **(Phần C)** Hai hiện tượng có cùng gốc không: lỗ giả có tập trung ở nơi gradient xung đột không?

Kết quả Phần A chọn nhánh phương pháp theo bảng quyết định mục 5 — **chốt trước khi có số**.

---

## 1. Checkpoint — dùng cái nào, vì sao

### 1.1. Kiểm kê (từ `doi-chieu-du-lieu-goc-va-hang-so-tham-chieu.md`, `run5-...`, `khe-danh-gia-run7-...`)

| Run | Checkpoint có | Iter | BoundaryHead đã huấn luyện? | Mask đã dump? |
|---|---|---|---|---|
| Baseline | `best_model.pth` | 40000 | ❌ | ✅ |
| BCE λ=0.4 | `best_model.pth` | 40000 | ✅ | ✅ |
| Static seed 19 | `best_model.pth` | **36000** (không có 40k) | ✅ | ✅ |
| Static seed 86 (Run 5) | `best_miou.pth` / `final_iter40000.pth` | 36000 / 40000 | ✅ | ✅ cả hai |
| Run 3b (α_aff 0.2) | 4 ckpt | 36000 / 40000 | ✅ | ✅ @40k |
| Run 4 (dynamic) | 4 ckpt | 32000 / 40000 | ✅ | ✅ @40k |
| Affinity-only s19, s86 (Run 7) | best = final | 40000 | ❌ (α_bce ≡ 0, head không nhận gradient) | ✅ |

### 1.2. Khuyến nghị cho Phần A (gradient)

| Bậc | Checkpoint | Vai trò | Cặp gradient đo |
|---|---|---|---|
| **A1 — bắt buộc** | **Static s19 @36000** (`best_model.pth`) | Đối tượng chính — cấu hình đang có đánh đổi precision | (BCE, Aff), (Region, Aff), (Region, BCE) |
| **A1 — bắt buộc** | **Static s86 @36000** (`best_miou.pth`) | Lặp lại độc lập, **cùng iter** với s19 — điều kiện bắt buộc của bảng quyết định | như trên |
| A2 — nên có | Static s86 @40000 (`final_iter40000.pth`) | Kiểm độ ổn định theo mốc (36k vs 40k cùng run) | như trên |
| A2 — nên có | **BCE λ=0.4 @40000** | **Đầu dò (probe):** affinity chưa từng được huấn luyện, nhưng `L_aff` vẫn tính được trên Fused Feature → đo xung đột **vốn có** tại điểm tối ưu của BCE, trước khi affinity "thương lượng" | (BCE, Aff-probe), (Region, Aff-probe), (Region, BCE) |
| A2 — nên có | Affinity-only s19 @40000, s86 @40000 | Affinity không có BCE — chỉ đo (Region, Aff). So với Static để biết BCE có làm đổi quan hệ Region–Aff không (liên hệ Nhánh R) | (Region, Aff) **chỉ cặp này** |
| A3 — tuỳ chọn | Run 3b @40000 | Liều affinity 0.2 — xung đột có đổi theo liều không | 3 cặp |
| A3 — tuỳ chọn | Baseline @40000 | Probe (Region, Aff) tại điểm không có loss biên | (Region, Aff-probe) |
| **Bỏ** | Run 4 | Lịch trình động gây nhiễu diễn giải; không cần cho câu hỏi này | — |

**Không tính bất kỳ cặp nào có `L_BCE` trên Baseline hoặc Affinity-only** — BoundaryHead ở đó chưa huấn luyện (khởi tạo ngẫu nhiên), gradient vô nghĩa. Script phải assert điều này (mục 4, cổng 4).

**Giới hạn đã biết:** đây là ảnh chụp tại điểm hội tụ. Xung đột có thể đã bị "giải quyết bằng thoả hiệp" trong lúc train (cái giá chính là precision bị mất). Probe trên BCE λ=0.4 bù một phần; Phần D (quỹ đạo) bù đầy đủ.

### 1.3. Khuyến nghị cho Phần B (lỗ giả) — CPU, mask đã có

Bắt buộc: Baseline@40k, BCE λ=0.4@40k, Static s19@36k, Static s86@36k, Affinity-only s19@40k, Affinity-only s86@40k.
Nên có: Static s86@40k, Run 3b@40k. Tuỳ chọn: Run 4@40k.

Thiết kế này cho đủ bảng 2×2 (có/không BCE × có/không affinity) **và** hai cặp đo nhiễu liên-seed (Static s19/s86, AffOnly s19/s86).

---

## 2. Phần A — Đo xung đột gradient

### 2.1. Thiết lập tính toán

- Ảnh: 384 ảnh val, **cùng tiền xử lý như huấn luyện** (resize 1024×1024, ToTensor, chuẩn hoá ImageNet; không augment, không crop).
- `model.eval()` — BN dùng running stats, để kết quả tất định. Ghi rõ lựa chọn này trong `summary`.
- Loss: gọi **đúng** module loss của repo (`src/losses/total_loss.py`, `boundary_bce.py`, `affinity.py`), lấy riêng ba số hạng chưa nhân trọng số rồi nhân trọng số hiệu dụng của checkpoint: `w_region=1`, `w_bce=α·λ1`, `w_aff=α·λ2` (Static: 0.4/0.4; probe: dùng 0.4 cho số hạng probe). Cosine không phụ thuộc trọng số; tỉ lệ độ lớn thì có.
- Nếu `affinity.py` lấy mẫu ngẫu nhiên: cố định `torch.manual_seed(19 + image_idx)` trước mỗi ảnh, **giống hệt nhau giữa các checkpoint**.
- Điểm dùng chung = Fused Feature `F` (đầu vào chung của segmentation head và BoundaryHead). Lấy bằng forward hook.
- Với mỗi số hạng k ∈ {region, bce, aff}:
  ```python
  g_F[k], g_theta[k] = torch.autograd.grad(w_k * L_k, [F] + shared_params, retain_graph=True)
  ```
  `shared_params` = mọi tham số **phía trước** F (encoder + decoder tới khối fuse). **Không** gồm segmentation head và BoundaryHead (riêng từng số hạng).

### 2.2. Mức tham số (toàn cục — mức PCGrad tác động)

- Chia 384 ảnh thành batch cố định, kích thước = `TRAIN.BATCH_SIZE` của config huấn luyện (nếu không đọc được, dùng 8 và ghi rõ). Phân hoạch bằng `np.random.default_rng(19).permutation(384)`, giống nhau cho mọi checkpoint.
- Mỗi batch b: cộng gradient tham số theo ảnh → `G_k(b)`; tính `cos_param(b) = cos(G_bce(b), G_aff(b))` và tương tự cho hai cặp còn lại; cùng tỉ lệ độ lớn `‖G_aff‖/‖G_bce‖`.
- Báo cáo: trung vị, CI 95% bootstrap trên các batch (10.000 lần, rng 19), tỉ lệ batch có cos < 0.
- **Null:** ghép `G_bce(b)` với `G_aff(b')`, b' ≠ b (mọi cặp lệch) → phân phối cosine "không liên quan". Báo cáo khoảng 2.5–97.5%.

### 2.3. Mức đặc trưng, theo không gian (độ phân giải stride-4)

Với mỗi vị trí p trên lưới stride-4 và một cặp (k1, k2): `dot_p = <g_F[k1]_p, g_F[k2]_p>`, `n1_p = ‖g_F[k1]_p‖²`, `n2_p = ‖g_F[k2]_p‖²` (vector 64 chiều).

**Phân tầng (strata)** — tính từ GT full-res rồi hạ về stride-4 bằng **nearest** (cùng quy ước với affinity):

| Trục | Các tầng |
|---|---|
| Khoảng cách tới biên GT (px full-res) | [0,4] · (4,8] · (8,16] · (16,32] · >32 |
| Lớp GT tại vị trí | 9 lớp, `CLASS_NAMES` chuẩn |
| Kích thước thành phần GT chứa vị trí (8-liên thông, theo lớp, diện tích full-res) | nhỏ < 4096 · vừa 4096–65536 · lớn > 65536 |

Với mỗi ảnh i và tầng s, lưu thống kê đủ: `Σdot, Σn1, Σn2`, số vị trí, số vị trí "hỗ trợ" (cả hai chuẩn > 1e-12), số vị trí hỗ trợ có cos_p < 0.

**Đại lượng chính — cosine có trọng số năng lượng, gộp micro** (đúng quy ước micro của dự án):
`cosW_s = Σ_i Σdot / sqrt(Σ_i Σn1 · Σ_i Σn2)`
Phụ: tỉ lệ vị trí cos_p < 0; tỉ lệ độ lớn `sqrt(Σn2/Σn1)`; độ phủ hỗ trợ.

- Bootstrap ghép cặp per-image (10.000, rng 19) cho `cosW_s`.
- **Null:** hoán vị ngẫu nhiên vị trí của `g_F[k2]` **trong cùng tầng, cùng ảnh** (rng 19, 20 lần) → khoảng null 2.5–97.5%.
- Tầng có độ phủ hỗ trợ < 1% số vị trí → đánh dấu **"không đủ hỗ trợ"**, loại khỏi quyết định (lưu ý: gradient affinity có thể chỉ khác 0 gần biên).

### 2.4. Hình minh hoạ — danh sách ảnh chốt TỪ GT, trước khi tính

6 ảnh: 2 ảnh có diện tích GT Agriculture lớn nhất, 2 ảnh có diện tích GT Water lớn nhất, 2 ảnh có số thành phần GT Building "nhỏ" nhiều nhất. Mỗi ảnh: RGB · GT · bản đồ cos_p (BCE, Aff) · bản đồ cos_p (Region, Aff), cho Static s19@36k. **Không đổi danh sách sau khi thấy bản đồ.**

---

## 3. Phần B — Lỗ giả (lỗi loại 5)

### 3.1. Định nghĩa vận hành

Với mỗi checkpoint, ảnh, lớp c:
1. `M_c` = mask dự đoán lớp c (từ PNG đã dump). Thành phần: 8-liên thông.
2. Lỗ = `binary_fill_holes(M_c) − M_c`, tách thành phần lỗ 4-liên thông (đối ngẫu topo chuẩn 8/4).
3. Giữ lỗ có diện tích ≥ `A_min`. **Chính: `A_min = 64` px full-res**; phụ: 16 và 256 (độ nhạy).
4. **Lỗ giả:** ≥ 50% pixel của lỗ có GT = c (GT nói ở đó là đặc).
5. **Thành phần vành khuyên:** thành phần dự đoán có tổng diện tích lỗ giả ≥ 10% diện tích sau khi lấp.
6. **Lỗ giả mới (do loss):** lỗ giả của model X **không chồng** (IoU < 0.1) với bất kỳ lỗ giả nào của Baseline trên cùng ảnh, cùng lớp.

Đại lượng per-image: `n_lo_gia`, `dien_tich_lo_gia`, `n_vanh_khuyen`, `n_lo_gia_moi` — tổng và theo lớp.

### 3.2. Cặp so sánh (bootstrap ghép cặp per-image, 10.000, rng 19)

| # | Cặp | Trả lời |
|---|---|---|
| H1 | BCE λ=0.4 − Baseline | BCE một mình có sinh lỗ giả không |
| H2 | AffOnly s19 − Baseline; H2' AffOnly s86 − Baseline | Affinity một mình có sinh không |
| H3 | Static s19 − Baseline; H3' Static s86 − Baseline | Loss ghép có sinh không |
| H4 | Static s19 − BCE λ=0.4 | Thêm affinity vào BCE |
| H5 | **Contrast:** (Static − BCE) − (AffOnly − Baseline), cho từng seed | Lỗ giả có phải hiệu ứng tương tác (như Nhánh R) không |
| N1 | Static s86 − s19 (@36k); N2 AffOnly s86 − s19 | **Biên độ liên-seed** — sàn nhiễu cho mọi cặp trên |

### 3.3. Cổng tần suất

Nếu Static s19@36k có **< 0.1 lỗ giả mới/ảnh** (tức < ~38 trên toàn val, ở `A_min=64`) ⇒ ghi **"lỗi loại 5 hiếm"** — có thể vẫn đúng về cơ chế nhưng **không đủ làm trục đóng góp**; chỉ dùng làm ví dụ định tính.

### 3.4. Phần B2 (tuỳ chọn, chỉ khi B không rơi vào "hiếm") — thử hậu xử lý

Lấp mọi lỗ có diện tích ≤ `A_fill` (∈ {64, 256, 1024}) trên mask dự đoán, **không dùng GT**; tính lại mIoU-9, precision/recall, BF, ASD hai chiều bằng chính các hàm đo đã có. So trước/sau lấp, bootstrap ghép cặp.

---

## 4. Cổng kiểm — bắt buộc trước khi tính trên 384 ảnh

1. **Nạp đúng checkpoint:** forward trên 384 ảnh tái lập `EXPECTED_MIOU9` (mục 3.0 `doi-chieu-...`): Static s19 = **0.6540486**, BCE λ=0.4 = **0.6566**, Baseline = **0.6550926**; Static s86@36k = **0.6557**. Dung sai 0.001. In cả mIoU-9 lẫn mIoU-8 kèm nhãn. Lệch ⇒ **DỪNG**.
2. **Đồng nhất thức loss** trên 8 ảnh đầu: `l_total == l_region + α(λ1·l_bce + λ2·l_aff)`, sai số < 1e-5.
3. **Tuyến tính gradient:** grad của tổng có trọng số == tổng các grad riêng (sai số tương đối < 1e-5) cho cả `F` và `shared_params`.
4. **Đúng phạm vi cặp:** assert không có cặp chứa `L_BCE` cho Baseline / Affinity-only; assert BoundaryHead có trọng số khác khởi tạo cho các checkpoint có BCE.
5. **Tất định:** chạy 2 lần trên 4 ảnh, mọi thống kê khớp tới 1e-6.
6. **Unit test lỗ (Phần B):** đĩa đặc → 0 lỗ; vành khuyên → 1 lỗ, đúng diện tích; hai đĩa rời → 0 lỗ; lỗ 63 px bị loại ở `A_min=64`.
7. **Danh sách ảnh** của mọi checkpoint khớp tuyệt đối (nội dung + thứ tự) với dump hiện có.

**Không đổi ngưỡng ở mục này sau khi thấy số.**

---

## 5. Bảng quyết định — CHỐT TRƯỚC KHI CÓ SỐ

### 5.1. Khi nào một xung đột được coi là "đã xác nhận"

Một cặp (k1, k2) tại một mức/tầng là **xung đột đã xác nhận** khi **đồng thời**:
1. CI 95% bootstrap có cận trên < 0;
2. giá trị quan sát nằm dưới cận 2.5% của phân phối null;
3. |cos| ≥ τ, với **τ_param = 0.05** (mức tham số, không gian rất nhiều chiều) và **τ_feat = 0.10** (mức đặc trưng);
4. tỉ lệ độ lớn nằm trong [0.1, 10] (xung đột với một gradient nhỏ hơn 10 lần thì không đáng kể về thực hành);
5. **đúng ở cả Static s19@36k và Static s86@36k** (cùng dấu, cả hai đạt 1–3).

### 5.2. Nhánh (đại lượng quyết định: Static, hai seed, @36k)

| Nhánh | Điều kiện | Đọc ra | Phương pháp cho bước tiếp |
|---|---|---|---|
| **G — Toàn cục** | Cặp (BCE, Aff) xung đột đã xác nhận ở **mức tham số** | Hai loss kéo tham số dùng chung ngược nhau | Chiếu gradient kiểu PCGrad giữa BCE và Aff; baseline so sánh: PCGrad thường, Du et al. 2018 |
| **L — Khu trú không gian** | Không đạt G, nhưng (BCE, Aff) xác nhận ở **≥1 tầng khoảng cách hoặc kích thước**, không phải mọi tầng | Xung đột chỉ ở một vùng (vd. dải sát biên, hoặc vật nhỏ) | Mặt nạ / trọng số affinity theo khoảng cách hoặc kích thước thành phần; chiếu gradient **chỉ trong vùng xung đột** |
| **C — Khu trú theo lớp** | Không đạt G/L, nhưng xác nhận ở **≤3 lớp** | Xung đột phụ thuộc lớp (ứng viên: Water, Agriculture, Bareland — khớp K12) | Trọng số affinity theo lớp |
| **RA — Affinity xung đột với nhiệm vụ chính** | (BCE, Aff) không xác nhận ở đâu, nhưng **(Region, Aff)** xác nhận ở mức tham số hoặc ở tầng khoảng cách > 8px | Cái giá precision/ruột vùng đến từ affinity đối nghịch `L_region`, không phải BCE. **Khung của Hướng B đổi**: "bảo vệ nhiệm vụ chính khỏi loss phụ" | Cổng gradient kiểu Du et al. (chỉ dùng gradient aff khi cos với region ≥ 0) hoặc chiếu aff lên region |
| **N — Không xung đột tại điểm hội tụ** | Không cặp nào đạt 5.1 | Chưa đóng: xung đột có thể đã bị thoả hiệp trong lúc train | **Chạy Phần D** trước khi kết luận. Nếu D cũng N ⇒ **dừng Hướng B**, đưa thành một đoạn Discussion |

Nếu nhiều nhánh cùng đạt: báo cáo **tất cả**; chọn phương pháp theo thứ tự **G > RA > L > C**, và ghi nhánh còn lại làm biến thể ablation.

### 5.3. Thông tin mô tả (không dùng để chọn nhánh)

- **Probe BCE λ=0.4:** nếu xung đột (BCE, Aff-probe) mạnh hơn rõ so với (BCE, Aff) trong Static ⇒ diễn giải "affinity đã thoả hiệp với BCE trong huấn luyện, cái giá là precision" — ủng hộ Phần D.
- **Affinity-only vs Static trên (Region, Aff):** nếu quan hệ đổi dấu khi có BCE ⇒ cơ chế ứng viên cho Nhánh R của Run 7.
- **Static s86 @36k vs @40k:** nếu kết luận đổi giữa hai mốc ⇒ ghi cảnh báo độ ổn định, không chọn mốc có lợi.

### 5.4. Phần B — quy trách nhiệm lỗi loại 5

Một chênh lệch là **đã xác nhận** khi CI loại trừ 0 **và** |Δ| > biên độ liên-seed tương ứng (N1 cho Static, N2 cho AffOnly) — quy tắc kép chuẩn của dự án.

| Kết quả | Kết luận |
|---|---|
| H2, H2' và H3, H3' xác nhận tăng; H1 không | Lỗi 5 **do affinity**, độc lập với BCE |
| H1 xác nhận tăng; H2/H2' không | Lỗi 5 **do BCE** |
| Chỉ H3/H3' xác nhận, H5 loại trừ 0 ở cả hai seed | Lỗi 5 là **hiệu ứng tương tác** — cùng họ với Nhánh R |
| Không cặp nào xác nhận | Lỗi 5 **không do loss**; ví dụ bồn chứa là trường hợp riêng lẻ hoặc có ở cả Baseline |

Cổng tần suất (3.3) áp dụng song song: "do X" + "hiếm" ⇒ đúng cơ chế, không làm trục đóng góp.

---

## 6. Phần C — Nối hai hiện tượng (mô tả, chạy khi B ≠ "không do loss")

Trên Static s19@36k: với các thành phần GT **nhỏ**, so `cosW` (BCE, Aff) và (Region, Aff) **bên trong** thành phần có lỗ giả vs thành phần không có lỗ giả cùng lớp. Bootstrap theo thành phần. Nếu bên có lỗ giả âm hơn rõ (CI chênh lệch loại trừ 0) ⇒ ủng hộ "lỗi 5 và xung đột cùng một gốc" → gộp lỗi 5 vào trục chính của Hướng B.

---

## 7. Phần D (tuỳ chọn, có cổng) — Quỹ đạo xung đột trong lúc huấn luyện

**Cổng:** chạy nếu Phần A rơi vào **N**, hoặc nếu A đạt G/L/C/RA và cần bằng chứng mạnh hơn cho bài tạp chí.

- Train lại **Static seed 19**, config `run3_static_boundary.yaml` **nguyên vẹn**, thêm callback mỗi 500 iter: đo mức tham số (mục 2.2) và mức đặc trưng theo tầng khoảng cách (mục 2.3) trên **một tập dò cố định 32 ảnh lấy từ train split** (rng 19, chốt trước), không dùng val.
- Lưu đủ 4 checkpoint theo quy tắc chuẩn — **lợi ích phụ: lấp lỗ hổng cũ "Static s19 không có checkpoint 40000"**.
- Sanity: mIoU-9 tại 36k phải nằm trong biên độ liên-seed so với run gốc (0.6540 ± 0.0017) — nếu không, callback đã làm đổi huấn luyện ⇒ DỪNG.
- Chi phí: ~3.5h GPU + thời gian callback (đo lại ở dry-run).

---

## 8. Output

```
grad_conflict/
  per_image_strata_<ckpt>.csv      # Σdot, Σn1, Σn2, n_pos, n_support, n_neg theo cặp × tầng
  per_batch_param_<ckpt>.csv       # cos_param, tỉ lệ độ lớn theo cặp × batch
  null_<ckpt>.csv
  summary_grad_conflict.md         # bảng cosW + CI + null + nhánh 5.2, từng checkpoint
  figures/                         # 6 ảnh chốt ở mục 2.4
spurious_holes/
  per_image_<ckpt>.csv
  pairs_bootstrap.md               # H1–H5, N1–N2, verdict 5.4, cổng tần suất
  fill_test.md                     # chỉ khi chạy B2
```

`summary_grad_conflict.md` dòng đầu ghi: checkpoint, iter, seed, chế độ `eval()`, batch size dùng ở 2.2, `A_min` chính.

---

## 9. KHÔNG ĐƯỢC ĐỔI

- Splits và 384 ảnh val; tiền xử lý (resize 1024, không augment/crop).
- Mọi file trong `src/losses/` — chỉ **gọi**, không sửa.
- Mọi kết quả, CSV, dump, `bootstrap_ci.*` đã có — chỉ đọc. Output mới ghi vào thư mục mới ở mục 8.
- Ngưỡng ở mục 4 và 5 sau khi thấy số. Nếu thấy ngưỡng có vẻ sai, ghi nhận xét bên cạnh, **giữ nguyên verdict**.
- Không chọn checkpoint/mốc/seed "đẹp": mọi checkpoint đã liệt kê bắt buộc ở 1.2/1.3 đều phải có trong báo cáo.

---

## 10. Checklist bàn giao

- [ ] Script `tools/grad_conflict_probe.py` (Phần A) + `tools/spurious_holes.py` (Phần B)
- [ ] Cổng kiểm mục 4: **7/7 PASS**, báo cáo từng cổng
- [ ] Phần A trên các checkpoint A1 (bắt buộc) và A2 (nên có); ghi thời gian thật/checkpoint
- [ ] Phần B trên các checkpoint bắt buộc ở 1.3; cổng tần suất 3.3
- [ ] Áp bảng 5.2 và 5.4 **nguyên văn**, ghi verdict
- [ ] Phần C nếu B ≠ "không do loss"; Phần B2 nếu B ≠ "hiếm"
- [ ] Quyết định có chạy Phần D theo cổng mục 7
- [ ] Gửi `summary_grad_conflict.md` + `pairs_bootstrap.md` để đánh giá trong một tài liệu project mới
