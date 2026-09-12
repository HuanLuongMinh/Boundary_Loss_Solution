# Spec bàn giao Claude Code — bản 2 (8/9/2026): Per-image sufficient statistics + Paired bootstrap + đếm mảnh giả Agriculture

**Repo:** `Boundary_Loss_Solution`
**Thay thế cho việc thực thi:** `spec-per-image-bootstrap-eval-ban-giao-claude-code.md` (7/9). Giữ nguyên mọi nguyên tắc thiết kế của bản đó; tài liệu này là **bản đầy đủ, tự chứa** để giao thẳng cho Claude Code — không cần đọc chéo hai file. Ba thay đổi so với bản gốc, đánh dấu 🆕 xuyên suốt:

1. Thêm **2 checkpoint của Run 5 (Static, seed 86)** vào danh sách dump — bản gốc viết trước khi Run 5 có kết quả nên thiếu.
2. Thêm **cặp bootstrap thứ 6: Static seed19 − Static seed86** — trả lời câu hỏi đang treo: biên độ 0.5343px đo được giữa hai seed (báo cáo `run5-static-seed86-ket-qua-va-danh-gia.md`) đến từ nhiễu tập đánh giá hay từ nhiễu huấn luyện thật.
3. Thêm **Phần D — đếm thành phần liên thông giả trên ảnh Agriculture (việc #3 / C4)**, dùng chung mask PNG mà Phần B xuất ra, không tốn thêm inference. Việc này đã được nâng ưu tiên sau phát hiện oracle C5b (Agriculture +2.36đ là hiệu ứng ruột vùng, khớp giả thuyết "mảnh giả giữa ruộng").

**Chi phí tổng:** ~2.5h CPU (dump 7 checkpoint thay vì 5, +50 phút) + phần phân tích vài phút + C4 ~30-60 phút. **0 GPU, 0 quota GPU.**
**Không train lại bất cứ thứ gì.**

---

## 0. Vì sao làm việc này

Toàn bộ hệ ngưỡng quyết định của dự án đang dùng σ = 0.149px, đo **giữa các vòng validate trong một run** — đó là phương sai của việc chọn checkpoint, không phải của hai nguồn còn lại:

| Nguồn phương sai | Đã đo? | Đo bằng gì |
|---|---|---|
| Chọn checkpoint nào trong một run | ✅ σ = 0.149px | mean±std 4 vòng cuối |
| Val set gồm 384 ảnh nào | ❌ | **Việc này (bootstrap)** — 0 GPU |
| Quỹ đạo huấn luyện / seed | 🔶 đo được 1 điểm (n=2) | Run 5 — biên độ 0.5343px trên `gt→pred`, **vượt xa** ngưỡng 0.25px đã đăng ký trước |

**Run 5 đã đổi ý nghĩa của việc này.** Trước 7/9, mục tiêu là "đóng lỗ hổng rẻ trước khi quyết có cần seed thứ hai". Bây giờ seed thứ hai đã có và cho kết quả xấu hơn dự kiến (nhánh III — xem `khung-dien-giai-va-trich-dan-cho-paper.md` bản 5, mục 0). Mục tiêu bây giờ là **tách hai nguồn phương sai ra khỏi nhau**: nếu bootstrap ghép cặp cho CI hẹp trên chính cặp Static seed19-seed86 (cặp #6 mục 3.4), biên độ 0.5343px chủ yếu đến từ huấn luyện, không phải từ việc chọn 384 ảnh nào — củng cố lý do cần seed thứ 3. Nếu CI rộng ngang biên độ đó, một phần đáng kể có thể quy về nhiễu đánh giá — tin tốt, rẻ hơn để đóng lại.

Việc này **không thay thế** run lặp seed thêm. Nó đóng một lỗ hổng rẻ, và bất đối xứng: nếu CI trùm 0 thì claim bị loại với 0 GPU; nếu không, có ngay thanh sai số in được vào bài.

---

## 1. Nguyên tắc thiết kế — đọc trước khi viết dòng code nào

### 1.1. Xuất **thống kê đủ** (sufficient statistics), KHÔNG xuất metric per-image đã chia

Đây là điểm quyết định việc này làm đúng hay làm hỏng.

Bootstrap trên ảnh chỉ hợp lệ nếu cách gộp lại **khớp đúng** cách `eval_boundary_metrics.py` hiện đang gộp. Có hai quy ước khả dĩ và chưa biết chắc code đang dùng cái nào:

- **macro** — trung bình các giá trị per-image
- **micro** — dồn tử số và mẫu số toàn dataset rồi mới chia

Nếu code dùng micro mà bootstrap trên giá trị per-image (hoặc ngược lại), mọi CI sẽ **sai** và không tái lập được số đã công bố.

**Giải pháp không cần đoán: xuất tử số và mẫu số riêng cho từng ảnh.** Cộng lại theo công thức đúng ra số gộp; bootstrap thì resample ảnh rồi cộng lại. Cách này đúng với **cả hai** quy ước, và giai đoạn 2 sẽ tự dò xem quy ước nào khớp số đã công bố (mục 3.2).

### 1.2. Thuần cộng thêm — không đụng đường tính cũ

Mọi thứ trong tài liệu này là **nhánh phụ** kích hoạt bằng cờ CLI. Đường tính mặc định của `eval_boundary_metrics.py` phải cho ra **byte-identical** kết quả so với trước khi sửa. Nếu số cũ đổi thì hỏng cả bảng hằng số tham chiếu và mọi tài liệu project đã viết dựa trên nó.

### 1.3. Chỉ dùng một nguồn: pipeline JSON hậu kỳ

Đã biết CSV (trong lúc train) và JSON (hậu kỳ) lệch ~0.0011px ở ASD (đo tại Run 3, iter 36000). Mọi số trong việc này lấy từ **pipeline JSON hậu kỳ**, không trộn với CSV. 🆕 **Run 5 hiện chỉ có CSV in-training** — dump ở Phần B chính là bước tạo ra JSON hậu kỳ còn thiếu của nó, đóng luôn khoảng trống mà `run5-static-seed86-ket-qua-va-danh-gia.md` mục 4 ghi là việc ưu tiên #1.

### 1.4. Tách hai giai đoạn: inference một lần, phân tích nhiều lần

Inference tốn ~24.5 phút/checkpoint (3.82 s/ảnh, CPU). Nếu nhét bootstrap thẳng vào vòng inference, mỗi lần đổi ý về cách tính lại mất thêm 25 phút.

- **Giai đoạn 1** (`--dump-preds`): chạy inference **một lần**, ghi mask dự đoán ra PNG + thống kê đủ per-image ra CSV.
- **Giai đoạn 2** (script riêng, đọc từ file đã dump): tính gộp, kiểm cổng, bootstrap. Chạy trong vài giây, lặp bao nhiêu lần cũng được.

---

## 2. PHẦN A — Thay đổi code trong `eval_boundary_metrics.py`

### 2.1. Thêm 3 cờ

```
--dump-preds DIR              # ghi mask dự đoán (PNG, uint8, giá trị 0-8, không palette)
--dump-per-image-stats FILE   # ghi CSV thống kê đủ per-image
--dump-only                   # chỉ dump, bỏ qua phần in báo cáo gộp (tuỳ chọn, tiện lợi)
```

Không có cờ ⇒ hành vi **y hệt** hiện tại (xem cổng bắt buộc ở mục 4.0 bên dưới).

### 2.2. Đặc tả CSV thống kê đủ — 384 hàng, một hàng một ảnh

Cột `image` là **tên file gốc** (không phải chỉ số), dùng làm khoá ghép cặp giữa các checkpoint.

**Khối chung:**

| Cột | Ý nghĩa |
|---|---|
| `image` | tên file ảnh val |
| `n_valid_px` | số pixel hợp lệ (sau `ignore_index`) |

**Khối IoU ngữ nghĩa — 9 lớp × 2 cột:**

| Cột | Ý nghĩa |
|---|---|
| `inter_<Class>` | \|pred = c ∧ gt = c\| |
| `union_<Class>` | \|pred = c ∨ gt = c\| |

**Khối Boundary IoU — 9 lớp × 3 ngưỡng × 2 cột:**

| Cột | Ý nghĩa |
|---|---|
| `biou_inter_<Class>_d{1,2,4}` | tử số Boundary IoU của lớp c tại ngưỡng d |
| `biou_union_<Class>_d{1,2,4}` | mẫu số |

**Khối BF-Score — tính trên bản đồ biên nhị phân gộp lớp (đúng như hiện tại, một số duy nhất, không per-class):**

| Cột | Ý nghĩa |
|---|---|
| `bf_tp_pred` | số pixel biên **pred** có match trong dung sai θ với biên gt |
| `bf_n_pred` | tổng số pixel biên pred |
| `bf_tp_gt` | số pixel biên **gt** có match trong dung sai θ với biên pred |
| `bf_n_gt` | tổng số pixel biên gt |

⇒ `precision = bf_tp_pred / bf_n_pred`, `recall = bf_tp_gt / bf_n_gt`. **Hai mẫu số khác nhau** — đây là lý do không được xuất precision/recall dạng đã chia.

**Khối ASD:**

| Cột | Ý nghĩa |
|---|---|
| `asd_sum_pred_to_gt` | tổng khoảng cách từ mỗi pixel biên pred tới biên gt gần nhất |
| `asd_n_pred` | số hạng tử trong tổng trên |
| `asd_sum_gt_to_pred` | tổng khoảng cách chiều ngược lại |
| `asd_n_gt` | số hạng tử |

**Khối cờ trạng thái rỗng — bắt buộc, đây là chỗ dễ sập nhất:**

| Cột | Ý nghĩa |
|---|---|
| `has_gt_boundary` | 0/1 — ảnh có pixel biên gt nào không |
| `has_pred_boundary` | 0/1 |
| `class_present_<Class>` | 0/1 — lớp c có trong gt của ảnh này không |

### 2.3. File JSON đi kèm — ghi lại quy ước, không để phải đoán lại sau

Cùng thư mục, `<stem>_meta.json`:

```json
{
  "checkpoint_path": "...",
  "checkpoint_iter": 36000,
  "run_name": "run3_static_boundary",
  "val_split_file": "...",
  "n_images": 384,
  "bf_tolerance_theta": <giá trị thật trong code>,
  "biou_dilations": [1, 2, 4],
  "connectivity": 4,
  "ignore_index": 255,
  "class_names": ["Background", "...", "Building"],
  "asd_distance_transform": "<tên hàm/thư viện dùng>",
  "empty_image_policy_observed": "<điền ở mục 3.2>",
  "git_commit": "...",
  "eval_wall_time_s": ...
}
```

### 2.4. Mask PNG

`<dump_dir>/<tên_file_gốc>.png`, uint8, giá trị 0–8, **không palette**, cùng kích thước ảnh gốc. Input cho Phần D.

---

## 3. PHẦN B — Dump 7 checkpoint

🆕 Bảng gốc có 5 checkpoint; đã thêm hai hàng cuối (Run 5, seed 86) — đây chính là mảnh còn thiếu mà `run5-static-seed86-ket-qua-va-danh-gia.md` mục 4 nêu là ưu tiên #1.

| # | Run | File checkpoint | Iter | Ưu tiên | ~Thời gian |
|---|---|---|---|---|---|
| 1 | Run 3 — Static α_aff 0.4, seed 19 | `best_model.pth` | 36000 | **bắt buộc** | 25' |
| 2 | Run 2b — BCE λ=0.4 | `best_model.pth` | 40000 | **bắt buộc** | 25' |
| 3 | Run 1 — Baseline | `best_model.pth` | 40000 | **bắt buộc** | 25' |
| 4 | Run 4 — Dynamic | `final_iter40000.pth` | 40000 | **bắt buộc** | 25' |
| 5 | Run 3b — α_aff 0.2 | `final_iter40000.pth` | 40000 | nên có | 25' |
| **6 🆕** | **Run 5 — Static α_aff 0.4, seed 86** | **`best_miou.pth`** | **36000** | **bắt buộc** | 25' |
| **7 🆕** | **Run 5 — Static α_aff 0.4, seed 86** | **`final_iter40000.pth`** | **40000** | nên có | 25' |

Tổng ~2h55' CPU cho cả 7 (bắt buộc: #1-4, #6 ⇒ ~2h05'; thêm #5, #7 ⇒ +50').

Nếu Kaggle cấp P100 (sm_60, không tương thích build PyTorch sm_70+) vẫn chạy được — đây là eval CPU thuần. **Không bật GPU cho notebook này**, để dành quota.

**Lưu ý mốc iteration:** Static (cả seed 19 và seed 86) chỉ tồn tại tại checkpoint **36000** cho `best_miou`/`best_model` — không có checkpoint 40000 riêng cho "best". Checkpoint #7 (`final_iter40000.pth` của seed 86) tồn tại vì Run 5 dùng protocol báo cáo mới (4 checkpoint/run, xem `run5-static-seed86-ket-qua-va-danh-gia.md` mục 3) — khác Run 3 (seed 19) vốn không có checkpoint 40000 nào cả. **Không nhầm hai việc này** khi đọc bảng.

**Lệnh mẫu cho mỗi checkpoint** (điều chỉnh path theo thực tế repo):

```bash
python eval_boundary_metrics.py \
  --checkpoint <path_to_checkpoint>.pth \
  --val-split <path_to_val_split_384>.txt \
  --dump-preds dump/<run_key>_iter<iter>/masks \
  --dump-per-image-stats dump/<run_key>_iter<iter>/per_image_stats.csv \
  --dump-only
```

`<run_key>` gợi ý: `run1_baseline`, `run2b_bce04`, `run3_static_s19`, `run3b_aff02`, `run4_dynamic`, `run5_static_s86_best`, `run5_static_s86_final`.

---

## 4. PHẦN C — Script giai đoạn 2: `tools/bootstrap_boundary_ci.py` (file mới)

### 4.0. Cổng bắt buộc trước tiên: không cờ ⇒ kết quả y hệt cũ

Trước khi tin bất cứ thứ gì khác: chạy `eval_boundary_metrics.py` **không** cờ mới trên baseline, so **byte-identical** JSON output với trước khi sửa code. Đây là điều kiện tiên quyết của mục 1.2 — nếu fail, dừng lại, không đi tiếp.

### 4.1. Tái lập số gộp từ thống kê đủ

Với mỗi checkpoint, tính lại **cả hai quy ước** rồi so với số đã công bố:

```
micro:  pred_to_gt = sum(asd_sum_pred_to_gt) / sum(asd_n_pred)
macro:  pred_to_gt = mean_i( asd_sum_pred_to_gt[i] / asd_n_pred[i] )
```

Tương tự cho `gt→pred`, precision, recall, IoU, BIoU (mỗi lớp), rồi macro-9 / macro-8 trên lớp.

### 4.2. Dò chính sách ảnh rỗng bằng dữ liệu, không bằng cách đọc code

Với ảnh có `asd_n_pred = 0` hoặc `class_present_c = 0`, thử lần lượt:

- `skip` — bỏ ảnh đó khỏi cả tử và mẫu
- `zero` — tính là 0
- `one` — tính là 1 (với IoU/BIoU)
- `diag` — gán bằng đường chéo ảnh (với ASD)

Chính sách nào **tái lập được** số công bố thì ghi vào `empty_image_policy_observed`. **Chính sách đó phải được dùng y nguyên trong bootstrap.**

### 4.3. Bootstrap ghép cặp

```python
rng = np.random.default_rng(19)          # cùng seed với toàn dự án
B = 10000
idx = rng.integers(0, n_images, size=(B, n_images))
```

Với **mỗi** vòng b: dùng **cùng một bộ chỉ số** `idx[b]` cho **cả hai** mô hình trong cặp, tính lại số gộp cho từng mô hình theo đúng quy ước đã dò (mục 4.2), rồi lấy hiệu Δ.

Ghép cặp là bắt buộc: hai mô hình cùng khó ở cùng những ảnh khó, nên CI ghép cặp hẹp hơn đáng kể so với hai CI riêng — và **Δ mới là thứ bài bảo vệ**, không phải giá trị tuyệt đối của từng mô hình.

Báo cáo cho mỗi Δ: **trung vị, CI 95% (percentile 2.5 / 97.5), và P(Δ < 0)**.

### 4.4. Các cặp phải tính — 6 cặp

🆕 Cặp #6 là bổ sung của bản này.

| # | Cặp | Vì sao |
|---|---|---|
| 1 | **Static s19@36k − BCE λ=0.4@40k** | ⭐ Cặp cô lập affinity — hai con số Abstract cũ nằm ở đây |
| 2 | Static s19@36k − Baseline@40k | Hiệu ứng tổng của loss biên đề xuất |
| 3 | **Run 4@40k − Baseline@40k** | ⭐ Ví dụ nghịch lý (bf_score tăng, asd_pred_to_gt xấu đi) |
| 4 | Run 4@40k − Static s19@36k | Verdict nhánh 4 của Run 4 (hiện đọc là "không kết luận được") |
| 5 | Run 3b@40k − BCE λ=0.4@40k | Điểm liều 0.2, để đường liều 3 điểm cũng có CI |
| **6 🆕** | **Static s19@36k − Static s86@36k** | **Câu hỏi trung tâm của bản này: biên độ 0.5343px quan sát ở `gt→pred` đến từ nhiễu tập đánh giá (384 ảnh nào) hay từ nhiễu huấn luyện thật?** Nếu CI hẹp hơn hẳn 0.5343px và không trùm 0 ⇒ phần lớn biên độ là thật (huấn luyện), không phải nhiễu đánh giá. Nếu CI trùm 0 hoặc rộng gần bằng 0.5343px ⇒ không loại trừ được rằng phần lớn là nhiễu đánh giá |

### 4.5. Các đại lượng phải tính CI cho mọi cặp

`asd_pred_to_gt`, `asd_gt_to_pred`, `asd`, `bf_precision`, `bf_recall`, `bf_score`, `mIoU-9`, `mIoU-8`, `mIoU loại Bareland`, `BIoU-8@{1,2,4}`, và per-class IoU cho **Bareland, Water, Agriculture** (ba lớp đang có tranh luận cơ chế).

### 4.6. Output

- `bootstrap_ci.csv` — một hàng mỗi (cặp × đại lượng): `median, ci_lo, ci_hi, p_delta_lt_0, observed_delta`
- `bootstrap_ci.md` — bảng đọc được, dán thẳng vào tài liệu project

---

## 5. Cổng kiểm — BẮT BUỘC PASS trước khi tin bất kỳ CI nào

**5.1. Tái lập Run 3 Static seed19 @36000** từ thống kê đủ, so với `run3_static_boundary_boundary_metrics.json`:

| Đại lượng | Giá trị phải ra |
|---|---|
| `asd_pred_to_gt` | 5.2157764 |
| `asd_gt_to_pred` | 4.0321485 |
| `asd` | 4.62396245 |
| `bf_score` | 0.6144298 |
| `bf_precision` / `bf_recall` | 0.6241 / 0.6051 |
| mIoU-9 | 0.65404855 |
| BIoU-8 @d1/d2/d4 (×100) | 5.270 / 10.428 / 19.651 |
| BIoU Background @d1/d2/d4 (×100) | 10.519 / 21.892 / 40.713 |

**Dung sai: 1e-6.** Lệch hơn ⇒ hiểu sai quy ước gộp, **DỪNG LẠI**, báo cáo, không chạy tiếp bootstrap.

**5.2. Tái lập BCE λ=0.4 @40000 và Baseline @40000** từ JSON tương ứng, cùng dung sai 1e-6.

**5.3. Run 4 và Run 3b — dung sai nới, ghi lại độ lệch.** Hai run này chưa có JSON hậu kỳ; số đối chiếu lấy từ CSV in-training, đã biết CSV/JSON lệch 0.0011px ở ASD. Dung sai **0.003px cho ASD, 1e-4 cho các đại lượng còn lại**.

**5.4. 🆕 Run 5 (seed 86) — cùng quy tắc dung sai nới như 5.3.** Số đối chiếu lấy từ `benchmark_results.csv` của Run 5 (vòng 9 = iter 36000 cho `best_miou`, vòng 10 = iter 40000 cho `final`). Giá trị tham chiếu (từ `run5-static-seed86-ket-qua-va-danh-gia.md` mục 1, đơn vị điểm/px):

| Đại lượng | Giá trị CSV tham chiếu @36000 |
|---|---|
| `asd_pred_to_gt` | 5.4879 |
| `asd_gt_to_pred` | 4.5664 |
| `bf_precision` / `bf_recall` | 0.6255 / 0.5868 |
| mIoU-9 | 0.6557 |

Dung sai **0.003px cho ASD, 1e-4 cho các đại lượng còn lại** — giống Run 4/3b.

**5.5. Ghi lại chính xác độ lệch CSV↔JSON quan sát được cho Run 4, Run 3b, và 🆕 Run 5.** Hiện dự án chỉ có **một** phép đo độ lệch này (0.0011px, Run 3 @36000). Ba phép đo nữa (Run 4, Run 3b, Run 5) biến nó từ giai thoại thành một con số trích được trong mục Reproducibility. Rẻ, và không lấy lại được nếu quên.

**5.6. Kiểm tính tất định:** chạy giai đoạn 2 hai lần trên cùng file dump phải ra kết quả **giống hệt** (bootstrap RNG seed cố định).

**5.7. Kiểm ghép cặp:** danh sách `image` của mọi checkpoint phải **giống hệt nhau về cả nội dung lẫn thứ tự**, kể cả 2 checkpoint Run 5 mới. Assert cứng, không sort ngầm.

**5.8. Kiểm mask:** với 3 ảnh bất kỳ, `inter_<Class>` / `union_<Class>` tính lại từ PNG đã dump phải khớp CSV. Đảm bảo mask dùng được cho Phần D.

---

## 6. Điều KHÔNG ĐƯỢC ĐỔI

- Đường tính gộp mặc định của `eval_boundary_metrics.py` — kết quả không cờ phải giống hệt trước khi sửa
- Val split (384 ảnh), thứ tự file, `seed = 19` (là seed của **script eval/bootstrap**, không nhầm với seed **huấn luyện** của Run 5, vốn là 86 — hai khái niệm khác nhau, ghi rõ trong output để người đọc không lẫn)
- `connectivity = 4`, `dilation_radius = 0`, `ignore_index = 255`, dung sai θ của BF-Score
- Bất kỳ checkpoint nào — **không train lại gì cả**
- `src/losses/*` — việc này không chạm vào loss
- Không sửa số đã công bố trong bất kỳ tài liệu nào của project dựa trên kết quả việc này; nếu phát hiện mâu thuẫn, **báo cáo, đừng sửa**

---

## 7. Quy tắc đọc kết quả — CHỐT TRƯỚC KHI CÓ SỐ

### 7.1. Bảng quyết định chính — CI 95% của Δ(`gt→pred`) cho cặp #1 (Static s19 − BCE λ=0.4)

Giá trị quan sát được: **−0.3570px**.

| Nhánh | Điều kiện | Kết luận | Bước tiếp |
|---|---|---|---|
| **A** | CI hoàn toàn < 0 **và** nửa độ rộng ≤ 0.15px | Nhiễu tập đánh giá không giải thích được hiệu ứng | In CI kèm số trong bài — **nhưng vẫn phải đọc thêm mục 7.5 dưới đây trước khi in**, vì bootstrap chỉ đóng được một trong hai nguồn phương sai |
| **B** | CI hoàn toàn < 0 **và** nửa độ rộng 0.15–0.35px | Claim sống nhưng mong manh | Bắt buộc in kèm CI, không in số trần ở Abstract; đọc thêm 7.5 |
| **C** | CI trùm 0 | Không claim được độ lớn | Δ ra khỏi Abstract. Bài lùi về phát biểu định tính về dấu |

### 7.2. Bảng phụ — `recall`, cặp #1

Giá trị quan sát: **+0.0094**. Cùng ba nhánh, ngưỡng nửa độ rộng **0.004 / 0.004–0.009 / trùm 0**.

### 7.3. Ví dụ nghịch lý — cặp #3 (Run 4 − Baseline)

Dùng được trong bài chỉ nếu **cả hai** điều kiện đồng thời: CI của Δ(`bf_score`) hoàn toàn > 0, **và** CI của Δ(`asd_pred_to_gt`) hoàn toàn > 0. Nếu một CI trùm 0 ⇒ không dùng ví dụ này, quay lại ví dụ nghịch lý Run3b–Static (đã có sẵn, không cần bootstrap thêm — xem `khung-dien-giai...` mục 2.6).

### 7.4. Verdict nhánh 4 của Run 4 — cặp #4

Nếu CI của Δ(`asd_pred_to_gt`) trùm 0 thì verdict "lịch trình có hại" **không được củng cố**. Ghi nhận xét bên cạnh, **không sửa verdict** đã đăng ký (verdict hiện đã là "không kết luận được" theo Run 5 — xem mục 1.3 tài liệu `run5-static-seed86-ket-qua-va-danh-gia.md`; việc này chỉ thêm bằng chứng, không đổi verdict).

### 7.5. 🆕 Cặp #6 (Static s19 − Static s86) — bảng đọc riêng, quyết seed thứ 3

Giá trị quan sát trên `gt→pred`: **+0.5343px** (theo chiều s86 − s19; giữ đúng dấu này khi báo cáo, đừng đảo).

| Nhánh | Điều kiện trên CI của Δ(`gt→pred`), cặp #6 | Đọc ra | Hệ quả cho seed thứ 3 (mục 3.3 tài liệu tổng hợp) |
|---|---|---|---|
| **X — nhiễu đánh giá nhỏ** | CI hoàn toàn cùng dấu với 0.5343 **và** nửa độ rộng ≤ 0.15px | Phần lớn biên độ 0.5343px là thật (đến từ huấn luyện/seed), không phải do chọn 384 ảnh nào | Củng cố mạnh: seed thứ 3 đáng chạy nếu muốn giữ số liệu độ lớn trong Abstract |
| **Y — nhiễu đánh giá đáng kể** | CI cùng dấu nhưng nửa độ rộng > 0.15px | Một phần biên độ có thể là nhiễu đánh giá, chưa tách bạch hết | Trung lập — quyết seed thứ 3 theo cân nhắc ngân sách GPU, không theo bằng chứng thống kê thêm |
| **Z — không phân biệt được** | CI trùm 0 | Không loại trừ được khả năng phần lớn 0.5343px là nhiễu đánh giá, không phải nhiễu huấn luyện | Yếu đi: nếu ngay cả *chọn ảnh nào* đã có thể tạo ra chênh lệch cỡ này, thêm một seed nữa (vẫn cùng 384 ảnh) sẽ không tách bạch được nguồn — cân nhắc **mở rộng val set** thay vì thêm seed |

⚠ Nhánh Z có hệ quả khác thường: nó gợi ý *bootstrap trên cùng 384 ảnh* không đủ để giải thích biên độ 0.5343px chỉ khi CI trùm 0 theo cách đặc biệt — cụ thể là nếu ngay cả việc **resample lại từ đúng 384 ảnh đó** (không phải một tập ảnh khác) đã cho biên độ ngang 0.5343px, thì đó là dấu hiệu bootstrap over-dispersion (vài ảnh outlier chi phối), không phải bằng chứng nhiễu tập đánh giá theo nghĩa "384 ảnh này không đại diện". Ghi rõ cách đọc này trong `bootstrap_ci.md`, đừng diễn giải quá tay.

### 7.6. 🆕 Đối chiếu bắt buộc với bảng biên độ liên-seed (Run 5) trước khi in bất kỳ CI nào vào bài

Bất kể CI ghép cặp nói gì ở cặp #1–#5, một con số cụ thể chỉ được coi là **"đã xác nhận độ lớn"** nếu đồng thời lớn hơn biên độ liên-seed đo được ở Run 5 (bảng dưới, từ `khung-dien-giai-va-trich-dan-cho-paper.md` mục 0.2):

| Đại lượng | Biên độ liên-seed (n=2) |
|---|---|
| `asd_gt_to_pred` | 0.5343px |
| `asd_pred_to_gt` | 0.2721px |
| `asd` gộp | 0.4032px |
| `bf_precision` | 0.0014 |
| `bf_recall` | 0.0183 |
| `bf_score` | 0.0089 |
| `mIoU-9` | 0.0017 |

Lý do: CI bootstrap chỉ đóng được câu hỏi *nhiễu tập đánh giá*. Nó **không tự động** đóng được câu hỏi *nhiễu giữa các lần huấn luyện* — hai nguồn phương sai độc lập (mục 0 bảng trên). CI hẹp từ bootstrap không đủ để tuyên bố một hiệu ứng "đã xác nhận" nếu độ lớn của hiệu ứng đó còn nhỏ hơn biên độ liên-seed đã đo trực tiếp.

### 7.7. Nhãn bắt buộc khi viết vào bài

Mọi CI từ việc này phải được gọi đúng tên: **"khoảng tin cậy bootstrap trên tập đánh giá 384 ảnh"**. Nó **không** phải khoảng tin cậy trên lần huấn luyện. Viết mập mờ để người đọc hiểu "chạy lại sẽ ra khoảng này" là phát biểu sai.

---

## 8. PHẦN D — Việc #3: Đếm thành phần liên thông giả trên ảnh Agriculture (0 GPU, ~30-60', sau Phần B)

### 8.1. Vì sao, và tại sao ưu tiên cao

Oracle C5b (`cong-c5b-oracle-bce-va-static-ket-qua-va-danh-gia.md` mục 4.1) đã lượng hoá: mức tăng Agriculture +2.364đ của Static so với BCE λ=0.4 gần như toàn bộ là hiệu ứng **ruột vùng** (+2.378đ ở `interior_d1`, chỉ +0.211đ ở `boundary_d1`). Điều này khớp giả thuyết đã ghi ở `phase1-run3b-lambda2-05-ket-qua-va-danh-gia.md` mục 4.1: *"cánh đồng lớn đồng nhất là nơi mảnh phân đoạn giả sinh sống"*. Việc này là phép kiểm định tính trực tiếp cho một hiệu ứng đã được lượng hoá — không còn là suy đoán.

### 8.2. Input

Mask PNG đã dump ở Phần B cho **4 cấu hình**: Baseline, BCE λ=0.4, Static (seed19), Run 3b (α_aff=0.2). Không cần Run 5/Run 4 cho việc này — giữ đúng 4 cấu hình theo kế hoạch gốc (`tong-ket-tien-do...` mục 5.1 #3).

### 8.3. Phương pháp

Với mỗi ảnh val có lớp Agriculture trong ground truth (`class_present_Agriculture = 1`, lấy từ CSV Phần B):

1. Trích binary mask `pred == Agriculture` từ PNG dump.
2. Trích binary mask `gt == Agriculture` (dựng lại từ label gốc, hoặc dùng script hiện có nếu đã có sẵn đường load label).
3. Đếm connected components (dùng `connectivity=4`, nhất quán với toàn dự án) trên cả hai mask.
4. Định nghĩa "mảnh giả": một connected component trong `pred` mà **không overlap** (hoặc overlap dưới ngưỡng nhỏ, ví dụ <5% diện tích component) với bất kỳ component nào trong `gt` — tức là vùng Agriculture mà mô hình dự đoán ra nhưng không tương ứng với một vùng Agriculture thật nào ở gần đó. Ghi rõ ngưỡng đã chọn trong output.
5. Xuất per-image: `n_components_pred`, `n_components_gt`, `n_fake_components`, tổng diện tích mảnh giả (px).
6. Gộp theo cấu hình: trung bình và tổng số mảnh giả trên toàn bộ ảnh có Agriculture.

### 8.4. Output

- `agriculture_fake_fragments.csv` — per-image, per-config
- `agriculture_fake_fragments_summary.md` — bảng gộp 4 cấu hình, đọc trực tiếp: cấu hình nào có ít mảnh giả nhất, có khớp thứ tự Δ Agriculture (+2.364đ cho Static, xem bảng oracle) hay không
- Nếu dữ liệu cho phép, 1-2 ảnh ví dụ minh hoạ trực quan (dùng được cho Figure 3 của outline paper — mục 5.6 `cvis-paper-outline.md`)

### 8.5. Không cần ngưỡng đăng ký trước cho việc này

Khác với Phần C, đây là việc mô tả/kiểm định tính, không phải phép thử giả thuyết nhị phân — báo cáo số liệu và để người đọc (Huan) diễn giải cùng với bảng oracle đã có.

---

## 9. Checklist bàn giao

- [ ] **Phần A:** thêm 3 cờ vào `eval_boundary_metrics.py`, thuần cộng thêm
- [ ] Xác minh: chạy **không** cờ cho kết quả giống hệt trước khi sửa (so byte JSON output) — mục 4.0
- [ ] **Phần B:** dump 7 checkpoint (mục 3) — mask PNG + CSV thống kê đủ + `_meta.json` cho mỗi checkpoint, **gồm 2 checkpoint Run 5 seed 86**
- [ ] **Phần C:** `tools/bootstrap_boundary_ci.py` — tái lập gộp, dò chính sách ảnh rỗng, bootstrap ghép cặp cho **6 cặp**
- [ ] **Cổng 5.1–5.8 PASS toàn bộ** — báo cáo từng cổng, không gộp; đặc biệt cổng 5.4 (Run 5) và 5.5 (độ lệch CSV↔JSON cho cả Run 4/3b/5)
- [ ] Xuất `bootstrap_ci.csv` + `bootstrap_ci.md`
- [ ] Áp dụng bảng 7.1–7.6 **nguyên văn**, ghi verdict vào tài liệu mới trong project (đặt tên gợi ý: `claude/ket-qua-bootstrap-ci-ket-luan-do-lon.md`)
- [ ] Xác nhận mask PNG dùng được cho Phần D (cổng 5.8)
- [ ] **Phần D:** chạy đếm mảnh giả Agriculture trên 4 cấu hình, xuất CSV + summary
- [ ] Báo cáo cuối: verdict cặp #6 (mục 7.5) — vì nó quyết định có nên đề nghị Huan chạy seed thứ 3 hay không
