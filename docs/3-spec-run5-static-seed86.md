# Spec bàn giao Claude Code — Run 5: Lặp lại Static (α_aff = 0.4) ở seed 86

**Ngày:** 7/9/2026
**Repo:** `Boundary_Loss_Solution`
**Chi phí:** ~3.5h GPU (2×T4), vừa một phiên Kaggle.
**Chạy song song với:** notebook CPU bootstrap (`spec-per-image-bootstrap-eval-...`) và Run 6 — (C) sửa vùng lấy mẫu affinity. Ba việc độc lập tài nguyên. Điều kiện an toàn: **ngưỡng của cả ba đã chốt trước khi thấy bất kỳ con số nào**.

---

## 0. Run này trả lời đúng một câu hỏi

**σ giữa các lần huấn luyện là bao nhiêu?**

Toàn bộ hệ ngưỡng của dự án đang dùng σ = 0.149 px, đo **giữa các vòng validate trong một run** — tức phương sai của việc *chọn checkpoint nào*, không phải của việc *train lại*. Run 4 (7/9) làm lộ vấn đề: Run 3 và Run 4 kết thúc với ba số hạng loss lệch ≤ 0.85% nhưng `pred→gt` lệch **0.765 px**.

Nếu σ thật ở cỡ đó thì hai con số trung tâm của bài — `gt→pred` **−0.357**, `recall` **+0.0094** — chưa đứng vững, **và** verdict nhánh 4 của Run 4 cũng không đứng vững. Không phép đo nào khác trong dự án trả lời được câu này.

**Đây là run hiệu chỉnh thước đo.** Mọi run sau nó, kể cả Run 6 (C), sẽ được đọc bằng ngưỡng mà nó tạo ra.

### 0.1. Vì sao chỉ lặp arm Static, chưa lặp BCE λ=0.4

Đây là **thiết kế tuần tự có chủ đích**, không phải cắt xén:

- Nếu `|Static₈₆ − Static₁₉|` **lớn** (> 0.25 px) ⇒ claim độ lớn chết ngay, **không cần** chạy BCE ở seed 86. Tiết kiệm 3.5h.
- Nếu **nhỏ** ⇒ lúc đó mới đáng chi 3.5h cho BCE λ=0.4 seed 86, để có `Δ₈₆` — một phép tái lập độc lập của chính hiệu số trong Abstract.

Ghi rõ hạn chế hiện tại để không quên: với một arm, `Var(Δ) = Var(Static) + Var(BCE) − 2Cov` chỉ ước lượng được số hạng đầu; số hạng thứ hai đang phải **giả định** bằng nó. Giả định đó chưa được kiểm, và nếu run này cho kết quả tốt thì nó trở thành việc phải làm tiếp, không phải việc bỏ qua.

---

## 1. Cấu hình — clone Run 3, đổi đúng một dòng

```yaml
# configs/research_boundary/phase1_fixed_split/run5_static_seed86.yaml
# Ke thua NGUYEN config Run 3 (static_boundary), doi dung SEED.

TRAIN:
  SEED: 86                      # <── BIEN DUY NHAT DOI so voi Run 3 (was 19)
  MAX_ITERS: 40000              # GIU NGUYEN

BOUNDARY_LOSS:
  USE_BCE: true                 # GIU NGUYEN
  USE_AFFINITY: true            # GIU NGUYEN
  DYNAMIC_WEIGHTS: false        # GIU NGUYEN (tinh)
  ALPHA: 0.4                    # GIU NGUYEN
  LAMBDA1_STATIC: 1.0           # GIU NGUYEN
  LAMBDA2_STATIC: 1.0           # GIU NGUYEN
  WINDOW_K: 5                   # GIU NGUYEN
  DISTANCE: cosine              # GIU NGUYEN
  MARGIN: 1.0                   # GIU NGUYEN
  CONNECTIVITY: 4               # GIU NGUYEN
  DILATION_RADIUS: 0            # GIU NGUYEN
  POS_WEIGHT_MAX: 20.0          # GIU NGUYEN

OUTPUT:
  WORK_DIR: work_dirs/phase1/run5_static_seed86
```

**Seed 86 được chọn tuỳ ý, TRƯỚC khi có bất kỳ kết quả nào.** Ghi câu này vào `summary.txt` và vào phần Reproducibility của bài. Không được thử nhiều seed rồi lấy cái đẹp — đó là p-hacking, và nó phá hỏng đúng thứ run này sinh ra để bảo vệ.

### 1.1. Splits — điểm dễ sai nhất của run này

`create_splits.py` và `build_clean_val_split.py` đã sinh splits với `--seed 19`. **KHÔNG sinh lại splits.** Train split và val split (384 ảnh) phải **giống hệt** Run 3 — nếu val set đổi thì run này đo lẫn nhiễu val set với nhiễu huấn luyện, và mất luôn ý nghĩa.

`SEED: 86` chỉ được phép ảnh hưởng: khởi tạo trọng số, thứ tự lấy batch, lấy mẫu augmentation. **Không** ảnh hưởng việc chia dữ liệu.

Kiểm tra bắt buộc ở dry-run: hash danh sách file val của run này phải khớp hash của Run 3.

---

## 2. Checkpoint & log — theo đặc tả Run 3b/Run 4, không rút gọn

- **4 checkpoint** + `checkpoint_index.json`: `best_miou.pth`, `best_bfscore.pth`, `best_bareland.pth`, **`final_iter40000.pth`**.
- **`benchmark_results.csv` 10 hàng đủ cột** — per-class kèm tên lớp, ASD **hai chiều**, `bf_precision`/`bf_recall`, per-class BIoU 3 ngưỡng. Đây là điều Run 3 **không có**, và là lý do phép so sánh chính của Run 4 bị ép về mốc 36000.
- **Boundary metrics đầy đủ cả 10 vòng** nếu một vòng validate < ~5 phút (Run 4 làm được, tốn thêm ~12 phút tổng).
- `summary.txt`: khối 4-checkpoint + khối mean±std 4 vòng cuối **ddof = 1**.
- **Sau khi train xong**, chạy `eval_boundary_metrics.py --dump-preds --dump-per-image-stats` (spec bootstrap) trên `best_miou.pth` (nếu ≠ 36000 thì thêm checkpoint gần 36000 nhất) **và** `final_iter40000.pth`.

**Vì sao cần cả 36000 và 40000:** Static seed 19 chỉ tồn tại tại 36000. Run này có cả hai mốc ⇒ so được ở **cùng 36000** (đối chứng trực tiếp) *và* ở 40000 (mốc chuẩn của mọi run khác). Đây cũng là lần đầu dự án vá được lỗ hổng checkpoint của Run 3, với chi phí bằng 0.

---

## 3. Cổng dry-run — `MAX_ITERS=100`

1. `grep "SEED: 86"` trên config trả kết quả; log in ra seed 86.
2. **Hash danh sách file val khớp Run 3** (mục 1.1). Lệch ⇒ **DỪNG**.
3. Đồng nhất thức `l_total = l_region + α(λ₁·l_bce + λ₂·l_affinity)`, sai số < 1e-5, mọi vòng.
4. `alpha_bce_effective = 0.4` và `alpha_affinity_effective = 0.4` ở **mọi** vòng (tĩnh, không lịch trình).
5. `pos_weight` in ra nằm trong **10.8–11.1** (4 run trước: 11.0392 / 10.8799 / 11.0014). Ngoài khoảng ⇒ nghi ngờ splits đã đổi, quay lại cổng 2.
6. 4 checkpoint + `checkpoint_index.json` ghi ra đúng, load lại được.
7. CSV đủ cột; đo thời gian một vòng validate để quyết boundary metrics đủ 10 vòng hay 4 vòng cuối.
8. `nvidia-smi` ra **2×T4**, không phải P100.

---

## 4. Quy tắc đọc kết quả — CHỐT TRƯỚC KHI CÓ SỐ

### 4.1. Bảng quyết định chính — biên độ giữa hai lần train

Đại lượng: `|Static₈₆ − Static₁₉|` tại **cùng iter 36000**, trên **`asd_gt_to_pred`** (chiều mang claim trung tâm). Giá trị seed 19: **4.0321**.

| Nhánh | Biên độ | Kết luận | Hệ quả bắt buộc |
|---|---|---|---|
| **I — σ nhỏ** | ≤ 0.10 px | Ngưỡng 0.149 đang dùng là hợp lệ | Mọi verdict cũ đứng, gồm nhánh 4 của Run 4. Chạy BCE λ=0.4 seed 86 (3.5h) để có Δ₈₆ — tái lập trực tiếp con số Abstract |
| **II — σ trung bình** | 0.10 – 0.25 px | Ngưỡng cũ quá lỏng | **Thay ngưỡng của mọi run sau** bằng giá trị mới. Claim −0.357 vẫn sống nhưng **bắt buộc in kèm biên độ**. Vẫn nên chạy BCE seed 86 |
| **III — σ lớn** | > 0.25 px | Ngưỡng cũ vô hiệu | Không claim được **độ lớn** của `gt→pred`. Δ ra khỏi Abstract. **Mọi verdict dựa trên ngưỡng 0.149 — gồm nhánh 4 của Run 4 — ghi lại là "không kết luận được"**. Không chạy BCE seed 86; bài lùi về phát biểu định tính về dấu |

### 4.2. Bảng phụ — báo cáo biên độ cho mọi đại lượng, không chỉ đại lượng chính

Ghi `|seed86 − seed19|` tại cùng 36000 cho: `asd_pred_to_gt`, `asd_gt_to_pred`, `asd`, `bf_precision`, `bf_recall`, `bf_score`, `mIoU-9`, `mIoU-8`, `mIoU loại Bareland`, `BIoU-8@{1,2,4}`, và per-class IoU **đủ 9 lớp**.

Ba con số trong bảng này quan trọng riêng:

- **σ(mIoU-9) giữa hai lần train** — thay thế con số 0.0052 (giữa các vòng) đang dùng làm sàn 0.52đ trong giao thức giảm nhiễu mục 5.6. Nếu nó lớn hơn nhiều, toàn bộ mục 5.6 phải viết lại và số run cần cho một claim mIoU tăng theo bình phương.
- **Biên độ Bareland** — lớp này có σ nội tại 0.0735 giữa các vòng. Nếu biên độ giữa hai lần train cũng ở cỡ đó, giả thuyết "Bareland là bộ phát nhiễu chung" được xác nhận ở một chiều độc lập, và việc loại nó khỏi mIoU có căn cứ mạnh hơn.
- **Biên độ Water** — chẩn đoán teo tín hiệu ở stride-4 (mục 3.6b) dựa vào mức sụt Water 2.75đ. Nếu Water dao động ≥ 2đ giữa hai seed thì chẩn đoán đó mất chỗ dựa, và **kết quả của Run 6 (C) cũng phải đọc lại**.

### 4.3. Không được làm

- **Không** so `Static₈₆@40000` với `Static₁₉@36000` rồi kết luận — đó đúng là artifact chọn checkpoint đã làm đảo dấu BIoU giữa Run 2 và Run 3. So cùng 36000 là phép chính; 40000 chỉ dùng khi cả hai đều có.
- **Không** đổi ngưỡng ở mục 4.1 sau khi thấy số. Nếu ngưỡng có vẻ sai, ghi nhận xét bên cạnh và giữ verdict.
- **Không** kết luận "hai run giống nhau ⇒ σ nhỏ" từ n = 2 như thể đó là một ước lượng σ. Với hai điểm, cái thu được là **biên độ**, không phải độ lệch chuẩn. Viết đúng chữ trong bài: *"hai lần huấn luyện độc lập lệch X px"*, không phải *"σ = X"*.

---

## 5. KHÔNG ĐƯỢC ĐỔI

- Splits (train + val 384 ảnh), sinh với `--seed 19` — **không sinh lại**
- `MAX_ITERS = 40000`, batch size, optimizer, LR schedule, augmentation policy
- `ALPHA = 0.4`, `LAMBDA1_STATIC = 1.0`, `LAMBDA2_STATIC = 1.0`
- `WINDOW_K = 5`, `DISTANCE = cosine`, `MARGIN = 1.0`, `CONNECTIVITY = 4`, `DILATION_RADIUS = 0`
- Cách downsample nhãn cho affinity: **vẫn nearest** về stride-4 — việc sửa nó là Run 6 (C), **không gộp vào đây**
- `src/losses/*` — không đụng vào
- Mọi file của Track A-D và Track ResNet18-baseline

---

## 6. Checklist bàn giao

- [ ] Config `run5_static_seed86.yaml` — chỉ đổi `SEED` và `WORK_DIR`
- [ ] Xác nhận **không** sinh lại splits; hash val list khớp Run 3
- [ ] Dry-run: **8/8 cổng PASS**, báo cáo từng cổng
- [ ] Xác nhận GPU 2×T4 rồi mới submit
- [ ] Train ~3.5h
- [ ] 4 checkpoint + `checkpoint_index.json`; CSV 10 hàng đủ cột; `summary.txt` ddof=1
- [ ] Chạy `--dump-preds --dump-per-image-stats` trên checkpoint 36000 và 40000
- [ ] Ghi kết quả vào tài liệu mới trong project, áp dụng bảng 4.1 **nguyên văn**, kèm bảng biên độ đầy đủ mục 4.2
