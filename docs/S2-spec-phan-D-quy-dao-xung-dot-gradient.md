# Spec bàn giao Claude Code — Phần D: Quỹ đạo xung đột gradient trong lúc huấn luyện (Static seed 19)

**Ngày:** 13/10/2026 — viết **sau** Bước 1 (verdict N), **trước** khi chạy Phần D. Mọi ngưỡng và luật đọc ở §6 là tiền đăng ký; không sửa sau khi thấy số.
**Bản sửa 1 (13/10/2026, trước khi chạy bất cứ thứ gì):** đổi cách lấy quỹ đạo từ **callback trong vòng train** sang **lưu checkpoint theo lịch + đo offline** bằng chính `Tools/grad_conflict_probe.py` của Bước 1. Lý do: (i) lưu `state_dict` không chạm RNG, `.grad`, optimizer, BN hay đồng bộ DDP ⇒ bỏ được toàn bộ yêu cầu cách ly và rủi ro chen code vào train loop; (ii) tái dùng script đã qua 7 cổng thay vì viết logic đo thứ hai; (iii) đo lại được bất kỳ lúc nào mà không train lại. Luật đọc §6 **không đổi**.
**Repo:** `Boundary_Loss_Solution` (commit Bước 1: `8d52089`).
**Căn cứ:** `S1-spec-chan-doan-xung-dot-gradient-va-lo-gia.md` §7 (cổng Phần D); `ket-qua-buoc1-xung-dot-gradient-va-lo-gia-danh-gia.md` §5 (quyết định chạy D + 3 bổ sung); `tong-quan-huong-nghien-cuu-va-quy-trinh-xu-ly-13-10-2026.md` §10.
**Chi phí ước tính:** Phiên 1 — train ~3.5h (2×T4) + eval. Phiên 2 — đo 35 snapshot × 32 ảnh ≈ 75 phút trên 1×T4 (≈ 40 phút nếu chia 2 GPU; ước từ Bước 1: ~3.9 s/ảnh cho 3 số hạng). Lưu trữ ~35 × 47 MB ≈ 1.7 GB. Đo lại ở dry-run.

---

## 0. Câu hỏi Phần D trả lời

Bước 1 không thấy xung đột (BCE, Aff), (Region, Aff) tại điểm hội tụ (36k/40k). Nhưng cos (BCE, Aff) mức tham số của cùng run s86 nhảy từ 0.477 (36k) xuống 0.012 (40k) — ảnh chụp một mốc không ổn định. Phần D hỏi:

1. **D-Q1:** Có **cửa sổ iter** nào trong quá trình train mà (BCE, Aff) hoặc (Region, Aff) xung đột theo tiêu chí 5.1 không (xung đột đã bị "thoả hiệp" trước điểm hội tụ)?
2. **D-Q2:** Tỉ lệ độ lớn ‖G_aff‖/‖G_region‖ và ‖G_bce‖/‖G_region‖ biến đổi thế nào theo iter? (Bước 1: 0.021 và 0.093 tại 36k.)
3. **D-Q3:** Quan hệ hợp tác (Region, BCE) (Bước 1: +0.29) hình thành từ khi nào? Giả thuyết mở bàn giao §11.1 — "BCE định hình Fused Feature quanh biên trước, affinity phát huy sau" — có dấu hiệu thời gian không?

Lợi ích phụ: lấp lỗ hổng "Static s19 không có checkpoint 40000".

---

## 1. Phiên 1 — Huấn luyện (KHÔNG ĐƯỢC ĐỔI cấu hình)

- Config: đúng config của Static s19 gốc. Theo `done_static_s19_36k.json` của Bước 1 là `configs/unet_former_resnet18_static_boundary/static_boundary.yaml` (S1 §7 ghi `run3_static_boundary.yaml` — **kiểm hai tên trỏ cùng nội dung**; nếu khác, dùng file đã sinh `best_model_Static_S19.pth` và ghi `md5` config vào log).
- Seed 19; 2×T4 DDP; batch 4/GPU; AMP; 40.000 iter; validation như run gốc (10 vòng); `pos_weight` như run gốc (11.0014).
- Lưu **4 checkpoint chuẩn**: `best_miou`, `best_bfscore`, `best_bareland`, `final_iter40000`.
- Log CSV mọi vòng validation với đầy đủ chỉ số biên.
- `src/losses/*`: chỉ gọi, không sửa.

### 1.1. Snapshot theo lịch — thay đổi duy nhất trong train loop

- **Lịch (35 mốc):** `{0, 250, 500, 750, 1000}` ∪ `{1500, 2000, …, 8000}` (bước 500) ∪ `{10000, 12000, …, 40000}` (bước 2000). Dày ở đầu vì giả thuyết "thoả hiệp" đặt xung đột vào giai đoạn sớm.
- **Mốc 0** = trọng số trước bước optimizer đầu tiên. Mốc t > 0 = **sau** bước optimizer của iter t.
- **Nội dung lưu:** chỉ `model.module.state_dict()` (gồm tham số + buffer BN), kèm `{"iter": t, "seed": 19, "config_md5": …}`. Không lưu optimizer.
- **Cách lưu:** chỉ rank 0, `torch.save` ra `snapshots/iter_<t>.pth`; các rank khác không làm gì (không cần barrier — `torch.save` không gọi collective). Không gọi forward, backward hay bất kỳ hàm ngẫu nhiên nào.
- **Bắt buộc có** `snapshots/iter_36000.pth` (để so với `best_model_Static_S19.pth` @36k).
- Cuối phiên: lưu thư mục `snapshots/` thành Kaggle Dataset (hoặc output của notebook) để phiên 2 đọc.

---

## 2. Phiên 2 — Đo offline trên snapshot

### 2.1. Tập dò (chốt trước phiên 1, lưu file)

- **32 ảnh từ train split**, chọn bằng `np.random.default_rng(19).choice(N_train, 32, replace=False)` trên danh sách train đã sắp xếp theo tên. Lưu `probe_ids.txt` **trước khi khởi chạy phiên 1**. Không dùng ảnh val.
- Tiền xử lý: resize 1024×1024, ToTensor, chuẩn hoá ImageNet.
- 4 batch cố định × 8 ảnh (`default_rng(19).permutation(32)`), giống mọi snapshot.

### 2.2. Mở rộng tối thiểu cho `Tools/grad_conflict_probe.py`

Giữ nguyên toàn bộ logic đo đã qua 7 cổng (tách số hạng, trọng số hiệu dụng `w_region=1, w_bce=0.4, w_aff=0.4`, hook Fused Feature, `shared_params`, tầng khoảng cách/lớp/kích thước, `model.eval()`, fp32). Chỉ thêm:
1. `--image-list probe_ids.txt --split train`: đo trên danh sách ảnh cho trước.
2. `--snapshot-dir snapshots/`: lặp qua mọi `iter_<t>.pth`, ghi output theo t.
3. **Đầu ra Gram** (mục 2.3).

Không đổi mặc định của các chế độ cũ — chạy lại Bước 1 phải cho kết quả y hệt.

### 2.3. Đại lượng đo

**Mức tham số (qua ma trận Gram, cho bootstrap chính xác):**
- Với mỗi ảnh i và số hạng k: `a_k,i = ∂(w_k L_k,i)/∂θ_shared` (fp32, chuyển về CPU).
- Ma trận Gram 32×32 cho mọi cặp và từng số hạng: `G_k1k2[i,j] = <a_k1,i , a_k2,j>`. Có thể tính tăng dần để không giữ cả 32 gradient trong RAM; kết quả phải trùng (cổng D3).
- Cosine của tổng trên tập ảnh có vector trọng số w: `wᵀG_12w / sqrt(wᵀG_11w · wᵀG_22w)`.
- Báo cáo tại mỗi snapshot, cho 3 cặp (BCE,Aff), (Region,Aff), (Region,BCE):
  - `cos_all` — cosine của tổng 32 ảnh;
  - cos của 4 batch cố định (mô tả, giống Bước 1);
  - **CI 95%** bootstrap trên ảnh (10.000 lần, rng 19) của `cos_all`;
  - **null:** 1.000 lần chia ngẫu nhiên 32 ảnh thành hai nửa rời nhau H1, H2 (rng 19), cos(Σ_{H1} a_k1, Σ_{H2} a_k2) → khoảng 2.5–97.5% (tương tự null b ≠ b' của S1 §2.2);
  - tỉ lệ độ lớn `‖Σa_k2‖/‖Σa_k1‖`.

**Mức đặc trưng:** đúng S1 §2.3 (cosW gộp micro theo tầng khoảng cách / lớp / kích thước; bootstrap per-image; null hoán vị 20 lần trong tầng; "không đủ hỗ trợ" < 1%).

**Snapshot 0:** BoundaryHead còn khởi tạo — vẫn đo và báo cáo, **không dùng cho quyết định**.

---

## 3. Bổ sung: kiểm vùng hỗ trợ của affinity (Bước 1 §2.4)

Bước 1 thấy độ phủ hỗ trợ (cặp có Aff) ở tầng (4,8] chỉ 1.3%, trái với dự đoán ~15.3 px của tổng quan §4.7. Chạy trong dry-run, trên `best_model_Static_S19.pth` @36k và 2 ảnh val đầu tiên trong `figure_images.json`:
- bản đồ ‖∂L_aff/∂F(p)‖ trên lưới stride-4; khoảng cách lớn nhất (px full-res) tới biên GT có chuẩn > 1e-12;
- tỉ lệ cặp có hinge hoạt động: cặp khác lớp có D < margin; cặp cùng lớp có gradient khác 0;
- tách đóng góp gradient vào **tâm** vs **láng giềng**.

Mô tả, không là cổng. Ghi vào `affinity_support.md`.

---

## 4. Dry-run — 5 cổng, tất cả PASS trước phiên 1 đầy đủ

| # | Cổng | Ngưỡng |
|---|---|---|
| D1 | **Lưu snapshot không đổi huấn luyện.** Chạy 3 lần 500 iter cùng seed: OFF-a, OFF-b, ON (lưu snapshot tại 0, 250, 500) | max\|ON − OFF-a\| (loss từng iter) ≤ max\|OFF-b − OFF-a\|. Nếu OFF-a ≡ OFF-b thì yêu cầu ON ≡ OFF-a |
| D2 | **Snapshot nạp đúng.** Nạp `iter_500.pth` vào model mới; so với trọng số trong bộ nhớ của run ON tại iter 500 | Trùng bit (mọi tham số và buffer) |
| D3 | **Gram đúng.** Trên `iter_500.pth`: cos_all từ Gram = cos tính trực tiếp trên tổng gradient; Gram tăng dần = Gram đầy đủ | < 1e-6 |
| D4 | **Cổng S1 trên ảnh train.** Cổng 2 (đồng nhất thức loss), 3 (tuyến tính gradient), 5 (tất định) của S1 trên 4 ảnh tập dò, snapshot `iter_500.pth` | Như S1 (< 1e-5; < 1e-5; < 1e-6) |
| D5 | **Thời gian.** s/iter train; s/snapshot đo | Phiên 1 ≤ 10h; phiên 2 ≤ 10h. Vượt ⇒ giãn lịch §1.1 thành {0–1000 bước 250; 1000–8000 bước 1000; 8000–40000 bước 4000}, ghi lại |

Thêm (sau phiên 1, trước khi đọc quỹ đạo): **D6 — nạp đúng snapshot 36k.** Dùng `grad_conflict_probe.py` chế độ cổng 1 trên val 384 ảnh với `iter_36000.pth`: mIoU-9 phải bằng mIoU-9 của vòng validation 36000 trong log train của chính run này, dung sai 0.001.

Không đổi ngưỡng sau khi thấy số.

---

## 5. Sanity sau phiên 1

- **mIoU-9 tại iter 36000 nằm trong 0.6540 ± 0.0017** (run gốc 0.6540486; 0.0017 = \|s86 − s19\| @36k). Trượt ⇒ **DỪNG**, không diễn giải quỹ đạo. Lưu ý: AMP + cuDNN không tái lập bit-by-bit, nên run này là **một** run Static s19 mới, không phải bản sao run gốc; dung sai trên là để kiểm điều đó.
- Ghi thêm, không là cổng: mIoU-9 @40k; BF precision/recall @36k so run gốc.
- Snapshot 36k đo trên tập dò train, Bước 1 đo trên val — chỉ so **dấu** và bậc độ lớn, không so số.

---

## 6. Luật đọc kết quả — CHỐT TRƯỚC (không đổi so với bản đầu)

### 6.1. Xung đột tại một snapshot

Cặp (k1, k2) tại snapshot t là **xung đột tại t** khi đồng thời (tiêu chí 5.1 của S1, bỏ điều kiện 5 vì chỉ có một seed):
1. cận trên CI 95% của `cos_all` < 0;
2. `cos_all` < cận 2.5% của null;
3. \|cos_all\| ≥ τ_param = 0.05 (mức tham số) / τ_feat = 0.10 (mức đặc trưng, theo tầng);
4. tỉ lệ độ lớn trong [0.1, 10].

### 6.2. Cửa sổ xung đột

**Cửa sổ xung đột** = ≥ 3 snapshot **liên tiếp** (không tính snapshot 0) cùng thoả 6.1 cho cùng (mức, cặp, trục, tầng).

### 6.3. Bảng quyết định

| Mã | Điều kiện | Đọc ra | Bước tiếp |
|---|---|---|---|
| **D-G** | Có cửa sổ (BCE, Aff) ở **mức tham số**, kết thúc trước 36000 | Xung đột có thật nhưng bị thoả hiệp trong train — khớp verdict N của Bước 1 | Hướng B sống, khung "xung đột theo thời gian". Ứng viên: chiếu gradient kiểu PCGrad (Yu et al., NeurIPS 2020) **chỉ trong cửa sổ**. Bắt buộc lặp quỹ đạo seed 86 trước khi viết spec phương pháp |
| **D-RA** | Không D-G; có cửa sổ (Region, Aff) mức tham số | Affinity cản nhiệm vụ chính ở một giai đoạn | Cổng gradient kiểu Du et al. (arXiv:1812.02224) trong cửa sổ; lặp seed 86 trước |
| **D-L** | Không D-G/D-RA; có cửa sổ (BCE, Aff) ở ≥ 1 tầng khoảng cách/kích thước (mức đặc trưng, tầng đủ hỗ trợ) | Xung đột khu trú theo không gian và thời gian | Như nhánh L (tổng quan §5-L), giới hạn trong cửa sổ; lặp seed 86 trước |
| **D-dai-dang** | Có cửa sổ kéo tới ≥ 36000 | Mâu thuẫn Bước 1 | **Dừng diễn giải.** Kiểm khác biệt train vs val (tập dò) trước mọi kết luận |
| **D-N** | Không có cửa sổ nào | Không có xung đột cả trong quá trình train | **Dừng Hướng B phần phương pháp** (kịch bản S3 cho phương pháp). Quỹ đạo vào phần phân tích bài 2 |

Thứ tự khi nhiều mã cùng đạt: D-G > D-RA > D-L. Một seed ⇒ mọi kết quả dương chỉ là **ứng viên**; không viết claim trước khi có seed 86.

### 6.4. Mô tả (không dùng để quyết định)

- Đường cong ‖G_aff‖/‖G_region‖, ‖G_bce‖/‖G_region‖, ‖G_aff‖/‖G_bce‖ theo iter. Tại 36k Bước 1 đo trên val: 0.021 / 0.093 / 0.239.
- Đường cong cos (Region, BCE): iter đầu tiên CI loại 0 về phía dương.
- Giả thuyết bàn giao §11.1 (mô tả): so iter đầu tiên cos (Region, BCE) có CI > 0 với iter đầu tiên cos (Region, Aff) có CI > 0. BCE sớm hơn rõ (≥ 2 snapshot) ⇒ "phù hợp với" giả thuyết, không "chứng minh".
- Độ ổn định: độ lệch chuẩn của `cos_all` (BCE, Aff) qua 5 snapshot cuối (32000–40000), so với khoảng 0.477 → 0.012 của Bước 1.

---

## 7. Output

```
phanD/
  probe_ids.txt                     # 32 ảnh train, lưu TRƯỚC phiên 1
  gates_dryrun.json                 # D1–D5 (+ D6 sau phiên 1)
  snapshots/iter_<t>.pth            # 35 file, ~1.7 GB — lưu thành Kaggle Dataset
  traj_param.csv                    # t × cặp: cos_all, ci_lo, ci_hi, null_lo, null_hi, ratio, cos_batch_1..4
  traj_feat_strata.csv              # t × cặp × trục × tầng: Σdot, Σn1, Σn2, n_pos, n_support, n_neg, cosW, ci, null
  gram/iter_<t>.npz                 # Gram 32×32 mọi cặp/số hạng (tính lại thống kê không cần GPU)
  affinity_support.md               # mục 3
  summary_phanD.md                  # dòng đầu: config + md5, seed, chế độ eval khi đo; sanity §5; bảng 6.3; mô tả 6.4; hình
  figures/traj_cos_param.png        # 3 cặp, CI + dải null theo iter
  figures/traj_ratio.png            # 3 tỉ lệ độ lớn theo iter (thang log)
checkpoints/ best_miou, best_bfscore, best_bareland, final_iter40000
logs/ train log, val CSV 10 vòng
```

## 8. KHÔNG ĐƯỢC ĐỔI

- Config, seed, splits, tiền xử lý, `src/losses/*`.
- Train loop ngoài việc gọi `torch.save` ở rank 0 theo lịch §1.1.
- Logic đo của `grad_conflict_probe.py` (chỉ thêm tuỳ chọn ở §2.2).
- Tập dò sau khi đã lưu `probe_ids.txt`; lịch snapshot (trừ trường hợp D5 cho phép).
- Ngưỡng §4, §5, §6 sau khi thấy số. Nếu thấy ngưỡng có vẻ sai: ghi nhận xét bên cạnh, **giữ nguyên verdict**.
- Không chọn snapshot/cửa sổ "đẹp": báo cáo mọi snapshot.

## 9. Checklist bàn giao

- [ ] `probe_ids.txt` sinh và lưu trước mọi thứ khác
- [ ] Cờ `--save-snapshots <lịch>` trong script train (OFF mặc định)
- [ ] Mở rộng `grad_conflict_probe.py` theo §2.2; chạy lại 1 checkpoint Bước 1 để xác nhận output cũ không đổi
- [ ] Dry-run: D1–D5 PASS, báo cáo từng cổng; mục 3 (vùng hỗ trợ affinity)
- [ ] Phiên 1: train đầy đủ + snapshot; sanity §5; D6
- [ ] Phiên 2: đo 35 snapshot
- [ ] Áp bảng 6.3 nguyên văn; mô tả 6.4
- [ ] Gửi `summary_phanD.md`, `gates_dryrun.json`, `traj_param.csv`, `affinity_support.md`, hai hình, log
