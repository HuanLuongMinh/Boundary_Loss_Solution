# Spec bàn giao Claude Code — P0 (JSON hậu kỳ run Static chạy lại) + R0 (P-CRF)

**Ngày:** 16/10/2026 · **Trạng thái:** **BÀN GIAO** — chạy được ngay, không phụ thuộc quyết định R1/R3.
**Repo:** `Boundary_Loss_Solution` (commit hiện hành; ghi `git_commit` vào mọi output).
**Căn cứ:** `S3-spec-nhap-tinh-chinh-hau-ky-R0-R1.md` §2–§3; `K2-ke-hoach-kiem-lo-gia-va-quyet-dinh-huong-giai-phap.md` §5.1 (luật P-CRF, chốt 15/10) và §5.2 (bản sửa 2: P-CRF là baseline không học, **không** là cổng dừng).
**Tài nguyên:** Kaggle, CPU 4 nhân đủ (GPU nếu có thì dùng cho bước forward, không bắt buộc). Ước tính: P0 ~1h, R0 ~2–3h. P0 và R0 độc lập, chạy song song được.

---

## 0. Bối cảnh ngắn

- Bài 2 chuyển sang **tinh chỉnh biên hậu kỳ**. R0 đo baseline không học (DenseCRF) trên đúng giao thức đánh giá của dự án; R1 (pilot có học kiểu SegFix) sẽ có spec riêng.
- Run Static s19 train lại ở Phần D là lần lặp Static thứ ba. Chỉ số biên của nó hiện chỉ có từ CSV val trong train (4 chữ số); P0 xuất JSON hậu kỳ để xác nhận bảng biên độ n = 3.

## 1. Dữ liệu và công cụ dùng chung

- **Dataset:** `/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap`, layout `images/{train,val}`, `label/{train,val}` (số ít `label/`). Val 384 ảnh.
- **Tiền xử lý:** resize 1024×1024, ToTensor, chuẩn hoá ImageNet — đúng như mọi run.
- **Đánh giá:** `Tools/per_image_dump.compute_per_image_stats` + `Tools/bootstrap_boundary_ci`, quy ước micro, bootstrap ghép cặp per-image **10.000 lần, rng 19**. Chỉ số: mIoU-9, mIoU-8, BF precision / recall / F (θ = 2 px), BIoU@{1, 2, 4}, ASD `pred→gt` / `gt→pred`.
- **Eval mode:** `model.eval()`, fp32.
- **Không sửa:** `src/losses/*`, `src/models/*`, công cụ đánh giá, checkpoint.

## 2. P0 — JSON hậu kỳ cho run Static chạy lại

### 2.1. Đầu vào

| Nhãn | File | Nguồn |
|---|---|---|
| `static_s19r_36k` | `snapshots/iter_36000.pth` | output notebook Phần D (`notebook5ab020caa1`), thư mục `output/phanD/snapshots/` |
| `static_s19r_40k` | `snapshots/iter_40000.pth` | như trên |

Fingerprint kỳ vọng (kích thước + sha1 của 4 MB đầu nối 4 MB cuối):
- `iter_36000.pth`: `47002106:cf74b5adfb7a29371263482fc74e3768dfb67150`
- `iter_40000.pth`: `47002106:d363a0a36dac08c2b74df13769dbbad49fb4df7d`

Snapshot chỉ chứa `state_dict` (không optimizer). Config: `configs/unet_former_resnet18_static_boundary/static_boundary.yaml` (md5 `d7985f64d90402dda025a7bd702b719e`).

### 2.2. Việc

1. Kiểm fingerprint; lệch ⇒ dừng.
2. Forward 384 ảnh val, dump mask PNG (cùng định dạng dump của Bước 1 / K2), xuất JSON eval đầy đủ + per-image stats như các run khác.

### 2.3. Cổng

| # | Cổng | Ngưỡng |
|---|---|---|
| P0-g1 | mIoU-9 hậu kỳ khớp CSV val trong train | 0.6404 (@36k), 0.6548 (@40k); dung sai 0.001 |
| P0-g2 | Fingerprint snapshot | khớp tuyệt đối |

### 2.4. Mô tả (không cổng)

So BF-P, BF-R, BF-F, BIoU@{1,2,4}, ASD hai chiều hậu kỳ với CSV val (`PhanD/VERSION 1/output/phanD/logs/benchmark_results.csv`, hàng iter 36000 và 40000); báo cáo chênh lệch tuyệt đối từng chỉ số.

## 3. R0 — P-CRF (thực hiện nguyên văn K2 §5.1)

### 3.1. Checkpoint

| Vai trò | Nhãn | Checkpoint |
|---|---|---|
| **Quyết định của luật** | `static_s19_36k` | `best_model_Static_S19.pth` (iter 36000) |
| **Quyết định của luật** | `static_s86_36k` | Static s86 `best_miou` @36000 |
| Mô tả | `baseline_40k` | Baseline @40000 |
| Mô tả | `bce04_40k` | BCE λ0.4 @40000 |
| Mô tả | `static_s19r_36k` | snapshot Phần D @36000 (§2.1) |

Đường dẫn: cùng bảng checkpoint đã dùng ở Bước 1 và K2-a (dataset `huanluong/checkpoint-sumary`). Kiểm mIoU-9 "trước CRF" khớp bảng `CHECKPOINTS.expected_miou9` (dung sai 0.001) trước khi chạy CRF.

### 3.2. Đầu vào CRF

- **Chế độ chính:** softmax của checkpoint (forward lại, fp32), độ phân giải 1024×1024. Unary = −log(p), kẹp p ≥ 1e-8.
- **Dự phòng** (chỉ khi chế độ chính không khả thi về tài nguyên): nhãn đã dump, unary từ nhãn với `gt_prob` = 0.7. Ghi rõ chế độ đã dùng trong mọi output.
- Ảnh RGB cho kernel song phương: ảnh gốc 1024×1024, uint8, **chưa** chuẩn hoá.

### 3.3. Thư viện

`pydensecrf` (cài đặt của Krähenbühl & Koltun, NeurIPS 2011). Gợi ý cài: `pip install git+https://github.com/lucasb-eyer/pydensecrf.git`. Ghi phiên bản/commit. Cài không được ⇒ dừng và báo, **không** tự viết CRF thay thế.

### 3.4. Chọn siêu tham số — chỉ trên train

- **Tập chọn:** 32 ảnh train trong `probe_ids.txt` của Phần D (`output/phanD/probe_ids.txt`).
- **Lưới chốt trước (27 tổ hợp):** `bi_xy_std` ∈ {40, 80, 121}, `bi_rgb_std` ∈ {3, 5, 10}, `bi_w` ∈ {3, 4, 5}. Cố định: `pos_xy_std` = 3, `pos_w` = 3, 10 vòng lặp.
- **Tiêu chí:** nhỏ nhất của ASD hai chiều trung bình = (`pred→gt` + `gt→pred`) / 2, tính micro trên 32 ảnh bằng `compute_per_image_stats`.
- **Chọn riêng cho từng checkpoint** (mỗi checkpoint lấy tổ hợp tốt nhất của chính nó trên tập chọn). Hoà ⇒ lấy tổ hợp xuất hiện trước theo thứ tự lưới (`bi_xy_std`, `bi_rgb_std`, `bi_w` tăng dần).
- **Tuyệt đối không** dùng ảnh val để chọn.

### 3.5. Đo trên val

384 ảnh val, tổ hợp đã chọn, Δ = sau CRF − trước, bootstrap ghép cặp per-image (§1). Báo cáo mọi chỉ số §1 cho cả 5 checkpoint.

### 3.6. Luật đọc — nguyên văn K2 §5.1, chỉ trên hai checkpoint quyết định

| Mã | Điều kiện (cả hai seed Static) |
|---|---|
| **R-có** | (i) Δ `pred→gt` < 0 với CI loại 0; (ii) Δ BIoU@d2 > 0 với CI loại 0; (iii) Δ `gt→pred` không dương quá 0.534 px; (iv) Δ recall có cận dưới CI > −0.0094 |
| **R-không** | Không đạt (i) hoặc (ii) ở cả hai seed |
| **R-đánh-đổi** | Đạt (i)–(ii) nhưng trượt (iii) hoặc (iv) |

Ghi rõ trong kết quả: theo K2 §5.2, mọi mã của R0 là **mô tả**; R0 là baseline, không dừng hướng tinh chỉnh.

### 3.7. Cổng

| # | Cổng | Ngưỡng |
|---|---|---|
| R0-g1 | mIoU-9 "trước CRF" khớp bảng checkpoint | dung sai 0.001 |
| R0-g2 | CRF với `bi_w` = 0 và `pos_w` = 0 (chỉ unary) trả về argmax softmax | trùng ≥ 99.99% pixel trên 4 ảnh |
| R0-g3 | Tất định: chạy lại 2 ảnh với cùng tham số | trùng tuyệt đối |
| R0-g4 | Không rò val: log danh sách ảnh dùng để chọn tham số | đúng 32 id của `probe_ids.txt`, không id val nào |

### 3.8. Mô tả kèm (không dùng quyết định)

1. Δ BIoU@2 theo từng lớp; gộp nhóm stuff (Rangeland, Agriculture, Bareland, Water) và thing (Building, Road); Developed, Tree báo cáo riêng.
2. Tỉ lệ pixel đổi nhãn sau CRF; tỉ lệ đó nằm trong dải biên GT ≤ 2 px / 2–8 px / > 8 px.
3. Thành phần dự đoán ≤ 700 px: tỉ lệ bị xoá sau CRF.
4. So với B2 (lấp mù lỗ ≤ 64 px, Static s19@36k: ΔBF-P +0.0130, ΔBF-R −0.0163, Δ `gt→pred` +0.693 px): CRF nằm ở đâu trên trục đánh đổi precision ↔ recall.
5. Bảng kết quả lưới trên tập chọn (27 tổ hợp × 5 checkpoint).

## 4. KHÔNG ĐƯỢC ĐỔI

- Checkpoint, tiền xử lý, công cụ đánh giá, quy ước micro, rng 19.
- Lưới siêu tham số, tiêu chí chọn, tập chọn.
- Luật §3.6 sau khi thấy số. Thấy ngưỡng có vẻ sai: ghi nhận xét bên cạnh, giữ nguyên mã.
- Không chọn ảnh/lớp "đẹp"; báo cáo đủ.

## 5. Output

```
refine/
  p0_rerun/
    static_s19r_36k/  static_s19r_40k/     # JSON eval đầy đủ + per_image + masks/
    p0_compare_csv.md                       # §2.4
    gates_p0.json
  r0_crf/
    crf_grid_train.csv                      # 27 × 5, ASD hai chiều trung bình
    crf_selected.json                       # tổ hợp chọn cho từng checkpoint
    crf_val.csv, crf_val.md                 # bảng §3.6 + mô tả §3.8
    per_class.csv, changed_pixels.csv, small_regions.csv
    figures/                                # 3 ảnh trong figure_images.json: trước / sau CRF / GT
    gates_r0.json
    run_log.txt                             # chế độ đầu vào (softmax / gt_prob), phiên bản pydensecrf, git_commit
  summary_P0_R0.md
```

## 6. Checklist

- [ ] P0: fingerprint → forward → JSON + per-image → cổng P0-g1, g2 → so CSV
- [ ] R0: cài `pydensecrf` → cổng R0-g1 → R0-g2, g3 → lưới trên 32 ảnh train (R0-g4) → đo val → áp §3.6 → mô tả §3.8
- [ ] Gửi `summary_P0_R0.md`, `gates_p0.json`, `gates_r0.json`, `crf_val.md`, `crf_selected.json`, log
