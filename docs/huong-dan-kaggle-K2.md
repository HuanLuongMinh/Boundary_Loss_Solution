# Hướng dẫn chạy K2 trên Kaggle — Kiểm lỗ giả (oracle lấp lỗ + mẫu kiểm bằng mắt)

- Kế hoạch: `docs/K2-ke-hoach-kiem-lo-gia-va-quyet-dinh-huong-giai-phap.md`
- Script chạy (một lệnh duy nhất): `scripts/kaggle_K2.sh`
- Tool:
  - `Tools/hole_oracle_fill.py` (K2-a)
  - `Tools/hole_audit_sample.py` (K2-b)
  - `Tools/test_hole_oracle.py` (self-test)

K2 chạy **chỉ bằng CPU** và **không cần checkpoint**. Mask dự đoán được lấy từ bản dump của Bước 1 đã commit trong repo (`data/dump_parts`). Vì vậy K2 chạy được song song với Phần D (GPU) trong một notebook khác. Một lần commit sẽ chạy liền các bước:

```
self-test → giải nén dump → đối chiếu ảnh val → K2-a (oracle lấp lỗ) → K2-b (chọn mẫu + crop) → in kết quả
```

---

## Cách gọn nhất: 2 cell

Tạo notebook mới với **Accelerator: None (CPU)** và **Internet: On**.

```bash
# Cell 1
!cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
# Cell 2
!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_K2.sh
```

Chạy bằng **Save Version → Save & Run All (Commit)**. Khi thành công, dòng cuối của log là:

```
==> K2: XONG sau N phut. (K2-a rc=0, K2-b rc=0)
```

Có thể thêm tuỳ chọn vào cuối lệnh ở Cell 2:

| Tuỳ chọn | Khi nào dùng |
|---|---|
| `--dry-run 8` | Chạy thử trên 8 ảnh (khoảng 5 phút), output vào `output/hole_oracle_dryrun/`. Cổng K-a1 và K-b0 bị SKIP, số liệu không dùng được |
| `--stages "a"` hoặc `--stages "b"` | Chỉ chạy K2-a, hoặc chỉ chạy K2-b |
| `--workers N` | Số tiến trình. Mặc định bằng số CPU (Kaggle CPU có 4) |
| `--resume` | Chạy tiếp version bị ngắt (xem mục 4) |

---

## 0. Chuẩn bị trên máy bạn

1. Push code:
   ```bash
   git add Tools/hole_oracle_fill.py Tools/hole_audit_sample.py Tools/test_hole_oracle.py Tools/per_image_dump.py \
           scripts/kaggle_K2.sh docs/huong-dan-kaggle-K2.md
   git commit -m "K2: oracle lap lo co chon loc + mau kiem bang mat"
   git push origin main
   ```
   `Tools/per_image_dump.py` chỉ thay đổi một chỗ: khi `boundary_distances=()` thì bỏ qua khối Boundary IoU, nhờ đó K2 nhanh gấp khoảng 3 lần. Mọi lời gọi cũ đều truyền `(1, 2, 4)`, nên kết quả của chúng không đổi.
2. `data/dump_parts/` phải có trong repo. Thư mục này đã được commit từ Bước 1, nên không cần làm gì thêm.
3. Tuỳ chọn: chạy test trên máy (CPU, khoảng 1 phút):
   ```bash
   python Tools/test_hole_oracle.py        # 37/37 PASS
   ```

---

## 1. Add Input

| Input | Bắt buộc | Dùng để |
|---|---|---|
| Dataset **OpenEarthMap** (cùng bản đã dùng ở Bước 1) | **Có** | Đọc GT val (`label/val`) và ảnh RGB val để cắt crop |
| **Output của version Phiên 3 Bước 1** (Your Work → Notebooks → notebook CPU Phần B) | Không | Nếu có, script tự tìm `spurious_holes/fill_test.csv` và `spurious_holes/holes_static_s19_36k.csv`, rồi dùng chúng để đối chiếu **từng dòng** ở cổng K-a1 và K-b0. Nếu không có, hai cổng này đối chiếu với các con số đã ghi trong kế hoạch |

**Không cần** dataset checkpoint.

---

## 2. Script làm gì, mất bao lâu

| Bước | Việc | Thời gian (Kaggle CPU, 4 tiến trình) | Dòng log mong đợi |
|---|---|---|---|
| Cài thư viện | `pip install -r requirements.txt` | 1–2 phút | `torch ... cuda: False` |
| Self-test | `Tools/test_hole_oracle.py` | khoảng 1 phút | `[OK]    37/37 PASS` |
| Dump | Ghép, kiểm sha256, giải nén `data/dump_parts` vào `/tmp/dump_repo` | khoảng 1 phút | `val tif = 384, dump png = 384, trung khop = True` |
| **K2-a** | Đo 4 checkpoint × 384 ảnh × (trước + 4 biến thể × 2 A_fill), rồi bootstrap 10.000 lần | **khoảng 1–2 giờ** | `[static_s19_36k] xong sau .. phut — F-all lech bit: 0, K-a2 loi: 0` (lặp 4 lần), sau đó `[PASS] K-a1`, `[PASS] K-a2` |
| **K2-b** | Quần thể lỗ giả của s19, cổng K-b0, chốt mẫu, rồi tạo khoảng 146 crop | khoảng 10 phút | `[PASS] K-b0: 8836 lo gia`, `Da chot mau: 146 muc (20 lap an)` |

Ở lần chạy đầu, GT 1024 được lưu cache vào `/tmp/k2_gt1024`. Bước này đọc 384 file tif, mất vài phút.

Cổng K-a1 hoặc K-a2 FAIL **không** dừng K2-b, vì hai phần độc lập với nhau. Tuy vậy, dòng cuối sẽ báo `rc` khác 0, và notebook kết thúc với trạng thái lỗi để bạn để ý.

---

## 3. Output (`Boundary_Loss_Solution/output/hole_oracle/`)

### K2-a: đọc `oracle_fill.md` trước

| File | Nội dung |
|---|---|
| `oracle_fill.md` | Bảng **Cổng**; mục **§4.2** gồm dòng in đậm `Don bay lo gia ton tai: CO/KHONG`, bảng hai điều kiện cho từng seed Static, và bảng kiểm dự đoán "mất recall nằm ở F-thật"; tiếp theo là bảng Δ kèm CI cho từng checkpoint và từng A_fill |
| `oracle_fill.csv` | Mọi số: `label, a_fill, variant, quantity, before, after, delta, ci_lo, ci_hi, p_delta_lt_0, ci_excludes_0, n_holes_filled, hole_px, changed_px` |
| `gates_oracle.json` | Cổng K-a1, K-a2, K-a0 và kết quả `lever_42`, dạng máy đọc được |
| `per_image/<nhan>__<bien the>_a<A>.csv` | Số liệu thô từng ảnh, dùng được cho phân tích sau mà không phải đo lại |
| `done_<nhan>.json` | Đánh dấu checkpoint đã đo xong (để resume), số ảnh đã kiểm bit, thời gian |

Bốn biến thể (chọn lỗ bằng GT, nên đây là **trần**, không phải phương pháp):

| Mã trong file | Kế hoạch | Lấp lỗ nào (diện tích ≤ A_fill) |
|---|---|---|
| `F-all` | F-all | Mọi lỗ. Tái lập B2 |
| `F-gia` | F-giả | `frac_gt_c` ≥ 0,5. Oracle chính |
| `F-gia95` | F-giả95 | `frac_gt_c` ≥ 0,95 |
| `F-that` | F-thật | `frac_gt_c` < 0,5 |

Các cổng:

| Cổng | Kiểm gì | Chặn |
|---|---|---|
| **K-a1** | F-all của Static s19 với A_fill 64 tái lập `fill_test.md`: Δprecision 0,01304 và Δrecall −0,01633, sai số < 1e-4. Nếu có `fill_test.csv` thì đối chiếu thêm **mọi** dòng trùng (nhãn, A_fill, đại lượng). Ngoài ra, mask F-all phải **trùng bit** với `spurious_holes.fill_small_holes`: kiểm trên mọi ảnh của s19 và 24 ảnh đầu của mỗi nhãn còn lại | Có |
| **K-a2** | F-giả ∩ F-thật = ∅ và F-giả ∪ F-thật = F-all. Kiểm cả số lỗ và khoá (lớp, hole_id) trên từng ảnh. Thêm điều kiện F-giả95 ⊆ F-giả | Có |
| K-a0 | mIoU-9 "trước" khớp `expected_miou9` của Bước 1, sai số 1e-3 | Không (chỉ thông tin) |

### K2-b: tải `audit_cham.zip` về để chấm

| File | Ai được xem |
|---|---|
| `audit_cham.zip` = `audit_crops/H001.png…` + `audit_crops/HUONG_DAN_CHAM.md` + `audit_sample_rating.csv` (trống) | **Người chấm** |
| `audit_sample.csv`: khoá mẫu (ảnh, lớp, bbox, `frac_gt_c`, tầng, `duplicate_of`) | **KHÔNG mở cho tới khi chấm xong** (§2.1) |
| `audit_population.csv`: N quần thể của 18 ô (dùng làm trọng số cho p_B) | Không cần xem |
| `gates_audit.json`: cổng K-b0 | Không cần xem |

- **K-b0** là cổng chặn. Quần thể lỗ giả (`is_spurious = 1`, A_min = 64) của Static s19@36k phải có **đúng 8.836 lỗ**, chia theo nhóm lớp: Rangeland 3.458, Developed 2.767, Tree 1.081, Building 938, Agriculture 320, Khác 272. Nếu có `holes_static_s19_36k.csv` thì tập lỗ phải trùng với file đó. Khi K-b0 FAIL, script **không** rút mẫu, vì nhiều khả năng đang dùng sai bản dataset.
- Mẫu được chốt **trước** khi sinh crop. Nếu `audit_sample.csv` đã tồn tại (ví dụ khi chạy `--resume`), script giữ nguyên mẫu đó và chỉ vẽ những crop còn thiếu. `audit_sample_rating.csv` **không bao giờ** bị ghi đè.
- Mỗi crop có 2 ô. Ô trái là RGB gốc. Ô phải là RGB cộng viền đỏ quanh lỗ, với cửa sổ vuông bằng 3 lần bbox (tối thiểu 128 px full-res), phóng lên 400 px. Tiêu đề chỉ ghi mã mục và lớp c.

### Cách chấm (tóm tắt, bản đầy đủ nằm trong `HUONG_DAN_CHAM.md` của gói zip)

Mở từng `H###.png` và trả lời câu hỏi: *"Bên trong viền đỏ, trên ảnh thực, có vật hoặc lớp phủ khác lớp c chiếm phần lớn vùng không?"* Sau đó điền vào `audit_sample_rating.csv`:
- cột `nhan`: **A** = mô hình sai (vùng đồng nhất với lớp c; bóng đổ trên nền cùng lớp vẫn tính là A), **B** = nhãn thô (có vật thật khác lớp), **C** = không phân định được;
- cột `vat`: chỉ điền khi chọn B (cây / nhà / nước / đất trống / đường / khác);
- cột `ghi_chu`: tuỳ ý.

Khoảng 20 mục là bản lặp ẩn. Hãy chấm từng mục độc lập, không đối chiếu với các mục khác. Khi xong, gửi lại `audit_sample_rating.csv`. Bước tính p_B, p_C, κ (§2.4) và áp §4.1, §4.3 làm sau.

---

## 4. Bị ngắt giữa chừng: resume

K2-a ghi `done_<nhan>.json` sau khi đo xong mỗi checkpoint, nên chạy lại sẽ bỏ qua các checkpoint đã xong.

1. Mở notebook → **Add Input** → Your Work → Notebooks → chọn **version bị ngắt** của chính notebook này.
2. Cell 2: `!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_K2.sh --resume`

Script chép `output/hole_oracle` của version trước vào, rồi chạy tiếp. Mẫu K2-b đã chốt và file chấm vẫn được giữ nguyên.

---

## 5. Chạy trên máy (không qua Kaggle)

```bash
python Tools/dump_archive.py join --parts-dir data/dump_parts --dest /tmp/dump_repo   # in DUMP_ROOT=...
python Tools/hole_oracle_fill.py measure --dump-root /tmp/dump_repo/dump --data-root <OpenEarthMap> --workers 4
python Tools/hole_audit_sample.py --dump-root /tmp/dump_repo/dump --data-root <OpenEarthMap> --workers 4
python Tools/hole_oracle_fill.py report        # chỉ tổng hợp lại từ per_image/ (vd đổi --n-boot)
```

---

## Phụ lục — Lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `khong tim thay dataset OpenEarthMap (images/val)` | Chưa Add Input dataset OpenEarthMap |
| `danh sach anh dump khac anh val` | Đang gắn nhầm bản dataset (tập val khác 384 ảnh của Bước 1) |
| `khong co data/dump_parts/manifest.json` | Clone sai repo hoặc sai nhánh. Dump phải nằm trong repo |
| **K-a1 FAIL** với `abs_err` nhỏ (khoảng 1e-4) | Kiểm `gates_oracle.json`: nếu chỉ lệch do làm tròn 5 chữ số của kế hoạch, hãy Add Input output Phiên 3 Bước 1 để đối chiếu bằng `fill_test.csv` gốc |
| **K-a1 FAIL** với `F-all lech bit > 0` | Định nghĩa lấp đã lệch khỏi B2. **Không dùng số liệu.** Gửi `gates_oracle.json` để kiểm |
| **K-a2 FAIL** | Lỗi phân loại lỗ (không nên xảy ra). Gửi `gates_oracle.json` và `done_*.json` |
| K-a0 `WARN` | mIoU-9 "trước" lệch Bước 1 quá 1e-3. Không chặn, nhưng nên kiểm lại bản dataset |
| **K-b0 FAIL** (số lỗ lệch) | Sai bản dataset/GT hoặc sai dump. Script không rút mẫu. Chỉ dùng `--allow-count-mismatch` (khi gọi tool trực tiếp) nếu biết chắc lý do |
| Hết 12 giờ (rất khó xảy ra) | Dùng `--resume` |
| Chậm hơn nhiều so với bảng ở mục 2 | Kiểm `--workers`. Notebook CPU của Kaggle có 4 lõi |
