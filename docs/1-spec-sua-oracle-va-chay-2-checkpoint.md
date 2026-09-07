# Spec bàn giao Claude Code — SỬA cổng kiểm oracle-d và chạy nốt 2 checkpoint

**Ngày:** 7/9/2026
**Repo:** `Boundary_Loss_Solution`
**Chi phí:** sửa code ~15 phút + **~57 phút CPU** (2 checkpoint), hoặc ~85 phút nếu chạy thêm Run 4. **0 GPU, 0 quota.**
**Ưu tiên:** cao — đây là **cổng thứ hai chặn Run 6**, và là phép thử quyết định ứng viên (B) sống hay chết.

---

## 0. Tình trạng

Oracle-d đã chạy **thành công trên baseline** (kết quả: `oracle-d-ket-qua-baseline-va-danh-gia.md`). Hai checkpoint còn lại **fail ở cổng 5.1**:

```
[bce_lambda04] CONG 5.1 FAIL: none.mIoU-9=0.6566 lech 0.0383
               so voi expected=0.6183 (nguong 0.001). DUNG — moi so oracle vo gia tri.
```

**Cổng làm đúng việc của nó, nhưng bắt nhầm thủ phạm. Run oracle chạy ĐÚNG — chỉ hằng số đối chiếu sai.**

---

## 1. Chẩn đoán — đã xác minh, không cần điều tra lại

`0.6183` là **mIoU-8** của BCE λ=0.4 (đã bỏ Background), không phải mIoU-9. Nó bị lấy nhầm từ bảng "các quy ước mIoU" thay vì bảng mIoU-9.

Chứng minh bằng quan hệ `mIoU-9 = (8·mIoU-8 + IoU_Background)/9`:

```
BCE λ=0.4 : (8 × 0.6183    + 0.9634   ) / 9 = 0.65664    ≈ 0.6566     ✓ khop so oracle tinh ra
Baseline  : (8 × 0.6168605 + 0.9609492) / 9 = 0.65509258 = 0.6550926  ✓ khop tuyet doi
```

Độ lệch quan sát: BCE 0.0383, baseline 0.0382 — **bằng nhau**, vì cùng do Background kéo lên. Không phải trùng hợp.

**Kết luận: không có gì sai với checkpoint, dataset, hay pipeline eval. Sửa hằng số, chạy lại.**

---

## 2. Hằng số ĐÚNG cho cổng 5.1

Nguồn chuẩn: `doi-chieu-du-lieu-goc-va-hang-so-tham-chieu.md` **mục 3.0** (mới thêm 7/9). **Không lấy số từ mục 3.2** — mục đó là bảng bốn quy ước mIoU khác nhau và chính là cái bẫy vừa sập.

```python
# Hang so cong kiem eval — mIoU-9, KHONG phai mIoU-8.
# Nguon: doi-chieu-du-lieu-goc-va-hang-so-tham-chieu.md muc 3.0
EXPECTED_MIOU9 = {
    "baseline":      (40000, 0.6550926),
    "bce_lambda02":  (40000, 0.6490),
    "bce_lambda04":  (40000, 0.6566),   # <── SUA: truoc day la 0.6183 (mIoU-8)
    "static":        (36000, 0.6540486),  # <── LUU Y: iter 36000, KHONG phai 40000
    "run3b":         (40000, 0.6442),
    "run4":          (40000, 0.6491),
}
TOL_MIOU = 0.001    # KHONG duoc noi
```

### 2.1. Hai chỗ dễ vấp tiếp theo

- **Static là checkpoint iter 36000, không phải 40000.** `best_model.pth` của Run 3 dừng ở vòng 9. Nếu cổng Static đang dùng `0.6151` (mIoU-8 của Static) thì nó sẽ fail y hệt với độ lệch ~0.0389 — **sửa cả hai cùng lúc**.
- Đừng nhầm 0.6540486 (JSON @36000) với 0.6538 (@40000). Cả hai đều lọt dung sai 0.001, nhưng **ghi đúng mốc iter trong log**.

### 2.2. KHÔNG nới dung sai

`0.001` là đúng. Lỗi nằm ở hằng số. Nới ngưỡng vừa che lỗi này vừa vô hiệu hoá cổng cho lần sau.

---

## 3. Ba yêu cầu siết cổng — bắt buộc, đây là lần thứ hai dự án dính lỗi nhãn

Lần đầu: hoán đổi Water/Agriculture trong `phase1-run2-...` (phát hiện 4/9). Cùng loại lỗi: **số đúng, nhãn sai**.

### 3.1. Tên biến tường minh + thông báo lỗi tự chẩn đoán

```python
computed_miou9 = ...
computed_miou8 = ...
iter_expected, expected_miou9 = EXPECTED_MIOU9[model_key]

if abs(computed_miou9 - expected_miou9) > TOL_MIOU:
    raise SystemExit(
        f"[{model_key}] CONG 5.1 FAIL\n"
        f"  computed mIoU-9   = {computed_miou9:.7f}\n"
        f"  computed mIoU-8   = {computed_miou8:.7f}   <- neu EXPECTED trung so nay, "
        f"ban dang so nham QUY UOC (xem doi-chieu muc 3.0)\n"
        f"  EXPECTED_MIOU9    = {expected_miou9:.7f}  @ iter {iter_expected}\n"
        f"  lech              = {abs(computed_miou9-expected_miou9):.7f} (nguong {TOL_MIOU})"
    )
```

Nếu thông báo lỗi hôm qua có in `computed mIoU-8 = 0.6183`, thủ phạm đã lộ ngay lập tức.

### 3.2. Cổng vector per-class, không chỉ cổng scalar

Cổng scalar **không** bắt được lỗi hoán đổi lớp. Thêm:

```python
EXPECTED_PER_CLASS = {
  "baseline": {"Background":0.9609492,"Bareland":0.3646590,"Rangeland":0.5253651,
               "Developed":0.5354177,"Road":0.6156210,"Tree":0.6819640,
               "Water":0.7278831,"Agriculture":0.7263622,"Building":0.7576119},
  # bce_lambda04 va static: lay tu JSON hau ky tuong ung, dung TEN LOP lam khoa
}
# assert tung lop, dung sai 0.001, va bao cao TAT CA lop lech chu khong dung o lop dau
```

Khoá phải là **tên lớp**, không phải chỉ số. Nếu JSON gốc chỉ có mảng thuần, ánh xạ qua `CLASS_NAMES` và **in ra bảng tên–giá trị** trong log để người đọc kiểm được bằng mắt.

### 3.3. Tự kiểm quy ước

```python
assert abs(computed_miou9 - (8*computed_miou8 + per_class["Background"])/9) < 1e-9, \
    "Quan he mIoU-9/mIoU-8 khong thoa — kiem lai cach tinh macro."
```

Nếu ai đó lại lấy nhầm hằng số sang quy ước khác, độ lệch sẽ đúng bằng khoảng cách quan sát được và thông báo nói thẳng nguyên nhân.

---

## 4. Checkpoint phải chạy

| # | Model key | Checkpoint | iter | Ưu tiên | ~Thời gian |
|---|---|---|---|---|---|
| 1 | `static` | Run 3 `best_model.pth` | **36000** | **bắt buộc** | ~28' |
| 2 | `bce_lambda04` | Run 2b `best_model.pth` | 40000 | **bắt buộc** | ~28' |
| 3 | `run4` | Run 4 `final_iter40000.pth` | 40000 | nên có | ~28' |

Baseline **đã xong**, không chạy lại.

Lý do nên thêm Run 4: nó là cấu hình có **Water cao nhất toàn bảng** (0.7430) trong khi Static có Water thấp nhất (0.6991). Nó là điểm đối chứng tự nhiên cho đúng câu hỏi mà mục 5 đặt ra.

Giữ nguyên mọi biến thể oracle đã dùng cho baseline: `none`, `boundary_d{0,1,2,4,8}`, `interior_d{1,2,4}`, `all` — để bảng so được theo hàng.

---

## 5. Câu hỏi mà hai checkpoint này trả lời — CHỐT TRƯỚC KHI CÓ SỐ

Đại lượng quyết định: **mức sụt Water 2.75đ của Static so với BCE λ=0.4 rơi vào đâu.**

| Mức sụt Water rơi vào | Chẩn đoán được ủng hộ | Hệ quả |
|---|---|---|
| **`O_b`** — ruột vùng | **(B) xung đột gradient** | Run 6 (C) mất phần lớn cơ sở; ưu tiên (B) projection head |
| **`Gap`** — dải biên | **(C) teo vùng lấy mẫu** | Run 6 (C) đúng hướng, chạy |

Và tiêu chí huỷ (B), đã chốt từ 6/9 ở tổng kết mục 5.2, **giữ nguyên văn**:

> *Nếu `O_b(Static) ≈ O_b(BCE λ=0.4)` thì chẩn đoán xung đột gradient sai và **(B) bị huỷ**, tiết kiệm 3.5h GPU cộng thời gian code.*

**Bối cảnh từ kết quả baseline (đã có):** trần biên d=1 của Water chỉ **+5.54đ**. Mức sụt 2.75đ bằng **50% toàn bộ trần biên** của lớp đó — lớn bất thường cho một cơ chế thuần biên, gợi ý thiệt hại lan vào ruột vùng, tức nghiêng về **(B)**. Đây là **lập luận về độ lớn, không phải chứng minh** (oracle chặn trên mức *lợi* khi sửa hoàn hảo, không chặn trên mức *hại* khi làm hỏng). Hai checkpoint này **đo trực tiếp** điều đó.

**Không sửa tiêu chí sau khi thấy số.**

---

## 6. Output

- `oracle_results_static.json`, `oracle_results_bce_lambda04.json` (+ `_run4.json` nếu chạy), cùng schema với file baseline
- `oracle_table_<key>.md` cho mỗi checkpoint, cùng định dạng Bảng 1 / Bảng 2
- **Một bảng gộp mới** `oracle_compare.md`: mỗi hàng một checkpoint, các cột `mIoU-9 none`, `boundary_d1`, `boundary_d2`, `interior_d1`, `P_band@d1`, `E_band@d1`, và **per-class Water: none / boundary_d1 / interior_d1** — đây là bảng dùng để đọc mục 5
- Log đầy đủ của cả ba cổng ở mục 3, PASS/FAIL từng cái

---

## 7. KHÔNG ĐƯỢC ĐỔI

- Đường tính oracle, danh sách biến thể, cách dựng dải `boundary_d` / `interior_d` — chỉ sửa **hằng số và cổng kiểm**
- Kết quả baseline đã có — không chạy lại, không ghi đè
- Dung sai 0.001
- Val split 384 ảnh, `seed = 19`
- Không bật GPU cho notebook này

---

## 8. Checklist bàn giao

- [ ] Thay `EXPECTED_MIOU9` theo mục 2; kiểm cả `static` (0.6540486 @ **36000**) lẫn `bce_lambda04` (0.6566)
- [ ] Thông báo lỗi in cả `computed mIoU-9` và `computed mIoU-8` kèm nhãn (mục 3.1)
- [ ] Thêm cổng vector per-class theo **tên lớp** (mục 3.2)
- [ ] Thêm assert quan hệ mIoU-9/mIoU-8 (mục 3.3)
- [ ] Chạy lại `static` và `bce_lambda04`; thêm `run4` nếu thời gian cho phép
- [ ] Xuất `oracle_compare.md` theo mục 6
- [ ] Áp dụng bảng mục 5 **nguyên văn**, báo cáo `O_b` vs `Gap` cho Water
- [ ] Báo kết quả để hợp với cổng `r_edge` theo quy tắc gộp 4 ô (`oracle-d-ket-qua-baseline-va-danh-gia.md` mục 7) — quyết Run 6 chạy hay huỷ
