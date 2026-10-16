# Hướng dẫn chạy P0 + R0 trên Kaggle — JSON hậu kỳ run Static chạy lại + P-CRF

- Spec: `docs/S4-spec-P0-R0-ban-giao-claude-code.md`
- Script chạy (một lệnh duy nhất): `scripts/kaggle_P0_R0.sh`
- Tool:
  - `Tools/p0_rerun_eval.py` (P0)
  - `Tools/r0_pcrf.py` (R0)
  - `Tools/refine_summary.py` (tổng hợp `summary_P0_R0.md`)
  - `Tools/refine_common.py` (phần dùng chung)
  - `Tools/test_refine.py` (self-test)

Một lần commit chạy liền mọi bước, không dừng giữa chừng:

```
cài thư viện + pydensecrf → self-test → DRY-RUN (cổng chặn, ước tính thời gian)
  → P0 (fingerprint → forward 2 snapshot → JSON + per-image + masks → cổng P0-g1, g2 → so CSV)
  → R0 (g1 → g2, g3 → lưới 27 tổ hợp trên 32 ảnh train → chọn (g4) → val 384 ảnh → bootstrap, §3.6, §3.8)
  → summary_P0_R0.md
```

---

## Cách gọn nhất: 2 cell

Tạo notebook mới với **Accelerator: GPU T4** và **Internet: On**. GPU chỉ dùng cho bước forward. CRF luôn chạy trên CPU (4 lõi của notebook).

```bash
# Cell 1
!cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
# Cell 2
!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_P0_R0.sh
```

Chạy bằng **Save Version → Save & Run All (Commit)**. Khi thành công, dòng cuối của log là:

```
==> P0+R0: XONG sau N phut. (P0 rc=0, R0 rc=0)
```

### Nên làm: kiểm nhanh trước bằng `--dryrun`

```bash
# Cell 2 (lần đầu)
!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_P0_R0.sh --dryrun
```

Chế độ này mất khoảng 10–15 phút. Script cài đặt, chạy self-test, rồi chạy P0 và R0 trên 4 ảnh val và 2 ảnh train, đủ mọi bước kể cả bootstrap, bảng và hình. Kết quả ghi vào `output/refine_dryrun/`, xong thì **dừng**. Đọc `output/refine_dryrun/summary_P0_R0.md`: mục R0 có dòng **"Uoc tinh chay that: luoi ~X ph, val ~Y ph"**, tính từ thời gian CRF thật đo được.

Nếu dry-run PASS, chạy hoàn chỉnh. Có thể bỏ dry-run nhờ `--skip-dryrun`, hoặc giữ nguyên lệnh mặc định: dry-run lặp lại chỉ tốn khoảng 10 phút và vẫn đóng vai trò cổng chặn.

### Tuỳ chọn (thêm vào cuối lệnh ở Cell 2)

| Tuỳ chọn | Khi nào dùng |
|---|---|
| `--dryrun` | Chỉ kiểm nhanh (cài đặt, self-test, dry-run P0 + R0) rồi dừng |
| `--skip-dryrun` | Bỏ dry-run, vì version trước đã PASS |
| `--stages "p0"` / `--stages "r0"` | Chỉ chạy P0, hoặc chỉ chạy R0. Mặc định chạy cả hai. P0 và R0 độc lập nhau |
| `--resume` | Chạy tiếp version bị ngắt (xem mục 4) |
| `--workers N` | Số tiến trình CRF. Mặc định bằng số CPU |
| `--dry-n N` | Số ảnh val của dry-run. Mặc định 4 |
| `--input-mode gt_prob` | **Chế độ dự phòng §3.2**: unary tính từ nhãn với `gt_prob` = 0,7. Chỉ dùng khi chế độ softmax không chạy nổi. Chế độ đã dùng được ghi trong mọi output |

---

## 0. Chuẩn bị trên máy bạn

1. Push code:
   ```bash
   git add Tools/refine_common.py Tools/p0_rerun_eval.py Tools/r0_pcrf.py Tools/refine_summary.py Tools/test_refine.py \
           scripts/kaggle_P0_R0.sh docs/huong-dan-kaggle-P0-R0.md docs/S4-spec-P0-R0-ban-giao-claude-code.md
   git commit -m "S4: P0 (JSON hau ky Static chay lai) + R0 (P-CRF)"
   git push origin main
   ```
   Không sửa `src/losses/*`, `src/models/*`, công cụ đánh giá hay checkpoint.
2. Tuỳ chọn: chạy test trên máy (CPU, khoảng 1 phút):
   ```bash
   python Tools/test_refine.py          # 27/27 PASS (test CRF that SKIP neu may chua co pydensecrf)
   ```

---

## 1. Add Input

| Input | Bắt buộc | Dùng để |
|---|---|---|
| Dataset **OpenEarthMap** (`aletbm/global-land-cover-mapping-openearthmap`) | **Có** | `images/{train,val}`, `label/{train,val}` |
| Dataset **checkpoint** (`huanluong/checkpoint-sumary`) | **Có** cho R0 | `best_model_Static_S19.pth`, `best_miou_Static_s86.pth`, `best_model_baseline.pth`, `best_model_BCE_04.pth` |
| **Dataset `Boundary_PhanD`** (5 file để phẳng), **hoặc** output notebook Phần D (`notebook5ab020caa1`, version đã train xong) | **Có** | `output/phanD/snapshots/iter_36000.pth` và `iter_40000.pth` (P0 + nhãn `static_s19r_36k` của R0), `probe_ids.txt` (32 ảnh train để chọn tham số), `figure_images.json`, `logs/benchmark_results.csv` (§2.4) |

Dataset `Boundary_PhanD` để phẳng, gồm 5 file trong cùng một thư mục: `iter_36000.pth`, `iter_40000.pth`, `probe_ids.txt`, `figure_images.json`, `benchmark_results.csv`. Hai file snapshot phải là **file gốc**, không được lưu lại, vì cổng P0-g2 so fingerprint tuyệt đối. Script cũng nhận cấu trúc output gốc của Phần D (`snapshots/`, `logs/`).

Script tự tìm mọi đường dẫn trong `/kaggle/input` và in ra hai dòng `SNAP = …`, `PHD = …` ở đầu log. Nếu cần, ghi đè bằng biến môi trường: `CK=... DATA=... SNAP=<thu muc chua 2 snapshot> PHD=<thu muc chua probe_ids.txt>`.

Nếu thiếu `figure_images.json`, script chọn lại từ GT val theo quy tắc Bước 1 (`Tools/select_figure_images.py`). Nếu thiếu `benchmark_results.csv`, §2.4 chỉ in giá trị hậu kỳ.

---

## 2. Script làm gì, mất bao lâu

| Bước | Việc | Thời gian (GPU T4 + 4 CPU) | Dòng log mong đợi |
|---|---|---|---|
| Cài thư viện | `requirements.txt` + `pydensecrf`, thử theo thứ tự (xem mục 2.1) | 2–6 phút | `[OK]    lan N OK`, `[OK]    pydensecrf: ...` |
| Self-test | `Tools/test_refine.py --require-crf` | khoảng 1 phút | `[OK]    34/34 PASS` |
| **Dry-run** | P0 + R0 trên 4 ảnh val, 2 ảnh train, bootstrap 200 lần | 10–15 phút | `[OK]    DRY-RUN PASS` |
| **P0** | Fingerprint → forward 2 snapshot × 384 ảnh → JSON + per-image + masks | khoảng 15–25 phút | `[PASS] P0-g2: 2/2 snapshot khop fingerprint`, `[PASS] P0-g1: ...` |
| R0-g1 | Forward val của 5 checkpoint, kiểm mIoU-9 trước CRF | khoảng 10 phút | `[static_s19_36k] mIoU-9 truoc CRF 0.65405 (ky vong 0.6540486)` |
| R0-g2, g3 | CRF chỉ unary bằng argmax; chạy lại thì trùng tuyệt đối | 1–2 phút | `[PASS] R0-g2 ...`, `[PASS] R0-g3 ...` |
| **R0 lưới** | 5 checkpoint × 32 ảnh train × 27 tổ hợp | **khoảng 1–2 giờ** (xem ước tính của dry-run) | `[static_s19_36k] luoi 32/32 anh (...s, CRF x.xxs/lan)` |
| R0 chọn | Tổ hợp tốt nhất cho từng checkpoint + R0-g4 | dưới 1 phút | `[PASS] R0-g4: ... id val bi dung: 0` |
| **R0 val** | 5 checkpoint × 384 ảnh với tổ hợp đã chọn | **khoảng 40–80 phút** | `[bce04_40k] val 384/384 anh` |
| R0 báo cáo | Bootstrap 10.000 lần, §3.6, §3.8, hình | 2–5 phút | `§3.6 (mo ta): R-...` |

Tổng thời gian khoảng 2,5–4 giờ, nằm trong giới hạn 12 giờ của một phiên.

- **Cổng chặn**: dry-run FAIL thì không chạy thật. P0-g2 FAIL (fingerprint lệch) thì P0 dừng trước khi forward. R0-g1 FAIL thì R0 dừng **trước khi chạy CRF**. R0-g2, g3, g4, g5 FAIL thì R0 dừng.
- P0 FAIL **không** dừng R0, vì hai phần độc lập. Tuy vậy, dòng cuối sẽ báo `rc` khác 0.
- Thiếu `pydensecrf` (cả 4 lần cài đều lỗi) thì script **dừng và báo**, đúng §3.3. Script không tự viết CRF thay thế. Khi đó vẫn chạy được P0 riêng bằng `--stages "p0"`.

### 2.1. Cài `pydensecrf`: 4 lần thử, giữ log mọi lần

Kaggle hiện chạy **Python 3.13**. Repo `lucasb-eyer` ghim `Cython==0.29.36` trong `pyproject.toml` (commit e22f579: "Temp fix for cython 3.0.0 breaking changes"), mà Cython 0.29 không hỗ trợ Python 3.13. Vì thế cách cài thông thường lỗi.

| Lần | Cách cài | Ghi chú |
|---|---|---|
| 1 | `pip install -v git+https://github.com/lucasb-eyer/pydensecrf.git` | Nguồn gợi ý của §3.3. Trên Python 3.13: lỗi (Cython 0.29.36) |
| 2 | Như lần 1, nhưng Cython < 3 và `--no-build-isolation` | Trên Python 3.13: lỗi (Cython 0.29.37) |
| **3** | **Cùng mã nguồn lucasb-eyer**, ghim commit `2723c7f`, **không sửa dòng nào**. File `.cpp` được sinh bằng Cython 3 (hỗ trợ Python 3.13) với chỉ thị tương thích ngược `legacy_implicit_noexcept=True`, `language_level=2` (đúng ngữ nghĩa Cython 0.29). Sau đó gỡ Cython để `setup.py` gốc build từ `.cpp` (nhánh có sẵn của chính `setup.py`) | Log ghi commit, `git status` (sạch), phiên bản Cython, các file `.cpp` sinh ra và output trình biên dịch |
| 4 | `pip install -v pydensecrf2==1.1` (fork của adaptivemotorcontrollab) | Cũng ghim Cython 0.29.36, nên trên Python 3.13 gần như chắc chắn lỗi. Chỉ giữ làm bằng chứng |

Hành vi của bản build được kiểm bằng R0-g2 (chỉ unary bằng argmax), R0-g3 (tất định) và R0-g5 (dấu vân tay số liệu).

- Mỗi lần thử ghi **full log** vào `r0_crf/install_logs/attempt<N>_*.log`, kể cả khi lần sau thành công. Nhờ vậy có bằng chứng cụ thể vì sao không dùng được bản gốc.
- `install_logs/install_summary.json` ghi lần nào lỗi, cách nào được dùng và dòng `pip freeze`. Nguồn và phiên bản CRF cũng được ghi vào `run_log.txt`, `gates_r0.json` (mục `crf_backend`, `crf_install`) và dòng đầu của `crf_val.md`.
- Console in các dòng lỗi chính của mỗi lần thử thất bại.

---

## 3. Output (`Boundary_Loss_Solution/output/refine/`)

Đọc `summary_P0_R0.md` trước, sau đó đến `r0_crf/crf_val.md`.

```
refine/
  summary_P0_R0.md
  p0_rerun/
    static_s19r_36k/  static_s19r_40k/   eval.json, per_image_stats.csv (+ _meta.json), masks/*.png
    p0_compare_csv.md (+ .csv)            §2.4: hậu kỳ vs CSV val trong train, |Δ| từng chỉ số
    gates_p0.json                         P0-g1, P0-g2 (+ md5 config, đối chiếu per-image ↔ eval.json)
    run_log.txt
  r0_crf/
    crf_grid_train.csv                    27 tổ hợp × 5 checkpoint (+ hàng combo = -1 = trước CRF), ASD hai chiều TB
    crf_selected.json                     tổ hợp chọn cho từng checkpoint + danh sách 32 ảnh dùng để chọn
    crf_val.csv, crf_val.md               bảng §3.6 + mọi chỉ số §1 + mô tả §3.8
    per_class.csv                         Δ BIoU@2 từng lớp + nhóm stuff / thing (CI)
    changed_pixels.csv                    tỉ lệ pixel đổi nhãn; phân theo dải ≤2 / 2–8 / >8 px tới biên GT
    small_regions.csv                     thành phần dự đoán ≤ 700 px bị xoá sau CRF (từng lớp + ALL)
    figures/<nhan>__<anh>.png             6 ảnh × 2 checkpoint quyết định: trước CRF / sau CRF / GT
    masks/<nhan>/*.png                    mask sau CRF (384 ảnh × 5)
    grid/, val/                           số liệu thô từng ảnh (tổng hợp lại không cần chạy lại)
    gates/R0-g{1..5}.json, gates_r0.json  cổng, mã §3.6, chế độ đầu vào, nguồn + phiên bản pydensecrf, log cài đặt
    crf_fingerprint.json, crf_fingerprint/  dấu vân tay số liệu CRF (R0-g5): 3 mask + sha256
    install_logs/                         full log mọi lần cài pydensecrf + install_summary.json
    run_log.txt                           chế độ đầu vào (softmax / gt_prob), pydensecrf version + commit, git_commit
```

### Cổng

| Cổng | Kiểm gì | Ngưỡng |
|---|---|---|
| **P0-g1** | mIoU-9 hậu kỳ so với CSV val trong train | 0,6404 (@36k), 0,6548 (@40k), dung sai 0,001 |
| **P0-g2** | Fingerprint snapshot (kích thước + sha1 của 4 MB đầu và 4 MB cuối) | khớp tuyệt đối |
| **R0-g1** | mIoU-9 trước CRF so với bảng checkpoint (`static_s19r_36k` dùng 0,6404) | dung sai 0,001 |
| **R0-g2** | CRF với `bi_w` = `pos_w` = 0 cho kết quả bằng argmax softmax | ≥ 99,99% pixel trên 4 ảnh |
| **R0-g3** | Chạy lại 2 ảnh với cùng tham số | trùng tuyệt đối |
| **R0-g4** | Tập chọn đúng 32 id của `probe_ids.txt`, không có id val nào. Kèm điều kiện `probe_ids` sinh lại (rng 19) khớp file | khớp |
| **R0-g5** | Dấu vân tay số liệu CRF. Chạy CRF trên 3 ảnh val đầu (theo tên) với tham số cố định (`bi_xy_std` 80, `bi_rgb_std` 5, `bi_w` 4, `pos` 3/3, 10 vòng lặp). Unary dựng **từ GT** (0,7 / 0,3÷8), nên không phụ thuộc model hay GPU. Lưu mask và sha256. Lần đầu: `RECORDED`. Lần chạy thật so với lần dry-run cùng notebook; resume so với lần trước | sha256 trùng tuyệt đối với mọi tham chiếu |

### Luật §3.6: mọi mã đều là **mô tả** (K2 §5.2)

Luật chỉ áp lên hai checkpoint quyết định, `static_s19_36k` và `static_s86_36k`:

- (i) Δ ASD `pred→gt` < 0, CI loại 0;
- (ii) Δ BIoU@d2 > 0, CI loại 0;
- (iii) Δ `gt→pred` ≤ 0,534 px;
- (iv) cận dưới CI của Δ BF-recall > −0,0094.

Mã: **R-có** (cả hai seed đạt cả 4 điều kiện), **R-không** (ở mỗi seed, trượt (i) hoặc trượt (ii)), **R-đánh-đổi** (cả hai seed đạt (i)–(ii) nhưng trượt (iii) hoặc (iv)). Trường hợp một seed đạt (i)–(ii) còn seed kia không thì luật không phủ, script ghi **KHÔNG PHÂN LOẠI ĐƯỢC**.

Hai quy ước được chốt khi lập trình. Cả hai đều ghi trong `crf_val.md`:
- **BIoU@d2 của (ii)** là `biou_8_d2`, tức trung bình 8 lớp không gồm Background, theo quy ước của `bootstrap_boundary_ci` dùng ở mọi CI của dự án. Bảng còn có thêm dòng **độ nhạy** với `biou_9_d2` (9 lớp, đúng định nghĩa cột `biou_d2` của CSV train). Dòng này không dùng để quyết định.
- **Nhóm** ở §3.8.1 là trung bình BIoU@2 của các lớp trong nhóm. Thành phần nhỏ ở §3.8.3 được tách theo 8-liên thông theo lớp; "bị xoá" nghĩa là sau CRF không còn pixel nào của lớp đó trong thành phần.

---

## 4. Bị ngắt giữa chừng: resume

P0 ghi `eval.json` cho từng snapshot. R0 ghi `grid/done_<nhan>.json` và `val/done_<nhan>.json` cho từng checkpoint. Khi chạy lại, những phần đã xong được bỏ qua. Riêng g1, g2, g3 luôn chạy lại vì chúng nhanh.

1. Mở notebook → **Add Input** → Your Work → Notebooks → chọn **version bị ngắt** của chính notebook này.
2. Cell 2: `!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_P0_R0.sh --resume --skip-dryrun`

---

## 5. Chạy trên máy (không qua Kaggle)

```bash
python Tools/p0_rerun_eval.py --snapshot-dir <phanD>/snapshots --data-root <OpenEarthMap> \
    --bench-csv <phanD>/logs/benchmark_results.csv --output-dir output/refine/p0_rerun
python Tools/r0_pcrf.py run --ckpt-dir <checkpoint> --snapshot-dir <phanD>/snapshots --data-root <OpenEarthMap> \
    --probe-ids <phanD>/probe_ids.txt --figure-images <phanD>/figure_images.json --output-dir output/refine/r0_crf --workers 4
python Tools/r0_pcrf.py report ...            # chi tong hop lai tu val/ (vd doi --n-boot)
python Tools/refine_summary.py --root output/refine
```

---

## Phụ lục — Lỗi thường gặp

| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `khong tim thay iter_36000.pth + iter_40000.pth` | Chưa Add Input dataset `Boundary_PhanD` hoặc output Phần D, hoặc hai snapshot không nằm cùng một thư mục |
| `cai pydensecrf KHONG duoc (4 lan ...)` | Kiểm Internet: On. Gửi thư mục `output/refine/r0_crf/install_logs/` (đủ các file `attempt*.log`, đã có output trình biên dịch nhờ `pip -v`). **Không** thay bằng CRF tự viết (§3.3). Có thể chạy riêng P0 bằng `--stages "p0"` |
| **P0-g2 FAIL** | Snapshot không đúng file của Phần D (`notebook5ab020caa1`). Kiểm version của output đã Add Input |
| **P0-g1 FAIL** | mIoU-9 hậu kỳ lệch CSV quá 0,001. Kiểm bản dataset và config (md5 nằm trong `gates_p0.json`, mục `P0-info`) |
| **R0-g1 FAIL** | Checkpoint hoặc dataset sai bản. Xem `r0_crf/gates/R0-g1.json`. R0 đã dừng trước khi chạy CRF |
| **R0-g2 / g3 FAIL** | Bản cài `pydensecrf` hành xử lạ. Gửi `gates/R0-g2.json`, `R0-g3.json` và `run_log.txt` |
| **R0-g5 FAIL** | Bản CRF (hoặc dataset) đã khác lần chạy tham chiếu, nên số liệu không so sánh được với nhau. Xem `gates/R0-g5.json` (`compared_with`) và `install_logs/install_summary.json`. Không ép chạy tiếp |
| **R0-g4 FAIL** | `probe_ids.txt` không phải bản của Phần D, hoặc có ảnh val lẫn vào. Không chạy tiếp |
| Hết RAM (Killed) | Giảm `--workers`. Mỗi tiến trình CRF giữ một softmax 1024² × 9 (khoảng 38 MB) cộng với lưới permutohedral |
| Chậm hơn ước tính nhiều | Xem `CRF x.xxs/lan` trong log lưới. Nếu quá lâu, dùng `--resume` ở phiên sau. Không đổi lưới hay số vòng lặp (§4) |
