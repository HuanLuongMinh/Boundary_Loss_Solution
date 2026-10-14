# Hướng dẫn chạy Phần D trên Kaggle — Quỹ đạo xung đột gradient (Static s19)

- Spec: `docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md`
- Script chạy (một lệnh duy nhất): `scripts/kaggle_phanD.sh`

Phần D gồm hai phiên trong spec: Phiên 1 train, Phiên 2 đo. Ở đây cả hai được gộp vào **một lần commit Kaggle**. Script tự chạy theo thứ tự sau, không dừng giữa chừng:

```
prep → dryrun (cổng chặn) → train (Phiên 1) → post (sanity §5 + D6) → measure (Phiên 2) → summarize
```

---

## Cách gọn nhất: 2 cell

```bash
# Cell 1
!cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
# Cell 2
!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_phanD.sh
```

Chạy bằng **Save Version → Save & Run All (Commit)**. Khi thành công, dòng cuối của log là:

```
==> PHAN D: XONG sau N phut. Verdict: D-... | dien giai duoc: True/False
```

Có thể thêm tuỳ chọn vào cuối dòng ở Cell 2:

| Tuỳ chọn | Khi nào dùng |
|---|---|
| `--dryrun-only` | Chỉ chạy prep + dry-run (khoảng 45 phút) để xem `gates_dryrun.json` và thời gian D5 trước khi chạy thật |
| `--skip-dryrun` | Version trước đã dry-run PASS. Dùng cùng `--resume` |
| `--resume` | Chạy tiếp từ output của một version bị ngắt (xem mục 4) |
| `--stages "measure summarize"` | Chỉ chạy các stage được liệt kê. `prep` luôn chạy |

---

## 0. Chuẩn bị trên máy bạn

1. Push code:
   ```bash
   git add scripts/kaggle_phanD.sh src/utils/phanD_snapshots.py src/train_static_boundary.py \
           src/train_static_boundary_weighted.py Tools/grad_conflict_probe.py Tools/grad_conflict_common.py \
           Tools/phanD_probe.py Tools/phanD_traj.py Tools/phanD_dryrun.py Tools/affinity_support.py \
           Tools/test_phanD.py docs/huong-dan-kaggle-phanD.md docs/S2-spec-phan-D-quy-dao-xung-dot-gradient.md
   git commit -m "Phan D: snapshot theo lich + do quy dao xung dot gradient"
   git push origin main
   ```
2. **Không clone với `--depth`.** Cổng R0 cần commit `8d52089` của Bước 1 để so code cũ với code mới.
3. Dùng lại **dataset checkpoint** của lần chạy Bước 1. Phần D chỉ cần hai file trong đó:
   - `best_model_Static_S19.pth`: dùng cho R0 và §3;
   - `summary_static_s19.txt`: script kiểm có dòng `pos_weight used ... 11.0014`.
4. Tuỳ chọn: chạy test trên máy, khoảng 1 phút, chỉ cần CPU:
   ```bash
   python Tools/test_phanD.py
   python Tools/test_grad_conflict_probe.py
   ```

## 1. Tạo notebook

1. **Create → New Notebook**, đặt tên ví dụ `phanD-traj`.
2. **Settings:** Accelerator **GPU T4 x2** (bắt buộc 2 GPU vì train dùng DDP, batch 4/GPU), Internet **On**.
3. **Add Input:** dataset OpenEarthMap (bản đã dùng lúc train) và dataset checkpoint.
4. Tạo 2 cell như trên, rồi **Save Version → Save & Run All (Commit)**.

Ước tính thời gian: tổng khoảng 5,5–6 giờ, dưới giới hạn 12 giờ của Kaggle.

| Stage | Làm gì | Thời gian | Dòng log mong đợi |
|---|---|---|---|
| prep | Cài thư viện, kiểm 2×T4, kiểm checkpoint/summary, md5 config, sinh `probe_ids.txt` (32 ảnh train) và `figure_images.json` | 3–5 phút | `[OK] probe_ids.txt: ...` |
| dryrun | R0, D0, D1, D2, D3, D4, D5, §3 | 40–50 phút | `DRY-RUN: tat ca cong PASS — tiep tuc Phien 1` |
| train | Phiên 1: 40.000 iter, lưu snapshot theo lịch | khoảng 3,5 giờ | `[Val round 10/10, iter 40000]`, `[OK] snapshot 35/35` |
| post | D6 + sanity §5 | khoảng 10 phút | `Sanity §5: PASS ...; D6: PASS; dien giai duoc: CO` |
| measure | Phiên 2: đo 35 snapshot × 32 ảnh trên 2 GPU song song | 40–75 phút | `[OK] do xong 35 snapshot` |
| summarize | Tạo bảng 6.3, mô tả 6.4 và hình | 1–3 phút | `VERDICT: ...` |

## 2. Dry-run: các cổng chặn

**Hai cờ cố định cho mọi lệnh train** (dry-run và Phiên 1), cả hai đều mặc định OFF trong script train:

- `--seed-augment`: albumentations 2.x **bỏ qua** seed toàn cục (`random`, `np.random`, `torch.manual_seed`). Vì vậy hai lần chạy cùng seed 19 vẫn crop/flip khác nhau; lần chạy Kaggle đầu tiên cho thấy W-a và W-b lệch tới 0.69. Cờ này seed transform của từng worker DataLoader từ `torch.initial_seed()` của worker đó. Kết quả là cùng SEED thì ra cùng augmentation. Chính sách augmentation không đổi. Run Static s19 gốc **không** có cờ này, nên run Phần D vẫn là một run s19 mới, đúng như spec §5 đã lưu ý.
- `--pos-weight 11.0014`: ghim pos_weight bằng giá trị của run gốc (spec §1). Bước ước lượng 100 batch vẫn chạy như cũ để giữ nguyên vị trí luồng dữ liệu. Giá trị ước lượng chỉ được ghi log (`Phan D: --pos-weight GHIM ...`) và lưu vào `gates/D0.json` để tham khảo.


Chỉ cần một cổng FAIL là script dừng **trước** Phiên 1, nên không tốn quota cho lần train 3,5 giờ.

| Cổng | Kiểm gì | Ngưỡng | File |
|---|---|---|---|
| **R0** | Bản mở rộng `grad_conflict_probe.py` không làm đổi kết quả chế độ cũ. Chạy `measure`+`summarize` trên `static_s19_36k`, 4 ảnh val, bằng code commit `8d52089` và bằng code mới | sai khác ≤ 1e-6 × biên độ cột, ở mọi cột của mọi CSV | `gates/R0.json` |
| **D0** | Chọn script train. (1) **Phần tĩnh** (`Tools/phanD_d0_static.py`, kiểm tự động trên AST của chính code sẽ chạy): ① lịch LR + optimizer + AMP; ② thứ tự khởi tạo model / RNG; ③ sampler / worker seed; ④ nguồn pos_weight và λ (kèm tính `l_total` thật ra trùng bit); ⑤ lịch validation / chọn checkpoint. Kết quả từng mục nằm trong `d0_script_diff.md` và `d0_static.json`. (2) **Thực nghiệm** 500 iter, seed 19: W-a, W-b (weighted), O-a (gốc) | Phần tĩnh: cả 5 mục GIỐNG. Thực nghiệm: **loss iter 1 của W-a và O-a trùng bit** (4 số hạng); `max\|W-a − O-a\| ≤ max\|W-b − W-a\|` (nếu W-a ≡ W-b thì yêu cầu W-a ≡ O-a); pos_weight **dùng để train** = 11.0014 (đã ghim) | `gates/D0.json` |
| **D1** | Lưu snapshot không làm đổi quá trình train (run ON lưu ở 0/250/500) | `max\|ON − OFF-a\| ≤ max\|OFF-b − OFF-a\|`. Nếu OFF-a ≡ OFF-b thì yêu cầu ON ≡ OFF-a | `gates/D1.json` |
| **D2** | Nạp `iter_500.pth` vào model mới, so với model trong bộ nhớ của run ON | trùng bit với mọi tham số và buffer | `gates/D2.json` |
| **D3** | `cos_all` tính từ Gram bằng cos tính trực tiếp; Gram tính tăng dần bằng Gram tính đầy đủ | < 1e-6 | `gates/D3.json` |
| **D4** | Cổng 2/3/5 của S1 trên 4 ảnh dò (train), `iter_500.pth` | < 1e-5 / < 1e-5 / < 1e-6 | `gates/D4.json` |
| **D5** | Thời gian: s/iter và s/snapshot, dự báo cho từng phiên | mỗi phiên ≤ 10 giờ. Nếu vượt thì tự chuyển sang lịch thưa và ghi lại trong `schedule.txt` | `gates/D5.json` |
| §3 | Vùng hỗ trợ affinity. Chỉ mô tả, **không chặn** | — | `affinity_support.md` |

**Nếu D0 FAIL:** script **không dừng**. Nó in `D0 FAIL -> chuyen sang script goc`, chạy thêm O-b, rồi dùng `src/train_static_boundary.py` với cờ `--report-ckpts` (vẫn có 4 checkpoint, `checkpoint_index.json` và CSV mở rộng). Khác biệt giữa hai script được ghi trong `d0_script_diff.md`. D0 chỉ chặn khi pos_weight sai.

## 3. Sau Phiên 1: sanity §5 và D6 (không chặn)

- **Sanity §5:** mIoU-9 tại vòng validation 36000 phải nằm trong 0.6540 ± 0.0017.
- **D6:** mIoU-9 của `snapshots/iter_36000.pth`, đo trên val 384 ảnh, phải bằng giá trị ở vòng 36000 trong CSV, dung sai 0.001.
- Nếu một trong hai FAIL, script **vẫn chạy tiếp Phiên 2**. Khi đó `summary_phanD.md` ghi `dien giai duoc: KHONG` kèm cảnh báo in đậm. **Đừng diễn giải quỹ đạo** của run đó (spec §5).

## 4. Khi bị ngắt giữa chừng

1. Tạo version mới, chọn **Add Input → Your Work → Notebooks →** version bị ngắt.
2. Sửa Cell 2 thành:
   ```bash
   !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_phanD.sh --resume
   ```
   Thêm `--skip-dryrun` nếu không muốn chạy lại dry-run. Thật ra không cần: script tự bỏ qua dry-run nếu `gates_dryrun.json` đã PASS.
3. Cách resume hoạt động:
   - train tự resume từ `work/latest_checkpoint.pth` (lưu sau mỗi vòng validation, tức mỗi 4000 iter);
   - snapshot có mốc **lớn hơn** điểm resume thuộc quỹ đạo cũ đã bị ngắt, nên được **xoá rồi lưu lại** để hai quỹ đạo không bị trộn;
   - snapshot đã đo xong (`done/iter_<t>.json` khớp dấu vân tay) thì được bỏ qua.

## 5. Kết quả: đọc gì trước

Vào version → tab **Output** → `Boundary_Loss_Solution/output/phanD/`:

| File | Nội dung |
|---|---|
| **`summary_phanD.md`** | Dòng đầu: config + md5, seed, script train, chế độ đo, có diễn giải được không. Sau đó: sanity §5/D6, **verdict bảng 6.3**, các cửa sổ xung đột, bảng mức tham số theo iter, mô tả 6.4 |
| `gates_dryrun.json` | R0, D0–D5 (từng cổng), `meta` (script train, lịch snapshot), `post` (sanity §5, D6) |
| `traj_param.csv` | t × cặp: `cos_all, ci_lo, ci_hi, null_lo, null_hi, ratio, cos_batch_1..4`, tiêu chí 6.1, `conflict` |
| `traj_feat_strata.csv` | t × cặp × trục × tầng: `sum_dot, sum_n1, sum_n2, n_pos, n_support, n_neg, cosW`, CI, null |
| `windows.csv` | Các cửa sổ xung đột (≥ 3 snapshot liên tiếp, không tính t=0) |
| `figures/traj_cos_param.png`, `figures/traj_ratio.png` | Hai hình theo §7 |
| `affinity_support.md` | §3 |
| `d0_script_diff.md` | D0: khác biệt giữa hai script train |
| `probe_ids.txt` | 32 ảnh dò. Nên commit file này vào `phanD/probe_ids.txt` trong repo; các lần chạy sau sẽ phải sinh ra đúng file đó |
| `snapshots/iter_<t>.pth` | 35 snapshot, khoảng 1,7 GB. Nên lưu thành Kaggle Dataset để đo lại mà không cần train lại |
| `gram/iter_<t>.npz` | Gram 96×96 cho mỗi snapshot. Có thể tính lại mọi thống kê mức tham số trên CPU |
| `checkpoints/` | `best_miou`, `best_bfscore`, `best_bareland`, `final_iter40000` + `checkpoint_index.json` |
| `logs/` | `train_log.txt`, `benchmark_results.csv` (10 vòng), log đo `measure_cuda*.log`, `train_stdout.log` |

Đọc verdict:

- **D-G / D-RA / D-L**: có xung đột trong một cửa sổ trước 36000. Mới chỉ là **ứng viên**: bắt buộc lặp lại quỹ đạo với seed 86 trước khi viết spec phương pháp.
- **D-dai-dang**: có cửa sổ kéo tới ≥ 36000. **Dừng diễn giải** và kiểm khác biệt train vs val trước. Trong báo cáo, mã này ghi đè các mã khác; dòng "Ma theo dung thu tu" vẫn cho biết mã nào đạt theo nguyên văn bảng.
- **D-N**: không có cửa sổ nào. Dừng phần phương pháp của Hướng B.

Tính lại thống kê trên CPU mà không cần GPU (ví dụ trên máy bạn, sau khi tải `output/phanD/`):

```bash
python Tools/phanD_traj.py --output-dir output/phanD --gates-json output/phanD/gates_dryrun.json
```

## 6. Chạy Phiên 2 độc lập (Phiên 1 đã xong ở version trước)

Dùng khi Phiên 1 (train + snapshot) đã chạy xong nhưng Phiên 2 (đo) bị lỗi hoặc bị ngắt. Script này **không train lại** và **không cần dataset checkpoint**.

1. Tạo notebook mới (hoặc version mới), chọn **GPU T4 x2**, Internet On.
2. **Add Input:**
   - dataset OpenEarthMap;
   - **output của version đã chạy Phiên 1** (Your Work → Notebooks → version đó). Có thể thay bằng một Kaggle Dataset chứa thư mục `phanD/` có `snapshots/` và `probe_ids.txt`.
3. Hai cell:
   ```bash
   !cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
   !bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_phanD_phien2.sh
   ```
4. **Save & Run All (Commit).** Thời gian khoảng 40–75 phút.

Script tự làm các việc sau:
- tìm `phanD/probe_ids.txt`, thư mục `snapshots/` (bắt buộc có `iter_36000.pth`) và dataset;
- chép các kết quả nhỏ của Phiên 1 (`gates_dryrun.json`, `meta.json`, `post/`, `logs/`, và `gram/ feat/ done/` nếu đã đo dở) vào `output/phanD/`;
- đọc snapshot **ngay tại chỗ**, không chép 1,7 GB;
- kiểm dung lượng đĩa cho memmap (khoảng 4,6 GB mỗi GPU);
- đo các snapshot còn thiếu, rồi tổng hợp bảng 6.3.

Dòng cuối khi thành công: `==> PHAN D — PHIEN 2: XONG ... Verdict: ...`

Tuỳ chọn:

| Tuỳ chọn | Khi nào dùng |
|---|---|
| `--summarize-only` | Đã đo đủ snapshot, chỉ cần tổng hợp lại |
| `--prev DIR`, `--snapshot-dir DIR` | Script không tự tìm được đúng thư mục |
| `SCRATCH=/duong/dan` (đặt trước `bash`) | `/tmp` không đủ chỗ cho memmap |

**Nếu bước đo lỗi:** script in 40 dòng cuối của `logs/measure_cuda0.log`, `logs/measure_cuda1.log` và `measure_errors.txt`, trong đó có traceback của từng snapshot lỗi. Các snapshot đã đo xong **được giữ lại**. Sửa lỗi xong thì chạy lại script với Add Input là output mới nhất; nó chỉ đo phần còn thiếu.

## Phụ lục — Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `can 2 GPU ... dang co 1` | Chọn sai accelerator | Chọn **GPU T4 x2** |
| `GPU khong chay duoc voi ban PyTorch nay` | Máy P100 | Chọn T4 x2 |
| `summary_static_s19.txt khong co pos_weight 11.0014` | Đặt nhầm file summary | Đổi tên đúng file như bước 0.3 của hướng dẫn Bước 1 |
| `khong tao duoc worktree commit 8d52089` | Clone shallow | Clone lại, không dùng `--depth` |
| `R0 FAIL` | Bản mở rộng probe đã làm đổi kết quả chế độ cũ | **Không chạy tiếp.** Gửi `gates/R0.json` |
| `D0: pos_weight dung de train cua W-a khong phai 11.0014` | Lệnh train thiếu `--pos-weight 11.0014` (code cũ?) | Clone lại code mới nhất. Gửi `gates/D0.json` + `dryrun/W-a/train_log.txt` |
| `D0 FAIL` vì loss iter 1 không trùng | Forward trên GPU không tất định, hoặc augmentation chưa được seed | Xem `gates/D0.json` → `value.iter1.A_equals_B`. Nếu W-a ≠ W-b ngay ở iter 1 thì nguyên nhân nằm ở GPU, không phải ở script. Script tự chuyển sang script gốc. Gửi file JSON để quyết định |
| `D1 FAIL` | Lưu snapshot làm đổi quá trình train | **Không nới ngưỡng.** Gửi `gates/D1.json` |
| `D2 FAIL` | Snapshot nạp lại khác trọng số trong bộ nhớ | Gửi `gates/D2.json` |
| `D3 FAIL` | Gram sai | Gửi `gates/D3.json` + `dryrun/d3.log` |
| `D4 FAIL` | Cổng S1 trượt trên ảnh train | Gửi `gates/D4.json` |
| `D5 FAIL` | Cả lịch thưa cũng vượt 10 giờ | Gửi `gates/D5.json` |
| `final o iter ... (early stopping?)` | Early stopping (patience 4) dừng trước 40k | Gửi `logs/train_log.txt`. Spec cần đủ 40k |
| `CUDA out of memory` ở D4 | Cổng 3 chạy float64 | Gửi log |
| `No space left on device` | `/tmp` đầy (memmap khoảng 4,5 GB/snapshot/GPU) | Gửi log |
| Bị ngắt (hết 12 giờ, mất kết nối) | — | Mục 4 (`--resume`) |
| `tail: option used in invalid context -- 2` (code cũ) | Lỗi của chính script khi in log đo: `tail -20` không nhận nhiều file. Lỗi gốc của bước đo vẫn nằm trong `logs/measure_cuda*.log` | Đã sửa. Chạy Phiên 2 độc lập (mục 6) |
| `CUDA out of memory` ở bước đo, sau khoảng 9 snapshot (code cũ) | Mỗi snapshot rò khoảng 1,4 GB GPU: Probe cũ không được giải phóng | Đã sửa (`Probe.close()`). Log mỗi snapshot giờ in `GPU dinh ... con giu ...`; mức "con giu" phải đứng yên giữa các snapshot |
| `khong du dia cho memmap` | `/tmp` còn dưới khoảng 5 GB mỗi GPU | Đặt `SCRATCH=<thư mục khác>` (mục 6) |
| `do snapshot chua xong: N/35` | Có snapshot bị lỗi khi đo | Xem traceback trong `measure_errors.txt` và log được in ra. Gửi các file đó. Chạy lại mục 6 để đo tiếp |
