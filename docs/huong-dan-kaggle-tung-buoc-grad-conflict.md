# Hướng dẫn từng bước trên Kaggle — Chẩn đoán xung đột gradient (Phần A) + lỗ giả (Phần B, C, B2)

Tài liệu này đi từ bước đầu tiên đến khi có kết quả cuối cùng. Làm lần lượt từ trên xuống, không bỏ bước.

- Spec: `docs/spec-chan-doan-xung-dot-gradient-va-lo-gia.md`
- Tài liệu tra cứu (mô tả từng file kết quả, cách xử lý khi cổng FAIL): `docs/huong-dan-chay-grad-conflict-va-lo-gia.md`

---

## Cách gọn nhất: mỗi phiên một script

Mỗi phiên Kaggle chỉ cần **2 cell**. Script tự làm hết các bước của phiên đó và **dừng ngay khi gặp lỗi** (in `[LOI]` kèm lý do).

```bash
# Cell 1 (giống nhau ở cả 3 phiên)
!cd /kaggle/working && rm -rf Boundary_Loss_Solution && git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
# Cell 2 — chọn đúng 1 dòng theo phiên
!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_session1_check.sh    # PHIÊN 1 (CPU)
!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_session2_partA.sh    # PHIÊN 2 (GPU T4 x2, commit)
!bash /kaggle/working/Boundary_Loss_Solution/scripts/kaggle_session3_partB.sh    # PHIÊN 3 (CPU, commit)
```

| Script | Làm gì | Dòng cuối khi thành công |
|---|---|---|
| `kaggle_session1_check.sh` | Cài thư viện → tìm đường dẫn → kiểm 8 checkpoint + 4 summary → giải nén dump → đối chiếu ảnh val → chạy test → in trước bảng ghép + pos_weight. **Không đo gì.** | `==> PHIEN 1: TAT CA OK` |
| `kaggle_session2_partA.sh` | Phần chuẩn bị như phiên 1, cộng kiểm GPU → **chạy thử 4 ảnh** (cổng chặn: có lỗi thì dừng) → **chạy thật 384 ảnh** (2 GPU song song) → xác nhận 8 checkpoint PASS trên cuda | `==> PHIEN 2: XONG sau N phut` |
| `kaggle_session3_partB.sh` | Phần chuẩn bị → chép output phiên 2 → Phần B, C, B2 → tổng hợp lại → in kết luận A/B/C/B2 | `==> PHIEN 3: XONG sau N phut` |

Tuỳ chọn của phiên 2, thêm vào cuối dòng Cell 2:
- `--dryrun-only`: chỉ chạy thử 4 ảnh, để xem thời gian ước tính.
- `--skip-dryrun`: bỏ bước chạy thử.
- `--resume`: chạy tiếp từ output của một version trước (Add Input version đó).

Phần dưới đây mô tả chi tiết từng bước mà 3 script đang tự động làm. Hãy dùng nó khi cần chạy tay hoặc khi gặp lỗi. Phần chuẩn bị ở máy bạn (**PHIÊN 0**: push code, upload checkpoint + summary) vẫn phải làm trước.

---

## Tổng quan

### Dữ liệu nằm ở đâu

| Dữ liệu | Nằm ở | Ghi chú |
|---|---|---|
| Code | GitHub repo → `git clone` trên Kaggle | |
| **Dump** (mask dự đoán + `per_image_stats.csv` của 7 run) | **Trong repo**: `data/dump_parts/` | `dump.zip` (167 MB) vượt giới hạn 100 MB/file của GitHub, nên được chia thành 4 phần ≤ 45 MB. Trên Kaggle, script **tự ghép, kiểm sha256 và giải nén**. |
| 8 checkpoint + 4 file summary (pos_weight) | **Kaggle Dataset** của bạn | Checkpoint quá lớn để đưa vào git |
| Ảnh + nhãn OpenEarthMap | Kaggle Dataset bạn dùng lúc train | |

### Bốn phiên, chỉ một phiên tốn quota GPU

| Phiên | Máy | Tốn quota GPU? | Làm gì | Thời gian |
|---|---|---|---|---|
| **0** | Máy của bạn | — | Push code + dump lên GitHub; cập nhật dataset checkpoint | 15 phút |
| **1** | Kaggle **CPU** (tương tác) | Không | Clone, giải nén dump, kiểm tra mọi thứ, chạy test, xem lệnh | 15–20 phút |
| **2** | Kaggle **GPU T4 x2** (commit) | **Có** | Phần A: chạy thử 4 ảnh (cổng chặn), sau đó chạy thật 384 ảnh | khoảng 1,5–2,5 giờ |
| **3** | Kaggle **CPU** (commit) | Không | Phần B, C, B2, rồi tổng hợp lại báo cáo | khoảng 1–2 giờ |

Nguyên tắc tiết kiệm quota:
- Việc gì không cần GPU thì làm ở phiên CPU.
- Phiên GPU chạy bằng **Save & Run All (Commit)**: chạy xong là tự tắt.
- Nếu bước chạy thử 4 ảnh có lỗi, notebook dừng ngay và không chạy phần 384 ảnh.

---

## PHIÊN 0 — Trên máy của bạn

### 0.1. Dump đã được chia sẵn

Thư mục `data/` có:

```
data/
├── dump.zip                    ← file gốc 167 MB — KHÔNG commit (đã ghi trong .gitignore)
└── dump_parts/                 ← CÁI NÀY được commit
    ├── dump.zip.part000  (45 MB)
    ├── dump.zip.part001  (45 MB)
    ├── dump.zip.part002  (45 MB)
    ├── dump.zip.part003  (31.9 MB)
    └── manifest.json           ← kích thước + sha256 từng phần và cả file
```

Tôi đã chạy lệnh chia, rồi thử ghép lại và giải nén. Kết quả: sha256 khớp, 7 dump cần dùng đều đủ 384 PNG + CSV.

**Chỉ khi bạn thay `data/dump.zip` bằng bản khác** mới cần chia lại:

```bash
python Tools/dump_archive.py split            # đọc data/dump.zip -> ghi lại data/dump_parts/
```

### 0.2. Commit và push

Mở terminal tại `D:\Claude\BoundaryLossSolution`:

```bash
git add .gitignore .gitattributes Tools/dump_archive.py Tools/grad_conflict_common.py \
        scripts/run_grad_conflict_kaggle.sh data/dump_parts \
        docs/huong-dan-chay-grad-conflict-va-lo-gia.md docs/huong-dan-kaggle-tung-buoc-grad-conflict.md
git status          # data/dump.zip KHÔNG được xuất hiện trong danh sách "to be committed"
git commit -m "Add dump parts (split zip) and auto-extract for Kaggle runs"
git push origin main
```

- Lần push này tải lên khoảng 167 MB nên sẽ lâu hơn bình thường.
- Nếu push báo `File ... exceeds GitHub's file size limit of 100 MB`, nghĩa là bạn đã vô tình add `data/dump.zip`. Chạy `git rm --cached data/dump.zip`, rồi `git commit --amend` và push lại.
- Kiểm tra trên GitHub: thư mục `data/dump_parts/` có 4 file `.part` và `manifest.json`.

Repo phải **public** để Kaggle clone được mà không cần mật khẩu. Nếu repo private: tạo Personal Access Token trên GitHub (quyền `repo`), lưu vào **Kaggle → Add-ons → Secrets** với tên `GH_TOKEN`, rồi dùng lệnh clone thay thế ghi ở Phụ lục B.

### 0.3. Chuẩn bị Kaggle Dataset (checkpoint + summary)

Dataset checkpoint trên Kaggle **không cần chứa dump nữa**, vì dump đã nằm trong repo. Nếu trước đó bạn đã upload `dump.zip` vào dataset này thì cứ để đó; script không dùng tới nó.

Dataset cần chứa, **cùng một thư mục**:

**(a) 8 checkpoint**, giữ đúng tên:

```
best_model_Static_S19.pth      best_miou_Static_s86.pth      final_iter40000_static_s86.pth
best_model_BCE_04.pth          best_model_BCE_02.pth
best_miou_aff_s19.pth          best_miou_aff_s86.pth         best_model_baseline.pth
```

**(b) 4 file summary đã đổi tên**, để script tự đọc pos_weight:

| Tên trong dataset | Lấy từ file gốc | pos_weight |
|---|---|---|
| `summary_static_s19.txt` | `D:\Nghien Cuu Sinh\Lab\Boundary\Boundary_Static\unetformer-resnet18-static-boundary-openearthmap\work_dirs\static_boundary\summary.txt` | 11.0014 |
| `summary_static_s86.txt` | `D:\Nghien Cuu Sinh\Lab\Boundary\Boundary_Static_seed86\run5_static_seed86\summary.txt` | 10.9749 |
| `summary_bce04.txt` | `D:\Nghien Cuu Sinh\Lab\Boundary\BCE_Lambda_0.4\unetformer-resnet18-bce-edge-openearthmap\work_dirs\bce_edge\summary.txt` | 10.8799 |
| `summary_bce02.txt` | `D:\Nghien Cuu Sinh\Lab\Boundary\BCE_Lambda_0.2\unetformer-resnet18-bce-edge-openearthmap\work_dirs\bce_edge_lambda0.20\summary.txt` | 11.0392 |

Lệnh PowerShell để copy và đổi tên. Thay `<THU_MUC_CHECKPOINT>` bằng thư mục chứa 8 file `.pth` mà bạn dùng để upload:

```powershell
$B = "D:\Nghien Cuu Sinh\Lab\Boundary"
$DST = "<THU_MUC_CHECKPOINT>"
Copy-Item "$B\Boundary_Static\unetformer-resnet18-static-boundary-openearthmap\work_dirs\static_boundary\summary.txt" "$DST\summary_static_s19.txt"
Copy-Item "$B\Boundary_Static_seed86\run5_static_seed86\summary.txt" "$DST\summary_static_s86.txt"
Copy-Item "$B\BCE_Lambda_0.4\unetformer-resnet18-bce-edge-openearthmap\work_dirs\bce_edge\summary.txt" "$DST\summary_bce04.txt"
Copy-Item "$B\BCE_Lambda_0.2\unetformer-resnet18-bce-edge-openearthmap\work_dirs\bce_edge_lambda0.20\summary.txt" "$DST\summary_bce02.txt"
Select-String -Path "$DST\summary_*.txt" -Pattern "pos_weight used"
```

Dòng cuối phải in ra 4 dòng có 11.0014 / 10.9749 / 10.8799 / 11.0392.

Upload:
- Dataset **mới**: Kaggle → **Datasets → New Dataset** → kéo thả 12 file → đặt tên, ví dụ `boundary-ckpts` → **Create**.
- Dataset **đã có**: mở dataset → **New Version** → thêm 4 file summary → **Create**.

---

## PHIÊN 1 — Kaggle CPU: chuẩn bị và kiểm tra (không tốn quota)

### 1.1. Tạo notebook

1. **Create → New Notebook**, đặt tên ví dụ `grad-conflict-run`.
2. Panel phải → **Settings**: **Accelerator: None** · **Internet: On** · Language: Python.
3. Panel phải → **Input → + Add Input**, thêm:
   - dataset OpenEarthMap mà bạn đã dùng lúc train;
   - dataset checkpoint (`boundary-ckpts`).
4. Xoá cell mẫu.

### 1.2. Cell 1 — Clone code + cài thư viện

```bash
%%bash
set -eo pipefail
cd /kaggle/working
rm -rf Boundary_Loss_Solution
git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
cd Boundary_Loss_Solution
pip install -q -r requirements.txt
ls -la data/dump_parts/
python -c "import torch, timm, albumentations, rasterio; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
```

**Mong đợi:**
- `data/dump_parts/` có 4 file `dump.zip.part000…003`, kích thước 45 MB / 45 MB / 45 MB / ~32 MB, cùng `manifest.json`.
- Dòng `torch ... | cuda: False`. Đây là phiên CPU nên `False` là đúng.

Nếu các file `.part` chỉ nặng vài trăm byte, đó là file con trỏ Git LFS. Bạn cần push lại theo bước 0.2, không dùng LFS.

### 1.3. Cell 2 — Tìm đường dẫn, giải nén dump, kiểm tra dữ liệu

Dán **nguyên** cell:

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution

# (a) Đường dẫn dataset (checkpoint + OpenEarthMap)
CK=$(dirname "$(find /kaggle/input -name best_model_Static_S19.pth | head -1)")
DATA=$(dirname "$(dirname "$(find /kaggle/input -maxdepth 6 -type d -path '*images/val' | head -1)")")
echo "CK   = $CK"; echo "DATA = $DATA"
[ -n "$CK" ] && [ -d "$DATA/images/val" ] || { echo "LOI: khong tim thay checkpoint hoac dataset"; exit 1; }
printf 'CK=%q\nDATA=%q\n' "$CK" "$DATA" > /tmp/paths.env

# (b) 8 checkpoint + 4 summary
for f in best_model_Static_S19.pth best_miou_Static_s86.pth final_iter40000_static_s86.pth best_model_BCE_04.pth \
         best_model_BCE_02.pth best_miou_aff_s19.pth best_miou_aff_s86.pth best_model_baseline.pth; do
  [ -f "$CK/$f" ] && echo "[OK]    $f" || { echo "[THIEU] $f"; exit 1; }
done
for f in summary_static_s19.txt summary_static_s86.txt summary_bce04.txt summary_bce02.txt; do
  [ -f "$CK/$f" ] && echo "[OK]    $f -> $(grep 'pos_weight used' "$CK/$f")" || { echo "[THIEU] $f (buoc 0.3)"; exit 1; }
done

# (c) Ghép + kiểm sha256 + giải nén dump từ repo, kiểm 7 dump
python Tools/dump_archive.py join --parts-dir data/dump_parts --dest /tmp/dump_repo | tee /tmp/join.log
DP=$(sed -n 's/^DUMP_ROOT=//p' /tmp/join.log | tail -1)

# (d) Tên ảnh trong dump khớp với ảnh val của dataset?
python - "$DP" "$DATA" <<'EOF'
import os, sys
dp, data = sys.argv[1:]
val = sorted(f[:-4] for f in os.listdir(os.path.join(data, 'images/val')) if f.lower().endswith('.tif'))
png = sorted(f[:-4] for f in os.listdir(os.path.join(dp, 'run3_static_s19_iter36000/masks')))
print(f"val tif = {len(val)}, dump png = {len(png)}, trung khop = {val == png}")
sys.exit(0 if val == png else 1)
EOF
echo "==> TAT CA KIEM TRA DU LIEU: OK"
```

**Mong đợi:**
- `CK = ...` và `DATA = ...` đều không rỗng.
- 8 dòng `[OK]` cho checkpoint, 4 dòng `[OK]` cho summary, mỗi dòng summary kèm giá trị pos_weight.
- Các dòng `Ghep 4 phan ... sha256 OK` và `Da giai nen vao /tmp/dump_repo`.
- 7 dòng `[OK]` cho dump (`png=384 csv=384`), rồi dòng `DUMP_ROOT=/tmp/dump_repo/dump`.
- Dòng `trung khop = True`.
- Dòng cuối `==> TAT CA KIEM TRA DU LIEU: OK`.

**Nếu lỗi:**
- `DATA` rỗng: chạy `!find /kaggle/input -maxdepth 6 -type d -name val` rồi gửi tôi kết quả.
- `[THIEU]` checkpoint hoặc summary: tên file trong dataset sai. Đổi tên trong dataset (New Version).
- `sai kich thuoc/sha256`: repo clone về bị lỗi. Chạy lại Cell 1.
- `trung khop = False`: đang gắn nhầm bản dataset OpenEarthMap.

### 1.4. Cell 3 — Test (khoảng 1 phút)

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution
python Tools/test_spurious_holes.py
python Tools/test_grad_conflict_probe.py
```

**Mong đợi:** cả hai lệnh kết thúc bằng dòng `Tat ca self-test ... PASS.`

### 1.5. Cell 4 — Xem trước lệnh sẽ chạy (không chạy gì thật)

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution
source /tmp/paths.env
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" --make-missing-dumps --print-only
```

**Mong đợi:**
- Dòng `DUMP_ROOT = /tmp/dump_repo/dump`. Script tự lấy dump từ repo, không cần `--dump-root`.
- Bảng `== Ghep checkpoint / dump` có **8 dòng**, dòng nào cũng có `ckpt:` và `dump:`. Riêng `bce02_40k` có `dump: output/dump/run2_bce02_iter40000`: dump này sẽ được tạo trên GPU ở phiên 2.
- 5 dòng `pos_weight = ...`, lần lượt **11.0014 / 10.9749 / 10.9749 / 10.8799 / 11.0392**.
- Không có dòng `(KHONG CO)` và không có dòng `LOI:`.

Khi Cell 2, 3, 4 đều đạt, bấm **Stop session**.

---

## PHIÊN 2 — Kaggle GPU: Phần A (tốn quota)

### 2.1. Chuẩn bị notebook

1. Mở notebook ở phiên 1, hoặc **File → Copy** thành `grad-conflict-gpu`. Giữ nguyên input.
2. **Settings → Accelerator: GPU T4 x2**, Internet: On.
3. **Xoá hết cell cũ**, rồi tạo đúng 4 cell dưới đây. Commit chạy trên một máy mới hoàn toàn, nên mỗi lần chạy đều phải clone lại và giải nén lại.

**Cell 1 — Clone + cài + kiểm tra GPU**

```bash
%%bash
set -eo pipefail
cd /kaggle/working
rm -rf Boundary_Loss_Solution
git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
cd Boundary_Loss_Solution
pip install -q -r requirements.txt
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'KHONG CO GPU'; print(torch.cuda.device_count(), 'GPU OK')"
```

**Cell 2 — Đường dẫn + giải nén dump**

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution
CK=$(dirname "$(find /kaggle/input -name best_model_Static_S19.pth | head -1)")
DATA=$(dirname "$(dirname "$(find /kaggle/input -maxdepth 6 -type d -path '*images/val' | head -1)")")
printf 'CK=%q\nDATA=%q\n' "$CK" "$DATA" | tee /tmp/paths.env
[ -f "$CK/summary_bce02.txt" ] && [ -d "$DATA/images/val" ]
python Tools/dump_archive.py join --parts-dir data/dump_parts --dest /tmp/dump_repo
```

**Cell 3 — Chạy thử 4 ảnh (cổng chặn)**

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution
source /tmp/paths.env
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" \
     --make-missing-dumps --dry-run 4 --stages "figures gates A" 2>&1 | tee dryrun_A.log
echo "---- thiet bi va thoi gian (4 anh) ----"
grep -h '"label"\|"gpu"\|"wall_time_s"' output/grad_conflict_dryrun/done_*.json
```

Cell này làm 3 việc:
- Tạo dump cho BCE λ=0.2 (khoảng 3 phút, chỉ forward).
- Chốt 6 ảnh minh hoạ.
- Chạy các cổng kiểm và đo Phần A trên 4 ảnh cho đủ 8 checkpoint.

Nếu **bất kỳ cổng nào FAIL**, cell báo lỗi và notebook dừng; Cell 4 không chạy.

**Cell 4 — Chạy thật Phần A trên 2 GPU, rồi tổng hợp**

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution
source /tmp/paths.env
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" \
     --make-missing-dumps --parallel --stages "summarize" 2>&1 | tee full_A.log
echo "---- kiem tra ----"
grep -h '"label"\|"gpu"\|"wall_time_s"' output/grad_conflict/done_*.json
head -40 output/grad_conflict/summary_grad_conflict.md
```

- Với `--parallel`: 4 checkpoint chạy trên `cuda:0` và 4 checkpoint chạy trên `cuda:1`, song song.
- `--stages "summarize"`: bước gộp cuối **chỉ** tổng hợp Phần A. Phần B để dành cho phiên CPU.
- Dump của BCE λ=0.2 đã được tạo ở Cell 3, nên không tạo lại.

### 2.2. Chạy

1. Bấm **Save Version** → **Save & Run All (Commit)** → **Save**.
2. Có thể tắt trình duyệt. Muốn theo dõi thì vào trang notebook → **Versions** → bấm version đang chạy → **Log**.
3. Ước tính: Cell 3 khoảng 20–40 phút, Cell 4 khoảng 1–2 giờ.

### 2.3. Kiểm tra khi xong

Vào version → tab **Output** → `Boundary_Loss_Solution/`:

| Kiểm tra | Ở đâu | Phải thấy |
|---|---|---|
| Không lỗi | `full_A.log`, `output/parallel_cuda0.log`, `output/parallel_cuda1.log` | Mỗi checkpoint có `TONG: PASS`. Không có `Traceback`. |
| Chạy trên GPU | đầu log và `output/grad_conflict/done_*.json` | `Device: cuda:0` hoặc `cuda:1`; `"gpu": "Tesla T4"` |
| Đủ 8 checkpoint | `output/grad_conflict/` | 8 file `done_*.json` và 8 file `gates_*.json` |
| Kết quả Phần A | `output/grad_conflict/summary_grad_conflict.md` | Mục "Nhanh phuong phap (bang 5.2)" |

**Nếu dừng ở Cell 3:** mở `output/grad_conflict_dryrun/gates_<nhan>.json`, tìm cổng có `"status": "FAIL"`, rồi gửi tôi nội dung file. Đừng chạy lại khi chưa rõ nguyên nhân.

**Nếu bị ngắt giữa Cell 4:**
1. Tạo version mới, **Add Input → Your Work → Notebooks →** chọn version bị ngắt.
2. Thêm cell sau, đặt **sau Cell 2**:
   ```bash
   %%bash
   OLD=$(find /kaggle/input -type d -path '*Boundary_Loss_Solution/output' | head -1)
   cp -r "$OLD" /kaggle/working/Boundary_Loss_Solution/ && ls /kaggle/working/Boundary_Loss_Solution/output/grad_conflict
   ```
3. Xoá Cell 3 rồi commit lại. Checkpoint nào đã có `done_*.json` sẽ được bỏ qua.

---

## PHIÊN 3 — Kaggle CPU: Phần B, C, B2 (không tốn quota)

### 3.1. Notebook

1. **Create → New Notebook**, đặt tên ví dụ `grad-conflict-cpu-B`.
2. **Accelerator: None**, Internet: On.
3. **Add Input**, thêm:
   - dataset OpenEarthMap;
   - dataset checkpoint;
   - **output của version GPU vừa xong** (**Your Work → Notebooks →** `grad-conflict-gpu`).

### 3.2. Các cell

**Cell 1 — Clone + cài**

```bash
%%bash
set -eo pipefail
cd /kaggle/working
rm -rf Boundary_Loss_Solution
git clone https://github.com/HuanLuongMinh/Boundary_Loss_Solution.git
cd Boundary_Loss_Solution
pip install -q -r requirements.txt
```

**Cell 2 — Đường dẫn, giải nén dump, chép kết quả Phần A về**

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution
CK=$(dirname "$(find /kaggle/input -name best_model_Static_S19.pth | head -1)")
DATA=$(dirname "$(dirname "$(find /kaggle/input -maxdepth 6 -type d -path '*images/val' | head -1)")")
printf 'CK=%q\nDATA=%q\n' "$CK" "$DATA" | tee /tmp/paths.env
python Tools/dump_archive.py join --parts-dir data/dump_parts --dest /tmp/dump_repo

# Chép output phiên GPU: grad_conflict/ + dump BCE λ=0.2 (output/dump/) + cache GT
OLD=$(find /kaggle/input -type d -path '*Boundary_Loss_Solution/output' | head -1)
echo "Output phien GPU: $OLD"
cp -r "$OLD" ./
echo "so done_*.json = $(ls output/grad_conflict/done_*.json | wc -l)  (phai = 8)"
ls -d output/dump/run2_bce02_iter40000/masks
```

**Cell 3 — Phần B, C, B2, rồi tổng hợp lại báo cáo Phần A**

```bash
%%bash
set -eo pipefail
cd /kaggle/working/Boundary_Loss_Solution
source /tmp/paths.env
bash scripts/run_grad_conflict_kaggle.sh --ckpt-dir "$CK" --data-root "$DATA" \
     --stages "B C B2" 2>&1 | tee part_B.log
# Tong hop lai Phan A de bao cao co dong "Cong 6" (cong 6 vua chay o Phan B)
python Tools/grad_conflict_probe.py summarize --output-dir output/grad_conflict --holes-dir output/spurious_holes
echo "---- ket qua ----"
head -30 output/spurious_holes/pairs_bootstrap.md
```

Script tự quyết định có chạy C và B2 hay không, theo kết quả Phần B:
- Phần B kết luận "không do loss" → bỏ qua C.
- Cổng tần suất ra "hiếm" → bỏ qua B2.

Chạy bằng **Save Version → Save & Run All (Commit)**. Phiên này không tốn quota GPU, chỉ là bạn không phải giữ trình duyệt mở.

---

## BƯỚC CUỐI — Lấy và đọc kết quả

Vào version của phiên 3 → tab **Output** → `Boundary_Loss_Solution/output/`:

| File | Nội dung | Đọc gì trước |
|---|---|---|
| `grad_conflict/summary_grad_conflict.md` | **Phần A** | Dòng in đậm dưới "Nhanh phuong phap (bang 5.2)" (G / L / C / RA / N). Bảng cổng kiểm phải toàn PASS. |
| `spurious_holes/pairs_bootstrap.md` | **Phần B** (+ Phần C ở cuối) | Dòng in đậm "Verdict bang 5.4" và "Cong tan suat 3.3" |
| `spurious_holes/fill_test.md` | **Phần B2** | Chỉ có khi không "hiếm" |
| `grad_conflict/figures/*.png` | 6 hình minh hoạ | Bản đồ cos_p: đỏ = ngược hướng, xanh = cùng hướng |
| `grad_conflict/summary_<nhan>.csv`, `spurious_holes/pairs_bootstrap.csv` | Số chi tiết | Mô tả cột ở mục 6 của `docs/huong-dan-chay-grad-conflict-va-lo-gia.md` |

Bước tiếp theo:
- Phần A ra **N**: chạy Phần D (train lại có callback, mục 7 spec). Phần này chưa có code, báo tôi khi cần.
- Phần A ra **G / RA / L / C**: đó là nhánh phương pháp được chọn.

---

## Phụ lục A — Lỗi thường gặp

| Hiện tượng | Nguyên nhân | Cách xử lý |
|---|---|---|
| `git push` báo file > 100 MB | Lỡ add `data/dump.zip` | `git rm --cached data/dump.zip`, `git commit --amend`, push lại |
| `git clone` hỏi mật khẩu / treo | Repo private | Để public, hoặc làm theo Phụ lục B |
| `Khong thay data/dump_parts/manifest.json` | Chưa push `data/dump_parts` | Làm lại bước 0.2 |
| `Phan dump.zip.partXXX sai kich thuoc/sha256` | Clone lỗi, hoặc file thành con trỏ LFS | Clone lại; kiểm `ls -la data/dump_parts` (phải 45 MB/phần) |
| `LOI: [...] khong thay .../summary_....txt` | Thiếu file summary trong dataset | Bước 0.3 |
| `LOI: [...] khong phai summary cua run nay` | Đặt nhầm tên summary | Đặt lại tên theo bảng ở 0.3 |
| `CUDA khong dung duoc ... chuyen sang CPU` | Sai loại GPU | Accelerator: **GPU T4 x2** |
| `CUDA out of memory` | Không đủ bộ nhớ GPU (cổng 3 chạy float64) | Gửi log cho tôi |
| Cổng 1 FAIL | Nạp sai checkpoint hoặc nhầm quy ước 8/9 lớp | Gửi `gates_<nhan>.json` |
| Cổng 5 FAIL | Kernel CUDA không tất định | Không nới ngưỡng; gửi `max_rel_diff` |
| Cổng 7 FAIL | Dump không khớp danh sách ảnh val | Kiểm lại Cell 2 (d) của phiên 1 |
| `No space left on device` | `/tmp` đầy | Gửi log |

## Phụ lục B — Clone repo private bằng token

1. Trên GitHub: **Settings → Developer settings → Personal access tokens** → tạo token có quyền `repo`.
2. Trên Kaggle: notebook → **Add-ons → Secrets** → thêm secret tên `GH_TOKEN`, dán token vào. Bật cho notebook.
3. Thay dòng `git clone` trong mọi Cell 1 bằng hai cell sau. Cell Python lấy token từ Secrets và ghi ra file tạm:

```python
from kaggle_secrets import UserSecretsClient
open('/tmp/gh_token', 'w').write(UserSecretsClient().get_secret('GH_TOKEN'))
```

```bash
%%bash
cd /kaggle/working && rm -rf Boundary_Loss_Solution
git clone "https://$(cat /tmp/gh_token)@github.com/HuanLuongMinh/Boundary_Loss_Solution.git"
rm -f /tmp/gh_token
```
