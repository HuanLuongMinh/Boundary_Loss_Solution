# Workflow: Từ Baseline đến Full Pipeline "Dynamic Boundary-Aware Loss"

Tài liệu này là **bản đặc tả công việc (task spec)** để đưa cho một phiên Claude Code (agent viết code) thực hiện tuần tự, sinh ra source code sẵn sàng chạy trên Kaggle (2× T4 GPU — hạ tầng hiện tại của bạn, xem `requirements.txt` đã pre-pin cho stack Kaggle). Không cần chạy thủ công từng lệnh — đưa nguyên file này (hoặc từng Bước) cho Claude Code trong repo `Unerformer_Satellite_Image` và yêu cầu thực hiện đúng thứ tự.

**Ràng buộc bắt buộc, áp dụng cho MỌI bước dưới đây:**
- **`seed = 19`, cố định tuyệt đối** — mọi script/config mới cho nghiên cứu này (splits, model init, training, augmentation shuffle) đều dùng seed 19, không dùng 42 (42 là seed riêng của Track A-D cũ, giữ nguyên không đổi). Baseline đã bàn giao (`configs/unet_former_resnet18_combineLoss/*.yaml`, `Tools/build_clean_val_split.py`) **đã được cập nhật `SEED: 19`** — Claude Code chỉ cần giữ nguyên quy ước này khi sinh thêm config/script mới.
- **Dataset dùng đúng bộ có sẵn trên Kaggle** (`dyiyacao/openearthmap`, đã dùng xuyên suốt Track A-D) — KHÔNG tải/tạo dataset mới, KHÔNG đổi nguồn dữ liệu.
- **Quy ước "file độc lập"** đã dùng trong repo: mọi thứ mới cho nghiên cứu boundary-loss nằm trong file/thư mục MỚI, không sửa các file Track A-D hoặc Track ResNet18-baseline hiện có (để không phá kết quả cũ).

---

## 0. Trạng thái hiện tại (đã hoàn thành, không cần làm lại)

| Hạng mục | File | Trạng thái |
|---|---|---|
| Model baseline (ResNet-18 + GLTB decoder) | `src/models/unet_former_resnet18.py` | ✅ Đã build, đã verify forward/backward pass thật |
| Training script baseline | `src/train_unet_former_resnet18.py` | ✅ Đã có, seed mặc định = 19 |
| Loss CE+Dice (dùng cho baseline, tính trên Fused Feature) | `src/utils/losses.py` (`CombinedLoss`) | ✅ Đã có sẵn trong repo, không cần sửa |
| Config baseline (smoke-test) | `configs/unet_former_resnet18_combineLoss/luot{1,2,3}_*.yaml` | ✅ Đã có, `SEED: 19` |
| Launcher | `run_unet_former_resnet18_combineLoss_experiment.sh` | ✅ Đã có |
| Tool tạo val sạch (không augment-pad) | `Tools/build_clean_val_split.py` | ✅ Đã có, `--seed` mặc định = 19 |
| Tool chia 5-fold spatial CV theo vùng | `Tools/build_spatial_folds.py` | ✅ Đã có, đã unit-test (cân bằng fold ~1.01, không rò rỉ vùng) |
| Phân tích kiến trúc + chiến lược dataset | `phan_tich_baseline_va_dataset.md` | ✅ Đã có |
| **Run 1 — Baseline (α=0), kết quả thật** | `work_dirs/.../benchmark_results.csv` | ✅ Đã chạy xong 28/8/2026, best mIoU = 0.6551 — xem `phase1-run1-baseline-ket-qua.md`. (Việc chọn báo cáo mIoU trên bộ lớp nào cho bài báo — Huan tự tính lại khi viết bài, không phải việc của pipeline lúc chạy.) |
| **"Baseline vs BCE Loss" (Run 2, α cố định qua λ_edge=0.4, chưa alpha-sweep)** | `src/losses/boundary_bce.py`, `src/train_bce_edge.py`, `configs/unet_former_resnet18_bce_edge/bce_edge.yaml`, `Tools/measure_edge_ratio.py`, `scripts/run_bce_edge.sh` / `resume_bce_edge.sh` | ✅ Đã triển khai 29/8/2026 (repo `BoundaryLossSolution`, tách biệt hoàn toàn với repo/pipeline `Unerformer_Satellite_Image` mô tả trong tài liệu này) — spec `L_BCE_edge` đúng theo mục 3.1 bên dưới. Chạy `bash scripts/run_bce_edge.sh` (full) hoặc `bash scripts/run_bce_edge.sh --dry-run`; `bash scripts/run_bce_edge.sh <lambda>` để ablation λ_edge nhanh. Xem README.md mục "Baseline vs BCE Loss". **`summary.txt` của Run 2 để N/A ở Boundary IoU/BF-Score — xem TODO ở mục 4 bên dưới.** |

Các bước dưới đây (1→9) là phần **CHƯA làm** tính đến thời điểm Run 1 hoàn tất (28/8/2026) — cần Claude Code sinh tiếp để có full pipeline sẵn sàng chạy 18 runs theo `idead_research.md`. Cụ thể: `src/losses/*.py` (Bước 5) — **đã có `boundary_bce.py`, còn thiếu `affinity.py`/`dynamic_weighting.py`/`total_loss.py`**, `src/utils/boundary_metrics.py` (Bước 6) — **chưa làm, xem TODO mục 4**, `src/train_boundary_research.py` (Bước 7) — thay bằng `src/train_bce_edge.py` (chỉ Run 2, chưa tổng quát hoá cho cả 8+10 run), 12 config Phase 1+2 (Bước 8, gồm `run2_bce_edge.yaml`) — **đã có 1 config riêng `bce_edge.yaml` cho Run 2**, `Tools/aggregate_results.py` (Bước 9) — chưa làm.

---

## 1. Cấu trúc thư mục mục tiêu

Chạy (hoặc yêu cầu Claude Code chạy) trước tiên để dựng khung thư mục:

```bash
cd Unerformer_Satellite_Image   # repo gốc

mkdir -p src/losses
mkdir -p src/utils
mkdir -p configs/research_boundary/phase1_fixed_split
mkdir -p configs/research_boundary/phase2_5fold_cv
mkdir -p Tools
mkdir -p dataset/spatial_folds        # output của build_spatial_folds.py sẽ nằm ở đây (trên Kaggle: /kaggle/working/.../dataset/spatial_folds)
mkdir -p docs
mkdir -p work_dirs/phase1
mkdir -p work_dirs/phase2
```

Cây thư mục cuối cùng (sau khi hoàn thành hết Bước 1-9):

```
Unerformer_Satellite_Image/
├── docs/
│   ├── idead_research.md                          # đề cương gốc (đã có)
│   ├── phan_tich_baseline_va_dataset.md            # phân tích đã có — copy vào đây
│   └── workflow.md                                 # chính file này
├── src/
│   ├── models/
│   │   └── unet_former_resnet18.py                 # ✅ đã có — dùng chung cho mọi run Phase 1&2
│   ├── losses/                                      # 🆕 package mới, KHÔNG sửa src/utils/losses.py cũ
│   │   ├── __init__.py
│   │   ├── boundary_bce.py                          # L_BCE_edge  (Bước 5.1)
│   │   ├── affinity.py                              # L_Affinity  (Bước 5.2)
│   │   ├── dynamic_weighting.py                     # λ1(t), λ2(t) scheduler (Bước 5.3)
│   │   └── total_loss.py                            # L_total = L_region + α·L_boundary_dynamic (Bước 5.4)
│   ├── utils/
│   │   ├── losses.py                                # ✅ CombinedLoss (CE+Dice) — KHÔNG sửa
│   │   ├── metrics.py                                # ✅ mIoU/per-class IoU — KHÔNG sửa
│   │   └── boundary_metrics.py                       # 🆕 Boundary IoU(d)/BF-Score/ASD (Bước 6)
│   └── train_boundary_research.py                    # 🆕 script train dùng chung cho cả 8 (Phase1) + 10 (Phase2) run,
│                                                       #    khác biệt hoàn toàn tham số hoá qua config (Bước 7)
├── configs/
│   ├── unet_former_resnet18_combineLoss/             # ✅ đã có — baseline smoke-test, giữ nguyên
│   └── research_boundary/                            # 🆕
│       ├── phase1_fixed_split/
│       │   ├── run1_baseline.yaml                    # L_region thuần, α=0
│       │   ├── run2_bce_edge.yaml                     # + BCE edge, α=α*
│       │   ├── run3_static_boundary.yaml              # + BCE + Affinity, λ1=λ2=1, α=α*
│       │   ├── run4_dynamic_boundary.yaml             # Ours: λ1(t),λ2(t) động, α=α*
│       │   ├── alpha_sweep_0.1.yaml
│       │   ├── alpha_sweep_0.5.yaml
│       │   ├── alpha_sweep_1.0.yaml
│       │   └── alpha_sweep_2.0.yaml
│       └── phase2_5fold_cv/
│           ├── fold1_baseline.yaml ... fold5_baseline.yaml
│           └── fold1_proposed.yaml ... fold5_proposed.yaml
├── Tools/
│   ├── create_splits.py                              # ✅ đã có (dùng chung, Track A-D) — KHÔNG sửa
│   ├── build_clean_val_split.py                       # ✅ đã có — dùng cho Phase 1
│   ├── build_spatial_folds.py                         # ✅ đã có — dùng cho Phase 2
│   └── aggregate_results.py                           # 🆕 gộp benchmark_results.csv của các run thành Bảng 1/Bảng 2 (Bước 9)
├── dataset/
│   ├── val_2000_fixed.txt                             # cũ, Track A-D — KHÔNG dùng cho nghiên cứu mới
│   ├── val_clean_538.txt                              # 🆕 sinh bởi build_clean_val_split.py (Bước 2.3)
│   ├── train_<N>_fixed.txt                             # 🆕 full pool, sinh bởi create_splits.py (Bước 2.2)
│   └── spatial_folds/
│       ├── fold1_train.txt ... fold5_train.txt         # 🆕 sinh bởi build_spatial_folds.py (Bước 2.4)
│       └── fold1_val.txt   ... fold5_val.txt
└── work_dirs/
    ├── phase1/run{1..4}_.../                           # output từng run: best_model.pth, benchmark_results.csv, ...
    └── phase2/fold{1..5}_{baseline,proposed}/
```

---

## 2. Chuẩn bị dữ liệu (seed = 19 xuyên suốt)

### 2.1. Kết nối dataset Kaggle

Nếu chạy trong **Kaggle Notebook** (giống các track cũ): attach dataset `dyiyacao/openearthmap` qua panel "Add Input" — dataset sẽ tự mount tại `/kaggle/input/datasets/dyiyacao/openearthmap`, không cần tải gì thêm.

Nếu chạy trên **server ngoài** (không phải Kaggle Notebook) nhưng vẫn muốn dùng đúng bộ dataset đó, tải qua Kaggle API:

```bash
pip install kaggle
# cần ~/.kaggle/kaggle.json (API token từ trang Kaggle account)
kaggle datasets download -d dyiyacao/openearthmap -p /data/openearthmap --unzip
# sau đó set --data-root /data/openearthmap ở mọi lệnh Tools/*.py bên dưới thay vì /kaggle/input/...
```

### 2.2. Sinh full train split (thay cho 500/1000/1500 cũ)

```bash
python Tools/create_splits.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/research-boundary-openearthmap \
    --seed 19
```

Script sẽ log ra `Found N training images` và ghi `dataset/train_N_fixed.txt` (N = toàn bộ pool train thật, dùng làm tập train cố định cho **cả 8 run Phase 1**). Ghi lại N, rồi tạo alias tên cố định để config không phải sửa lại nếu N thay đổi:

```bash
ln -sf train_${N}_fixed.txt /kaggle/working/research-boundary-openearthmap/dataset/train_full_fixed.txt
```

(Trên Kaggle không hỗ trợ symlink tốt trong mọi trường hợp — nếu lỗi, dùng `cp` thay `ln -sf`.)

### 2.3. Sinh val "sạch" (không augment-pad) — thay cho `val_2000_fixed.txt`

```bash
python Tools/build_clean_val_split.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/research-boundary-openearthmap \
    --seed 19
# → dataset/val_clean_538.txt (hoặc số thực tế nếu mirror khác 538)
```

Dùng `val_clean_<N>.txt` này làm `VAL_SPLIT_FILE` cho **toàn bộ 8 run Phase 1** — đảm bảo Phase 1 so sánh 4 biến thể loss trên đúng 1 tập train + 1 tập val cố định, chỉ khác nhau ở cấu hình loss/α.

### 2.4. Sinh 5-fold spatial CV — dùng cho Phase 2

```bash
python Tools/build_spatial_folds.py \
    --data-root /kaggle/input/datasets/dyiyacao/openearthmap \
    --output-dir /kaggle/working/research-boundary-openearthmap \
    --n-folds 5
# → dataset/spatial_folds/fold{1..5}_{train,val}.txt
```

(Script này không cần `--seed` vì thuật toán chia fold là greedy tất định trên input đã sort — chạy lại nhiều lần luôn ra cùng kết quả.)

**Kiểm tra bắt buộc trước khi train:** đọc log "Spatial-leakage check: 0 region(s) duplicated" — nếu khác 0, DỪNG lại, không chạy Phase 2 cho tới khi sửa.

---

## 3. Bước 5 — Cài đặt các thành phần Loss (phần code MỚI, quan trọng nhất)

Đây là phần Claude Code cần **viết mới hoàn toàn**, bám sát đúng công thức trong `idead_research.md` mục 2.2. Đặc tả chi tiết dưới đây để tránh suy diễn sai công thức.

### 3.1. `src/losses/boundary_bce.py` — L_BCE_edge

**Quyết định thiết kế quan trọng (đọc kỹ):** code tham khảo gốc (`geoseg/losses/useful_loss.py::EdgeLoss` trong repo `ContrastiveGeoSeg`) trích biên dự đoán bằng cách áp Laplacian kernel lên **argmax** của logits — thao tác này **không lan truyền gradient** (argmax không khả vi), nên về bản chất chỉ hoạt động như một số hạng phạt gần-tĩnh, không thực sự dạy mô hình định vị biên tốt hơn qua backprop. **KHÔNG copy lại cách này.**

Thay vào đó, thêm một **boundary head nhỏ** (1 nhánh conv phụ, tương tự `AuxHead` đã định nghĩa nhưng chưa dùng trong `UNetFormer.py` gốc) gắn trực tiếp lên **Fused Feature** của decoder (`unet_former_resnet18.py`, tham số `return_fused_feature=True` đã sẵn có ở model — dùng luôn, không cần sửa model), sinh ra `edge_logits` (1 kênh, cùng spatial size với Fused Feature, upsample về kích thước input):

```python
class BoundaryHead(nn.Module):
    def __init__(self, in_channels=64):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, fused_feature, out_size):
        x = self.conv_out(self.drop(self.conv(fused_feature)))
        return F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)  # (B,1,H,W) edge_logits
```

**Spec đầy đủ cho `edge_gt`, đo tỉ lệ mất cân bằng, và balanced BCE — thay thế hoàn toàn placeholder Sobel/Laplacian trước đây** (nguồn: ghi chú LaTeX "Edge-pixel ratio measurement and balanced BCE edge loss", 29/8/2026). Đây thuần tuý là spec kỹ thuật cho thuật toán edge-loss, độc lập với việc paper báo cáo mIoU trên bộ lớp nào:

**1) Validity mask:** `m[b,h,w] = 1[y[b,h,w] != y_ignore]`, dùng đúng giá trị `y_ignore` mà pipeline hiện có đang dùng (xác nhận giá trị thực tế, ví dụ 255, trong `Tools/create_splits.py`/dataset loader trước khi code — chỉ cần biết đúng giá trị, không cần tranh luận lại việc Background có phải lớp thứ 9 hay không).

**2) Edge target** (4-connected, có thể chọn 8-connected nhưng phải cố định 1 lựa chọn cho toàn bộ nghiên cứu, ghi rõ trong config/paper): `e[b,h,w] = m[b,h,w] · 1[tồn tại ít nhất 1 hàng-xóm 4-connected VALID có nhãn khác y[b,h,w]]` — cả pixel trung tâm và hàng xóm đều phải valid, nếu không sẽ tạo edge giả tại ranh giới vùng ignore/padding.

```python
def extract_edge_gt(mask, ignore_index=255, connectivity=4, dilation_radius=1):
    # mask: (B,H,W) int64. Trả về (B,1,H,W) float {0,1}.
    # 1) validity mask m = (mask != ignore_index)
    # 2) so sánh mask với 4 (hoặc 8) hàng xóm dịch 1 pixel, chỉ tính cặp mà CẢ HAI đều valid
    # 3) e = m * OR(các so sánh khác nhãn)
    # 4) nếu dilation_radius > 1: dilate binary e, rồi nhân lại với m để loại pixel dilate lấn vào vùng ignore
    ...
```

Giữ `connectivity` và `dilation_radius` cố định xuyên suốt mọi run/ablation (không đổi edge width giữa các run) — cả `boundary_bce.py` và `affinity.py` (3.2) PHẢI dùng chung 1 hàm `extract_edge_gt` này để đảm bảo nhất quán.

**3) Đo tỉ lệ edge-pixel ở mức toàn dataset** (KHÔNG average theo batch — cộng dồn count vì các batch có số pixel valid khác nhau):

```python
# Chạy 1 epoch đầy đủ qua train_full_fixed (seed=19, augmentation cố định), dùng đúng extract_edge_gt sẽ dùng khi train:
N_valid = sum(m) qua toàn bộ epoch
N_edge  = sum(e) qua toàn bộ epoch
r_edge  = N_edge / max(N_valid, 1)
N_nonedge = N_valid - N_edge
# Log & lưu: r_edge, N_edge, N_nonedge, connectivity, dilation_radius — 1 lần, trước khi chạy Run 2.
```

**4) `pos_weight` cho balanced BCE** (edge pixel luôn là thiểu số — nếu không cân bằng, BCE dễ hội tụ về nghiệm tầm thường "toàn bộ không phải edge"):

```
pos_weight = N_nonedge / max(N_edge, 1) = (1 - r_edge) / max(r_edge, 1e-6)
pos_weight = clip(pos_weight, 1.0, w_max)   # w_max mặc định 20, để config được
```

**5) `L_BCE_edge`** — dùng `F.binary_cross_entropy_with_logits(edge_logits, edge_gt, reduction='none', pos_weight=pos_weight)`, nhân với validity mask, chia cho `valid_mask.sum().clamp_min(1)`. KHÔNG áp `sigmoid` thủ công trước khi gọi hàm này (mất ổn định số học) — đưa thẳng `edge_logits` (raw logits từ `BoundaryHead`).

**6) `L_total` cho riêng thí nghiệm BCE-edge (Run 2):** `L_total = L_segmentation + λ_edge · L_edge`. λ_edge=0.4 là giá trị khởi tạo đề xuất cho lần chạy Run 2 đầu tiên (smoke-test trước khi có hạ tầng alpha-sweep đầy đủ) — **không thay thế** thiết kế α*-từ-alpha-sweep của Bảng 1 (mục 3.4/Bước 8): để Bảng 1 so sánh công bằng Run 2/3/4 với cùng 1 α, giá trị α cuối cùng dùng cho `run2_bce_edge.yaml` vẫn phải là α* chọn từ 4 run `alpha_sweep_*` (chạy trên cấu hình Run 4). Dùng 0.4 chỉ cho lần chạy thăm dò/dry-run đầu tiên của riêng BCE-edge, không dùng làm số cuối cùng trong Bảng 1 nếu α* khác 0.4.

**Sanity checks bắt buộc trước khi chạy full Run 2** (gate, giống triết lý dry-run đã có ở Bước 8):
- Shape `edge_gt`/`edge_logits` = `(B,1,H,W)`, giá trị `edge_gt` chỉ ∈ {0,1}.
- Pixel ignore/invalid có `edge_gt=0` và đóng góp 0 vào `L_edge` (kiểm tra bằng cách set toàn bộ 1 ảnh là ignore, verify loss = 0/không NaN).
- `0 < r_edge < 1` với crop đất phủ bình thường (không phải toàn ignore).
- `pos_weight` hữu hạn và dương.
- Input vào `BCEWithLogitsLoss` là raw logits, không phải xác suất đã qua sigmoid.
- Visualize overlay `edge_gt` lên vài ảnh OEM thật, xác nhận trùng khớp ranh giới đối tượng thật bằng mắt.
- `BoundaryHead` nhận gradient khác 0 sau `L_edge.backward()`.

### 3.2. `src/losses/affinity.py` — L_Affinity

Đúng công thức mục 2.2 của đề cương:

L_Affinity = (1/|P|) * Σ_(i,j)∈P [ A_ij · D(f_i,f_j) + (1-A_ij) · max(0, m - D(f_i,f_j)) ]

- `f_i, f_j`: lấy từ **Fused Feature** (64 kênh, stride 4) — KHÔNG dùng logits (số kênh bằng số lớp, đã mất nhiều thông tin biểu diễn).
- `P`: tập cặp pixel (i,j) trong cửa sổ cục bộ K×K (khuyến nghị K=5 hoặc K=7, để config được), **chỉ lấy mẫu ở vùng lân cận biên** — dùng **đúng hàm `extract_edge_gt` đã định nghĩa ở 3.1** (cùng `connectivity`/`dilation_radius`) để mask vùng cần tính, đảm bảo nhất quán giữa 2 loss — tránh tính toàn ảnh, vừa đúng tinh thần "boundary-aware" vừa tiết kiệm compute.
- `A_ij ∈ {0,1}`: 1 nếu i,j cùng nhãn ground-truth, 0 nếu khác. Loại hoàn toàn cặp pixel có 1 trong 2 pixel là ignore (dùng `m` từ 3.1).
- `D(f_i,f_j)`: cosine distance hoặc L2 — để tham số hoá qua config (`AFFINITY_DISTANCE: cosine|l2`).
- `m`: margin, mặc định 1.0 nếu dùng cosine distance (range [0,2]) — để tham số hoá qua config (`AFFINITY_MARGIN`).

Cài đặt bằng `F.unfold` để lấy các patch K×K hiệu quả trên GPU thay vì vòng lặp Python.

### 3.3. `src/losses/dynamic_weighting.py` — λ1(t), λ2(t)

Đề cương chỉ mô tả định tính ("ban đầu λ1 chiếm ưu thế, về sau λ2 tăng dần") — cần cụ thể hoá thành công thức tường minh, đề xuất (curriculum tuyến tính theo iteration, đơn giản, dễ tái lập và dễ diễn giải trong bài báo):

```python
def lambda_schedule(cur_iter: int, max_iters: int,
                     lambda1_start=1.0, lambda1_end=0.3,
                     lambda2_start=0.0, lambda2_end=1.0,
                     warmup_frac=0.1):
    """Tuyến tính: λ1 giảm dần, λ2 tăng dần theo tiến trình training.
    warmup_frac: tỉ lệ iteration đầu giữ λ2=0 hoàn toàn (chỉ học định vị biên
    thô qua BCE trước, đúng mô tả 'giai đoạn đầu áp affinity sớm dễ lan
    truyền nhiễu' trong idead_research.md mục 1)."""
    progress = cur_iter / max(max_iters, 1)
    if progress < warmup_frac:
        return lambda1_start, 0.0
    p = (progress - warmup_frac) / (1 - warmup_frac)
    lambda1 = lambda1_start + (lambda1_end - lambda1_start) * p
    lambda2 = lambda2_start + (lambda2_end - lambda2_start) * p
    return lambda1, lambda2
```

Đây là baseline hợp lý cho "Dynamic" (Variant 4/Ours) — nếu sau này muốn thử biến thể Gradient Magnitude Balancing (nêu trong đề cương như 1 lựa chọn khác), tách thành 1 file `dynamic_weighting_gradnorm.py` riêng, không sửa file này (để so sánh ablation 2 cách curriculum nếu cần).

### 3.4. `src/losses/total_loss.py` — ghép toàn bộ

```python
class DynamicBoundaryTotalLoss(nn.Module):
    """L_total = L_region + alpha * (lambda1(t)*L_BCE_edge + lambda2(t)*L_Affinity)
    L_region = CombinedLoss (CE+Dice) đã có sẵn trong src/utils/losses.py — tái sử dụng, không viết lại."""
    def __init__(self, num_classes, alpha, use_bce=True, use_affinity=True,
                 dynamic_weights=True, ignore_index=255, ...):
        ...
    def forward(self, logits, edge_logits, fused_feature, masks, cur_iter, max_iters):
        l_region = self.combined_loss(logits, masks)
        if not self.use_bce and not self.use_affinity:
            return l_region, {'l_region': l_region.item()}
        if self.dynamic_weights:
            lam1, lam2 = lambda_schedule(cur_iter, max_iters)
        else:
            lam1, lam2 = 1.0, 1.0   # Variant "Static Boundary" (Run 3)
        l_bce = self.bce_edge(edge_logits, masks) if self.use_bce else 0.
        l_aff = self.affinity(fused_feature, masks) if self.use_affinity else 0.
        l_total = l_region + self.alpha * (lam1 * l_bce + lam2 * l_aff)
        return l_total, {'l_region': ..., 'l_bce': ..., 'l_affinity': ..., 'lambda1': lam1, 'lambda2': lam2}
```

`ignore_index` ở đây phải khớp với giá trị thực tế pipeline đang dùng (xem 3.1, mục 1).

4 cấu hình `use_bce/use_affinity/dynamic_weights` tương ứng đúng 4 hàng Bảng 1 của đề cương:

| Run | use_bce | use_affinity | dynamic_weights | α |
|---|---|---|---|---|
| 1 — Baseline | False | False | — | 0 |
| 2 — +BCE Edge | True | False | — | α* |
| 3 — +Static Boundary | True | True | False (λ1=λ2=1) | α* |
| 4 — +Dynamic Boundary (Ours) | True | True | True | α* |

---

## 4. Bước 6 — Boundary metrics (`src/utils/boundary_metrics.py`)

Đề cương mục 4 yêu cầu 3 nhóm chỉ số ranh giới, chưa có trong `src/utils/metrics.py` hiện tại (chỉ có mIoU/per-class IoU):

- **Boundary IoU (trimap-based)** tại d ∈ {1,3,5} pixel — thuật toán chuẩn: dilate GT boundary và pred boundary bằng disk radius d, tính IoU trong dải trimap đó. Tham khảo định nghĩa gốc: Cheng et al., "Boundary IoU: Improving Object-Centric Image Segmentation Evaluation", CVPR 2021.
- **BF-Score** (Boundary F1): precision/recall giữa tập pixel biên dự đoán và GT trong ngưỡng khoảng cách cho phép.
- **ASD** (Average Surface Distance): trung bình khoảng cách Euclidean từ mỗi pixel biên dự đoán tới pixel biên GT gần nhất (và ngược lại), dùng `scipy.ndimage.distance_transform_edt` để tính hiệu quả (KHÔNG vòng lặp pixel-by-pixel).

Thiết kế class `BoundaryMetrics` theo đúng pattern `SegmentationMetrics` hiện có (`update()` tích luỹ theo batch, `compute()` trả dict) để cắm thẳng vào `validate()` của training script mới mà không đổi cấu trúc vòng lặp. Loại pixel ignore khỏi cả GT lẫn prediction trước khi dilate/so sánh (cùng `ignore_index` như 3.1).

**TODO (chốt với Huan 29/8/2026):** `src/utils/boundary_metrics.py` **chưa được xây** — thực nghiệm "Baseline vs BCE Loss" (Run 2, `src/train_bce_edge.py`, repo `BoundaryLossSolution`) đã cố tình bỏ qua bước này để giữ đúng phạm vi (chỉ BCE edge loss). `summary.txt` do `train_bce_edge.py` sinh ra hiện điền `N/A` cho 4 trường "Best validation BFScore"/"Best Boundary IoU @1/@2/@4". Khi làm Bước 6 này, cần quay lại: (1) implement `BoundaryMetrics` theo đúng spec ở trên, (2) cắm vào `validate()` của `src/train_bce_edge.py` (và mọi script `train_*_boundary*.py` sau này), (3) sửa `write_bce_edge_summary()` trong `src/train_bce_edge.py` để điền số thật thay vì N/A.

---

## 5. Bước 7 — Training script dùng chung (`src/train_boundary_research.py`)

Copy nguyên `src/train_unet_former_resnet18.py` làm khung (đã có DDP/AMP/checkpoint/early-stopping/CSV logging đúng chuẩn), sửa 3 chỗ:
1. Model: gọi `model(images, return_fused_feature=True)` để lấy cả `logits` và `fused_feature`, thêm `BoundaryHead` (3.1) chạy trên `fused_feature` để có `edge_logits`.
2. Loss: thay `CombinedLoss` bằng `DynamicBoundaryTotalLoss` (3.4), đọc cấu hình 4 cờ (`USE_BCE/USE_AFFINITY/DYNAMIC_WEIGHTS/ALPHA`) từ YAML.
3. Validate: gọi thêm `BoundaryMetrics` (Bước 6) song song `SegmentationMetrics`, log cả 2 nhóm chỉ số vào `benchmark_results.csv`.

Script phải nhận thêm 1 arg mới `--fold` (optional) để dùng chung được cho cả Phase 1 (không cần) và Phase 2 (chọn đúng `fold{N}_train.txt`/`fold{N}_val.txt`) mà không phải viết 2 script riêng.

---

## 6. Bước 8 — Sinh 12 file config (8 Phase 1 + 10 Phase 2, seed=19 ở mọi file)

Mẫu chung mọi config kế thừa từ `configs/unet_former_resnet18_combineLoss/luot1_500.yaml` (giữ nguyên `MODEL`, `OPTIMIZER`, phần lớn `TRAIN`), chỉ đổi:
- `DATASET.TRAIN_SPLIT_FILE` / `VAL_SPLIT_FILE` → trỏ đúng `train_full_fixed.txt` + `val_clean_<N>.txt` (Phase 1) hoặc `spatial_folds/fold{i}_{train,val}.txt` (Phase 2).
- `TRAIN.SEED: 19` (bắt buộc, mọi file).
- Thêm block mới `BOUNDARY_LOSS:` (USE_BCE, USE_AFFINITY, DYNAMIC_WEIGHTS, ALPHA, WINDOW_K, DISTANCE, MARGIN, CONNECTIVITY, DILATION_RADIUS, POS_WEIGHT_MAX) theo bảng ở mục 3.4 và spec ở 3.1.
- `OUTPUT.WORK_DIR` → `work_dirs/phase1/run{1..4}...` hoặc `work_dirs/phase2/fold{1..5}_{baseline,proposed}`.

Phase 2 "baseline" (5 run) dùng đúng cấu hình Run 1 (α=0), "proposed" (5 run) dùng đúng cấu hình Run 4 với α* đã chọn từ Phase 1 — KHÔNG re-tune α trong Phase 2 (đúng thiết kế 2 giai đoạn của đề cương: Phase 1 tune, Phase 2 chỉ benchmark).

---

## 7. Bước 9 — Tổng hợp kết quả (`Tools/aggregate_results.py`)

Đọc toàn bộ `work_dirs/phase1/*/benchmark_results.csv` (lấy dòng `is_best=True`) → xuất đúng format Bảng 1 (mIoU, Boundary IoU d=3, BF-Score theo 4 run). Đọc `work_dirs/phase2/*/benchmark_results.csv` → xuất Bảng 2 (mIoU từng fold + Mean±Std cho baseline và proposed, cột "Cải thiện Δ"). Xuất ra cả `.csv` và bảng markdown in thẳng được vào bài báo. Việc chọn báo cáo mIoU trên bộ lớp nào (nhiều paper OpenEarthMap báo cáo song song vài cách tính) để lúc viết bài tự tính lại từ per-class IoU đã có trong CSV, không cần pipeline tự quyết định.

---

## 8. Thứ tự chạy thực tế trên Kaggle (sau khi code đã sinh xong)

```bash
# 1. Splits (1 lần, seed=19)
python Tools/create_splits.py --data-root <root> --output-dir <work_base> --seed 19
python Tools/build_clean_val_split.py --data-root <root> --output-dir <work_base>
python Tools/build_spatial_folds.py --data-root <root> --output-dir <work_base>

# 2. Dry-run TỪNG config trước khi chạy thật (bắt lỗi shape/path trong <2 phút, không tốn quota GPU)
torchrun --nproc_per_node=2 src/train_boundary_research.py --config configs/research_boundary/phase1_fixed_split/run1_baseline.yaml --dry-run
# ... lặp lại --dry-run cho toàn bộ 12 config trước khi submit thật

# 3. Phase 1 — 8 run tuần tự (hoặc song song nếu có nhiều notebook Kaggle)
for cfg in configs/research_boundary/phase1_fixed_split/*.yaml; do
    torchrun --nproc_per_node=2 src/train_boundary_research.py --config "$cfg"
done

# 4. Chọn alpha* tốt nhất từ 4 run alpha_sweep, cập nhật vào run4_dynamic_boundary.yaml + toàn bộ config Phase 2 "proposed"

# 5. Phase 2 — 10 run (5 fold x baseline/proposed)
for cfg in configs/research_boundary/phase2_5fold_cv/*.yaml; do
    torchrun --nproc_per_node=2 src/train_boundary_research.py --config "$cfg"
done

# 6. Tổng hợp
python Tools/aggregate_results.py --phase1-dir work_dirs/phase1 --phase2-dir work_dirs/phase2 --out docs/results
```

---

## 9. Checklist trước khi giao cho Claude Code chạy full

- [ ] `dataset/train_full_fixed.txt`, `dataset/val_clean_*.txt`, `dataset/spatial_folds/fold*_*.txt` đã sinh xong với `--seed 19`, đã kiểm tra log "0 region(s) duplicated".
- [ ] Đã xác nhận giá trị `ignore_index` thực tế mà pipeline hiện dùng (dùng cho `extract_edge_gt`/`DynamicBoundaryTotalLoss`, mục 3.1/3.4).
- [ ] `src/losses/*.py` (Bước 5) đã unit-test độc lập (forward + backward, giống cách `unet_former_resnet18.py` đã được verify) trước khi cắm vào training loop.
- [ ] Đã chạy đủ 7 sanity check liệt kê ở mục 3.1 cho `boundary_bce.py` trước khi submit Run 2 full.
- [ ] `src/utils/boundary_metrics.py` (Bước 6) đã test trên vài mask giả lập, so khớp thủ công ít nhất 1 case biết trước đáp số.
- [ ] Toàn bộ 12 config Phase 1 + Phase 2 đã `--dry-run` thành công (không lỗi shape/path) trước khi submit run thật (tránh cháy quota GPU Kaggle vì lỗi cấu hình).
- [ ] `SEED: 19` xuất hiện trong TẤT CẢ config mới — grep nhanh: `grep -L "SEED: 19" configs/research_boundary/**/*.yaml` phải trả về rỗng.
- [ ] Đã ước tính lại thời gian chạy thực tế bằng 1 run `--dry-run` đo tốc độ/iteration trước khi cam kết chạy đủ 40.000 iteration × 18 run trên Kaggle. **Đã đo thực tế ở Run 1: 0.193s/iter, ~2h08m/40k iter** — dùng số này để ước lượng ngân sách 8 run Phase 1 (~16-17h, cần ≥2 phiên Kaggle).
- [ ] (Optional) Phase 3 backbone check (mục 10) chỉ chạy SAU KHI Phase 1+2 đã xong và ổn định — không chạy song song để tránh lẫn ngân sách compute với pipeline chính.

---

## 10. (Tuỳ chọn) Phase 3 — Kiểm chứng tổng quát hoá qua backbone khác

**Chỉ thực hiện sau khi** Phase 1 (đã chọn α* và biến thể loss tốt nhất trong Run 1-4 + alpha sweep) và Phase 2 (5-fold CV) đã hoàn tất và ổn định trên ResNet-18. Mục đích của Phase 3 là kiểm tra cải thiện đo được từ boundary loss có còn giữ nguyên khi đổi backbone hay không — **không phải** để đi tìm backbone tốt nhất, và không thay thế Phase 1/2 làm trọng tâm nghiên cứu.

**Nguyên tắc thiết kế bắt buộc — 2 run cho mỗi backbone bổ sung, không phải 1:** Chạy cả baseline (α=0, giống cấu hình Run 1) và cấu hình tốt nhất đã chọn từ Phase 1 (α*, biến thể loss tốt nhất) trên backbone mới, cùng `seed=19`, cùng `train_full_fixed`/`val_clean`, cùng `MAX_ITERS=40000` như Phase 1. Không chạy 1 run duy nhất (chỉ config tốt nhất) rồi so tuyệt đối với số của ResNet-18 — như vậy không tách được phần cải thiện do loss khỏi phần cải thiện do backbone mạnh hơn (nhầm lẫn nhân quả). Chỉ số so sánh chính là **Δ mIoU (proposed − baseline) trên backbone mới**, đối chiếu với Δ mIoU tương ứng đã đo trên ResNet-18 ở Phase 1. Nếu Δ giữ nguyên hoặc lớn hơn → bằng chứng ủng hộ tính tổng quát hoá. Nếu Δ co lại đáng kể/về gần 0 → phát hiện đáng báo cáo khác (hiệu ứng phụ thuộc capacity backbone), đưa vào Limitations/Discussion, không cần mở rộng thêm backbone.

**Backbone đầu tiên (bắt buộc nếu làm Phase 3): ResNet-101.**
- Rẻ và ít rủi ro kỹ thuật nhất — cùng họ ResNet, sẵn trong `timm`, có tiền lệ dùng trong Track A của repo.
- **Cần adapter channel:** ResNet-101 dùng bottleneck block, output channels `[256,512,1024,2048]` khác hẳn ResNet-18 basic block `[64,128,256,512]` mà decoder GLTB hiện đang hardcode kỳ vọng. Cần parametrize input channels của decoder theo `backbone.feature_info` của timm, hoặc thêm 1×1 conv projection về đúng `[64,128,256,512]` trước khi đưa vào GLTB decoder hiện có.
- Theo đúng quy ước "file độc lập": tạo file model mới `src/models/unet_former_resnet101.py`, KHÔNG sửa `unet_former_resnet18.py`.
- Ước lượng compute (ngoại suy thô từ Run 1 đo thực tế 2h08m/40k iter trên ResNet-18, ResNet-101 nặng hơn ~3.8 lần tham số — ~44.5M so với ~11.65M): khoảng 4-8h/run → **~8-16h cho cả 2 run**.

**Backbone thứ 2 (chỉ làm nếu ResNet-101 cho tín hiệu tích cực):** ưu tiên một kiến trúc khác họ để tăng sức thuyết phục tổng quát hoá — ví dụ Swin-Tiny hoặc EfficientNet, cả hai đều tích hợp sẵn qua `timm.create_model(..., features_only=True)`, rủi ro kỹ thuật thấp. **Không dùng Mamba ở giai đoạn Phase 3 này** — backbone dạng state-space model không có cấu trúc feature pyramid tương thích sẵn với decoder GLTB, cần thiết kế adapter riêng, thêm dependency mới (`mamba-ssm`, rủi ro compile trên môi trường Kaggle), và chưa có tiền lệ tích hợp trong repo — rủi ro kỹ thuật không tương xứng với mục đích chỉ là kiểm chứng nhanh. Nếu muốn khám phá Mamba, nên coi là một hướng nghiên cứu/track riêng, không gộp vào nghiên cứu boundary loss hiện tại.

**Điều kiện dừng:** nếu Δ trên ResNet-101 co lại đáng kể hoặc về gần 0, dừng lại, không mở rộng thêm backbone thứ 2.
