# Baseline UNetFormer (ResNet-18) — OpenEarthMap

Baseline cho đề tài **"Dynamic Boundary-Aware Loss via Adaptive BCE-Affinity Coupling"**
(xem `docs/idea_research.md`): kiến trúc UNetFormer (encoder ResNet-18 pretrained, decoder
GLTB paper-faithful), loss `L_region` thuần (CrossEntropy + Dice) — **chưa** có bất kỳ thành
phần boundary/affinity loss nào. Đây là mốc so sánh (hàng "Baseline" trong Bảng 1 của đề cương)
cho các biến thể boundary-loss sẽ làm sau, tách biệt hoàn toàn khỏi phần này.

> **Muốn xem nhanh tất cả thí nghiệm đã triển khai (cấu hình, kết quả, trạng thái) ở 1 chỗ?**
> Xem `docs/tong_quan_thuc_nghiem.md`. File README này chỉ mô tả chi tiết riêng thí nghiệm Baseline
> (mục 1-7) + thí nghiệm "+ BCE Edge" (mục 8, Run 2) + thí nghiệm "+ Static Boundary" (mục 9, Run 3)
> + thí nghiệm "λ₂=0.5" (mục 10, Run 3b) + thí nghiệm "Dynamic Boundary-Aware Loss" (mục 11, Run 4).

---

## 1. Tổng quan quá trình xử lý

```
                 ┌────────────────────────────┐
                 │  OpenEarthMap trên Kaggle   │
                 │  (images/train, images/val, │
                 │   labels/train, labels/val) │
                 └──────────────┬─────────────┘
                                │  resolve_dataset_paths()
                                │  (tự dò path mount thật +
                                │   "labels/" hay "label/")
                                ▼
                 ┌────────────────────────────┐
                 │   OpenEarthMapDataset       │  KHÔNG split/subset/augment-pad —
                 │   (đọc thẳng toàn bộ ảnh    │  đọc đúng số ảnh train/val mà
                 │    trong img_dir/mask_dir)  │  dataset đang có sẵn
                 └──────────────┬─────────────┘
                                ▼
        ┌───────────────────────────────────────────┐
        │  UNetFormer: encoder ResNet-18 (timm)      │
        │  → 3× GLTB (Global-Local Transformer Block)│
        │  → Weighted Fusion + FeatureRefinementHead │
        │  → Fused Feature (stride 4) → classifier   │
        └──────────────────────┬──────────────────────┘
                                ▼
        Loss = CombinedLoss = CrossEntropy + Dice (trên logits)
                                │
                 AdamW + poly-LR decay (warmup 500 iter)
                                │
              ┌─────────────────┴─────────────────┐
              │   Training loop: 40.000 iteration   │
              │   Cứ 4.000 iteration → 1 lần val    │
              └─────────────────┬─────────────────┘
                                ▼
       Mỗi lần val: SegmentationMetrics (mIoU + per-class IoU)
       → visualizer segmentation (vis/) + visualizer boundary (vis/boundary/)
       → ghi benchmark_results.csv, train_log.txt
       → lưu best_model.pth (nếu mIoU cải thiện) + latest_checkpoint.pth
                                ▼
       Kết thúc (đủ 40k iter hoặc early-stopping):
       learning_curves.png, per_class_iou_best.png, final_summary.txt
```

**Các quyết định thiết kế quan trọng của baseline này:**

- **Không split/subset/augment dataset**: dùng thẳng toàn bộ `images/train` + `images/val`
  (và `labels/train` + `labels/val`) đúng như dataset đã có sẵn trên Kaggle — không sinh file
  split trung gian, không cắt bớt ảnh, không lật/xoay thêm ảnh để "độn" số lượng.
- **Dataset root/tên thư mục label được resolve động lúc chạy** (`resolve_dataset_paths()` trong
  `src/train_unet_former_resnet18.py`): Kaggle có thể mount dataset ở path khác nhau tuỳ phiên,
  và các mirror khác nhau đặt tên thư mục nhãn là `labels/` (số nhiều) hoặc `label/` (số ít) —
  script tự dò ra cấu trúc thật thay vì yêu cầu sửa tay YAML mỗi lần.
- **Checkpoint/resume**: sau mỗi lần validation, `latest_checkpoint.pth` được ghi đè (kèm trạng
  thái optimizer/scaler/early-stopping/best-score) — nếu quá trình train bị ngắt (Kaggle hết giờ,
  mất kết nối, ...), chạy lại đúng lệnh cũ sẽ **tự động resume** từ đó, không cần thêm tham số.
- **Visualizer sau mỗi lần val**: 4 mẫu ảnh val được vẽ lại 2 kiểu — segmentation
  (Ảnh gốc | Ground Truth | Prediction, tô màu theo 9 lớp) và boundary (biên GT vs biên Pred
  overlay lên ảnh gốc) — để theo dõi trực quan model học được gì qua từng mốc 4.000 iteration.
- **Log bền vững**: toàn bộ log được ghi ra `train_log.txt` (không chỉ in ra màn hình) để xem lại
  được sau khi phiên Kaggle đã đóng.

---

## 2. Kiến trúc & Loss

| Thành phần | Chi tiết |
|---|---|
| Encoder | `resnet18.fb_swsl_ig1b_ft_in1k` (timm, pretrained), 4 stage stride 4/8/16/32 |
| Decoder | 3× **GLTB** (Global-Local Transformer Block: window attention + local conv) phân cấp, ghép bằng **WF** (Weighted Fusion) có trọng số học được, tầng nông nhất dùng **FeatureRefinementHead** (spatial + channel attention) → sinh ra "Fused Feature" (stride 4) |
| Số tham số | ~11.65M (encoder 11.18M + decoder 0.47M) — nhẹ hơn track ResNeXt101_32x16d cũ trong project ~16-17 lần |
| Loss | `CombinedLoss` = CrossEntropy + Dice (trọng số 1:1, `ignore_index=255`), tính trên logits suy ra từ Fused Feature — **không** có nhánh aux loss |
| Optimizer | AdamW, `weight_decay=0.01`, `grad_clip=5.0` |
| LR schedule | poly decay (`power=0.9`), warmup 500 iteration đầu |
| Seed | **19**, cố định cho mọi phần (model init, shuffle dữ liệu, augmentation) |

Chi tiết đầy đủ (lý do chọn ResNet-18, đối chiếu với paper gốc UNetFormer) xem
`docs/phan_tich_baseline_va_dataset.md` phần 1.

---

## 3. Cấu trúc thư mục

```
BoundaryLossSolution/
├── README.md                          # file này
├── requirements.txt                   # deps thêm (torch/torchvision đã có sẵn trên Kaggle)
├── configs/
│   └── unet_former_resnet18_combineLoss/
│       └── baseline.yaml              # config DUY NHẤT của baseline (seed=19, 40k iter, val/4k)
├── scripts/
│   ├── run_baseline.sh                # 1 lệnh: cài deps + train (full hoặc --dry-run)
│   └── resume_baseline.sh             # resume sau khi mất /kaggle/working (session mới)
├── src/
│   ├── models/unet_former_resnet18.py # kiến trúc UNetFormer (đã verify forward/backward)
│   ├── data/
│   │   ├── dataset.py                 # OpenEarthMapDataset — đọc thẳng thư mục, không split
│   │   └── transforms.py              # augmentation (train) / chỉ resize+normalize (val)
│   ├── utils/
│   │   ├── losses.py                  # CombinedLoss (CE + Dice)
│   │   ├── metrics.py                 # SegmentationMetrics (mIoU + per-class IoU)
│   │   ├── callbacks.py               # EarlyStopping
│   │   ├── visualizer.py              # save_visualization (Original | GT | Prediction)
│   │   └── boundary_visualizer.py     # save_boundary_visualization (biên GT vs Pred)
│   └── train_unet_former_resnet18.py  # script train chính (DDP + AMP + checkpoint/resume)
├── Tools/
│   ├── get_resume_checkpoint.py       # dùng bởi resume_baseline.sh (resume qua session mới)
│   ├── build_clean_val_split.py       # thuộc Phase 2 nghiên cứu boundary-loss (chưa dùng ở đây)
│   └── build_spatial_folds.py         # thuộc Phase 2 nghiên cứu boundary-loss (chưa dùng ở đây)
└── docs/                              # đề cương nghiên cứu + phân tích (bối cảnh, không phải code)
```

`Tools/build_clean_val_split.py`, `Tools/build_spatial_folds.py` và toàn bộ nội dung `docs/`
thuộc phạm vi nghiên cứu Dynamic Boundary-Aware Loss (Phase 1/2) sẽ làm **sau**, không liên quan
tới việc chạy baseline này.

---

## 4. Cài đặt & chạy trên server (Kaggle, 2× T4)

### 4.1. Chuẩn bị dataset

Trên Kaggle Notebook: attach dataset qua panel **"Add Input"** →
`aletbm/global-land-cover-mapping-openearthmap`. Dataset sẽ tự mount tại
`/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap` (đúng giá trị mặc định
trong `configs/unet_former_resnet18_combineLoss/baseline.yaml`). Nếu Kaggle mount ở path khác
(tuỳ phiên/cách attach), **không cần sửa YAML** — xem mục 4.3.

### 4.2. Chạy full training (40.000 iteration)

```bash
cd BoundaryLossSolution
bash scripts/run_baseline.sh
```

Script sẽ tự: (1) `pip install -r requirements.txt`, (2) chạy
`torchrun --nproc_per_node=2 src/train_unet_former_resnet18.py --config configs/unet_former_resnet18_combineLoss/baseline.yaml`.
Không cần bước tạo split file nào — dataset được đọc trực tiếp.

### 4.3. Nếu Kaggle mount dataset ở path khác

```bash
DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_baseline.sh
```

`DATA_ROOT` được truyền thẳng vào `--data-root` của train script, ghi đè `ROOT_DIR`/`VAL_ROOT_DIR`
trong config — không tạo file trung gian nào.

### 4.4. Dry-run (kiểm tra luồng trước khi chạy full)

```bash
bash scripts/run_baseline.sh --dry-run
```

Chạy 5 iteration, val 2 lần (mỗi lần 4 ảnh) — dùng để phát hiện lỗi shape/path trong vài phút,
không tốn quota GPU. Nên chạy bước này trước khi chạy full 40k iteration.

### 4.5. Chạy trực tiếp bằng `torchrun` (không qua script)

```bash
torchrun --nproc_per_node=2 src/train_unet_former_resnet18.py \
    --config configs/unet_former_resnet18_combineLoss/baseline.yaml
```

Tham số dòng lệnh của `train_unet_former_resnet18.py`:

| Tham số | Ý nghĩa |
|---|---|
| `--config` (bắt buộc) | Đường dẫn file YAML config |
| `--dry-run` | Smoke-test: 5 iteration, val mỗi 2 iteration, 4 ảnh mỗi split |
| `--resume <path>` | Resume thủ công từ 1 file checkpoint cụ thể |
| `--data-root <path>` | Ghi đè `ROOT_DIR`/`VAL_ROOT_DIR` trong config |

Trên máy không có GPU (chỉ để dev/test cục bộ), script tự chuyển sang backend `gloo` + CPU,
tắt AMP — hành vi trên Kaggle (GPU thật, backend `nccl` + AMP) không đổi.

---

## 5. Checkpoint & Resume

Có 2 tình huống cần resume, xử lý khác nhau:

### 5.1. Bị ngắt nhưng `/kaggle/working` vẫn còn (cùng phiên notebook)

Chạy lại **đúng lệnh cũ** — train script tự phát hiện `latest_checkpoint.pth` đã có sẵn trong
`WORK_DIR` và **tự động resume**, không cần thêm tham số gì:

```bash
bash scripts/run_baseline.sh
```

Log sẽ hiện dòng `Auto-resume: found existing checkpoint at ... — resuming from it.`

### 5.2. Session Kaggle mới, `/kaggle/working` đã bị xoá sạch

Cần đã tải `latest_checkpoint.pth` về máy từ phiên trước (hoặc lưu làm Kaggle Dataset/Output).
Upload lại file đó lên `/kaggle/working/...` (qua panel "Data" hoặc File Browser), rồi:

```bash
bash scripts/resume_baseline.sh
# hoặc bỏ qua bước hỏi đường dẫn nếu đã biết trước:
bash scripts/resume_baseline.sh --path /kaggle/working/.../latest_checkpoint.pth
```

Upload trực tiếp trong 1 cell notebook (thay vì gõ đường dẫn qua terminal):

```python
%run "Tools/get_resume_checkpoint.py" --browser --default-path <WORK_DIR>/latest_checkpoint.pth
# sau khi thấy "✔ Đã lưu checkpoint" → chạy: bash scripts/resume_baseline.sh
```

Checkpoint lưu đầy đủ: trọng số model, optimizer, GradScaler (AMP), trạng thái early-stopping,
best mIoU + iteration/lần-val đạt best, và toàn bộ lịch sử val trước đó (đọc lại từ
`benchmark_results.csv`) — resume xong, `final_summary.txt`/`learning_curves.png` vẫn tính đúng
trên toàn bộ quá trình train (không chỉ đoạn sau khi resume).

---

## 6. Kết quả đầu ra (`OUTPUT.WORK_DIR` trong config)

| File/thư mục | Nội dung |
|---|---|
| `best_model.pth` | Trọng số model tại lần val có mIoU cao nhất |
| `latest_checkpoint.pth` | Checkpoint đầy đủ để resume (ghi đè sau mỗi lần val) |
| `benchmark_results.csv` | 1 dòng / lần val: `iter`, `val_round`, `mIoU`, `val_loss`, IoU từng lớp trong 9 lớp, `is_best` |
| `train_log.txt` | Toàn bộ log console, có timestamp, append qua các lần resume |
| `final_summary.txt` | Tổng kết cuối: **best mIoU đạt ở lần val thứ mấy + iteration nào**, IoU từng lớp tại đó, tổng thời gian train |
| `learning_curves.png` | 2 đồ thị: val mIoU và val Loss theo từng lần val (mỗi 4.000 iteration) |
| `per_class_iou_best.png` | Biểu đồ cột IoU 9 lớp tại checkpoint tốt nhất |
| `time_<tag>.txt` | Giờ bắt đầu/kết thúc + tổng thời gian train |
| `vis/iterNNNNNN_sK.png` | Visualizer segmentation: Ảnh gốc \| Ground Truth \| Prediction (4 ảnh mẫu / lần val) |
| `vis/boundary/iterNNNNNN_sK.png` | Visualizer boundary: biên GT (cyan) vs biên Pred (magenta) vs trùng nhau (trắng), overlay lên ảnh gốc |

---

## 7. Ghi chú phạm vi

Baseline này **chỉ** gồm `L_region` (CE + Dice) — KHÔNG có boundary loss / affinity loss / λ
động / DAPCN. Các thành phần đó thuộc nghiên cứu "Dynamic Boundary-Aware Loss" mô tả trong
`docs/idea_research.md` và `docs/workflow_2.md`, được xây dựng trong các file/thư mục **mới**
riêng biệt (không sửa bất kỳ file nào trong `src/` liệt kê ở mục 3), để không ảnh hưởng tới kết
quả baseline này khi so sánh. Thực nghiệm đầu tiên thuộc hướng này — "+ BCE Edge" (Run 2 của
Bảng 1) — xem mục 8 bên dưới. Thực nghiệm thứ hai — "+ Static Boundary" (Run 3, cộng thêm
affinity loss lên trên Run 2) — xem mục 9.

---

## 8. Thực nghiệm 2 — "+ BCE Edge" (Run 2, Bảng 1 `docs/idea_research.md`)

Thêm **duy nhất 1 thành phần** lên trên baseline (mục 1-7): boundary/edge supervision loss.

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda_{\text{edge}} \cdot \mathcal{L}_{\text{edge}}$$

- $\mathcal{L}_{\text{seg}}$ = `CombinedLoss` (CE + Dice) — **tái sử dụng nguyên**, không sửa, không viết lại.
- $\mathcal{L}_{\text{edge}}$ = balanced BCE trên `edge_logits` (sinh bởi 1 `BoundaryHead` phụ gắn trên
  Fused Feature của decoder) so với `edge_gt` (trích từ ground-truth mask, 4-connected, không dilate
  thêm) — pos_weight cân bằng lớp thiểu số "edge". Spec kỹ thuật đầy đủ: `docs/workflow_2.md` mục 3.1.
- $\lambda_{\text{edge}}$ = đã chạy thật **2 giá trị**: 0.2 và 0.4 (ablation, xem mục 8.2) — chưa
  phải α* từ alpha-sweep chính thức (hạ tầng đó thuộc Phase 1 đầy đủ, chưa xây), nhưng kết quả mục
  8.5 cho thấy **0.4 tốt hơn**, được chọn làm α kế thừa cho Run 3 (mục 9).

Đây là **thực nghiệm hoàn toàn độc lập** với baseline: file mới 100%, không sửa bất kỳ file nào liệt
kê ở mục 3 (baseline vẫn `bash scripts/run_baseline.sh` chạy lại y hệt bất kỳ lúc nào), work dir
riêng, script chạy riêng.

### 8.1. File mới

```
BoundaryLossSolution/
├── src/
│   ├── losses/                              # 🆕 package mới — KHÔNG sửa src/utils/losses.py cũ
│   │   ├── __init__.py
│   │   └── boundary_bce.py                  # extract_edge_gt, BoundaryHead, BalancedBCEEdgeLoss,
│   │                                          # EdgeStatsAccumulator, compute_pos_weight
│   └── train_bce_edge.py                    # script train — ĐỘC LẬP với train_unet_former_resnet18.py,
│                                              # chỉ dùng chung src/data, src/utils, model UNetFormer
├── configs/
│   └── unet_former_resnet18_bce_edge/
│       └── bce_edge.yaml                    # kế thừa mọi hyperparam từ baseline.yaml (seed=19, 40k
│                                              # iter...), thêm block BOUNDARY_LOSS
├── Tools/
│   └── measure_edge_ratio.py                # tool ĐỘC LẬP, tuỳ chọn: đo r_edge/pos_weight "chính
│                                              # thức" trên full epoch, KHÔNG bắt buộc chạy trước train
└── scripts/
    ├── run_bce_edge.sh                      # launcher riêng — KHÔNG gọi run_baseline.sh
    └── resume_bce_edge.sh                   # resume riêng — KHÔNG gọi resume_baseline.sh
```

### 8.2. Cách chạy

```bash
# Full training (40.000 iteration, lambda_edge=0.4 mặc định)
bash scripts/run_bce_edge.sh

# Dry-run (5 iteration, val mỗi 2, 4 ảnh/split — kiểm tra luồng + 7 sanity check trước khi chạy thật)
bash scripts/run_bce_edge.sh --dry-run

# Ablation nhanh lambda_edge, KHÔNG cần sửa YAML — WORK_DIR tự thêm hậu tố _lambdaX.XX
# (mỗi giá trị lambda_edge có checkpoint/CSV/summary.txt riêng, tự resume đúng của chính nó)
bash scripts/run_bce_edge.sh 0.2
bash scripts/run_bce_edge.sh 0.2 --dry-run

# Nếu Kaggle mount dataset ở path khác:
DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_bce_edge.sh

# Resume (session Kaggle mới, /kaggle/working đã bị xoá) — cùng cú pháp lambda:
bash scripts/resume_bce_edge.sh
bash scripts/resume_bce_edge.sh 0.2
bash scripts/resume_bce_edge.sh --path /kaggle/working/.../latest_checkpoint.pth
```

Cùng trong 1 session (chưa mất `/kaggle/working`): chạy lại đúng lệnh `run_bce_edge.sh` cũ sẽ tự
auto-resume, y hệt cơ chế của baseline (mục 5.1).

Trước khi chạy thật, có thể (tuỳ chọn, không bắt buộc) đo `r_edge`/`pos_weight` "chính thức" trên
toàn bộ epoch train:

```bash
python Tools/measure_edge_ratio.py \
    --config configs/unet_former_resnet18_bce_edge/bce_edge.yaml \
    --out configs/unet_former_resnet18_bce_edge/edge_ratio_stats.json
# rồi trỏ BOUNDARY_LOSS.EDGE_STATS_FILE trong bce_edge.yaml tới file JSON này
```

Không chạy tool trên cũng không sao — `train_bce_edge.py` tự ước lượng `pos_weight` lúc khởi động
(bounded, vài trăm batch đầu, mọi rank DDP tự tích luỹ rồi `all_reduce`), log rõ là ước lượng
xấp xỉ (APPROXIMATE), và lưu lại đúng giá trị đã dùng vào checkpoint để resume không đo lại.

### 8.3. Sanity checks tự động

Ngay khi khởi động (kể cả `--dry-run`), script tự chạy 7 sanity check theo `docs/workflow_2.md` mục
3.1 (shape/binary `edge_gt`, batch toàn-ignore → loss=0, `0 < r_edge < 1`, `pos_weight` hữu hạn dương,
raw-logits-không-sigmoid, overlay trực quan, gradient khác 0 trên `BoundaryHead`) — fail cứng thì
dừng training ngay (trừ overlay trực quan, chỉ định tính). Kết quả pass/fail được ghi vào
`summary.txt` (mục "PRE-FLIGHT VALIDATION").

### 8.4. Output đầu ra (thêm so với mục 6 của baseline)

| File/thư mục | Nội dung |
|---|---|
| `summary.txt` | Báo cáo đầy đủ: header thí nghiệm, best checkpoint (mIoU-9 + mIoU-8 hậu kỳ), label/valid-pixel policy, edge-target definition, dataset-level edge statistics, BCE imbalance correction, pre-flight validation |
| `final_summary.txt` | Cùng format baseline (tương thích ngược) |
| `benchmark_results.csv` | Thêm cột `l_seg`, `l_edge`, `l_total` bên cạnh các cột đã có ở baseline |
| `sanity/edge_gt_overlay_sK.png` | Overlay `edge_gt` (cyan) lên ảnh gốc — sanity check #6, tự kiểm tra bằng mắt |
| `vis/`, `vis/boundary/` | Y hệt baseline (visualizer không đổi) |

### 8.5. Kết quả thật (Kaggle 2× T4, 40.000 iteration mỗi run)

| Chỉ số | Baseline (α=0) | λ_edge=0.2 | λ_edge=0.4 |
|---|---:|---:|---:|
| mIoU-9 (best, chọn checkpoint) | 0.6551 | 0.6490 | **0.6566** |
| mIoU-8 (loại Background, hậu kỳ) | — | 0.6093 | 0.6183 |
| Boundary BFScore | — | 0.6082 | 0.6123 |
| Boundary IoU @1 | — | 0.0582 | 0.0585 |
| Boundary IoU @2 | — | 0.1169 | 0.1165 |
| Boundary IoU @4 | — | 0.2193 | 0.2186 |
| Thời gian train | 02h08m35s | 02h10m59s | 02h04m00s |

λ_edge=0.2 làm mIoU-9 giảm nhẹ so với baseline; λ_edge=0.4 cải thiện nhẹ. → **Chốt dùng λ_edge=0.4
làm α cho Run 3** (mục 9), kế thừa xuyên suốt tới khi có alpha-sweep chính thức trên cấu hình Run 4.
Checkpoint/CSV/ảnh visualizer/`summary.txt` đầy đủ lưu ngoài repo tại
`D:\Nghien Cuu Sinh\Lab\Boundary\BCE_Lambda_0.2\` và `BCE_Lambda_0.4\`.

**Ghi chú phạm vi (đã cập nhật — khoảng trống cũ đã lấp):** lúc `train_bce_edge.py` chạy thật,
`src/utils/boundary_metrics.py` **chưa tồn tại**, nên `summary.txt` sinh trực tiếp bởi script này
vẫn để `N/A` ở 4 trường "Best validation BFScore" / "Best Boundary IoU @1/@2/@4" — điều đó **không
đổi** (không sửa lại output cũ của 1 run đã hoàn tất). Sau khi `boundary_metrics.py` được xây xong
(Bước 6 của `docs/workflow_2.md`), số liệu thật cho Run 2 được tính hậu kỳ bằng
`Tools/eval_boundary_metrics.py` (forward lại `best_model.pth` qua val set) rồi điền vào 1 **file
mới** (`*_summary_boundary_filled.txt`, qua `Tools/patch_bce_edge_summary.py` — không ghi đè
`summary.txt` gốc) — đó là nguồn số liệu bảng trên. Từ Run 3 trở đi, `boundary_metrics.py` được
tích hợp trực tiếp vào `validate()`, không cần bước hậu kỳ này nữa (xem mục 9.5). Tiêu chí chọn
best checkpoint / early stopping vẫn dùng mIoU 9-class (giống hệt baseline) để so sánh công bằng —
mIoU-8 (loại Background) trong `summary.txt` chỉ để báo cáo, không dùng để chọn checkpoint.

---

## 9. Thực nghiệm 3 — "+ Static Boundary" (Run 3, Bảng 1 `docs/idea_research.md`)

Thêm **thành phần thứ 2** lên trên thực nghiệm 2 (mục 8): affinity loss (contrastive
feature-distance gần biên), trọng số tĩnh.

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{region}} + \alpha \cdot (\lambda_1 \cdot \mathcal{L}_{\text{BCE\_edge}} + \lambda_2 \cdot \mathcal{L}_{\text{Affinity}}), \quad \lambda_1 = \lambda_2 = 1 \text{ (cố định)}$$

- $\mathcal{L}_{\text{region}}$, $\mathcal{L}_{\text{BCE\_edge}}$ — y hệt mục 8, tái sử dụng nguyên.
- $\mathcal{L}_{\text{Affinity}}$ — so sánh khoảng cách đặc trưng (cosine hoặc L2) giữa mỗi pixel
  gần biên và các hàng-xóm trong cửa sổ K×K, tính trên **Fused Feature** (64 kênh, stride 4, decoder
  — KHÔNG dùng logits): kéo gần feature cùng nhãn GT, đẩy xa feature khác nhãn (có margin). Dùng
  chung `extract_edge_gt()` với $\mathcal{L}_{\text{BCE\_edge}}$ để đảm bảo 1 định nghĩa biên duy
  nhất xuyên suốt. Spec kỹ thuật đầy đủ: `docs/workflow_2.md` mục 3.2.
- $\lambda_1 = \lambda_2 = 1$ **cố định, tĩnh** — không có dynamic schedule theo iteration (dynamic
  $\lambda_1(t)/\lambda_2(t)$ là Run 4 "Ours", ngoài phạm vi thực nghiệm này).
- $\alpha = 0.4$, kế thừa trực tiếp giá trị $\lambda_{\text{edge}}$ tốt nhất đã đo ở Run 2 (mục 8.5)
  — **tạm thời**, chờ alpha-sweep chính thức chạy trên cấu hình Run 4.

Đây là **thực nghiệm hoàn toàn độc lập** với Run 1/Run 2: file mới 100%, không sửa bất kỳ file nào
của 2 thực nghiệm trước (baseline và Run 2 vẫn chạy lại y hệt bất kỳ lúc nào), work dir riêng,
script chạy riêng.

### 9.1. File mới

```
BoundaryLossSolution/
├── src/
│   ├── losses/
│   │   ├── affinity.py                       # 🆕 AffinityLoss — contrastive feature-distance gần biên
│   │   └── total_loss.py                     # 🆕 StaticBoundaryTotalLoss — gộp L_region + L_BCE_edge + L_Affinity
│   └── train_static_boundary.py              # 🆕 script train — ĐỘC LẬP với 2 script trước, chỉ dùng
│                                               # chung src/data, src/utils (kể cả boundary_metrics.py), model UNetFormer
├── configs/
│   └── unet_former_resnet18_static_boundary/
│       └── static_boundary.yaml              # 🆕 kế thừa mọi hyperparam từ bce_edge.yaml, mở rộng
│                                               # block BOUNDARY_LOSS (USE_AFFINITY, AFFINITY_*)
└── scripts/
    ├── run_static_boundary.sh                # 🆕 launcher riêng — KHÔNG gọi run_bce_edge.sh
    └── resume_static_boundary.sh             # 🆕 resume riêng — KHÔNG gọi resume_bce_edge.sh
```

`src/utils/boundary_metrics.py` (file có sẵn từ nhiệm vụ trước, xem mục 8.5) được bổ sung THÊM 2
method `counts_tensor()`/`load_counts_tensor()` (mirror đúng `EdgeStatsAccumulator` trong
`boundary_bce.py`) để all-reduce đúng accumulator qua các rank DDP trước khi `compute()` — mỗi rank
qua `DistributedSampler` chỉ thấy 1 phần val set, giống hệt lý do `SegmentationMetrics.confusion`
phải all-reduce trước khi tính mIoU. Thay đổi thuần **additive**, không đổi hành vi cũ (self-test
`python -m src.utils.boundary_metrics` PASS y hệt trước/sau) — có backup gốc tại
`Tools/backups/boundary_metrics.py.bak_*`.

### 9.2. Cách chạy

```bash
# Full training (40.000 iteration, alpha=0.4 mặc định)
bash scripts/run_static_boundary.sh

# Dry-run (5 iteration, val mỗi 2, 4 ảnh/split — kiểm tra luồng + 8 sanity check trước khi chạy thật)
bash scripts/run_static_boundary.sh --dry-run

# Ablation nhanh alpha, KHÔNG cần sửa YAML — WORK_DIR tự thêm hậu tố _alphaX.XX
bash scripts/run_static_boundary.sh 0.2
bash scripts/run_static_boundary.sh 0.2 --dry-run

# Nếu Kaggle mount dataset ở path khác:
DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_static_boundary.sh

# Resume (session Kaggle mới, /kaggle/working đã bị xoá) — cùng cú pháp alpha:
bash scripts/resume_static_boundary.sh
bash scripts/resume_static_boundary.sh 0.2
bash scripts/resume_static_boundary.sh --path /kaggle/working/.../latest_checkpoint.pth
```

Cùng trong 1 session (chưa mất `/kaggle/working`): chạy lại đúng lệnh `run_static_boundary.sh` cũ
sẽ tự auto-resume, y hệt cơ chế của baseline (mục 5.1) và Run 2.

### 9.3. Sanity checks tự động (8, thêm 1 so với Run 2)

7 check của Run 2 (mục 8.3) + **check #8 mới**: gradient khác 0 trên `model.base.frh` (decoder, nơi
sinh Fused Feature) qua $\mathcal{L}_{\text{Affinity}}$ — xác nhận affinity loss thực sự lan truyền
gradient tới decoder, không chỉ tới `BoundaryHead`. Fail cứng thì dừng training ngay (trừ overlay
trực quan). Kết quả pass/fail ghi vào `summary.txt` (mục "PRE-FLIGHT VALIDATION").

### 9.4. Output đầu ra (thêm so với mục 8.4 của Run 2)

| File/thư mục | Nội dung |
|---|---|
| `summary.txt` | Thêm khối "AFFINITY LOSS CONFIGURATION" (window/distance/margin), 4 thành phần loss (`L_region/L_bce/L_affinity/L_total`), **boundary metrics THẬT** (không N/A — xem mục 9.5) |
| `benchmark_results.csv` | Thêm cột `l_region`, `l_bce`, `l_affinity`, `l_total`, `bf_score`, `boundary_iou_d1/d2/d4`, `asd` |
| `vis/`, `vis/boundary/`, `sanity/edge_gt_overlay_sK.png` | Y hệt Run 2 |

### 9.5. Boundary metrics — tích hợp trực tiếp, không cần hậu kỳ

Khác Run 2 (phải chạy `Tools/eval_boundary_metrics.py` sau khi train xong rồi patch vào
`summary.txt`), `validate()` của Run 3 gọi `BoundaryMetrics(boundary_distances=(1,2,4))` **ngay
trong vòng lặp training**, all-reduce đúng qua các rank DDP (mục 9.1), ghi số liệu thật (BFScore,
Boundary IoU @1/@2/@4, ASD) vào `benchmark_results.csv`/`summary.txt` mỗi lần validate.

### 9.6. Trạng thái hiện tại

Code hoàn chỉnh (31/8/2026), đã verify cục bộ (máy dev không có GPU/dataset thật, chỉ chạy trên
Kaggle):

| Kịch bản | Kết quả |
|---|---|
| `python -m src.losses.affinity` | ✅ PASS |
| `python -m src.losses.total_loss` (model UNetFormer thật) | ✅ PASS — loss hữu hạn, gradient chảy tới cả backbone/BoundaryHead/decoder |
| `python -m src.utils.boundary_metrics` | ✅ PASS (không đổi sau khi thêm 2 method DDP) |
| `--dry-run`, 2 rank DDP thật (dataset GeoTIFF giả lập, CPU) | ✅ Không lỗi/NaN, boundary metrics tính đúng qua all-reduce (đã verify N_valid/N_edge nhân đôi chính xác giữa 1-rank và 2-rank) |

**Cập nhật 6/9/2026 — đã chạy xong 40.000 iteration thật trên Kaggle 2× T4.** `best_model.pth` (iter
36000, `is_best=True` lần cuối ở vòng 9/10) đã được đánh giá hậu kỳ bằng
`Tools/eval_boundary_metrics.py` — log raw tại `docs/results/run3_static_boundary_boundary_metrics.txt`:

| Chỉ số | Giá trị (iter 36000, 384 ảnh val) |
|---|---:|
| mIoU (9 lớp) | **0.6540** |
| Boundary IoU @1 / @2 / @4 | 0.0585 / 0.1170 / 0.2199 |
| BF-Score (precision / recall) | 0.6144 (0.6241 / 0.6051) |
| ASD | 4.6240 px |

So với `benchmark_results.csv` ghi trong lúc train (ASD=4.6251 tại cùng iter 36000): lệch 0.0011 px —
nhỏ hơn 2 bậc so với mọi hiệu ứng đang đo (0.2–0.36 px), không đổi kết luận nào, nhưng khi so ASD
giữa các run cần giữ nhất quán **một nguồn** (ở đây dùng JSON hậu kỳ). Đối chiếu 2 dự đoán của giả
thuyết cơ chế "BCE định vị biên, affinity dập biên giả" (mục 10.7): `asd_pred_to_gt=5.2158` >
`asd_gt_to_pred=4.0321` (lợi ích tập trung đúng chiều pred→gt như dự đoán) và precision (0.6241) tăng
nhiều hơn recall (0.6051) so với baseline (P 0.6188/R 0.5888) — **cả 2 dự đoán đều khớp**, ủng hộ giả
thuyết cơ chế.

---

## 10. Thực nghiệm 3b — "λ₂ = 0.5 cho Affinity" (Run 3b, điểm giữa đường quét affinity)

Đặc tả đầy đủ: `docs/run3b_spec_lambda2_05.md`. Đường quét hệ số affinity cần 3 điểm cách đều
(α_affinity ∈ {0, 0.2, 0.4}) để biết affinity loss có "vùng ngọt" hay chỉ là núm đánh đổi tuyến
tính. Hai đầu mút đã có (BCE λ=0.4 của mục 8 ↔ α_affinity=0; Run 3 mục 9 ↔ α_affinity=0.4) — Run 3b
là điểm giữa.

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{region}} + \alpha \cdot (\lambda_1 \cdot \mathcal{L}_{\text{BCE\_edge}} + \lambda_2 \cdot \mathcal{L}_{\text{Affinity}})$$

$$\alpha = 0.4,\quad \lambda_1 = 1.0 \text{ (y hệt Run 3)},\quad \lambda_2 = 0.5 \text{ (BIẾN DUY NHẤT thay đổi)}$$

$$\Rightarrow \alpha_{\text{bce, hiệu dụng}} = 0.4 \text{ (không đổi)}, \quad \alpha_{\text{affinity, hiệu dụng}} = 0.4 \times 0.5 = 0.2 \text{ (giảm một nửa so với Run 3)}$$

Công thức trong bài báo **không đổi một ký tự nào** — `L_region + α(λ₁L_BCE + λ₂L_Aff)` vốn đã có
sẵn chỗ cho λ₁/λ₂ (`docs/workflow_2.md` mục 3.3/3.4); Run 3 chỉ mới hard-code λ₁=λ₂=1. Run 3b báo
cáo λ₂=0.5 thay vì 1.0, không đẻ thêm tham số mới.

**Nguyên tắc cài đặt:** người dùng yêu cầu **không sửa** bất kỳ file nào của Run 3
(`src/losses/total_loss.py`, `src/train_static_boundary.py`, `configs/.../static_boundary.yaml`,
`scripts/run_static_boundary.sh`, `scripts/resume_static_boundary.sh`) — kể cả theo kiểu thêm tham
số mặc định tương thích ngược — để Run 3 tái lập lại được y hệt bất kỳ lúc nào, độc lập hoàn toàn
với Run 3b. Toàn bộ Run 3b là **file mới 100%**, giống đúng quy ước baseline/Run 2/Run 3 đã dùng
xuyên suốt repo.

### 10.1. File mới

```
BoundaryLossSolution/
├── src/
│   ├── losses/
│   │   └── total_loss_weighted.py            # 🆕 StaticBoundaryTotalLossWeighted — nhận
│   │                                          #    lambda1_static/lambda2_static (mặc định 1.0/1.0,
│   │                                          #    default = tái lập y hệt StaticBoundaryTotalLoss/Run 3)
│   └── train_static_boundary_weighted.py     # 🆕 script train — ĐỘC LẬP với train_static_boundary.py,
│                                              #    chỉ dùng chung src/data, src/utils, model UNetFormer,
│                                              #    src/losses/boundary_bce.py + affinity.py (đã là file
│                                              #    dùng chung sẵn có, không phải file riêng của Run 3)
├── configs/
│   └── unet_former_resnet18_static_boundary/
│       └── run3b_static_lambda2_05.yaml      # 🆕 kế thừa nguyên static_boundary.yaml, chỉ đổi
│                                              #    LAMBDA2_STATIC 1.0 -> 0.5 + WORK_DIR riêng
└── scripts/
    ├── run_static_boundary_weighted.sh       # 🆕 launcher riêng — KHÔNG gọi run_static_boundary.sh
    └── resume_static_boundary_weighted.sh    # 🆕 resume riêng — KHÔNG gọi resume_static_boundary.sh
```

Bổ sung **thuần additive** (không đổi hành vi cũ) vào `Tools/eval_boundary_metrics.py`: thêm
`--model-type static_boundary` (import `StaticBoundaryUNetFormer` từ `src/train_static_boundary.py`,
không sửa file đó) — dùng cho việc phụ mục 10.7 bên dưới, retro-fit per-class Boundary IoU/ASD 2
chiều cho checkpoint Run 2/Run 3 đã có sẵn mà không cần train lại.

### 10.2. Cách chạy

```bash
# Full training (40.000 iteration, alpha=0.4/lambda1=1.0/lambda2=0.5 mặc định trong config)
bash scripts/run_static_boundary_weighted.sh

# Dry-run (5 iteration, val mỗi 2, 4 ảnh/split — kiểm tra luồng + 12 sanity check trước khi chạy thật)
bash scripts/run_static_boundary_weighted.sh --dry-run

# Ablation nhanh alpha/lambda1/lambda2, KHÔNG cần sửa YAML — WORK_DIR tự thêm hậu tố
bash scripts/run_static_boundary_weighted.sh --lambda2 0.3
bash scripts/run_static_boundary_weighted.sh --lambda2 0.3 --dry-run

# Kiểm tra tương thích ngược (mục 6.3 spec) — trỏ CONFIG vào config Run 3 cũ, không đổi Run 3:
CONFIG=configs/unet_former_resnet18_static_boundary/static_boundary.yaml \
  bash scripts/run_static_boundary_weighted.sh --dry-run

# Nếu Kaggle mount dataset ở path khác:
DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_static_boundary_weighted.sh

# Resume (session Kaggle mới, /kaggle/working đã bị xoá):
bash scripts/resume_static_boundary_weighted.sh
bash scripts/resume_static_boundary_weighted.sh --lambda2 0.3
bash scripts/resume_static_boundary_weighted.sh --path /kaggle/working/.../latest_checkpoint.pth
```

### 10.3. Sanity checks tự động (12, thêm 2 so với Run 3)

10 check của Run 3 (mục 9.3 + 8 của Run 2) + **2 check mới** (mục 6.1/6.2 `docs/run3b_spec_lambda2_05.md`):

- **#9 — đồng nhất công thức loss:** `l_total` phải khớp đúng
  `l_region + alpha*(lambda1*l_bce + lambda2*l_affinity)` (sai số < 1e-5) — bắt lỗi hệ số không tới
  được loss.
- **#10 — λ₂ thực sự có hiệu lực:** `alpha_affinity_effective` (log tường minh mỗi 100 iter) phải
  đúng bằng `alpha*lambda2` tính từ config — với config mặc định phải ra 0.2.

Fail cứng thì dừng training ngay (trừ overlay trực quan). Kết quả pass/fail ghi vào `summary.txt`
(mục "PRE-FLIGHT VALIDATION").

### 10.4. 4 checkpoint thay vì 1 — khắc phục lệch mốc iteration

Run 3 báo cáo tại iter 36000 (best mIoU) trong khi Run 1/2 báo cáo tại iter 40000 — so sánh
boundary-metric giữa các run bị lệch mốc, có thể **đảo dấu** tuỳ checkpoint được chọn. Run 3b lưu
đủ 4 checkpoint (raw `state_dict`, `torch.save` thẳng — tương thích ngược với mọi tool eval hiện
có) trong `WORK_DIR`:

| File | Tiêu chí |
|---|---|
| `best_miou.pth` | mIoU-9 cao nhất |
| `best_bfscore.pth` | BFScore cao nhất |
| `best_bareland.pth` | IoU lớp Bareland cao nhất (lớp yếu nhất, biến động mạnh nhất) |
| `final_iter<N>.pth` | luôn lưu vòng cuối cùng, bất kể chỉ số — cho so sánh cùng iteration giữa mọi run |

Kèm `checkpoint_index.json` (map từng khoá ở trên → `{file, iter, round, metrics}`, `metrics` là
toàn bộ hàng CSV của vòng đó) — không còn phải đoán checkpoint nào ứng với iteration nào.

### 10.5. `benchmark_results.csv` — đủ cột cho mean±std hậu kỳ

10 hàng (1/vòng validate, không chỉ hàng best): `miou9`/`miou8`, 9 `iou_<ClassName>`, `bf_score`/
`bf_precision`/`bf_recall`, `biou_d1`/`d2`/`d4` (mean) + 27 cột `biou_dX_<ClassName>` (per-class,
lấy trực tiếp từ `BoundaryMetrics.compute()` — đã có sẵn, chỉ mới lộ ra CSV), `asd` + `asd_pred_to_gt`
+ `asd_gt_to_pred` (2 chiều — cũng đã có sẵn trong `BoundaryMetrics`, Run 3 chưa ghi ra), 4 thành
phần loss, `lambda1`/`lambda2`/`alpha` + 2 giá trị effective, và `is_best_miou`/`is_best_bfscore`/
`is_best_bareland`/`is_final`. Boundary metrics tính **đầy đủ mọi vòng** (không rút gọn 4 vòng cuối)
— Run 3 (kiến trúc/val-set/tham số boundary giống hệt) đã tính đủ mọi vòng và hoàn thành 40k
iteration trong 03h22m, nên Run 3b giữ nguyên cách làm đã kiểm chứng thay vì thêm nhánh rút gọn
chưa test.

### 10.6. `summary.txt` — khối 4-checkpoint + mean±std

Thay khối "FINAL EVALUATION — BEST CHECKPOINT" đơn lẻ của Run 3 bằng khối "EVALUATION — 4
CHECKPOINTS" (bảng 4 hàng) + khối "MEAN ± STD — 4 VÒNG CUỐI" (mIoU-9/mIoU-8/BFScore/BIoU@2/BIoU@4/
ASD/Bareland + toàn bộ 9 per-class IoU, kèm tên lớp), cộng 5 dòng hệ số (`alpha`/`lambda1`/`lambda2`/
2 giá trị effective).

### 10.7. Việc phụ — eval hậu kỳ trên checkpoint đã có (CPU, song song, không tốn GPU)

**Cập nhật 4/9/2026:** lỗ hổng ASD đã được lấp bằng dữ liệu thật (`ablation.xlsx`, ngoài repo).
Bảng 4 điểm cho thấy **ASD là chỉ số DUY NHẤT affinity tạo hiệu ứng thật ngoài nhiễu** (2.5σ, không
đảo dấu khi so cùng iteration), trong khi BFScore/BIoU@d gần như **mù** với hiệu ứng này (bão hoà
ngoài ngưỡng d) và mIoU chỉ đo cái giá phải trả:

| Method | ASD (px) | Δ vs baseline | Δ vs BCE λ=0.4 |
|---|---:|---:|---:|
| Baseline | 4.9732 | — | |
| BCE λ=0.2 | 4.8347 | −0.139 (−2.8%) | |
| BCE λ=0.4 | 4.9128 | −0.060 (−1.2%) | — |
| Static (α_aff=0.4) | 4.6251¹ | −0.348 (−7.0%) | **−0.288 (−5.9%)** |

¹ Từ `benchmark_results.csv` ghi trong lúc train tại iter 36000. JSON hậu kỳ (`Tools/eval_boundary_metrics.py`,
6/9/2026) cho 4.6240 tại đúng checkpoint đó — lệch 0.0011 px, không đổi kết luận nào (mục 9.6). Khi so
ASD giữa các run, giữ nhất quán một nguồn — bảng này dùng CSV cho cả 4 hàng.

BCE đóng góp 17% mức giảm ASD, **affinity đóng góp 83%** — gần như toàn bộ hiệu ứng. **Giả thuyết cơ
chế:** ASD nhạy với đuôi phân phối (mảnh biên giả xa mọi biên thật đóng góp rất lớn), còn BIoU@d/
BFScore bão hoà ngoài ngưỡng d nên mù với loại lỗi này ⇒ **BCE định vị biên, affinity dập biên giả ở
vùng trong** — hai vai trò bổ sung nhau, không chồng lấn.

**Cập nhật 6/9/2026 — cả 3 lệnh eval hậu kỳ đã chạy xong** (`Tools/eval_boundary_metrics.py`, mục
10.1). BCE λ=0.4/λ=0.2 chạy trước, JSON tại `docs/results/run2_lambda04_boundary_metrics.json` và
`docs/results/run2_lambda02_boundary_metrics.json`; Static (Run 3, checkpoint `best_model.pth` @iter
36000) chạy sau, log raw tại `docs/results/run3_static_boundary_boundary_metrics.txt` (số liệu đầy đủ
ở mục 9.6). Bảng ASD 4 điểm dưới đây nay đã đủ số:

Đối chiếu **2 dự đoán** trong JSON của BCE λ=0.4 và Static — **cả 2 đều khớp** (đã xác nhận 6/9/2026,
số liệu đầy đủ ở mục 9.6):

1. **`asd_pred_to_gt` vs `asd_gt_to_pred`:** đúng như dự đoán, lợi ích của affinity tập trung ở
   chiều **pred→gt** (biên dự đoán thừa, nằm xa GT — baseline: pred→gt 5.6797 vs gt→pred 4.2668;
   Static: pred→gt **5.2158** vs gt→pred **4.0321** — cả 2 chiều đều cải thiện nhưng pred→gt cải
   thiện mạnh hơn). Cơ chế "dập biên giả" được ủng hộ, không phải "bắt biên bị bỏ sót".
2. **`bf_precision` vs `bf_recall`:** đúng như dự đoán, affinity **nâng precision nhiều hơn recall**
   (baseline: P 0.6188/R 0.5888 → Static: P **0.6241**/R **0.6051** — ΔP=+0.0053, ΔR=+0.0163; xét
   theo % tương đối so baseline thì ΔP/P=+0.86% < ΔR/R=+2.77%, nhưng **so với BCE λ=0.4** — mốc
   BCE-chỉ-mình-nó không có affinity — P 0.6299→0.6241 giảm nhẹ còn R 0.5957→0.6051 tăng; diễn giải
   chi tiết theo mốc nào cần thận trọng, xem thêm `docs/spec-run4-dynamic-weighting.md` mục 2.2 cho
   bảng đối chiếu đầy đủ 4 cấu hình).

Câu hỏi per-class (mục 7.3 spec, đọc từ `boundary_iou_dX_per_class` trong JSON) — **Water có BIoU
tăng trong khi IoU vùng giảm không?** — chưa kiểm tra (cần mở trực tiếp
`run3_static_boundary_boundary_metrics.json` gốc trên Kaggle, log console hiện có ở
`docs/results/run3_static_boundary_boundary_metrics.txt` không in per-class).

### 10.8. Quy tắc đọc kết quả (chốt TRƯỚC khi có số)

**ASD là chỉ số CHÍNH của run này** (đổi so với bản spec trước) — lý do: đây là chỉ số duy nhất
affinity tạo hiệu ứng ngoài nhiễu và không đảo dấu (mục 10.7), trong khi BFScore/BIoU đã chứng minh
mù với hiệu ứng đó, còn mIoU đo cái giá phải trả. Câu hỏi thật của Run 3b:

> **Ở λ₂=0.5, giữ được bao nhiêu phần lợi ích ASD, và trả lại được bao nhiêu phần chi phí mIoU?**

So sánh dùng **mean±std 4 vòng cuối** và dòng **final@40000**, không dùng một checkpoint best đơn lẻ.

| | ASD (px) | mIoU-9 | mIoU-8 | BFScore | BIoU@4 | Bareland |
|---|---:|---:|---:|---:|---:|---:|
| **BCE λ=0.4** (α_aff=0, đầu mút trái) | 4.9128 | 0.6566 | 0.6183 | 0.6123 | 0.2186 | 0.3452 |
| **Static** (α_aff=0.4, đầu mút phải, mean±std 3 vòng cuối) | 4.713 ± 0.081 | 0.6503 ± 0.0062 | — | 0.6133 ± 0.0028 | 0.2147 ± 0.0058 | 0.320 ± 0.023 |
| **Run 3b** (α_aff=0.2) — mean±std 4 vòng cuối (round 7-10) | **5.0886 ± 0.0349** | **0.6400 ± 0.0055** | **0.5995 ± 0.0061** | **0.6090 ± 0.0056** | **0.2171 ± 0.0051** | **0.2537 ± 0.0312** |

Nội suy tuyến tính giữa 2 đầu mút: ASD ≈ **4.81**, mIoU-9 ≈ **0.6535** — mốc "không có gì đặc biệt".

| Kết quả tại λ₂=0.5 | Diễn giải | Bước tiếp |
|---|---|---|
| **ASD ≤ 4.75** và **mIoU-9 ≥ 0.6545** | Vùng ngọt tồn tại — quan hệ ASD–α phi tuyến, lợi ích bão hoà sớm còn chi phí tuyến tính | Chốt λ₂\*=0.5, sang Run 4 (Dynamic) với `λ₂_end=0.5` — trở thành cấu hình đề xuất của bài |
| ASD ≈ 4.78–4.84 và mIoU ≈ 0.653–0.654 (bám sát nội suy tuyến tính) | Affinity là núm đánh đổi tuyến tính ASD↔mIoU, không có vùng ngọt | Vẫn có giá trị: phát biểu được đánh đổi định lượng có kiểm soát. Chọn điểm vận hành theo mục tiêu (α_aff=0.4 nếu ưu tiên ASD), sang Run 4. **Không** quét thêm α |
| ASD ≥ 4.88 (mất gần hết lợi ích dù chỉ giảm nửa cường độ) | Hiệu ứng ASD có ngưỡng, cần cường độ cao | Giữ α_aff=0.4 làm cấu hình đề xuất; chuyển ngân sách sang thí nghiệm sửa vùng lấy mẫu (dilation/max-pool) thay vì hạ α |
| **ASD ≥ 4.91 (bằng/tệ hơn BCE) và mIoU cũng không hồi ← NHÁNH THỰC TẾ** | **Phi đơn điệu** — α_aff=0.2 bị α_aff=0.4 **chi phối hoàn toàn**: thua cả BCE λ=0.4 (ASD 4.9128) lẫn Static (ASD 4.713±0.081) trên ASD, và thua cả 2 mốc trên mIoU-9 (0.6566/0.6503 so với 0.6400) — còn yếu tố ngoài cường độ đơn thuần | ✅ **Đã thực hiện:** không quét thêm α theo hướng "tìm liều tối ưu tĩnh". Thay vào đó chốt **α_aff=0.4** làm liều cuối, chuyển câu hỏi sang **định thời** (liều nào áp dụng lúc nào trong quá trình train) — xem Run 4 (`docs/spec-run4-dynamic-weighting.md`), đã có code hoàn chỉnh |

**Kết luận Run 3b (6/9/2026):** rơi đúng vào nhánh cuối — kết quả âm nhưng có thông tin: đường quét
tĩnh không đơn điệu theo α_aff, và bản thân liều giữa (0.2) không phải điểm vận hành tốt ở bất kỳ chỉ
số nào. Đây là bằng chứng trực tiếp dẫn tới thiết kế lịch trình `ramp_hold` của Run 4 — "vượt nhanh
qua vùng liều thấp" thay vì "ramp tuyến tính suốt run" (spec Run 4 mục 0.1), vì vùng liều thấp
(λ₂<0.5, α_aff<0.2) đã đo được là **có hại**, không phải trung tính.

### 10.9. Trạng thái hiện tại

**Cập nhật 6/9/2026 — đã chạy xong 40.000 iteration thật trên Kaggle 2× T4** (03h 26m 39s, log đầy đủ
tại `docs/results/run3b_lambda2_05_summary.txt`).

| Kịch bản | Kết quả |
|---|---|
| `python -m src.losses.total_loss_weighted` (model UNetFormer thật) | ✅ PASS — default λ₁=λ₂=1.0 khớp tuyệt đối `StaticBoundaryTotalLoss` (Run 3) trên cùng input; λ₂=0.5 cho `alpha_affinity_effective=0.2` đúng công thức, gradient chảy tới cả backbone/BoundaryHead/decoder |
| `--dry-run`, 1 rank (dataset GeoTIFF giả lập, CPU, config Run 3b) | ✅ 12/12 sanity check PASS, đủ 4 checkpoint + `checkpoint_index.json` sinh đúng và `load_state_dict` được (strict), CSV đủ cột, `summary.txt` có khối 4-checkpoint + mean±std |
| `--dry-run`, 1 rank, trỏ vào `static_boundary.yaml` gốc (không có `LAMBDA1_STATIC`/`LAMBDA2_STATIC`) | ✅ Tự dùng default λ₁=λ₂=1.0, `alpha_affinity_effective=0.4` — xác nhận tương thích ngược (mục 6.3 spec) |
| Full 40.000 iteration (Kaggle 2×T4) | ✅ Chạy xong, `is_best_miou` chốt ở iter 36000 (mIoU-9=0.6458) — kết quả đầy đủ ở bảng mục 10.8 |
| Run 1/2/3 không bị ảnh hưởng | ✅ `git status` xác nhận không file nào của Run 1/2/3 bị sửa |

**Việc tiếp theo:** không cần chạy lại/quét thêm α cho Run 3b — kết luận đã chốt (mục 10.8). Bước kế
tiếp của dự án là **Run 4 (Dynamic Boundary-Aware Loss)**, xem `docs/spec-run4-dynamic-weighting.md`
và mục 11 dưới đây.

---

## 11. Thực nghiệm 4 — "Dynamic Boundary-Aware Loss, ramp → hold" (Run 4, Ours)

Đặc tả đầy đủ: `docs/spec-run4-dynamic-weighting.md`. Thành phần biến Run 3 (phân tích tĩnh) thành
Run 4 (phương pháp đề xuất của bài): thay λ1=λ2=1 cố định bằng lịch trình λ1(t)/λ2(t) theo iteration,
dựa trực tiếp trên bằng chứng của Run 3b (mục 10.8) — vùng liều affinity thấp (α_aff<0.2) đã đo là
**có hại**, nên lịch trình phải vượt qua vùng đó nhanh thay vì ramp tuyến tính suốt run.

$$\lambda_1(t) = 1.0 \text{ (phẳng)}, \quad
\lambda_2(t) = \begin{cases} 0 & p < 0.10 \text{ (warmup)} \\ \text{ramp tuyến tính } 0\to1.0 & 0.10 \le p < 0.30 \\ 1.0 \text{ (hold)} & p \ge 0.30 \end{cases}, \quad p = \frac{t}{40000}$$

Với `ALPHA=0.4`: α_affinity hiệu dụng đi từ 0 (iter 0–4000) lên 0.4 (đạt từ iter 12000, giữ tới hết) —
đúng liều tốt nhất đã xác nhận ở Run 3 tĩnh, chỉ khác ở **thời điểm** áp dụng.

### 11.1. File mới (không sửa file Run 1/2/3/3b)

```
src/losses/dynamic_weighting.py                    # 🆕 lambda_schedule (gốc, không dùng) +
                                                    #    lambda_schedule_ramp_hold (Run 4) + SCHEDULES
src/losses/total_loss_dynamic.py                   # 🆕 DynamicBoundaryTotalLoss — schedule động HOẶC
                                                    #    lambda tĩnh (dynamic_weights=False, tương thích
                                                    #    ngược Run 3)
src/train_dynamic_boundary.py                      # 🆕 script train, độc lập hoàn toàn 4 script trước
configs/unet_former_resnet18_dynamic_boundary/
    run4_dynamic_ramp_hold.yaml                    # 🆕 kế thừa nguyên static_boundary.yaml, chỉ đổi
                                                    #    khối BOUNDARY_LOSS
```

### 11.2. Đã kiểm thử (local CPU, dataset giả lập — chưa chạy Kaggle GPU thật)

| Kịch bản | Kết quả |
|---|---|
| `python -m src.losses.dynamic_weighting` | ✅ PASS — lịch trình đúng bảng đóng 6 mốc (spec mục 5.1) |
| `python -m src.losses.total_loss_dynamic` | ✅ PASS — warmup vô hiệu hoá affinity, hold≡Static, tương thích ngược `dynamic_weights=False`, gradient chảy đủ 3 nhánh |
| `--dry-run` (CPU, dataset synthetic, config Run 4) | ✅ 13/13 sanity check PASS, đủ 4 checkpoint, `lambda_schedule_log.csv` + hình lịch trình |
| `--dry-run`, trỏ vào `static_boundary.yaml` gốc (Run 3, `DYNAMIC_WEIGHTS: false`) | ✅ `lambda1=lambda2=1.0` cố định suốt run, bỏ qua schedule hoàn toàn — xác nhận tương thích ngược bằng cách chạy thật, không chỉ đọc code |
| Kiểm thử độc lập bởi subagent tester (không dựa báo cáo trước) | ✅ 7/7 tiêu chí mục 5 spec PASS, rà code thủ công không phát hiện lỗi thật |

### 11.3. Cách chạy

```bash
# Full training (40.000 iteration, alpha=0.4/schedule=ramp_hold mặc định trong config)
bash scripts/run_dynamic_boundary.sh

# Dry-run (100 iteration, val mỗi 25 — đúng mục 5 spec) trước khi chạy thật
bash scripts/run_dynamic_boundary.sh --dry-run

# Ablation nhanh alpha, KHÔNG cần sửa YAML — WORK_DIR tự thêm hậu tố _alphaX.XX
bash scripts/run_dynamic_boundary.sh --alpha 0.2

# Kiểm tra tương thích ngược (mục 5.3 spec) — trỏ CONFIG vào config Run 3 cũ:
CONFIG=configs/unet_former_resnet18_static_boundary/static_boundary.yaml \
  bash scripts/run_dynamic_boundary.sh --dry-run

# Nếu Kaggle mount dataset ở path khác:
DATA_ROOT=/kaggle/input/openearthmap bash scripts/run_dynamic_boundary.sh

# Resume (session Kaggle mới, /kaggle/working đã bị xoá):
bash scripts/resume_dynamic_boundary.sh
bash scripts/resume_dynamic_boundary.sh --path /kaggle/working/.../latest_checkpoint.pth
```

`run_dynamic_boundary.sh` tự kiểm `nvidia-smi` (dừng ngay nếu dính Tesla P100, sm_60 không tương
thích build PyTorch hiện tại) trước khi chạy training thật — bỏ qua bước này khi `--dry-run`. Cùng
trong 1 session (chưa mất `/kaggle/working`): chạy lại đúng lệnh `run_dynamic_boundary.sh` cũ sẽ tự
auto-resume, y hệt cơ chế của các run trước.

**Việc còn lại:** chạy full 40.000 iteration thật trên Kaggle 2× T4 (~3.5h dự kiến). Quy tắc đọc kết
quả (bảng quyết định 4 nhánh theo ASD/mIoU đối chiếu với Static cùng liều cuối) xem spec mục 7.2.
