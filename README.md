# Baseline UNetFormer (ResNet-18) — OpenEarthMap

Baseline cho đề tài **"Dynamic Boundary-Aware Loss via Adaptive BCE-Affinity Coupling"**
(xem `docs/idea_research.md`): kiến trúc UNetFormer (encoder ResNet-18 pretrained, decoder
GLTB paper-faithful), loss `L_region` thuần (CrossEntropy + Dice) — **chưa** có bất kỳ thành
phần boundary/affinity loss nào. Đây là mốc so sánh (hàng "Baseline" trong Bảng 1 của đề cương)
cho các biến thể boundary-loss sẽ làm sau, tách biệt hoàn toàn khỏi phần này.

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
`docs/idea_research.md` và `docs/workflow_1.md`, sẽ được xây dựng trong các file/thư mục **mới**
riêng biệt (không sửa bất kỳ file nào trong `src/` liệt kê ở mục 3), để không ảnh hưởng tới kết
quả baseline này khi so sánh.
