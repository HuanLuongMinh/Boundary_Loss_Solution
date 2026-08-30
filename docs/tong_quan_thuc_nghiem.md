# Tổng quan các thí nghiệm đã triển khai

Tài liệu này tổng hợp **tất cả thí nghiệm đã có code chạy được** trong repo `BoundaryLossSolution`
tính đến thời điểm cập nhật (31/8/2026), cho đề tài "Dynamic Boundary-Aware Loss via Adaptive
BCE-Affinity Coupling" (`docs/idea_research.md`). Đây là tài liệu **tra cứu nhanh** — chi tiết đầy
đủ từng phần nằm ở `README.md` (thí nghiệm 1-3) và `docs/workflow_2.md` (đặc tả kỹ thuật/kế hoạch
các bước tiếp theo).

---

## 1. Bức tranh tổng thể

```
L_total = L_seg + α · L_boundary_dynamic          (công thức đề cương, docs/idea_research.md)
```

| # | Thí nghiệm | Công thức loss thực tế đã chạy | Trạng thái | Kết quả (mIoU-9 best) |
|---|---|---|---|---|
| 1 | **Baseline** | `L_total = L_seg` (CE + Dice thuần, α=0) | ✅ Đã chạy xong 28/8/2026 | **0.6551** |
| 2 | **+ BCE Edge** (Run 2 Bảng 1) | `L_total = L_seg + λ_edge · L_edge` | ✅ Đã chạy xong 40k iter trên Kaggle — 2 giá trị λ_edge | λ=0.2: **0.6490** · λ=0.4: **0.6566** |
| 3 | **+ Static Boundary** (Run 3 Bảng 1) | `L_total = L_region + α·(L_BCE_edge + L_Affinity)`, λ1=λ2=1 cố định | ✅ Code hoàn chỉnh 31/8/2026, đã verify `--dry-run` cục bộ (CPU, dataset giả lập, cả 1-rank và 2-rank DDP thật). **Chưa chạy full 40k trên Kaggle.** | chưa có (chờ chạy full) |
| 4 | + Dynamic Boundary (Ours, λ1(t)/λ2(t)) | — | ❌ Chưa làm | — |
| — | Alpha-sweep (α ∈ {0.1, 0.5, 1.0, 2.0}) | — | ❌ Chưa làm (hạ tầng chưa xây) | — |
| — | Phase 2 — 5-fold spatial CV (10 run) | — | ❌ Chưa làm | — |

**Quyết định đã chốt:** Run 2 cho thấy λ_edge=0.4 tốt hơn λ_edge=0.2 (mIoU-9 0.6566 > 0.6490, cũng
nhỉnh hơn cả baseline 0.6551; λ=0.2 ngược lại **thấp hơn** baseline) → **dùng λ_edge=0.4 làm giá trị
α tạm thời cho Run 3** (và Run 4 sau này), kế thừa xuyên suốt cho tới khi có alpha-sweep chính thức
trên cấu hình Run 4. Cả 3 thí nghiệm đã triển khai (#1, #2, #3) dùng **chung 1 kiến trúc UNetFormer
(ResNet-18 + GLTB decoder)** và **chung mọi hyperparameter training** (seed, optimizer, lịch train)
— chỉ khác nhau ở thành phần loss cộng thêm. Mục đích: cô lập tác động của boundary loss, so sánh
công bằng (apples-to-apples).

---

## 2. Cấu hình dùng chung cho MỌI thí nghiệm

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Dataset | OpenEarthMap (`aletbm/global-land-cover-mapping-openearthmap` trên Kaggle) | Không split/subset — dùng thẳng toàn bộ `images/{train,val}` + `labels/{train,val}` có sẵn |
| Số lớp | 9 (Background + 8 lớp địa vật) | Background là **class supervised thật** (class 0), không map ignore_index |
| `ignore_index` | 255 | No-op thực tế — dataset hiện không sinh pixel giá trị này |
| Seed | **19**, cố định tuyệt đối | Model init, shuffle dữ liệu, augmentation — mọi script mới đều dùng seed này |
| Encoder | `resnet18.fb_swsl_ig1b_ft_in1k` (timm, pretrained SWSL) | ~11.65M tham số |
| Decoder | 3× GLTB (Global-Local Transformer Block) + WF (Weighted Fusion) + FeatureRefinementHead | `decode_channels=64`, `window_size=8`, `num_heads=8`, `mlp_ratio=4.0` — GLTB nằm hoàn toàn ở decoder, encoder ResNet-18 (backbone) là CNN thuần, không có transformer block |
| Optimizer | AdamW, `base_lr=6e-4`, `weight_decay=0.01`, `grad_clip=5.0` | poly-LR decay (power=0.9), warmup 500 iteration |
| Lịch train | 40.000 iteration, validate mỗi 4.000 iteration | `batch_size_per_gpu=4` — Kaggle 2× T4, `torchrun --nproc_per_node=2` |
| Early stopping | patience = 4 lần val liên tiếp mIoU giảm | |
| Augmentation train | Resize 1024 → RandomCrop 512 → HFlip/VFlip/Rotate90 (p=0.5) → ColorJitter/Gamma (p=0.5) | `src/data/transforms.py`, không đổi giữa các thí nghiệm |
| Augmentation val | Resize 1024, không crop/augment | |
| Metric chọn best checkpoint | mIoU 9-class (gồm Background) | Giữ nguyên giữa mọi thí nghiệm để so sánh công bằng |
| Định nghĩa biên (edge) | `extract_edge_gt()` (`src/losses/boundary_bce.py`), 4-connected, `dilation_radius=0` | Dùng CHUNG cho `L_BCE_edge`, `L_Affinity` VÀ `BoundaryMetrics` — đúng bất biến "1 định nghĩa biên duy nhất" |

---

## 3. Thí nghiệm 1 — Baseline

**Mã nguồn:** `src/train_unet_former_resnet18.py` · **Config:**
`configs/unet_former_resnet18_combineLoss/baseline.yaml` · **Chạy:** `bash scripts/run_baseline.sh`

| Thành phần | Giá trị |
|---|---|
| Loss | `CombinedLoss` = CrossEntropy + Dice (trọng số 1:1), tính trên logits suy ra từ Fused Feature |
| Boundary/affinity loss | Không có |
| Kết quả | Best mIoU-9 = **0.6551**, chạy xong 28/8/2026, 02h08m35s |

Đây là mốc so sánh — hàng "Baseline (`L_region`)" trong Bảng 1 của đề cương. Chi tiết đầy đủ (sơ
đồ pipeline, checkpoint/resume, output files) xem `README.md` mục 1-7.

---

## 4. Thí nghiệm 2 — "+ BCE Edge" (Run 2 Bảng 1)

**Mã nguồn:** `src/train_bce_edge.py` (độc lập hoàn toàn với script baseline) · **Config:**
`configs/unet_former_resnet18_bce_edge/bce_edge.yaml` · **Chạy:** `bash scripts/run_bce_edge.sh`

### 4.1. Công thức

```
L_total = L_seg + λ_edge · L_edge
```

- `L_seg` = `CombinedLoss` (CE+Dice) — tái sử dụng nguyên, không sửa.
- `L_edge` = balanced BCE giữa `edge_logits` (sinh bởi `BoundaryHead` — 1 nhánh conv phụ gắn trên
  Fused Feature, 64 kênh, stride 4, của decoder) và `edge_gt` (trích từ ground-truth mask).
- `λ_edge` chạy thật 2 giá trị: **0.2** và **0.4** (ablation, xem mục 4.4) — kết quả mục 4.5 cho
  thấy 0.4 tốt hơn, được chọn làm α kế thừa cho Run 3/4.

### 4.2. Cấu hình `BOUNDARY_LOSS` (block mới, so với baseline)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `LAMBDA_EDGE` | 0.4 (mặc định YAML; 0.2 chạy qua ablation) | Hệ số λ_edge trong `L_total` |
| `CONNECTIVITY` | 4 | Edge target 4-connected (không dùng 8-connected) |
| `DILATION_RADIUS` | 0 | Không dilate thêm — edge width = 1 pixel |
| `IGNORE_INDEX` | 255 | Khớp `CombinedLoss`/`SegmentationMetrics` |
| `POS_WEIGHT_MAX` | 20.0 | Clip `pos_weight` balanced-BCE về `[1.0, 20.0]` |
| `EDGE_STATS_FILE` | `""` (trống) | Nếu trống → tự ước lượng `pos_weight` lúc khởi động (bounded). Trỏ tới JSON do `Tools/measure_edge_ratio.py` sinh ra để dùng số "chính thức" (đo full-epoch) |
| `EDGE_STATS_NUM_BATCHES` | 100 | Số batch/rank dùng để ước lượng bounded khi không có `EDGE_STATS_FILE` |

Mọi tham số khác (`DATASET`, `MODEL`, `TRAIN`, `OPTIMIZER`) **giống hệt** `baseline.yaml`.

### 4.3. Edge target & balanced BCE (spec đầy đủ: `docs/workflow_2.md` mục 3.1)

- `edge_gt[b,h,w] = 1` nếu pixel valid (`≠ ignore_index`) VÀ có ít nhất 1 hàng-xóm 4-connected valid
  mang nhãn khác — cả pixel trung tâm lẫn hàng-xóm đều phải valid.
- `pos_weight = clip(N_nonedge / max(N_edge,1), 1.0, 20.0)` — cân bằng lớp "edge" (luôn thiểu số).
- `BCEWithLogitsLoss` trên **raw logits** (không sigmoid thủ công), nhân validity mask, chia
  `valid_mask.sum()`.
- 7 sanity check tự động chạy lúc khởi động (kể cả `--dry-run`): shape/binary `edge_gt`, batch
  toàn-ignore → loss=0, `0 < r_edge < 1`, `pos_weight` hữu hạn dương, raw-logits-không-sigmoid,
  overlay trực quan (`sanity/edge_gt_overlay_*.png`), gradient khác 0 trên `BoundaryHead` — fail
  cứng thì dừng training ngay.

### 4.4. Ablation λ_edge (không cần sửa YAML)

```bash
bash scripts/run_bce_edge.sh          # λ_edge = 0.4 (mặc định)
bash scripts/run_bce_edge.sh 0.2      # λ_edge = 0.2 → WORK_DIR tự thêm hậu tố _lambda0.20
```

Mỗi giá trị λ_edge có `WORK_DIR`/checkpoint/`benchmark_results.csv`/`summary.txt` riêng, tự resume
đúng của chính nó.

### 4.5. Kết quả thật (Kaggle 2× T4, 40.000 iteration, đã chạy xong 30/8/2026)

Boundary metrics (BFScore/Boundary IoU) được tính hậu kỳ trên `best_model.pth` của mỗi run bằng
`Tools/eval_boundary_metrics.py` (sau khi `src/utils/boundary_metrics.py` được xây xong, xem mục
4.6) rồi điền vào `summary.txt` gốc qua `Tools/patch_bce_edge_summary.py` — không cần train lại.

| Chỉ số | Baseline (α=0) | λ_edge=0.2 | λ_edge=0.4 |
|---|---:|---:|---:|
| mIoU-9 (best, chọn checkpoint) | 0.6551 | 0.6490 | **0.6566** |
| mIoU-8 (loại Background, hậu kỳ) | — | 0.6093 | 0.6183 |
| Boundary BFScore | — | 0.6082 | 0.6123 |
| Boundary IoU @1 | — | 0.0582 | 0.0585 |
| Boundary IoU @2 | — | 0.1169 | 0.1165 |
| Boundary IoU @4 | — | 0.2193 | 0.2186 |
| Thời gian train (40k iter) | 02h08m35s | 02h10m59s | 02h04m00s |

**Nhận xét:** λ_edge=0.2 làm mIoU-9 **giảm nhẹ so với baseline** (0.6490 < 0.6551), trong khi
λ_edge=0.4 cải thiện nhẹ (0.6566 > 0.6551). Boundary BFScore/Boundary IoU chênh lệch không đáng kể
giữa 2 giá trị λ. → **Chốt dùng λ_edge=0.4 làm α cho Run 3 (Static Boundary)**, kế thừa xuyên suốt
tới khi có alpha-sweep chính thức chạy trên cấu hình Run 4.

Kết quả raw (checkpoint, CSV, ảnh visualizer, `summary.txt` gốc + bản đã điền boundary metrics)
lưu ngoài repo tại `D:\Nghien Cuu Sinh\Lab\Boundary\BCE_Lambda_0.2\` và `BCE_Lambda_0.4\`.

### 4.6. `src/utils/boundary_metrics.py` — đã xây xong (khoảng trống cũ đã lấp)

Trước đây `summary.txt` sinh trực tiếp bởi `train_bce_edge.py` để `N/A` ở 4 trường "Best validation
BFScore" / "Best Boundary IoU @1/@2/@4" vì `BoundaryMetrics` (Boundary IoU/BF-Score/ASD) chưa tồn
tại lúc Run 2 khởi chạy. File này đã được xây xong sau đó (`class BoundaryMetrics`: Boundary IoU(d)
per-class, BF-Score, ASD — dùng chung `extract_edge_gt()` với `L_BCE_edge`), cho phép:

- Retrofit số liệu thật cho Run 2 (baseline/BCE edge) qua `Tools/eval_boundary_metrics.py` +
  `Tools/patch_bce_edge_summary.py` (kết quả ở mục 4.5) — `summary.txt` **gốc** của Run 2 vẫn giữ
  nguyên `N/A` (không sửa output cũ), số liệu thật nằm ở file `*_summary_boundary_filled.txt` mới,
  tách biệt.
- Tích hợp trực tiếp (không cần hậu kỳ) vào `validate()` của Run 3 trở đi — xem mục 5.3.

### 4.7. Đã kiểm thử

| Kịch bản | Kết quả |
|---|---|
| `--dry-run` full pipeline (cục bộ) | ✅ 7/7 sanity check PASS, sinh đủ CSV/summary.txt/checkpoint/charts |
| Full 40.000 iteration (Kaggle 2×T4) | ✅ Chạy xong cả 2 giá trị λ_edge, xem kết quả mục 4.5 |
| Auto-resume | ✅ Tái dùng đúng `pos_weight` đã lưu trong checkpoint, không đo lại |
| `--lambda-edge 0.2` (ablation) | ✅ Tạo `WORK_DIR` riêng, không đè lên run mặc định |
| Baseline không bị ảnh hưởng | ✅ `git status` xác nhận không file nào trong `src/train_unet_former_resnet18.py`, `src/models/`, `src/utils/`, `src/data/`, `configs/unet_former_resnet18_combineLoss/`, `scripts/run_baseline.sh` bị sửa |

---

## 5. Thí nghiệm 3 — "+ Static Boundary" (Run 3 Bảng 1)

**Mã nguồn:** `src/train_static_boundary.py` (độc lập hoàn toàn với 2 script trước) · **Config:**
`configs/unet_former_resnet18_static_boundary/static_boundary.yaml` · **Chạy:**
`bash scripts/run_static_boundary.sh`

### 5.1. Công thức

```
L_total = L_region + α · (λ1 · L_BCE_edge + λ2 · L_Affinity),   λ1 = λ2 = 1 CỐ ĐỊNH
```

- `L_region` = `CombinedLoss` (CE+Dice), `L_BCE_edge` = y hệt Run 2 — tái sử dụng nguyên.
- `L_Affinity` = contrastive feature-distance gần biên (`src/losses/affinity.py`, **mới**): so sánh
  khoảng cách đặc trưng (cosine hoặc L2) giữa mỗi pixel gần biên và các hàng-xóm trong cửa sổ K×K,
  trên **Fused Feature** (64 kênh, stride 4, decoder — KHÔNG dùng logits), kéo gần feature cùng
  nhãn và đẩy xa feature khác nhãn (margin).
- `λ1 = λ2 = 1` **tĩnh** (không dynamic schedule theo iteration — dynamic λ1(t)/λ2(t) là Run 4
  "Ours", ngoài phạm vi Run 3).
- `α = 0.4`, kế thừa trực tiếp từ giá trị λ_edge tốt nhất đã đo được ở Run 2 (mục 4.5) — **tạm
  thời**, chờ alpha-sweep chính thức chạy trên cấu hình Run 4.

3 loss được gộp bởi `StaticBoundaryTotalLoss` (`src/losses/total_loss.py`, **mới**).

### 5.2. File mới (không sửa file của Run 1/Run 2)

```
BoundaryLossSolution/
├── src/
│   ├── losses/
│   │   ├── affinity.py                      # 🆕 AffinityLoss — contrastive feature-distance
│   │   └── total_loss.py                    # 🆕 StaticBoundaryTotalLoss — gộp L_region+L_BCE_edge+L_Affinity
│   └── train_static_boundary.py             # 🆕 script train — độc lập, chỉ dùng chung src/data,
│                                              #    src/utils (kể cả boundary_metrics.py), model UNetFormer
├── configs/
│   └── unet_former_resnet18_static_boundary/
│       └── static_boundary.yaml             # 🆕 kế thừa mọi hyperparam từ bce_edge.yaml, mở rộng
│                                              #    block BOUNDARY_LOSS (USE_AFFINITY, AFFINITY_*)
└── scripts/
    ├── run_static_boundary.sh               # 🆕 launcher riêng
    └── resume_static_boundary.sh            # 🆕 resume riêng
```

`src/utils/boundary_metrics.py` (file có sẵn từ nhiệm vụ trước, xem mục 4.6) được bổ sung THÊM 2
method `counts_tensor()`/`load_counts_tensor()` (mirror đúng `EdgeStatsAccumulator`) để all-reduce
đúng qua các rank DDP — thay đổi thuần additive, không đổi hành vi cũ, có backup tại
`Tools/backups/boundary_metrics.py.bak_*`.

### 5.3. Cấu hình `BOUNDARY_LOSS` (so với Run 2)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `USE_BCE` / `USE_AFFINITY` | `true` / `true` | Bật cả 2 nhánh boundary loss |
| `DYNAMIC_WEIGHTS` | `false` | λ1=λ2=1 cố định — Run 3, KHÔNG phải Run 4 |
| `ALPHA` | 0.4 | Kế thừa λ_edge tốt nhất của Run 2 (mục 4.5), tạm thời |
| `CONNECTIVITY` / `DILATION_RADIUS` | 4 / 0 | Dùng CHUNG cho `L_BCE_edge`, `L_Affinity` VÀ `BoundaryMetrics` |
| `AFFINITY_WINDOW_K` | 5 | Cửa sổ K×K hàng-xóm cho `L_Affinity` |
| `AFFINITY_DISTANCE` | `cosine` | `"cosine"` hoặc `"l2"` |
| `AFFINITY_MARGIN` | 1.0 | Margin cho contrastive term (mặc định cosine) |

### 5.4. Sanity checks tự động (8, thêm 1 so với Run 2)

7 check của Run 2 (mục 4.3) + **check #8 mới**: gradient khác 0 trên `model.base.frh` (decoder,
nơi sinh Fused Feature) qua `L_affinity` — xác nhận affinity loss thực sự lan truyền gradient tới
decoder, không chỉ tới `BoundaryHead`.

### 5.5. Boundary metrics — tích hợp trực tiếp, không cần hậu kỳ

Khác Run 2 (phải chạy `Tools/eval_boundary_metrics.py` sau khi train xong), `validate()` của Run 3
gọi `BoundaryMetrics` (`boundary_distances=(1,2,4)`) **ngay trong vòng lặp training**, all-reduce
đúng qua các rank DDP, ghi số liệu thật (BFScore, Boundary IoU @1/@2/@4, ASD) vào
`benchmark_results.csv` và `summary.txt` mỗi lần validate — không còn `N/A`.

### 5.6. Đã kiểm thử (local, CPU, dataset GeoTIFF giả lập — chưa chạy Kaggle GPU thật)

| Kịch bản | Kết quả |
|---|---|
| `python -m src.losses.affinity` (self-test độc lập) | ✅ PASS (3 case: feature tách lớp tốt / random / toàn ignore) |
| `python -m src.losses.total_loss` (self-test tích hợp, model thật) | ✅ PASS — `l_region/l_bce/l_affinity/l_total` hữu hạn, gradient chảy tới cả backbone/BoundaryHead/decoder |
| `python -m src.utils.boundary_metrics` | ✅ PASS (không đổi sau khi thêm 2 method DDP) |
| `--dry-run`, 1 rank (WORLD_SIZE=1) | ✅ Không lỗi, không NaN ở 4 thành phần loss, boundary metrics tính được |
| `--dry-run`, 2 rank thật (gloo, CPU, tự spawn — thay `torchrun` do lỗi môi trường Windows/libuv không liên quan code) | ✅ Không lỗi/hang; N_valid/N_edge ở bản 2-rank đúng gấp đôi bản 1-rank → xác nhận all-reduce boundary metrics hoạt động đúng qua rank thật |
| Không OOM | ✅ Giữ nguyên `AFFINITY_WINDOW_K=5` mặc định, không cần giảm |
| Run 1/Run 2 không bị ảnh hưởng | ✅ `git status` xác nhận không file nào của Run 1/Run 2 bị sửa (ngoại trừ bổ sung thuần-additive vào `boundary_metrics.py`, đã backup) |

**Việc còn lại:** chạy full 40.000 iteration thật trên Kaggle 2× T4
(`bash scripts/run_static_boundary.sh`).

---

## 6. Sơ đồ file liên quan

```
BoundaryLossSolution/
├── configs/
│   ├── unet_former_resnet18_combineLoss/
│   │   └── baseline.yaml                  # Thí nghiệm 1
│   ├── unet_former_resnet18_bce_edge/
│   │   └── bce_edge.yaml                  # Thí nghiệm 2
│   └── unet_former_resnet18_static_boundary/
│       └── static_boundary.yaml           # 🆕 Thí nghiệm 3
├── src/
│   ├── train_unet_former_resnet18.py      # Thí nghiệm 1 — script train
│   ├── train_bce_edge.py                  # Thí nghiệm 2 — script train (độc lập)
│   ├── train_static_boundary.py           # 🆕 Thí nghiệm 3 — script train (độc lập)
│   ├── models/unet_former_resnet18.py     # Kiến trúc — DÙNG CHUNG cả 3, không đổi
│   ├── losses/
│   │   ├── boundary_bce.py                # Thí nghiệm 2+3 (BoundaryHead, L_BCE_edge, pos_weight)
│   │   ├── affinity.py                    # 🆕 Thí nghiệm 3 (L_Affinity)
│   │   └── total_loss.py                  # 🆕 Thí nghiệm 3 (StaticBoundaryTotalLoss)
│   ├── data/                              # DÙNG CHUNG (dataset/transforms), không đổi
│   └── utils/                             # DÙNG CHUNG — losses.py/metrics.py/callbacks.py/visualizer.py
│       └── boundary_metrics.py            # Boundary IoU/BF-Score/ASD — dùng bởi Run 3 trực tiếp,
│                                            # bởi Run 2 qua Tools/eval_boundary_metrics.py (hậu kỳ)
├── Tools/
│   ├── measure_edge_ratio.py              # Thí nghiệm 2 (đo r_edge/pos_weight, tuỳ chọn)
│   ├── eval_boundary_metrics.py           # Retrofit boundary metrics cho checkpoint Run 1/2
│   ├── patch_bce_edge_summary.py          # Điền boundary metrics vào summary.txt Run 2 (hậu kỳ)
│   └── backups/                           # Backup file trước khi sửa (vd. boundary_metrics.py)
├── scripts/
│   ├── run_baseline.sh / resume_baseline.sh               # Thí nghiệm 1
│   ├── run_bce_edge.sh / resume_bce_edge.sh               # Thí nghiệm 2
│   └── run_static_boundary.sh / resume_static_boundary.sh # 🆕 Thí nghiệm 3
└── docs/
    ├── idea_research.md                   # Đề cương gốc (công thức, 18-run protocol)
    ├── workflow_2.md                      # Đặc tả kỹ thuật + checklist các bước CHƯA làm
    └── tong_quan_thuc_nghiem.md           # File này
```

---

## 7. Việc tiếp theo (theo `docs/idea_research.md` Bảng 1 / `docs/workflow_2.md`)

1. Chạy full `bash scripts/run_static_boundary.sh` trên Kaggle — có kết quả mIoU/boundary metrics
   thật cho Run 3.
2. `dynamic_weighting.py` (λ1(t), λ2(t)) → Run 4 (Dynamic Boundary, Ours).
3. Alpha-sweep (α ∈ {0.1, 0.5, 1.0, 2.0}) chạy trên cấu hình Run 4 để chọn α* chính thức — hiện
   Run 3 mới dùng α=0.4 kế thừa tạm thời từ Run 2.
4. Phase 2 — 5-fold spatial CV (10 run: 5 baseline + 5 proposed).
5. `Tools/aggregate_results.py` — gộp kết quả thành Bảng 1/Bảng 2 cho bài báo.
