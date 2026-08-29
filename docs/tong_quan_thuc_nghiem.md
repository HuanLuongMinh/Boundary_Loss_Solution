# Tổng quan các thí nghiệm đã triển khai

Tài liệu này tổng hợp **tất cả thí nghiệm đã có code chạy được** trong repo `BoundaryLossSolution`
tính đến thời điểm cập nhật (29/8/2026), cho đề tài "Dynamic Boundary-Aware Loss via Adaptive
BCE-Affinity Coupling" (`docs/idea_research.md`). Đây là tài liệu **tra cứu nhanh** — chi tiết đầy
đủ từng phần nằm ở `README.md` (thí nghiệm 1-2) và `docs/workflow_2.md` (đặc tả kỹ thuật/kế hoạch
các bước tiếp theo).

---

## 1. Bức tranh tổng thể

```
L_total = L_seg + α · L_boundary_dynamic          (công thức đề cương, docs/idea_research.md)
```

| # | Thí nghiệm | Công thức loss thực tế đã chạy | Trạng thái | Kết quả |
|---|---|---|---|---|
| 1 | **Baseline** | `L_total = L_seg` (CE + Dice thuần, α=0) | ✅ Đã chạy xong 28/8/2026 | best mIoU = **0.6551** |
| 2 | **Baseline vs BCE Loss** (Run 2 Bảng 1) | `L_total = L_seg + λ_edge · L_edge`, λ_edge=0.4 cố định | ✅ Code hoàn chỉnh, đã smoke-test end-to-end 29/8/2026 (dry-run + resume + ablation λ + EDGE_STATS_FILE — xem mục 4.5). **Chưa chạy full 40k iteration trên Kaggle.** | chưa có (chờ chạy full) |
| 3 | + Static Boundary (BCE + Affinity, λ1=λ2=1) | — | ❌ Chưa làm | — |
| 4 | + Dynamic Boundary (Ours, λ1(t)/λ2(t)) | — | ❌ Chưa làm | — |
| — | Alpha-sweep (α ∈ {0.1, 0.5, 1.0, 2.0}) | — | ❌ Chưa làm (hạ tầng chưa xây) | — |
| — | Phase 2 — 5-fold spatial CV (10 run) | — | ❌ Chưa làm | — |

Cả 2 thí nghiệm đã triển khai (#1, #2) dùng **chung 1 kiến trúc UNetFormer (ResNet-18 + GLTB
decoder)** và **chung mọi hyperparameter training** (seed, optimizer, lịch train) — chỉ khác nhau
đúng 1 điểm: có/không có nhánh `BoundaryHead` + `L_edge`. Mục đích: cô lập tác động của boundary
loss, so sánh công bằng (apples-to-apples).

---

## 2. Cấu hình dùng chung cho MỌI thí nghiệm

| Thành phần | Giá trị | Ghi chú |
|---|---|---|
| Dataset | OpenEarthMap (`aletbm/global-land-cover-mapping-openearthmap` trên Kaggle) | Không split/subset — dùng thẳng toàn bộ `images/{train,val}` + `labels/{train,val}` có sẵn |
| Số lớp | 9 (Background + 8 lớp địa vật) | Background là **class supervised thật** (class 0), không map ignore_index |
| `ignore_index` | 255 | No-op thực tế — dataset hiện không sinh pixel giá trị này |
| Seed | **19**, cố định tuyệt đối | Model init, shuffle dữ liệu, augmentation — mọi script mới đều dùng seed này |
| Encoder | `resnet18.fb_swsl_ig1b_ft_in1k` (timm, pretrained SWSL) | ~11.65M tham số |
| Decoder | 3× GLTB (Global-Local Transformer Block) + WF (Weighted Fusion) + FeatureRefinementHead | `decode_channels=64`, `window_size=8`, `num_heads=8`, `mlp_ratio=4.0` |
| Optimizer | AdamW, `base_lr=6e-4`, `weight_decay=0.01`, `grad_clip=5.0` | poly-LR decay (power=0.9), warmup 500 iteration |
| Lịch train | 40.000 iteration, validate mỗi 4.000 iteration | `batch_size_per_gpu=4` (baseline) — Kaggle 2× T4, `torchrun --nproc_per_node=2` |
| Early stopping | patience = 4 lần val liên tiếp mIoU giảm | |
| Augmentation train | Resize 1024 → RandomCrop 512 → HFlip/VFlip/Rotate90 (p=0.5) → ColorJitter/Gamma (p=0.5) | `src/data/transforms.py`, không đổi giữa 2 thí nghiệm |
| Augmentation val | Resize 1024, không crop/augment | |
| Metric chọn best checkpoint | mIoU 9-class (gồm Background) | Giữ nguyên giữa mọi thí nghiệm để so sánh công bằng |

---

## 3. Thí nghiệm 1 — Baseline

**Mã nguồn:** `src/train_unet_former_resnet18.py` · **Config:**
`configs/unet_former_resnet18_combineLoss/baseline.yaml` · **Chạy:** `bash scripts/run_baseline.sh`

| Thành phần | Giá trị |
|---|---|
| Loss | `CombinedLoss` = CrossEntropy + Dice (trọng số 1:1), tính trên logits suy ra từ Fused Feature |
| Boundary/affinity loss | Không có |
| Kết quả | Best mIoU = **0.6551**, chạy xong 28/8/2026 (xem `phase1-run1-baseline-ket-qua.md`) |

Đây là mốc so sánh — hàng "Baseline (`L_region`)" trong Bảng 1 của đề cương. Chi tiết đầy đủ (sơ
đồ pipeline, checkpoint/resume, output files) xem `README.md` mục 1-7.

---

## 4. Thí nghiệm 2 — "Baseline vs BCE Loss" (Run 2 Bảng 1)

**Mã nguồn:** `src/train_bce_edge.py` (độc lập hoàn toàn với script baseline) · **Config:**
`configs/unet_former_resnet18_bce_edge/bce_edge.yaml` · **Chạy:** `bash scripts/run_bce_edge.sh`

### 4.1. Công thức

```
L_total = L_seg + λ_edge · L_edge
```

- `L_seg` = `CombinedLoss` (CE+Dice) — tái sử dụng nguyên, không sửa.
- `L_edge` = balanced BCE giữa `edge_logits` (sinh bởi `BoundaryHead` — 1 nhánh conv phụ gắn trên
  Fused Feature, 64 kênh, stride 4, của decoder) và `edge_gt` (trích từ ground-truth mask).
- `λ_edge = 0.4`, **cố định** cho lần chạy đầu tiên này (chưa phải α* từ alpha-sweep, hạ tầng đó
  chưa xây) — có thể ghi đè qua tham số dòng lệnh để ablation nhanh, xem mục 4.4.

### 4.2. Cấu hình `BOUNDARY_LOSS` (block mới, so với baseline)

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `LAMBDA_EDGE` | 0.4 | Hệ số λ_edge trong `L_total` |
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

### 4.5. Đã kiểm thử (local, CPU, dataset GeoTIFF giả lập — chưa chạy Kaggle GPU thật)

| Kịch bản | Kết quả |
|---|---|
| `--dry-run` full pipeline | ✅ 7/7 sanity check PASS, sinh đủ CSV/summary.txt/checkpoint/charts |
| Auto-resume | ✅ Tái dùng đúng `pos_weight` đã lưu trong checkpoint, không đo lại |
| `--lambda-edge 0.2` (ablation) | ✅ Tạo `WORK_DIR` riêng, không đè lên run mặc định |
| `Tools/measure_edge_ratio.py` | ✅ Chạy độc lập, sinh JSON đúng format |
| `EDGE_STATS_FILE` trỏ tới JSON trên | ✅ `train_bce_edge.py` đọc đúng `pos_weight` "chính thức" |
| Baseline không bị ảnh hưởng | ✅ `git status` xác nhận không file nào trong `src/train_unet_former_resnet18.py`, `src/models/`, `src/utils/`, `src/data/`, `configs/unet_former_resnet18_combineLoss/`, `scripts/run_baseline.sh` bị sửa |

**Việc còn lại:** chạy full 40.000 iteration thật trên Kaggle 2× T4 (`bash scripts/run_bce_edge.sh`).

### 4.6. Khoảng trống đã biết (cố ý, đã chốt phạm vi với Huan 29/8/2026)

`summary.txt` sinh ra sau mỗi run để `N/A` ở 4 trường "Best validation BFScore" / "Best Boundary
IoU @1/@2/@4" — 2 chỉ số này cần `src/utils/boundary_metrics.py` (Bước 6 của `docs/workflow_2.md`),
**chưa xây trong thí nghiệm này** (cố tình giữ đúng phạm vi: chỉ BCE edge loss). TODO đã ghi lại
trong `docs/workflow_2.md` mục 4 để quay lại sau.

---

## 5. Sơ đồ file liên quan

```
BoundaryLossSolution/
├── configs/
│   ├── unet_former_resnet18_combineLoss/
│   │   └── baseline.yaml                  # Thí nghiệm 1
│   └── unet_former_resnet18_bce_edge/
│       └── bce_edge.yaml                  # Thí nghiệm 2
├── src/
│   ├── train_unet_former_resnet18.py      # Thí nghiệm 1 — script train
│   ├── train_bce_edge.py                  # Thí nghiệm 2 — script train (độc lập)
│   ├── models/unet_former_resnet18.py     # Kiến trúc — DÙNG CHUNG cả 2, không đổi
│   ├── losses/boundary_bce.py             # 🆕 riêng Thí nghiệm 2 (BoundaryHead, L_edge, pos_weight)
│   ├── data/, utils/                      # DÙNG CHUNG (dataset/transforms/CombinedLoss/metrics/visualizer)
│   └── losses/                            # Package mới, chỉ chứa boundary_bce.py (chưa có affinity.py)
├── Tools/measure_edge_ratio.py            # 🆕 riêng Thí nghiệm 2 (đo r_edge/pos_weight, tuỳ chọn)
├── scripts/
│   ├── run_baseline.sh / resume_baseline.sh       # Thí nghiệm 1
│   └── run_bce_edge.sh / resume_bce_edge.sh       # Thí nghiệm 2
└── docs/
    ├── idea_research.md                   # Đề cương gốc (công thức, 18-run protocol)
    ├── workflow_2.md                      # Đặc tả kỹ thuật + checklist các bước CHƯA làm
    └── tong_quan_thuc_nghiem.md           # File này
```

---

## 6. Việc tiếp theo (theo `docs/idea_research.md` Bảng 1 / `docs/workflow_2.md`)

1. Chạy full `bash scripts/run_bce_edge.sh` trên Kaggle — có kết quả mIoU thật cho Run 2.
2. `src/losses/affinity.py` (L_Affinity) + `dynamic_weighting.py` (λ1(t), λ2(t)) → Run 3 (Static
   Boundary) và Run 4 (Dynamic Boundary, Ours).
3. Alpha-sweep (α ∈ {0.1, 0.5, 1.0, 2.0}) để chọn α* — hiện Run 2 mới dùng λ_edge=0.4 tạm thời.
4. `src/utils/boundary_metrics.py` (Boundary IoU d=1/3/5, BF-Score, ASD) — lấp khoảng trống N/A ở
   mục 4.6.
5. Phase 2 — 5-fold spatial CV (10 run: 5 baseline + 5 proposed).
6. `Tools/aggregate_results.py` — gộp kết quả thành Bảng 1/Bảng 2 cho bài báo.
