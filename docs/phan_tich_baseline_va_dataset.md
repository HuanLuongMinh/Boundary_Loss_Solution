# Baseline UNetFormer (ResNet-18 + GLTB) & Chiến lược Dataset OpenEarthMap

Tài liệu này gồm 2 phần: (1) xác nhận tính hợp lý của baseline vừa tái lập so với paper gốc UNetFormer và code GeoSeg chính thức, (2) phân tích và đề xuất chiến lược sử dụng dataset OpenEarthMap cho protocol thực nghiệm 2 giai đoạn trong `idead_research.md`.

Toàn bộ số liệu định lượng trong tài liệu này (73% ảnh val bị trùng lặp, 538 ảnh gốc, 71 vùng, số tham số model...) đều được đo/verify trực tiếp trên code và dữ liệu thật của bạn, không phải ước lượng.

---

## Phần 1 — Baseline: kiến trúc có hợp lý không?

### 1.1. Đối chiếu với paper gốc

Đã fetch trực tiếp bản PDF paper UNetFormer (Lancaster eprints) và code chính thức [`WangLibo1995/GeoSeg`](https://github.com/Sjyhne/ContrastiveGeoSeg) (fork bạn cung cấp) để xác nhận từng thành phần:

| Thành phần | Paper gốc nói gì | Code chính thức | Baseline mới build (`src/models/unet_former_resnet18.py`) |
|---|---|---|---|
| Backbone | "pre-trained ResNet18 ... significantly low computational cost" (ablation cho thấy tốt hơn ViT-Tiny/Swin-Tiny/CoaT-Mini về tốc độ) | `timm.create_model('swsl_resnet18', features_only=True, out_indices=(1,2,3,4))` | `resnet18.fb_swsl_ig1b_ft_in1k` (tên timm hiện đại của `swsl_resnet18`) — verify channels `[64,128,256,512]`, stride `[4,8,16,32]` |
| Attention block | **"Global-Local Transformer Block (GLTB)"** — nhánh global (window multi-head self-attention) + nhánh local (2 conv song song 3×3 và 1×1) | class `GlobalLocalAttention` + `Block` | Giữ nguyên logic, **đặt tên tường minh `class GLTB`** đúng thuật ngữ paper thay vì tên chung chung `Block` |
| Decoder | 3 GLTB phân cấp + skip connection có trọng số (weighted sum) + Feature Refinement Head | `Decoder`: `b4→p3→b3→p2→b2→p1(FRH)` | `b4→wf3→b3→wf2→b2→frh` — cùng cấu trúc |
| Loss | $\mathcal{L}_p = \mathcal{L}_{ce} + \mathcal{L}_{dice}$, $\mathcal{L}_{total} = \mathcal{L}_p + 0.4\,\mathcal{L}_{aux}$ | `UnetFormerLoss` = JointLoss(SoftCE, Dice, 1.0, 1.0) + 0.4×aux CE | `CombinedLoss` (đã có sẵn trong repo, `src/utils/losses.py`) = CE + Dice, trọng số 1:1, **không** có nhánh aux |

**Kết luận:** kiến trúc bạn build ra khớp đúng bản chất kỹ thuật của UNetFormer gốc. Khác biệt duy nhất có chủ đích là **bỏ nhánh aux loss** (0.4×CE phụ trên feature stride-4) — đây là lựa chọn ĐÚNG cho mục đích của bạn, vì `idead_research.md` định nghĩa baseline là $\mathcal{L}_{region}$ thuần (chỉ CE+Dice trên **Fused Feature** chính, không trộn thêm tín hiệu supervision khác) — nhánh aux của paper gốc là một "phụ gia" kiến trúc nằm ngoài phạm vi $\mathcal{L}_{region}$, giữ lại nó sẽ làm baseline không còn "sạch" để so sánh công bằng với 3 biến thể sau (Bảng 1: +BCE Edge, +Static Boundary, +Dynamic Boundary).

Về từ khóa **GLTB** bạn yêu cầu: paper xác nhận đây chính là tên chính thức của attention block (không phải tên tự đặt) — tôi đã đổi tên class tương ứng trong code từ `Block` (tên chung chung trong code gốc) thành `GLTB` để bạn dễ trace lại đúng thuật ngữ khi viết bài báo.

### 1.2. Verify bằng thực thi (không chỉ đọc code)

Đã cài `torch` + `timm` (CPU) trong sandbox và chạy forward/backward thật:

```
Total params: 11.65M  (encoder: 11.18M, decoder: 0.47M)
logits shape (B=2, 512x512 input): torch.Size([2, 9, 512, 512])     ✓
fused_feature shape:               torch.Size([2, 64, 128, 128])    ✓ (stride 4, đúng vị trí "Fused Feature")
odd-size input (500x517, không chia hết window_size=8): torch.Size([1, 9, 500, 517])  ✓ (pad/crop logic đúng)
CombinedLoss (CE+Dice) trên logits: giá trị hợp lệ, backward() chảy gradient tới 100% tham số  ✓
```

So với track UNetFormer/ResNeXt101_32x16d hiện có trong repo (≈194M tham số), baseline ResNet-18 chỉ **11.65M tham số (~16-17 lần nhẹ hơn)** — điểm này quan trọng cho Phần 2 vì ảnh hưởng trực tiếp đến khả năng chạy đủ 18 runs trong ngân sách compute Kaggle.

### 1.3. Deliverables đã tạo (đúng convention hiện có của repo — file độc lập, không sửa track cũ)

- `src/models/unet_former_resnet18.py` — model (đã verify)
- `src/train_unet_former_resnet18.py` — training script (DDP + AMP + checkpoint/resume, giữ nguyên pattern các script `train_*.py` khác)
- `configs/unet_former_resnet18_combineLoss/luot{1,2,3}_{500,1000,1500}.yaml` — config smoke-test, tương thích ngược
- `run_unet_former_resnet18_combineLoss_experiment.sh` — launcher
- `Tools/build_clean_val_split.py`, `Tools/build_spatial_folds.py` — công cụ mới, xem Phần 2

**Một điểm cần bạn cân nhắc:** paper gốc dùng LR phân lớp (`backbone_lr=6e-5`, thấp hơn 10× so với phần còn lại `lr=6e-4`, qua `layerwise_params` + optimizer `Lookahead` + `CosineAnnealingWarmRestarts`). Baseline mới dùng LR phẳng `6e-4` cho toàn bộ model với `AdamW` + `poly decay` — **giống hệt cách Track A (ResNet-101, decoder CNN) trong repo bạn đã làm**, nên nhất quán nội bộ với các track khác, nhưng là một simplification so với paper gốc. Vì ResNet-18 nhẹ hơn nhiều so với ResNeXt101_32x16d(SWSL) mà Track C/D dùng LR cực nhỏ `1e-5`, việc dùng LR phẳng cao hơn cho ResNet-18 là hợp lý — nhưng nếu muốn bám sát paper 100%, có thể thêm layerwise LR sau (tôi không tự ý thêm `Lookahead`/`CosineAnnealingWarmRestarts` vì các track khác trong repo đều dùng AdamW + poly decay thuần, muốn giữ nhất quán hạ tầng train).

---

## Phần 2 — Dataset OpenEarthMap: dùng hết hay chỉ split một phần?

### 2.1. Bối cảnh: 2 câu hỏi nghiên cứu khác nhau, không nên dùng chung 1 chiến lược data

Track A-D cũ trong repo bạn trả lời câu hỏi: *"Backbone/decoder/loss nào tốt hơn, và độ nhạy với lượng dữ liệu train ra sao?"* — nên **cố ý** sweep 500/1000/1500 ảnh làm biến thí nghiệm.

`idead_research.md` trả lời câu hỏi khác hẳn: *"Dynamic boundary loss có cải thiện được các biến thể loss khác một cách ĐÁNG TIN CẬY không?"* — ở đây lượng dữ liệu **không phải** biến cần sweep, mà là một yếu tố cần **cố định ở mức đủ lớn** để hiệu ứng cần đo (chênh lệch mIoU dự kiến trong Bảng 1 của đề cương chỉ ~0.65–1.85 điểm giữa các biến thể) không bị nhiễu bởi phương sai do thiếu dữ liệu. Vì vậy, tiếp tục dùng nguyên xi 500/1000/1500 cho nghiên cứu mới là **lệch mục tiêu thống kê**, dù nó đã đúng cho mục tiêu cũ.

### 2.2. Số liệu nền tảng — dataset OpenEarthMap chính thức

Đã tra cứu paper gốc OpenEarthMap (Xia et al., WACV 2023):

- Tổng: **5.000 ảnh**, 97 vùng địa lý, 44 quốc gia, 8 lớp đất phủ (repo bạn định nghĩa 9 lớp — có thể do mirror Kaggle gộp/tách khác, không ảnh hưởng logic phân tích).
- Split chính thức: **tỉ lệ 6:1:3** ("images from each region were randomly divided into training/validation/test with ratio 6:1:3") → **~3.000 train / ~500 val / ~1.500 test**, và quan trọng là **chia THEO TỪNG VÙNG** (region-stratified) để tránh rò rỉ không gian — đúng tinh thần "5 Folds phân vùng không gian độc lập" mà đề cương của bạn muốn làm ở Giai đoạn 2.
- Baseline công bố chính thức cho UNetFormer/ResNeXt101 trên toàn bộ split đó: **mIoU = 68.37%** — con số này là mốc tham chiếu hữu ích: nếu baseline ResNet-18 mới của bạn (train trên tập nhỏ) cho mIoU thấp hơn nhiều so với 68%, khó biện minh rằng cải thiện đo được ở Bảng 1/2 phản ánh đúng hiệu quả của boundary loss chứ không phải baseline bị "đói dữ liệu".

### 2.3. Phát hiện quan trọng (đã đo trực tiếp trên file bạn có): `val_2000_fixed.txt` KHÔNG phải 2000 ảnh độc lập

```
Tổng dòng trong dataset/val_2000_fixed.txt : 2000
Số dòng có hậu tố "_aug..."                : 1462   (73.1%)
Số ảnh GỐC thực sự khác nhau               : 538
Số vùng địa lý khác nhau trong 538 ảnh đó  : 71
```

`Tools/prepare_splits.py` (script cũ) tạo ra file này bằng cách: copy toàn bộ ảnh val thật đang có (538 ảnh — khớp gần đúng với con số chính thức ~500 val của paper gốc), rồi vì thiếu so với target 2000, **tự động sinh thêm ảnh bằng 6 phép biến đổi hình học** (hflip/vflip/rot90/rot180/rot270/hflip+rot90) áp lên chính các ảnh gốc đó để "độn" đủ 2000 dòng.

Điều này **hoàn toàn hợp lý** cho mục đích cũ (ước lượng mIoU ổn định hơn, ít nhạy nhiễu, cho Track A-D) — lật/xoay ảnh không đổi bản chất phân loại pixel, mIoU tính trên bản lật gần như tương đương bản gốc.

Nhưng nó **có vấn đề thực sự** cho nghiên cứu boundary loss mới:

1. **Pseudo-replication**: 2000 "mẫu" val chỉ đến từ 538 cấu hình biên độc lập. Bất kỳ tính toán mean/std nào coi 2000 mẫu này là i.i.d. (kể cả trong Phase 1 fixed-split hiện tại, chưa nói đến Phase 2 CV) đều đánh giá sai độ tin cậy thống kê — cỡ mẫu hiệu dụng thực tế nhỏ hơn con số hiển thị gần 4 lần.
2. **Đặc biệt nghiêm trọng với boundary-level metrics** (Boundary IoU d=1/3/5, BF-Score, ASD — đúng các chỉ số đề cương của bạn dùng): lật/xoay bảo toàn nguyên vẹn hình học biên (chỉ phản chiếu/xoay góc), tức **không hề tạo thêm dạng ranh giới mới** để kiểm chứng tính robust của loss — trong khi đây chính là thứ nghiên cứu của bạn muốn đo.
3. Ràng buộc thêm: `val_2000_fixed.txt` gộp 538 ảnh từ tổng cộng 71 vùng — nếu dùng file này làm 20% val cố định cho Phase 1 **và đồng thời** cũng đưa các vùng này vào pool chia 5-fold ở Phase 2, cần cẩn thận không để cùng 1 vùng vừa nằm trong "val cố định Phase 1" vừa nằm trong "train của 1 fold Phase 2" theo cách gây rò rỉ chéo giữa 2 giai đoạn (xem khuyến nghị 2.4 bên dưới).

### 2.4. Đề xuất cụ thể

**Không dùng nhị phân "toàn bộ dataset" hay "subset nhỏ" cho cả 2 giai đoạn — mỗi giai đoạn có logic dữ liệu riêng:**

**Giai đoạn 1 (Fixed 80/20, tuning α + ablation 4 biến thể, 8 runs):**
- **Bỏ** cách sweep 500/1000/1500. Thay vào đó **cố định một tập train lớn duy nhất** (khuyến nghị: toàn bộ pool train thật hiện có trên mirror Kaggle của bạn — chạy `Tools/create_splits.py`, xem file `train_<N>_fixed.txt` với N = tổng số ảnh nó tự log ra, thường ~3000 theo split chính thức) cho **cả 8 run** — vì mục tiêu là so sánh 4 biến thể loss ở cùng 1 điều kiện dữ liệu, không phải so sánh theo lượng dữ liệu.
- **Thay `val_2000_fixed.txt` bằng val "sạch"**: dùng script mới `Tools/build_clean_val_split.py` (đã tạo, đã unit-test) — quét lại đúng ảnh val gốc, **không augment-pad**, ghi ra `val_clean_<N>.txt` (N thực tế ~538 trên mirror của bạn, có thể set `--max-size` nếu muốn khớp đúng 500 của paper gốc). Nếu 538 mẫu bị coi là ít để ước lượng boundary-metric ổn định, giải pháp đúng là **thêm ảnh train thật vào val** (di một phần vùng từ train sang val) — KHÔNG phải augment ảnh cũ.
- Nếu hạ tầng Kaggle không kham nổi train full-size × 8 runs (mỗi run 40k iter), ưu tiên giảm `MAX_ITERS`/bật early-stopping sớm hơn là giảm lượng ảnh train — vì hiệu ứng bạn cần đo (boundary) nhạy với **đa dạng cấu hình biên** hơn là số bước tối ưu.

**Giai đoạn 2 (5-Fold Spatial CV, 10 runs):**
- Đây chính xác là lúc "dùng hết dataset" có ý nghĩa nhất — mục tiêu của CV là đo tính tổng quát hoá qua nhiều vùng địa lý khác nhau, mẫu càng nhỏ mỗi fold thì độ lệch chuẩn giữa 5 fold trong Bảng 2 càng phản ánh **nhiễu do thiếu dữ liệu** hơn là **biến thiên thật giữa các vùng** — làm yếu chính luận điểm "cải thiện ổn định qua các vùng" mà Bảng 2 muốn chứng minh.
- Đã tạo `Tools/build_spatial_folds.py`: gộp toàn bộ ảnh thật (train+val, không tính bản augment) theo vùng, chia 5 fold bằng thuật toán greedy cân bằng (đã unit-test: với phân bố vùng lệch giả lập, tỉ lệ fold lớn nhất/nhỏ nhất chỉ ~1.01), đảm bảo **1 vùng không bao giờ xuất hiện ở 2 fold** (chống rò rỉ không gian) — cùng nguyên tắc paper gốc OpenEarthMap đã dùng khi tạo split 6:1:3 chính thức.
- 10 run full-scale (40k iter × 5 fold × 2 phương pháp) là gánh nặng compute thật — đây là lý do việc bạn chuyển baseline từ ResNeXt101_32x16d(SWSL, ~194M tham số, từng mất 12-17h/run chỉ với 500-1500 ảnh) sang **ResNet-18 (11.65M tham số, nhẹ hơn ~17 lần)** không chỉnchỉ đúng theo paper gốc mà còn là quyết định **thực dụng bắt buộc** để 18 run của cả 2 giai đoạn còn khả thi trong giới hạn Kaggle 2×T4/phiên ~9h.

**Tránh rò rỉ chéo giữa 2 giai đoạn:** vì Phase 1 dùng 1 tập val cố định và Phase 2 dùng 5-fold trên toàn bộ pool, nếu chạy cả 2 trên cùng máy/thời điểm, nên coi **val cố định của Phase 1 là một phần cố định của pool trước khi chia fold Phase 2** (tức Phase 2 chia fold trên toàn bộ ảnh, bao gồm cả những ảnh dùng làm val Phase 1) — điều này chấp nhận được về mặt phương pháp luận vì Phase 1 (tuning) và Phase 2 (benchmark cuối) là 2 thí nghiệm độc lập, không cùng lúc dùng 1 con số để vừa tune vừa report — miễn là *trong nội bộ mỗi giai đoạn* không có rò rỉ.

### 2.5. Tóm tắt khuyến nghị

| | Cách cũ (Track A-D) | Đề xuất cho nghiên cứu boundary loss mới |
|---|---|---|
| Train Phase 1 | 500/1000/1500 (sweep) | 1 tập lớn cố định (toàn bộ pool train thật, ~3000) |
| Val Phase 1 | `val_2000_fixed.txt` (73% ảnh trùng lặp augment) | `val_clean_<N>.txt` — ảnh thật 100%, không augment-pad |
| Phase 2 | (chưa có) | 5-fold **theo vùng địa lý** trên toàn bộ ảnh thật (train+val), dùng `Tools/build_spatial_folds.py` |
| Backbone | ResNeXt101_32x16d (194M, lr=1e-5, 12-17h/run) | ResNet-18 (11.65M, lr=6e-4) — vừa đúng paper gốc, vừa giải phóng ngân sách compute cho 18 run |

Nói ngắn gọn: **không nên** tiếp tục subset nhỏ 500/1000/1500 cho nghiên cứu mới (nó phục vụ câu hỏi nghiên cứu cũ, không phải câu hỏi mới), nhưng cũng **không đơn giản là "dùng hết 5000 ảnh"** — cách đúng là dùng **tối đa dữ liệu thật sẵn có phù hợp với mục tiêu thống kê của từng giai đoạn**, đồng thời sửa lại tập val hiện tại vì nó không phải 2000 mẫu độc lập như con số cho thấy.
