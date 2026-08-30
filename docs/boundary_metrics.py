"""BoundaryMetrics — Boundary IoU(d) / BF-Score / ASD cho Bước 6 (docs/workflow_2.md
mục 4). File này là BẢN THAM KHẢO gửi để tích hợp vào repo Boundary_Loss_Solution —
đặt đúng vị trí src/utils/boundary_metrics.py, theo đúng pattern SegmentationMetrics
đã có (src/utils/metrics.py): update() tích luỹ theo batch (KHÔNG average theo batch,
cộng dồn pixel-count qua toàn bộ dataset rồi mới chia ở compute(), giống hệt cách
SegmentationMetrics dồn confusion matrix), compute() trả dict — cắm thẳng vào
validate() của train_bce_edge.py / train script mới mà không đổi cấu trúc vòng lặp.

KHÔNG CẦN GPU để TÍNH các chỉ số này — toàn bộ dùng numpy/scipy trên CPU
(distance_transform_edt). GPU chỉ cần cho bước forward pass ra `logits` (đã có sẵn
trong vòng lặp train/eval hiện tại); BoundaryMetrics chỉ nhận logits/targets đã có
sẵn (có thể trên GPU hoặc CPU, tự chuyển .cpu().numpy() bên trong), không tự chạy
model. Nói cách khác: retro-fit chỉ số này cho checkpoint Run 1/Run 2 cũ CHỈ cần 1
lượt forward pass (có GPU thì nhanh hơn, không có vẫn chạy được vì ResNet-18 nhẹ),
còn phần tính Boundary IoU/BF-Score/ASD từ predicted mask thì thuần CPU.

3 nhóm chỉ số, theo đúng spec docs/workflow_2.md mục 4:

1. Boundary IoU(d), d ∈ {1,3,5} — Cheng et al., "Boundary IoU: Improving
   Object-Centric Image Segmentation Evaluation", CVPR 2021. Định nghĩa THEO TỪNG
   LỚP (per-class, giống per_class_iou trong SegmentationMetrics): với 1 mask nhị
   phân M của lớp c, "dải biên" (boundary band) là tập pixel thuộc M và cách biên
   ngoài của M không quá d pixel (đo bằng distance_transform_edt tính BÊN TRONG
   mask — khoảng cách tới pixel nền/không-thuộc-M gần nhất). Boundary IoU(d) = IoU
   thông thường nhưng CHỈ tính trên phần giao 2 dải biên (GT và pred) — phạt nặng
   lỗi định vị biên, bỏ qua lỗi ở vùng lõi xa biên (đúng tinh thần "boundary-aware
   evaluation", khác focus với mIoU toàn vùng). Trung bình trên các lớp có mặt
   trong GT hoặc pred (union > 0 ở ít nhất 1 sample), giống valid_classes trong
   SegmentationMetrics.compute().

2. BF-Score (Boundary F1) — precision/recall giữa tập pixel "biên" của GT và của
   prediction trong ngưỡng khoảng cách `bf_tolerance` pixel. Biên ở đây dùng ĐÚNG
   1 ĐỊNH NGHĨA CHUNG với boundary_bce.py/affinity.py — `extract_edge_gt()`, biên =
   pixel có ít nhất 1 hàng-xóm 4-connected (hoặc 8, tuỳ config) VALID mang nhãn
   khác, KHÔNG phải biên theo từng lớp riêng lẻ. Bắt buộc dùng lại đúng hàm này
   (không định nghĩa "biên" kiểu khác cho riêng metric) — đây là yêu cầu tường
   minh trong docs/workflow_2.md mục 3.1 ("cả boundary_bce.py và affinity.py PHẢI
   dùng chung 1 hàm extract_edge_gt") và áp dụng logic tương tự cho metric.

3. ASD (Average Surface Distance) — trung bình khoảng cách Euclidean 2 chiều
   (pred→GT và GT→pred) giữa 2 tập pixel biên (cùng định nghĩa biên như BF-Score),
   dùng scipy.ndimage.distance_transform_edt (KHÔNG vòng lặp pixel-by-pixel).

Pixel ignore (`mask == ignore_index`) bị loại khỏi CẢ GT lẫn prediction trước khi
tính, dùng validity mask lấy từ chính extract_edge_gt trả về — không tự định
nghĩa lại (đúng yêu cầu docs/workflow_2.md mục 4: "Loại pixel ignore khỏi cả GT
lẫn prediction trước khi dilate/so sánh, cùng ignore_index như 3.1").

Tự-test: chạy trực tiếp `python -m src.utils.boundary_metrics` (từ repo root) để
verify các case biết trước đáp số (mask trùng khớp hoàn toàn -> BIoU/BF=1.0, ASD=0;
lệch 1 pixel có kiểm soát; toàn bộ ignore -> không NaN/crash; batch không lớp nào
xuất hiện -> compute() vẫn trả dict hợp lệ).
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from src.losses.boundary_bce import extract_edge_gt


class BoundaryMetrics:
    """Tích luỹ Boundary IoU(d)/BF-Score/ASD qua nhiều batch, cùng pattern
    SegmentationMetrics (src/utils/metrics.py). Chạy hoàn toàn trên CPU
    (numpy/scipy) — gọi update() ngay sau (hoặc cùng chỗ) SegmentationMetrics.update()
    hiện có trong validate(), không cần đổi vị trí gọi trong vòng lặp."""

    def __init__(self, num_classes: int, ignore_index: int = 255,
                 boundary_distances=(1, 3, 5), bf_tolerance: int = 2,
                 connectivity: int = 4, dilation_radius: int = 0):
        """
        num_classes/ignore_index: PHẢI khớp SegmentationMetrics đang dùng (9 và
            255 theo baseline hiện tại).
        boundary_distances: các giá trị d (pixel) để tính Boundary IoU(d) — mặc
            định (1,3,5) đúng docs/workflow_2.md mục 4.
        bf_tolerance: ngưỡng khoảng cách (pixel) khi SO KHỚP 2 tập biên đã trích
            xuất cho BF-Score — 1 pixel biên dự đoán được tính "khớp" nếu có >=1
            pixel biên GT trong bán kính này (và ngược lại cho recall). KHÔNG
            nhầm với `dilation_radius` của extract_edge_gt (đó là làm dày biên
            NGAY KHI ĐỊNH NGHĨA edge_gt, còn bf_tolerance là dung sai lúc so khớp
            2 tập biên đã có sẵn — 2 khái niệm độc lập, có thể cùng >0).
        connectivity/dilation_radius: PHẢI khớp giá trị đã cố định cho toàn
            nghiên cứu trong boundary_bce.py/affinity.py (mặc định connectivity=4,
            dilation_radius=0, đúng configs/unet_former_resnet18_bce_edge/bce_edge.yaml
            hiện tại) — lệch giá trị này giữa các run sẽ phá tính so sánh được
            của Bảng 1.
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.boundary_distances = tuple(boundary_distances)
        self.bf_tolerance = bf_tolerance
        self.connectivity = connectivity
        self.dilation_radius = dilation_radius
        self.reset()

    def reset(self):
        # Boundary IoU: cộng dồn intersection/union theo pixel-count, per-class,
        # per-distance — qua TOÀN BỘ dataset (không average theo batch/ảnh),
        # giống hệt triết lý confusion matrix trong SegmentationMetrics.
        self._biou_inter = {d: np.zeros(self.num_classes, dtype=np.int64) for d in self.boundary_distances}
        self._biou_union = {d: np.zeros(self.num_classes, dtype=np.int64) for d in self.boundary_distances}
        # BF-Score: cộng dồn true-positive/pred-count/gt-count theo pixel-count.
        self._bf_tp_pred = 0  # pixel biên PRED có match GT trong tolerance
        self._bf_tp_gt = 0    # pixel biên GT có match PRED trong tolerance
        self._bf_n_pred = 0
        self._bf_n_gt = 0
        # ASD: cộng dồn tổng khoảng cách + số pixel biên (lấy mean ở compute()).
        self._asd_sum_p2g = 0.0
        self._asd_n_pred = 0
        self._asd_sum_g2p = 0.0
        self._asd_n_gt = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """logits: (B,C,H,W) float (raw, chưa softmax — chỉ cần đúng thứ tự
        argmax). targets: (B,H,W) int64, cùng resolution với logits (đã qua
        transform, giống targets truyền cho SegmentationMetrics.update() và
        extract_edge_gt() ở boundary_bce.py)."""
        preds = logits.argmax(dim=1)  # (B,H,W)

        # ---------- Boundary IoU: per-sample, per-class (CPU/numpy) ----------
        preds_np = preds.cpu().numpy()
        targets_np = targets.cpu().numpy()
        valid_np = targets_np != self.ignore_index

        for b in range(preds_np.shape[0]):
            t, p, v = targets_np[b], preds_np[b], valid_np[b]
            for c in range(self.num_classes):
                gt_c = (t == c) & v
                pred_c = (p == c) & v
                if not gt_c.any() and not pred_c.any():
                    continue  # lớp không xuất hiện trong ảnh này ở cả 2 phía — bỏ qua
                # Tính distance_transform_edt 1 LẦN mỗi mask (không lặp lại cho
                # từng d) rồi threshold nhiều mức d — tiết kiệm ~3x so với gọi
                # scipy riêng cho mỗi d.
                gt_dist = distance_transform_edt(gt_c) if gt_c.any() else None
                pred_dist = distance_transform_edt(pred_c) if pred_c.any() else None
                for d in self.boundary_distances:
                    gt_band = (gt_c & (gt_dist <= d)) if gt_dist is not None else np.zeros_like(gt_c)
                    pred_band = (pred_c & (pred_dist <= d)) if pred_dist is not None else np.zeros_like(pred_c)
                    inter = np.logical_and(gt_band, pred_band).sum()
                    union = np.logical_or(gt_band, pred_band).sum()
                    self._biou_inter[d][c] += inter
                    self._biou_union[d][c] += union

        # ---------- BF-Score + ASD: dùng CHUNG 1 định nghĩa biên (extract_edge_gt) ----------
        gt_edge, gt_valid = extract_edge_gt(targets, self.ignore_index,
                                             self.connectivity, self.dilation_radius)
        pred_edge, _ = extract_edge_gt(preds, self.ignore_index,
                                        self.connectivity, self.dilation_radius)
        # Pixel ignore của GT áp luôn cho pred — không đánh giá ở đâu GT không hợp lệ.
        pred_edge = pred_edge * gt_valid

        gt_edge_np = gt_edge.squeeze(1).cpu().numpy().astype(bool)      # (B,H,W)
        pred_edge_np = pred_edge.squeeze(1).cpu().numpy().astype(bool)  # (B,H,W)

        for b in range(gt_edge_np.shape[0]):
            g, p = gt_edge_np[b], pred_edge_np[b]
            n_gt, n_pred = int(g.sum()), int(p.sum())
            self._bf_n_gt += n_gt
            self._bf_n_pred += n_pred
            if n_gt == 0 and n_pred == 0:
                continue  # ảnh không có biên hợp lệ nào (vd toàn ignore) — bỏ qua

            # distance_transform_edt(~X): khoảng cách từ MỌI pixel tới pixel biên
            # gần nhất trong X — tra cứu 1 lần cho toàn ảnh, không lặp pixel.
            if n_gt > 0:
                dist_to_gt = distance_transform_edt(~g)
                if n_pred > 0:
                    self._bf_tp_pred += int((dist_to_gt[p] <= self.bf_tolerance).sum())
                    self._asd_sum_p2g += float(dist_to_gt[p].sum())
                    self._asd_n_pred += n_pred
            if n_pred > 0:
                dist_to_pred = distance_transform_edt(~p)
                if n_gt > 0:
                    self._bf_tp_gt += int((dist_to_pred[g] <= self.bf_tolerance).sum())
                    self._asd_sum_g2p += float(dist_to_pred[g].sum())
                    self._asd_n_gt += n_gt

    def compute(self) -> dict:
        result = {}

        # Boundary IoU(d): trung bình trên các lớp CÓ xuất hiện (union>0 ở ít
        # nhất 1 sample) — giống valid_classes trong SegmentationMetrics.compute().
        for d in self.boundary_distances:
            inter = self._biou_inter[d].astype(np.float64)
            union = self._biou_union[d].astype(np.float64)
            valid = union > 0
            iou_per_class = np.full(self.num_classes, np.nan)
            iou_per_class[valid] = inter[valid] / union[valid]
            result[f'boundary_iou_d{d}'] = float(np.nanmean(iou_per_class)) if valid.any() else float('nan')
            result[f'boundary_iou_d{d}_per_class'] = iou_per_class.tolist()

        # BF-Score
        precision = self._bf_tp_pred / self._bf_n_pred if self._bf_n_pred > 0 else 0.0
        recall = self._bf_tp_gt / self._bf_n_gt if self._bf_n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        result['bf_precision'] = float(precision)
        result['bf_recall'] = float(recall)
        result['bf_score'] = float(f1)

        # ASD (symmetric): trung bình pred->gt và gt->pred.
        d_p2g = self._asd_sum_p2g / self._asd_n_pred if self._asd_n_pred > 0 else float('nan')
        d_g2p = self._asd_sum_g2p / self._asd_n_gt if self._asd_n_gt > 0 else float('nan')
        valid_dists = [x for x in (d_p2g, d_g2p) if not np.isnan(x)]
        result['asd_pred_to_gt'] = float(d_p2g)
        result['asd_gt_to_pred'] = float(d_g2p)
        result['asd'] = float(np.mean(valid_dists)) if valid_dists else float('nan')

        return result


if __name__ == "__main__":
    # Self-test — chạy `python -m src.utils.boundary_metrics` từ repo root.
    torch.manual_seed(19)
    NUM_CLASSES, IGNORE = 4, 255
    H = W = 32

    def make_logits_from_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Sinh logits "hoàn hảo" (argmax == labels) để test case khớp tuyệt đối."""
        onehot = torch.nn.functional.one_hot(labels.clamp(min=0), num_classes).permute(0, 3, 1, 2).float()
        return onehot * 10.0 - 5.0

    print("=== Case 1: prediction TRÙNG KHỚP TUYỆT ĐỐI với GT ===")
    labels = torch.zeros(2, H, W, dtype=torch.int64)
    labels[:, :, W // 2:] = 1
    labels[0, 5:10, 5:10] = 2
    logits = make_logits_from_labels(labels, NUM_CLASSES)
    bm = BoundaryMetrics(NUM_CLASSES, IGNORE, boundary_distances=(1, 3, 5), bf_tolerance=2)
    bm.update(logits, labels)
    res = bm.compute()
    print(res)
    assert abs(res['boundary_iou_d1'] - 1.0) < 1e-6, "khớp tuyệt đối phải cho Boundary IoU = 1.0"
    assert abs(res['bf_score'] - 1.0) < 1e-6, "khớp tuyệt đối phải cho BF-Score = 1.0"
    assert res['asd'] == 0.0, "khớp tuyệt đối phải cho ASD = 0"
    print("PASS\n")

    print("=== Case 2a: SAI NHÃN nhưng biên ở ĐÚNG VỊ TRÍ (đoán nửa phải thành lớp 3 thay vì 1) ===")
    wrong_class_labels = labels.clone()
    wrong_class_labels[:, :, W // 2:] = 3
    wrong_class_logits = make_logits_from_labels(wrong_class_labels, NUM_CLASSES)
    bm2a = BoundaryMetrics(NUM_CLASSES, IGNORE, boundary_distances=(1, 3, 5), bf_tolerance=2)
    bm2a.update(wrong_class_logits, labels)
    res2a = bm2a.compute()
    print(res2a)
    assert res2a['boundary_iou_d1'] < 1.0, "sai nhãn phải làm Boundary IoU (per-class) giảm"
    assert abs(res2a['bf_score'] - 1.0) < 1e-6, (
        "BF-Score/ASD chỉ quan tâm VỊ TRÍ biên (extract_edge_gt không phân biệt "
        "nhãn cụ thể là gì, chỉ cần khác nhãn hàng xóm) — biên vẫn đúng chỗ nên "
        "BF-Score PHẢI vẫn = 1.0 dù toàn bộ nửa phải bị đoán sai lớp. Đây là khác "
        "biệt CÓ CHỦ ĐÍCH giữa 2 nhóm chỉ số: Boundary IoU nhạy với đúng-lớp, "
        "BF-Score/ASD chỉ nhạy với đúng-vị-trí.")
    print("PASS (Boundary IoU giảm vì sai lớp, BF-Score giữ nguyên vì biên đúng vị trí)\n")

    print("=== Case 2b: biên bị LỆCH VỊ TRÍ (dự đoán dịch ranh giới sang phải 3 pixel) ===")
    shifted_labels = labels.clone()
    shifted_labels[:, :, W // 2 + 3:] = 1  # ranh giới 0|1 dịch từ W/2 sang W/2+3
    shifted_labels[:, :, :W // 2 + 3] = 0
    shifted_logits = make_logits_from_labels(shifted_labels, NUM_CLASSES)
    bm2b = BoundaryMetrics(NUM_CLASSES, IGNORE, boundary_distances=(1, 3, 5), bf_tolerance=2)
    bm2b.update(shifted_logits, labels)
    res2b = bm2b.compute()
    print(res2b)
    assert res2b['boundary_iou_d1'] < 1.0
    assert res2b['bf_score'] < 1.0, "biên lệch vị trí phải làm BF-Score giảm"
    assert res2b['asd'] > 0.0, "biên lệch vị trí phải làm ASD > 0"
    print("PASS (biên lệch vị trí làm cả Boundary IoU, BF-Score giảm và ASD > 0)\n")

    print("=== Case 3: toàn bộ batch là ignore_index — không được NaN/crash ===")
    ignore_labels = torch.full((1, H, W), IGNORE, dtype=torch.int64)
    ignore_logits = torch.randn(1, NUM_CLASSES, H, W)
    bm3 = BoundaryMetrics(NUM_CLASSES, IGNORE)
    bm3.update(ignore_logits, ignore_labels)
    res3 = bm3.compute()
    print(res3)
    assert res3['bf_precision'] == 0.0 and res3['bf_recall'] == 0.0
    print("PASS (không crash, precision/recall/bf_score = 0 khi không có biên hợp lệ nào)\n")

    print("=== Case 4: batch trộn ảnh có biên + ảnh toàn ignore (giống batch thật) ===")
    bm4 = BoundaryMetrics(NUM_CLASSES, IGNORE, boundary_distances=(1, 3, 5), bf_tolerance=2)
    bm4.update(logits, labels)          # ảnh hợp lệ
    bm4.update(ignore_logits, ignore_labels)  # ảnh toàn ignore — cộng dồn vào cùng accumulator
    res4 = bm4.compute()
    print(res4)
    assert not np.isnan(res4['bf_score'])
    print("PASS (cộng dồn qua nhiều update() không NaN)\n")

    print("Tất cả self-test PASS.")
