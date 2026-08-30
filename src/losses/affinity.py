"""L_Affinity — contrastive feature-distance gần biên, cho thực nghiệm
"+Static Boundary" (Run 3 của Bảng 1, docs/idea_research.md mục 2.2, spec đầy
đủ ở docs/workflow_2.md mục 3.2):

    L_Affinity = 1/|P| * sum_{(i,j) in P} [
        A_ij * D(f_i, f_j)  +  (1 - A_ij) * max(0, margin - D(f_i, f_j))
    ]

- f_i, f_j: vector đặc trưng lấy từ FUSED FEATURE (C=64, stride 4 so với ảnh
  input) của decoder UNetFormer (src/models/unet_former_resnet18.py,
  `return_fused_feature=True`) — KHÔNG dùng logits. Fused Feature sinh ra sau
  3 khối GLTB + WF + FeatureRefinementHead của decoder; backbone ResNet-18
  (encoder) không liên quan tới loss này.
- P: tập cặp pixel (i, j) với i là pixel "gần biên" (từ extract_edge_gt) và j
  là hàng xóm trong cửa sổ K x K của i (loại trừ chính i), cả i và j đều phải
  valid (khác ignore_index).
- A_ij = 1 nếu i, j cùng nhãn GT, 0 nếu khác.
- D = cosine distance (1 - cosine_similarity, range [0,2]) hoặc L2 distance,
  chọn qua config AFFINITY_DISTANCE.
- margin: mặc định 1.0 cho cosine (config AFFINITY_MARGIN); L2 không bị chặn
  trên nên cần tune margin riêng nếu dùng L2.

QUAN TRỌNG — nhất quán với src/losses/boundary_bce.py (L_BCE_edge): hàm
`extract_edge_gt` ở đây là hàm dùng chung (import lại, không copy), với cùng
`connectivity`/`dilation_radius` — đúng bất biến "1 định nghĩa biên duy nhất"
mà docs/workflow_2.md mục 3.1/3.2 yêu cầu. Điểm khác biệt duy nhất: L_BCE_edge
tính edge_gt ở độ phân giải INPUT (vì edge_logits được BoundaryHead upsample
về kích thước ảnh gốc), còn L_Affinity tính trực tiếp trên Fused Feature ở độ
phân giải stride-4 (không upsample) — vì vậy nhãn (mask) được downsample về
đúng kích thước Fused Feature bằng NEAREST (không dùng bilinear/area cho nhãn
categorical) trước khi gọi extract_edge_gt trên bản đã downsample.

Bộ nhớ: với K=5 (25 hàng xóm), C=64, batch=4, feature map 128x128 (input
512x512, stride 4) — tensor unfold chính chiếm ~400MB (fp32) / ~200MB (AMP
fp16). K=7 (49 hàng xóm) gần gấp đôi. Nếu OOM trên Kaggle T4, giảm K trước khi
giảm batch size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .boundary_bce import extract_edge_gt  # tái sử dụng — KHÔNG copy lại logic edge


class AffinityLoss(nn.Module):
    def __init__(
        self,
        window_size: int = 5,       # K trong K x K, phải lẻ và >= 3
        distance: str = "cosine",   # "cosine" | "l2"
        margin: float = 1.0,        # mặc định 1.0 cho cosine; cần tune riêng cho l2
        ignore_index: int = 255,
        connectivity: int = 4,      # PHẢI khớp giá trị dùng trong boundary_bce.py
        dilation_radius: int = 0,   # PHẢI khớp giá trị dùng trong boundary_bce.py —
                                     # config hiện tại (BOUNDARY_LOSS.DILATION_RADIUS)
                                     # dùng 0. Luôn truyền tường minh khi khởi tạo
                                     # (không dựa vào default của class này).
        eps: float = 1e-6,
    ):
        super().__init__()
        assert window_size % 2 == 1 and window_size >= 3, "window_size phải lẻ và >= 3"
        assert distance in ("cosine", "l2"), "distance phải là 'cosine' hoặc 'l2'"
        self.K = window_size
        self.distance = distance
        self.margin = margin
        self.ignore_index = ignore_index
        self.connectivity = connectivity
        self.dilation_radius = dilation_radius
        self.eps = eps
        self.pad = window_size // 2
        self.center_idx = (window_size * window_size) // 2  # vị trí offset (0,0) sau unfold

    def forward(self, fused_feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        fused_feature: (B, C, Hf, Wf) float — Fused Feature từ decoder (C=64, stride 4)
        mask:          (B, H, W) int64 — nhãn GT ở độ phân giải ảnh INPUT

        Trả về: scalar loss (0-d tensor). Nếu trong batch không có cặp pixel hợp lệ
        nào (ví dụ crop toàn ignore), trả về loss = 0 nhưng vẫn nằm trong autograd
        graph (qua feat_unf) để không làm DDP lỗi vì thiếu gradient trên 1 rank.
        """
        B, C, Hf, Wf = fused_feature.shape

        # 1) Downsample nhãn về đúng kích thước Fused Feature bằng NEAREST (không nội
        #    suy giá trị cho nhãn categorical).
        mask_ds = F.interpolate(
            mask.unsqueeze(1).float(), size=(Hf, Wf), mode="nearest"
        ).squeeze(1).long()  # (B, Hf, Wf)

        # 2) Sampling mask ("center" i hợp lệ): dùng đúng extract_edge_gt như
        #    boundary_bce.py, áp dụng trên bản mask đã downsample (xem docstring ở
        #    trên về lý do khác độ phân giải với BCE).
        edge_gt, _ = extract_edge_gt(
            mask_ds, ignore_index=self.ignore_index,
            connectivity=self.connectivity, dilation_radius=self.dilation_radius,
        )  # (B, 1, Hf, Wf) float {0,1}
        sample_mask = edge_gt.squeeze(1).bool()  # (B, Hf, Wf) — True = pixel này được làm "center" i

        valid = (mask_ds != self.ignore_index)  # (B, Hf, Wf) bool

        # 3) Unfold Fused Feature thành các cửa sổ KxK cục bộ quanh mỗi pixel.
        feat_unf = F.unfold(fused_feature, kernel_size=self.K, padding=self.pad)  # (B, C*K*K, Hf*Wf)
        feat_unf = feat_unf.view(B, C, self.K * self.K, Hf, Wf)

        # 4) Unfold nhãn VÀ unfold validity riêng — đây là điểm kỹ thuật quan trọng:
        #    F.unfold luôn zero-pad, nên ở ngoài biên ảnh, vị trí "hàng xóm" không tồn
        #    tại sẽ cho giá trị 0. Nếu chỉ dựa vào label_unf==0 để biết "không tồn tại"
        #    thì sẽ nhầm lẫn với trường hợp nhãn thật sự là lớp có index 0. Cách xử lý
        #    đúng: unfold thêm 1 bản validity float {0,1} riêng — vị trí ngoài biên chắc
        #    chắn là 0 trong bản này (đúng do là padding, không phải vì là nhãn thật),
        #    nên dùng validity_unf (không dùng label_unf) để xác định pixel hàng xóm có
        #    hợp lệ hay không.
        label_unf = F.unfold(mask_ds.float().unsqueeze(1), kernel_size=self.K, padding=self.pad)
        label_unf = label_unf.view(B, self.K * self.K, Hf, Wf).long()

        valid_unf = F.unfold(valid.float().unsqueeze(1), kernel_size=self.K, padding=self.pad)
        valid_unf = valid_unf.view(B, self.K * self.K, Hf, Wf).bool()

        # 5) Feature / nhãn / validity của pixel trung tâm, broadcast theo chiều K*K.
        center_feat = fused_feature.unsqueeze(2)   # (B, C, 1, Hf, Wf)
        center_label = mask_ds.unsqueeze(1)         # (B, 1, Hf, Wf)
        center_valid = valid.unsqueeze(1)           # (B, 1, Hf, Wf)

        # 6) Khoảng cách D(f_center, f_neighbor) cho toàn bộ K*K offset cùng lúc.
        if self.distance == "cosine":
            c = F.normalize(center_feat, dim=1, eps=self.eps)   # (B, C, 1, Hf, Wf)
            n = F.normalize(feat_unf, dim=1, eps=self.eps)      # (B, C, K*K, Hf, Wf)
            cos_sim = (c * n).sum(dim=1)                         # (B, K*K, Hf, Wf)
            D = 1.0 - cos_sim                                    # [0, 2]
        else:  # "l2"
            D = torch.norm(feat_unf - center_feat, p=2, dim=1)   # (B, K*K, Hf, Wf)

        # 7) Nhãn affinity A_ij và mask hợp lệ cho từng cặp.
        A = (label_unf == center_label).float()          # (B, K*K, Hf, Wf)
        pair_valid = valid_unf & center_valid              # (B, K*K, Hf, Wf) bool

        # loại cặp tầm thường (offset (0,0), i với chính nó) — khoảng cách luôn = 0,
        # không mang tín hiệu học, chỉ làm loãng loss.
        pair_valid = pair_valid.clone()
        pair_valid[:, self.center_idx, :, :] = False

        # chỉ tính các cặp mà pixel TRUNG TÂM nằm trong vùng sampling gần biên.
        pair_valid = pair_valid & sample_mask.unsqueeze(1)

        pair_valid_f = pair_valid.float()

        # 8) Loss theo cặp, có cân bằng same-class / different-class.
        same_term = A * D
        diff_term = (1.0 - A) * torch.clamp(self.margin - D, min=0.0)
        per_pair_loss = (same_term + diff_term) * pair_valid_f

        denom = pair_valid_f.sum().clamp_min(1.0)
        loss = per_pair_loss.sum() / denom

        return loss


if __name__ == "__main__":
    # ---- self-test tối thiểu (chạy: python -m src.losses.affinity) ----
    # Mục tiêu: verify shape/không NaN, verify gradient chảy tới fused_feature,
    # verify loss cao khi feature "sai" (cùng lớp nhưng feature random rất khác nhau,
    # khác lớp nhưng feature giống hệt nhau) và thấp khi feature "đúng".
    torch.manual_seed(0)
    B, C, Hf, Wf = 2, 64, 16, 16
    loss_fn = AffinityLoss(window_size=5, distance="cosine", margin=1.0, ignore_index=255)

    # mask: nửa trên là lớp 0, nửa dưới là lớp 1 (biên ngang giữa ảnh) — ảnh input
    # giả định cùng kích thước Fused Feature x4 để dùng interpolate thật.
    H, W = Hf * 4, Wf * 4
    mask = torch.zeros(B, H, W, dtype=torch.long)
    mask[:, H // 2 :, :] = 1

    # Case "đúng": feature cùng lớp giống nhau, khác lớp rất khác nhau -> loss thấp.
    base_a = torch.randn(1, C, 1, 1)
    base_b = torch.randn(1, C, 1, 1) * -1  # có hướng ngược lại
    feat_good = torch.zeros(B, C, Hf, Wf)
    feat_good[:, :, : Hf // 2, :] = base_a + 0.01 * torch.randn(B, C, Hf // 2, Wf)
    feat_good[:, :, Hf // 2 :, :] = base_b + 0.01 * torch.randn(B, C, Hf // 2, Wf)
    feat_good.requires_grad_(True)

    loss_good = loss_fn(feat_good, mask)
    loss_good.backward()
    assert torch.isfinite(loss_good), "loss phải là số hữu hạn"
    assert feat_good.grad is not None and torch.any(feat_good.grad != 0), \
        "gradient phải lan truyền tới fused_feature"
    print(f"[case đúng] loss = {loss_good.item():.4f} (kỳ vọng nhỏ, feature đã tách lớp tốt)")

    # Case "sai": feature random hoàn toàn, không liên quan gì đến lớp -> loss cao hơn hẳn.
    feat_bad = torch.randn(B, C, Hf, Wf, requires_grad=True)
    loss_bad = loss_fn(feat_bad, mask)
    print(f"[case sai]  loss = {loss_bad.item():.4f} (kỳ vọng lớn hơn case đúng)")
    assert loss_bad.item() > loss_good.item(), \
        "loss với feature random phải lớn hơn loss với feature đã tách lớp tốt"

    # Case toàn ignore: không được NaN, không được crash.
    mask_ignore = torch.full((B, H, W), 255, dtype=torch.long)
    feat_ignore = torch.randn(B, C, Hf, Wf, requires_grad=True)
    loss_ignore = loss_fn(feat_ignore, mask_ignore)
    assert torch.isfinite(loss_ignore) and loss_ignore.item() == 0.0, \
        "toàn bộ pixel ignore -> loss phải đúng bằng 0, không NaN"
    print(f"[case toàn ignore] loss = {loss_ignore.item():.4f} (kỳ vọng đúng 0)")

    print("Tất cả self-test PASS.")
