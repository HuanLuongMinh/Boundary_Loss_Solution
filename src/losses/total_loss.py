"""src/losses/total_loss.py — L_total cho Run 3 "+Static Boundary" (Bảng 1,
docs/idea_research.md mục 3 / docs/workflow_2.md mục 3.4):

    L_total = L_region + alpha * (lambda1 * L_BCE_edge + lambda2 * L_Affinity)

Run 3 dùng lambda1 = lambda2 = 1 CỐ ĐỊNH (static, không dynamic schedule theo
iteration — dynamic lambda1(t)/lambda2(t) là Run 4/"Ours", ngoài phạm vi file
này). Vì lambda1=lambda2=1 không đổi, code không cần tham số cur_iter/max_iters
như bản DynamicBoundaryTotalLoss tổng quát trong docs/workflow_2.md mục 3.4.

L_region tái sử dụng CombinedLoss (CE+Dice) đã có sẵn (src/utils/losses.py).
L_BCE_edge tái sử dụng BalancedBCEEdgeLoss (src/losses/boundary_bce.py).
L_Affinity dùng AffinityLoss (src/losses/affinity.py).

connectivity/dilation_radius PHẢI truyền giống hệt nhau cho cả bce_edge và
affinity (chỉ có 1 cặp connectivity/dilation_radius dùng chung cho cả hai,
không tách riêng — đúng bất biến "1 định nghĩa biên duy nhất" mà
docs/workflow_2.md mục 3.1/3.2 yêu cầu).
"""

import torch
import torch.nn as nn

from src.utils.losses import CombinedLoss
from src.losses.boundary_bce import BalancedBCEEdgeLoss
from src.losses.affinity import AffinityLoss


class StaticBoundaryTotalLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float, ignore_index: int = 255,
                 ce_weight: float = 1.0, dice_weight: float = 1.0,
                 connectivity: int = 4, dilation_radius: int = 0,
                 affinity_window_size: int = 5, affinity_distance: str = 'cosine',
                 affinity_margin: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.region_loss = CombinedLoss(
            num_classes=num_classes, ignore_index=ignore_index,
            ce_weight=ce_weight, dice_weight=dice_weight)
        self.bce_edge = BalancedBCEEdgeLoss(
            ignore_index=ignore_index, connectivity=connectivity, dilation_radius=dilation_radius)
        self.affinity = AffinityLoss(
            window_size=affinity_window_size, distance=affinity_distance, margin=affinity_margin,
            ignore_index=ignore_index, connectivity=connectivity, dilation_radius=dilation_radius)

    def forward(self, logits: torch.Tensor, edge_logits: torch.Tensor,
                fused_feature: torch.Tensor, masks: torch.Tensor, pos_weight: float):
        """logits: (B,num_classes,H,W). edge_logits: (B,1,H,W) — đã upsample về
        input size bởi BoundaryHead. fused_feature: (B,64,H/4,W/4) — chưa
        upsample, lấy trực tiếp từ decoder (return_fused_feature=True). masks:
        (B,H,W) int64, độ phân giải INPUT — AffinityLoss tự downsample về đúng
        kích thước fused_feature bên trong nó (không downsample ở đây).
        pos_weight: đã tính 1 lần lúc khởi động training (giống hệt cách
        train_bce_edge.py đang làm cho Run 2 — không tính lại mỗi iteration)."""
        l_region = self.region_loss(logits, masks)
        l_bce = self.bce_edge(edge_logits, masks, pos_weight)
        l_affinity = self.affinity(fused_feature, masks)
        l_total = l_region + self.alpha * (l_bce + l_affinity)
        return l_total, {
            'l_region': l_region.item(), 'l_bce': l_bce.item(),
            'l_affinity': l_affinity.item(), 'l_total': l_total.item(),
        }


if __name__ == "__main__":
    # Self-test tích hợp — chạy `python -m src.losses.total_loss` từ repo root.
    # Dùng THẬT model UNetFormer + BoundaryHead (không giả lập shape) để xác
    # nhận toàn bộ pipeline logits/edge_logits/fused_feature khớp nhau, loss
    # hữu hạn, gradient chảy tới cả 3 nhánh (backbone qua l_region, BoundaryHead
    # qua l_bce, decoder/fused_feature qua l_affinity).
    import torch.nn.functional as F
    from src.models.unet_former_resnet18 import UNetFormer
    from src.losses.boundary_bce import BoundaryHead, extract_edge_gt, compute_pos_weight

    torch.manual_seed(19)
    B, H, W, NUM_CLASSES = 2, 256, 256, 9  # đủ lớn cho GLTB window_size=8 ở stride 32 (256/32=8)

    model = UNetFormer(encoder_name='resnet18.fb_swsl_ig1b_ft_in1k', num_classes=NUM_CLASSES,
                        pretrained=False, decode_channels=64, window_size=8)
    boundary_head = BoundaryHead(in_channels=64)

    images = torch.randn(B, 3, H, W)
    masks = torch.randint(0, NUM_CLASSES, (B, H, W), dtype=torch.int64)
    masks[0, :5, :5] = 255  # vài pixel ignore, giống dữ liệu thật

    logits, fused_feature = model(images, return_fused_feature=True)
    edge_logits = boundary_head(fused_feature, images.shape[2:])
    print(f"logits {tuple(logits.shape)}  edge_logits {tuple(edge_logits.shape)}  "
          f"fused_feature {tuple(fused_feature.shape)}")
    assert logits.shape == (B, NUM_CLASSES, H, W)
    assert edge_logits.shape == (B, 1, H, W)
    assert fused_feature.shape[0] == B and fused_feature.shape[1] == 64
    assert fused_feature.shape[2] == H // 4 and fused_feature.shape[3] == W // 4, \
        "Fused Feature phải đúng stride 4 — nếu lệch, kiểm tra lại window_size/pad-crop của model"

    edge_gt, valid_mask = extract_edge_gt(masks, ignore_index=255, connectivity=4, dilation_radius=0)
    pos_weight = compute_pos_weight(edge_gt.sum().item(),
                                     (valid_mask.sum() - edge_gt.sum()).item())

    loss_fn = StaticBoundaryTotalLoss(num_classes=NUM_CLASSES, alpha=0.4, ignore_index=255,
                                       connectivity=4, dilation_radius=0)
    l_total, parts = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight)
    print("parts:", parts)
    assert torch.isfinite(l_total), "l_total phải hữu hạn"

    l_total.backward()
    assert model.encoder.conv1.weight.grad is not None and \
        torch.any(model.encoder.conv1.weight.grad != 0), \
        "gradient phải lan truyền tới tận backbone qua l_region"
    assert boundary_head.conv_out.weight.grad is not None and \
        torch.any(boundary_head.conv_out.weight.grad != 0), \
        "gradient phải lan truyền tới BoundaryHead qua l_bce"
    assert model.frh.proj[0].weight.grad is not None and \
        torch.any(model.frh.proj[0].weight.grad != 0), \
        "gradient phải lan truyền tới decoder (nơi sinh Fused Feature) qua l_affinity"

    print("PASS — l_total hữu hạn, gradient chảy tới cả backbone/BoundaryHead/decoder.")
