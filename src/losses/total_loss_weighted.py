"""src/losses/total_loss_weighted.py — L_total cho Run 3b "λ₂=0.5 cho Affinity"
(docs/run3b_spec_lambda2_05.md), điểm giữa của đường quét affinity 3 điểm
(α_affinity ∈ {0, 0.2, 0.4}):

    L_total = L_region + alpha * (lambda1 * L_BCE_edge + lambda2 * L_Affinity)

File MỚI, ĐỘC LẬP HOÀN TOÀN với src/losses/total_loss.py (Run 3,
StaticBoundaryTotalLoss — lambda1=lambda2=1 hard-code trong forward(), KHÔNG
sửa file đó để Run 3 tái lập được bất kỳ lúc nào). Khác biệt DUY NHẤT so với
bản gốc: lambda1/lambda2 là THAM SỐ có thể cấu hình (mặc định 1.0 mỗi khoá —
truyền mặc định thì công thức và số ra hệt StaticBoundaryTotalLoss gốc), thay
vì hằng số 1.0 cứng trong công thức.

L_region tái sử dụng CombinedLoss (CE+Dice) đã có sẵn (src/utils/losses.py).
L_BCE_edge tái sử dụng BalancedBCEEdgeLoss (src/losses/boundary_bce.py).
L_Affinity dùng AffinityLoss (src/losses/affinity.py). Cả 3 file phụ thuộc này
đều là file DÙNG CHUNG, không sửa.

connectivity/dilation_radius PHẢI truyền giống hệt nhau cho cả bce_edge và
affinity (đúng bất biến "1 định nghĩa biên duy nhất" mà
docs/workflow_2.md mục 3.1/3.2 yêu cầu, giữ nguyên từ Run 3).
"""

import torch
import torch.nn as nn

from src.utils.losses import CombinedLoss
from src.losses.boundary_bce import BalancedBCEEdgeLoss
from src.losses.affinity import AffinityLoss


class StaticBoundaryTotalLossWeighted(nn.Module):
    def __init__(self, num_classes: int, alpha: float, ignore_index: int = 255,
                 ce_weight: float = 1.0, dice_weight: float = 1.0,
                 connectivity: int = 4, dilation_radius: int = 0,
                 affinity_window_size: int = 5, affinity_distance: str = 'cosine',
                 affinity_margin: float = 1.0,
                 lambda1_static: float = 1.0, lambda2_static: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.lambda1_static = float(lambda1_static)
        self.lambda2_static = float(lambda2_static)
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
        """Xem docstring StaticBoundaryTotalLoss.forward() (total_loss.py) cho
        ý nghĩa từng tham số — giữ nguyên chữ ký, chỉ khác lambda1/lambda2
        không còn cố định =1."""
        l_region = self.region_loss(logits, masks)
        l_bce = self.bce_edge(edge_logits, masks, pos_weight)
        l_affinity = self.affinity(fused_feature, masks)
        lam1, lam2 = self.lambda1_static, self.lambda2_static
        l_total = l_region + self.alpha * (lam1 * l_bce + lam2 * l_affinity)
        return l_total, {
            'l_region': l_region.item(), 'l_bce': l_bce.item(),
            'l_affinity': l_affinity.item(), 'l_total': l_total.item(),
            'lambda1': lam1, 'lambda2': lam2,
            'alpha': self.alpha,
            'alpha_bce_effective': self.alpha * lam1,
            'alpha_affinity_effective': self.alpha * lam2,
        }


if __name__ == "__main__":
    # Self-test tích hợp — chạy `python -m src.losses.total_loss_weighted` từ
    # repo root. Dùng THẬT model UNetFormer + BoundaryHead (không giả lập
    # shape), giống hệt self-test của total_loss.py, cộng thêm đối chiếu trực
    # tiếp với StaticBoundaryTotalLoss gốc (Run 3) trên CÙNG input/seed để xác
    # nhận default lambda1=lambda2=1.0 tái lập ĐÚNG y hệt bản gốc (kiểm chứng
    # tương thích ngược ở mức unit-test, tách biệt khỏi dry-run cấp script).
    import torch.nn.functional as F
    from src.models.unet_former_resnet18 import UNetFormer
    from src.losses.boundary_bce import BoundaryHead, extract_edge_gt, compute_pos_weight
    from src.losses.total_loss import StaticBoundaryTotalLoss

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

    # ── Test 1: lambda1=lambda2=1.0 (default) phải khớp tuyệt đối với
    # StaticBoundaryTotalLoss gốc của Run 3 (cùng seed cho cả 2 module con) ──
    torch.manual_seed(123)
    loss_fn_weighted_default = StaticBoundaryTotalLossWeighted(
        num_classes=NUM_CLASSES, alpha=0.4, ignore_index=255,
        connectivity=4, dilation_radius=0)  # lambda1_static/lambda2_static mặc định 1.0
    torch.manual_seed(123)
    loss_fn_original = StaticBoundaryTotalLoss(
        num_classes=NUM_CLASSES, alpha=0.4, ignore_index=255,
        connectivity=4, dilation_radius=0)

    l_total_w, parts_w = loss_fn_weighted_default(logits, edge_logits, fused_feature, masks, pos_weight)
    l_total_o, parts_o = loss_fn_original(logits, edge_logits, fused_feature, masks, pos_weight)
    print("parts (weighted, default lambda=1,1):", parts_w)
    print("parts (original StaticBoundaryTotalLoss):", parts_o)
    assert abs(parts_w['l_total'] - parts_o['l_total']) < 1e-6, \
        "Tương thích ngược thất bại — default lambda1=lambda2=1.0 phải cho l_total y hệt Run 3 gốc"
    assert parts_w['lambda1'] == 1.0 and parts_w['lambda2'] == 1.0
    assert abs(parts_w['alpha_bce_effective'] - 0.4) < 1e-9
    assert abs(parts_w['alpha_affinity_effective'] - 0.4) < 1e-9
    print("PASS — default lambda1=lambda2=1.0 tái lập đúng y hệt StaticBoundaryTotalLoss (Run 3).\n")

    # ── Test 2: lambda2=0.5 (cấu hình Run 3b thật) — kiểm công thức + gradient ──
    loss_fn = StaticBoundaryTotalLossWeighted(
        num_classes=NUM_CLASSES, alpha=0.4, ignore_index=255,
        connectivity=4, dilation_radius=0,
        lambda1_static=1.0, lambda2_static=0.5)
    l_total, parts = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight)
    print("parts (Run 3b, lambda2=0.5):", parts)
    assert torch.isfinite(l_total), "l_total phải hữu hạn"
    expected = parts['l_region'] + parts['alpha'] * (
        parts['lambda1'] * parts['l_bce'] + parts['lambda2'] * parts['l_affinity'])
    assert abs(parts['l_total'] - expected) < 1e-5, "đồng nhất thức công thức loss thất bại"
    assert abs(parts['alpha_affinity_effective'] - 0.2) < 1e-9, \
        "alpha_affinity_effective phải = alpha*lambda2 = 0.4*0.5 = 0.2"

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

    print("PASS — l_total hữu hạn, đúng công thức, gradient chảy tới cả "
          "backbone/BoundaryHead/decoder (lambda2=0.5).")
