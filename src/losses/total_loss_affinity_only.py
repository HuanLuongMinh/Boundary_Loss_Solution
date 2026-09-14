"""src/losses/total_loss_affinity_only.py — L_total cho Run 7 "Affinity-only"
(docs/spec-run7-affinity-only-ban-giao-claude-code.md muc 1.1):

    L_total = L_region + ALPHA * LAMBDA2_STATIC * L_Affinity

KHONG co so hang BCE-edge nao trong cong thuc — khac voi
src/losses/total_loss_weighted.py (Run 3b, luon tinh l_bce roi nhan voi
lambda1) o cho: file nay KHONG nhan `edge_logits`/`pos_weight` trong forward(),
KHONG co submodule BalancedBCEEdgeLoss nao ca. Day la khac biet CO CHU Y voi
"nhan trong so 0" — muc 1.1 cua spec noi ro alpha_bce_effective=0 vi
"l_bce khong duoc tinh vao tong, khong phai chi nhan trong so 0". File nay
hien thuc dung y do: khong co tensor l_bce nao duoc tinh trong do thi tinh
toan, nen BoundaryHead (von khong ton tai o kien truc Run 7 — xem
src/train_affinity_only.py) khong bao gio nhan gradient.

File MOI, DOC LAP HOAN TOAN voi total_loss.py (Run 3) va total_loss_weighted.py
(Run 3b) — KHONG sua 2 file do de chung tai lap duoc bat ky luc nao.

L_region tai su dung CombinedLoss (CE+Dice) da co san (src/utils/losses.py).
L_Affinity dung AffinityLoss (src/losses/affinity.py) — module nay KHONG co
tham so hoc duoc (khong nn.Parameter/Conv/Linear nao), chi la ham loss chay
truc tiep tren fused_feature co san tu decoder. Vi vay kien truc model cua
Run 7 la UNetFormer tran, khong can head phu nao (khac Run 3/3b can them
BoundaryHead cho nhanh BCE).
"""

import torch
import torch.nn as nn

from src.utils.losses import CombinedLoss
from src.losses.affinity import AffinityLoss


class AffinityOnlyTotalLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float, ignore_index: int = 255,
                 ce_weight: float = 1.0, dice_weight: float = 1.0,
                 connectivity: int = 4, dilation_radius: int = 0,
                 affinity_window_size: int = 5, affinity_distance: str = 'cosine',
                 affinity_margin: float = 1.0,
                 lambda1_static: float = 1.0, lambda2_static: float = 1.0):
        super().__init__()
        self.alpha = alpha
        # lambda1_static duoc GIU LAI chi de log/CSV dong dang voi Run 3/3b
        # (muc 1.1 spec: "khong co tac dung vi USE_BCE=false, giu de log/so
        # sanh nhat quan") — KHONG xuat hien trong cong thuc forward() ben duoi.
        self.lambda1_static = float(lambda1_static)
        self.lambda2_static = float(lambda2_static)
        self.region_loss = CombinedLoss(
            num_classes=num_classes, ignore_index=ignore_index,
            ce_weight=ce_weight, dice_weight=dice_weight)
        self.affinity = AffinityLoss(
            window_size=affinity_window_size, distance=affinity_distance, margin=affinity_margin,
            ignore_index=ignore_index, connectivity=connectivity, dilation_radius=dilation_radius)

    def forward(self, logits: torch.Tensor, fused_feature: torch.Tensor, masks: torch.Tensor):
        """Khac chu ky StaticBoundaryTotalLoss(Weighted).forward(): KHONG nhan
        edge_logits/pos_weight — Run 7 khong co BoundaryHead nen khong co gi
        de truyen vao."""
        l_region = self.region_loss(logits, masks)
        l_affinity = self.affinity(fused_feature, masks)
        lam2 = self.lambda2_static
        l_total = l_region + self.alpha * lam2 * l_affinity
        return l_total, {
            'l_region': l_region.item(), 'l_bce': 0.0,
            'l_affinity': l_affinity.item(), 'l_total': l_total.item(),
            'lambda1': self.lambda1_static, 'lambda2': lam2,
            'alpha': self.alpha,
            'alpha_bce_effective': 0.0,
            'alpha_affinity_effective': self.alpha * lam2,
        }


if __name__ == "__main__":
    # Self-test tich hop — chay `python -m src.losses.total_loss_affinity_only`
    # tu repo root. Dung THAT model UNetFormer (khong gia lap shape), doi
    # chieu cong thuc + gradient — khong can BoundaryHead, khong can GPU.
    from src.models.unet_former_resnet18 import UNetFormer

    torch.manual_seed(19)
    B, H, W, NUM_CLASSES = 2, 256, 256, 9  # 256/32=8, du lon cho window_size=8 o stride 32

    model = UNetFormer(encoder_name='resnet18.fb_swsl_ig1b_ft_in1k', num_classes=NUM_CLASSES,
                        pretrained=False, decode_channels=64, window_size=8)

    images = torch.randn(B, 3, H, W)
    masks = torch.randint(0, NUM_CLASSES, (B, H, W), dtype=torch.int64)
    masks[0, :5, :5] = 255  # vai pixel ignore, giong du lieu that

    logits, fused_feature = model(images, return_fused_feature=True)
    print(f"logits {tuple(logits.shape)}  fused_feature {tuple(fused_feature.shape)}")
    assert logits.shape == (B, NUM_CLASSES, H, W)
    assert fused_feature.shape[0] == B and fused_feature.shape[1] == 64
    assert fused_feature.shape[2] == H // 4 and fused_feature.shape[3] == W // 4, \
        "Fused Feature phai dung stride 4"

    # Kiem tra khong co tham so hoc duoc nao trong AffinityLoss (ly do kien
    # truc Run 7 khong can BoundaryHead).
    loss_fn = AffinityOnlyTotalLoss(
        num_classes=NUM_CLASSES, alpha=0.4, ignore_index=255,
        connectivity=4, dilation_radius=0)
    n_affinity_params = sum(1 for _ in loss_fn.affinity.parameters())
    assert n_affinity_params == 0, \
        f"AffinityLoss phai khong co tham so hoc duoc, thay {n_affinity_params}"
    print(f"PASS — AffinityLoss khong co tham so hoc duoc ({n_affinity_params} params).")

    l_total, parts = loss_fn(logits, fused_feature, masks)
    print("parts:", parts)
    assert torch.isfinite(l_total), "l_total phai huu han"
    expected = parts['l_region'] + parts['alpha'] * parts['lambda2'] * parts['l_affinity']
    assert abs(parts['l_total'] - expected) < 1e-5, "dong nhat thuc cong thuc loss that bai"
    assert parts['l_bce'] == 0.0 and parts['alpha_bce_effective'] == 0.0, \
        "l_bce/alpha_bce_effective phai la hang so 0.0 tuong minh"
    assert abs(parts['alpha_affinity_effective'] - 0.4) < 1e-9, \
        "alpha_affinity_effective phai = alpha*lambda2 = 0.4*1.0 = 0.4"

    l_total.backward()
    assert model.encoder.conv1.weight.grad is not None and \
        torch.any(model.encoder.conv1.weight.grad != 0), \
        "gradient phai lan truyen toi tan backbone qua l_region"
    assert model.frh.proj[0].weight.grad is not None and \
        torch.any(model.frh.proj[0].weight.grad != 0), \
        "gradient phai lan truyen toi decoder (noi sinh Fused Feature) qua l_affinity"

    print("PASS — l_total huu han, dung cong thuc, gradient chay toi ca "
          "backbone (qua l_region) va decoder/frh (qua l_affinity), khong co BoundaryHead.")
