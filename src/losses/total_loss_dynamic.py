"""src/losses/total_loss_dynamic.py — L_total cho Run 4 "Dynamic Boundary-Aware
Loss" (docs/spec-run4-dynamic-weighting.md):

    L_total = L_region + alpha * (lambda1(t) * L_BCE_edge + lambda2(t) * L_Affinity)

lambda1(t)/lambda2(t) đọc từ src/losses/dynamic_weighting.py::SCHEDULES theo
tên lịch trình (mặc định 'ramp_hold', duy nhất được dùng ở Run 4 — 'linear'
tồn tại trong registry cho đủ nhưng không dùng ở đây, xem docstring
dynamic_weighting.py).

File MỚI, ĐỘC LẬP HOÀN TOÀN với src/losses/total_loss.py (Run 3, lambda1=
lambda2=1 hard-code) và src/losses/total_loss_weighted.py (Run 3b, lambda1/
lambda2 static có thể cấu hình nhưng KHÔNG đổi theo iteration) — không sửa,
không import chéo 2 file đó. Khác biệt DUY NHẤT so với 2 bản trên: lambda1/
lambda2 là HÀM của cur_iter (qua schedule), không phải hằng số.

L_region tái sử dụng CombinedLoss (CE+Dice) đã có sẵn (src/utils/losses.py).
L_BCE_edge tái sử dụng BalancedBCEEdgeLoss (src/losses/boundary_bce.py).
L_Affinity dùng AffinityLoss (src/losses/affinity.py). Cả 3 file phụ thuộc
này đều là file DÙNG CHUNG, không sửa.

L_Affinity LUÔN được tính mỗi forward() (không if/else theo lambda2) — kể cả
trong warmup khi lambda2=0, để có l_affinity thô cho lambda_schedule_log.csv
(hình lịch trình của bài báo, spec mục 3.2/7.4). Cái giá ~9 phút GPU chấp
nhận được so với việc mất đường cong đó vĩnh viễn.

connectivity/dilation_radius PHẢI truyền giống hệt nhau cho cả bce_edge và
affinity (đúng bất biến "1 định nghĩa biên duy nhất", giữ nguyên từ Run 3).
"""

import torch
import torch.nn as nn

from src.utils.losses import CombinedLoss
from src.losses.boundary_bce import BalancedBCEEdgeLoss
from src.losses.affinity import AffinityLoss
from src.losses.dynamic_weighting import SCHEDULES


class DynamicBoundaryTotalLoss(nn.Module):
    def __init__(self, num_classes: int, alpha: float, max_iters: int,
                 ignore_index: int = 255,
                 ce_weight: float = 1.0, dice_weight: float = 1.0,
                 connectivity: int = 4, dilation_radius: int = 0,
                 affinity_window_size: int = 5, affinity_distance: str = 'cosine',
                 affinity_margin: float = 1.0,
                 dynamic_weights: bool = True,
                 schedule_name: str = 'ramp_hold', schedule_kwargs: dict = None,
                 lambda1_static: float = 1.0, lambda2_static: float = 1.0):
        """dynamic_weights=True (mac dinh, dung cho Run 4): lambda1(t)/
        lambda2(t) doc tu SCHEDULES[schedule_name]. dynamic_weights=False:
        dung hang so lambda1_static/lambda2_static (mac dinh 1.0/1.0) - tuong
        duong StaticBoundaryTotalLoss goc cua Run 3, dung de kiem tuong thich
        nguoc (docs/spec-run4-dynamic-weighting.md muc 3.2/5.3) bang CACH
        DIEM cung mot script/class, khong phai chay lai script Run 3 rieng."""
        super().__init__()
        self.alpha = alpha
        self.max_iters = int(max_iters)
        self.dynamic_weights = dynamic_weights
        self.schedule_name = schedule_name
        self.schedule_fn = SCHEDULES[schedule_name]
        self.schedule_kwargs = dict(schedule_kwargs) if schedule_kwargs else {}
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
                fused_feature: torch.Tensor, masks: torch.Tensor, pos_weight: float,
                cur_iter: int):
        """Xem docstring StaticBoundaryTotalLoss.forward() (total_loss.py) cho
        ý nghĩa logits/edge_logits/fused_feature/masks/pos_weight — giữ
        nguyên chữ ký, cộng thêm cur_iter (0-indexed, vd cur_iter=0 ở bước
        train đầu tiên) để tính lambda1(t)/lambda2(t) qua schedule."""
        l_region = self.region_loss(logits, masks)
        l_bce = self.bce_edge(edge_logits, masks, pos_weight)
        l_affinity = self.affinity(fused_feature, masks)
        if self.dynamic_weights:
            lam1, lam2 = self.schedule_fn(cur_iter, self.max_iters, **self.schedule_kwargs)
        else:
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
    # Self-test tích hợp — chạy `python -m src.losses.total_loss_dynamic` từ
    # repo root. Dùng THẬT model UNetFormer + BoundaryHead (không giả lập
    # shape), giống hệt self-test của total_loss.py/total_loss_weighted.py.
    from src.models.unet_former_resnet18 import UNetFormer
    from src.losses.boundary_bce import BoundaryHead, extract_edge_gt, compute_pos_weight
    from src.losses.total_loss import StaticBoundaryTotalLoss
    from src.losses.dynamic_weighting import lambda_schedule_ramp_hold

    torch.manual_seed(19)
    B, H, W, NUM_CLASSES = 2, 256, 256, 9  # đủ lớn cho GLTB window_size=8 ở stride 32 (256/32=8)
    MAX_ITERS = 40000

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

    loss_fn = DynamicBoundaryTotalLoss(
        num_classes=NUM_CLASSES, alpha=0.4, max_iters=MAX_ITERS, ignore_index=255,
        connectivity=4, dilation_radius=0, schedule_name='ramp_hold')

    # ── Test 1: warmup (cur_iter=0) — alpha_affinity_effective phải =0, l_affinity vẫn hữu hạn dương ──
    l_total_warmup, parts_warmup = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight, cur_iter=0)
    print("parts (cur_iter=0, warmup):", parts_warmup)
    assert parts_warmup['lambda2'] == 0.0
    assert abs(parts_warmup['alpha_affinity_effective'] - 0.0) < 1e-12
    assert torch.isfinite(torch.tensor(parts_warmup['l_affinity'])) and parts_warmup['l_affinity'] > 0.0, \
        "l_affinity phải vẫn được tính (hữu hạn, dương) ngay cả khi lambda2=0 trong warmup"
    print("PASS — warmup vo hieu hoa affinity (alpha_affinity_effective=0) nhung l_affinity van tinh.\n")

    # ── Test 2: hold (cur_iter=40000) — phải khớp StaticBoundaryTotalLoss gốc (lambda1=lambda2=1) ──
    torch.manual_seed(123)
    loss_fn_hold = DynamicBoundaryTotalLoss(
        num_classes=NUM_CLASSES, alpha=0.4, max_iters=MAX_ITERS, ignore_index=255,
        connectivity=4, dilation_radius=0, schedule_name='ramp_hold')
    torch.manual_seed(123)
    loss_fn_static = StaticBoundaryTotalLoss(
        num_classes=NUM_CLASSES, alpha=0.4, ignore_index=255,
        connectivity=4, dilation_radius=0)
    l_total_hold, parts_hold = loss_fn_hold(logits, edge_logits, fused_feature, masks, pos_weight, cur_iter=MAX_ITERS)
    l_total_static, parts_static = loss_fn_static(logits, edge_logits, fused_feature, masks, pos_weight)
    print("parts (cur_iter=40000, hold):", parts_hold)
    print("parts (StaticBoundaryTotalLoss, Run 3 goc):", parts_static)
    assert parts_hold['lambda1'] == 1.0 and parts_hold['lambda2'] == 1.0
    assert abs(parts_hold['l_total'] - parts_static['l_total']) < 1e-6, \
        "O hold (lambda1=lambda2=1), l_total phai khop tuyet doi StaticBoundaryTotalLoss goc"
    print("PASS — hold (cur_iter=40000) tai lap dung Run 3 static (lambda1=lambda2=1).\n")

    # ── Test 3: công thức đồng nhất tại điểm giữa ramp (cur_iter=8000, lambda2=0.5) ──
    l_total_mid, parts_mid = loss_fn(logits, edge_logits, fused_feature, masks, pos_weight, cur_iter=8000)
    lam1_exp, lam2_exp = lambda_schedule_ramp_hold(8000, MAX_ITERS)
    assert abs(parts_mid['lambda1'] - lam1_exp) < 1e-9 and abs(parts_mid['lambda2'] - lam2_exp) < 1e-9
    expected = parts_mid['l_region'] + parts_mid['alpha'] * (
        parts_mid['lambda1'] * parts_mid['l_bce'] + parts_mid['lambda2'] * parts_mid['l_affinity'])
    assert abs(parts_mid['l_total'] - expected) < 1e-5, "dong nhat thuc cong thuc loss that bai"
    assert torch.isfinite(l_total_mid), "l_total phai huu han"
    print(f"PASS — dong nhat thuc cong thuc loss dung tai cur_iter=8000 (lambda2={lam2_exp}).\n")

    # ── Test 4: gradient chảy tới cả backbone/BoundaryHead/decoder (dùng loss ở ramp giữa) ──
    l_total_mid.backward()
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
          "backbone/BoundaryHead/decoder (Run 4, DynamicBoundaryTotalLoss).\n")

    # ── Test 5: tương thích ngược (spec mục 3.2/5.3) — dynamic_weights=False,
    # lambda1_static/lambda2_static mặc định 1.0/1.0 -> phải khớp tuyệt đối
    # StaticBoundaryTotalLoss gốc, BẤT KỂ cur_iter truyền vào là gì (schedule
    # phải bị bỏ qua hoàn toàn) ──
    torch.manual_seed(123)
    loss_fn_static_mode = DynamicBoundaryTotalLoss(
        num_classes=NUM_CLASSES, alpha=0.4, max_iters=MAX_ITERS, ignore_index=255,
        connectivity=4, dilation_radius=0, dynamic_weights=False)  # lambda1_static/lambda2_static mac dinh 1.0
    l_total_static_mode, parts_static_mode = loss_fn_static_mode(
        logits, edge_logits, fused_feature, masks, pos_weight, cur_iter=123)  # cur_iter bat ky, phai bi bo qua
    print("parts (dynamic_weights=False, cur_iter=123 bi bo qua):", parts_static_mode)
    assert parts_static_mode['lambda1'] == 1.0 and parts_static_mode['lambda2'] == 1.0
    assert abs(parts_static_mode['l_total'] - parts_static['l_total']) < 1e-6, \
        "Tuong thich nguoc that bai - dynamic_weights=False (default 1.0/1.0) phai cho l_total y het Run 3 goc"
    print("PASS — dynamic_weights=False tai lap dung Run 3 static, bo qua schedule/cur_iter hoan toan.")
