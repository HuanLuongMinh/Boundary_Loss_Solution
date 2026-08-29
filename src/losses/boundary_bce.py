"""L_BCE_edge — boundary/edge supervision loss cho thực nghiệm "Baseline vs BCE
Loss" (Run 2 của Bảng 1, docs/idea_research.md): L_total = L_seg + lambda_edge * L_edge.

Spec kỹ thuật đầy đủ nằm ở docs/workflow_2.md mục 3.1 (transcribe từ file LaTeX
"Edge-pixel ratio measurement and balanced BCE edge loss" người dùng cung cấp —
đã đối chiếu, không có mâu thuẫn giữa 2 nguồn). Tóm tắt các quyết định đã chốt:

  - Background (class 0) là 1 class supervised THẬT (NUM_CLASSES=9, đúng
    convention baseline hiện có trong src/models/unet_former_resnet18.py /
    src/data/dataset.py) — KHÔNG map sang ignore_index. Chuyển tiếp
    Background -> lớp địa vật khác VẪN tính là 1 boundary hợp lệ.
  - `ignore_index=255` giữ để tương lai an toàn dù hiện tại là no-op (dataset
    OpenEarthMap hiện dùng không sinh ra pixel giá trị 255).
  - Edge target dùng 4-connected mặc định, `dilation_radius=0` (không dilate
    thêm) cho thực nghiệm Run 2 này — edge width đúng 1 pixel, khớp
    `sumary_template.txt` ("Edge dilation radius: 0 pixels").
  - Không bao giờ sigmoid thủ công trước khi gọi BCEWithLogits — luôn truyền
    raw logits.

File này ĐỘC LẬP, không sửa src/utils/losses.py (CombinedLoss cũ, vẫn dùng
nguyên cho L_seg) hay src/models/unet_former_resnet18.py (chỉ import lại
`ConvBNReLU` — 1 khối conv+BN+ReLU6 sẵn có — để BoundaryHead dùng đúng style
conv của decoder, không định nghĩa lại).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet_former_resnet18 import ConvBNReLU

# 4-connected: lên/xuống/trái/phải. 8-connected thêm 4 đường chéo. Chọn 1 lần,
# cố định cho toàn bộ nghiên cứu (mặc định 4-connected).
_OFFSETS_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_OFFSETS_8 = _OFFSETS_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def extract_edge_gt(mask: torch.Tensor, ignore_index: int = 255,
                     connectivity: int = 4, dilation_radius: int = 0):
    """(B,H,W) int64 label mask -> (edge_gt, valid_mask), cả hai (B,1,H,W) float {0,1}.

    QUAN TRỌNG: chỉ gọi hàm này trên `masks` tensor SAU khi đã qua
    src/data/transforms.py (resize/crop) trong training loop — KHÔNG gọi trên
    raster gốc đọc thẳng từ OpenEarthMapDataset — vì edge_gt phải cùng
    resolution với `edge_logits` (suy ra từ ảnh đã transform) để BCE tính đúng
    pixel-to-pixel.

    Validity: m = (mask != ignore_index). Edge target 4/8-connected: pixel
    trung tâm phải valid VÀ có ít nhất 1 hàng-xóm valid mang nhãn khác. Padding
    ở biên ảnh dùng `ignore_index` nên hàng-xóm ngoài biên tự động invalid,
    không tạo cạnh giả ở viền ảnh.

    `dilation_radius > 0`: dilate edge (max-pool nhị phân) rồi nhân lại với
    validity mask để loại phần dilate lấn vào vùng ignore. `dilation_radius=0`
    (mặc định) = không dilate thêm, edge_gt = biên 1-pixel thô.
    """
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    assert mask.dim() == 3, f"expected (B,H,W) mask, got shape {tuple(mask.shape)}"
    B, H, W = mask.shape

    valid = mask != ignore_index                                   # (B,H,W) bool

    padded_mask = F.pad(mask.unsqueeze(1).float(), (1, 1, 1, 1),
                        mode='constant', value=float(ignore_index)).squeeze(1)
    padded_valid = F.pad(valid.unsqueeze(1).float(), (1, 1, 1, 1),
                         mode='constant', value=0.0).squeeze(1).bool()

    offsets = _OFFSETS_4 if connectivity == 4 else _OFFSETS_8
    diff_any = torch.zeros_like(valid)
    for dy, dx in offsets:
        n_mask  = padded_mask[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        n_valid = padded_valid[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        diff = (mask.float() != n_mask) & n_valid
        diff_any = diff_any | diff

    edge_gt = (valid & diff_any).float().unsqueeze(1)               # (B,1,H,W)
    valid_mask = valid.float().unsqueeze(1)                         # (B,1,H,W)

    if dilation_radius > 0:
        k = 2 * dilation_radius + 1
        edge_gt = F.max_pool2d(edge_gt, kernel_size=k, stride=1, padding=dilation_radius)
        edge_gt = edge_gt * valid_mask

    return edge_gt, valid_mask


class BoundaryHead(nn.Module):
    """Nhánh conv phụ nhỏ gắn lên Fused Feature (64 kênh, stride 4) của decoder
    UNetFormer (`unet_former_resnet18.py`, `return_fused_feature=True` — model
    KHÔNG bị sửa) để sinh `edge_logits`. Đây là giải pháp khả vi thay cho cách
    trích biên bằng Laplacian-trên-argmax (không lan truyền gradient), xem
    giải thích đầy đủ ở docs/workflow_2.md mục 3.1."""

    def __init__(self, in_channels: int = 64, dropout: float = 0.1):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(dropout)
        self.conv_out = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, fused_feature: torch.Tensor, out_size) -> torch.Tensor:
        x = self.conv_out(self.drop(self.conv(fused_feature)))
        # Raw logits — KHÔNG sigmoid ở đây. BalancedBCEEdgeLoss dùng
        # binary_cross_entropy_with_logits trực tiếp trên output này.
        return F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)


def compute_pos_weight(n_edge: float, n_nonedge: float, w_max: float = 20.0) -> float:
    """pos_weight = N_nonedge / max(N_edge,1), clip về [1.0, w_max]."""
    raw = n_nonedge / max(n_edge, 1.0)
    return float(min(max(raw, 1.0), w_max))


class BalancedBCEEdgeLoss(nn.Module):
    """L_BCE_edge — balanced BCE trên edge_logits vs edge_gt (trích từ masks
    qua extract_edge_gt), pos_weight cố định truyền vào forward() (tính 1 lần
    lúc khởi động training, xem train_bce_edge.py — không tính lại pos_weight
    mỗi iteration)."""

    def __init__(self, ignore_index: int = 255, connectivity: int = 4,
                 dilation_radius: int = 0):
        super().__init__()
        self.ignore_index = ignore_index
        self.connectivity = connectivity
        self.dilation_radius = dilation_radius

    def forward(self, edge_logits: torch.Tensor, masks: torch.Tensor,
                pos_weight: float) -> torch.Tensor:
        edge_logits = edge_logits.float()  # guard NaN under AMP float16
        edge_gt, valid_mask = extract_edge_gt(
            masks, self.ignore_index, self.connectivity, self.dilation_radius)
        pw = torch.as_tensor(pos_weight, dtype=edge_logits.dtype, device=edge_logits.device)
        loss = F.binary_cross_entropy_with_logits(
            edge_logits, edge_gt, reduction='none', pos_weight=pw)
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp_min(1.0)


class EdgeStatsAccumulator:
    """Cộng dồn N_valid/N_edge qua nhiều batch mask KHÔNG average theo batch
    (docs/workflow_2.md mục 3.1 điểm 3 — các batch có số pixel valid khác
    nhau). Dùng chung bởi Tools/measure_edge_ratio.py (1 tiến trình, 1 epoch
    đầy đủ) và train_bce_edge.py (mỗi rank tự tích luỹ rồi all_reduce SUM)."""

    def __init__(self, ignore_index: int = 255, connectivity: int = 4,
                 dilation_radius: int = 0):
        self.ignore_index = ignore_index
        self.connectivity = connectivity
        self.dilation_radius = dilation_radius
        self.n_valid = 0.0
        self.n_edge = 0.0
        self.n_batches = 0

    @torch.no_grad()
    def update(self, mask: torch.Tensor):
        edge_gt, valid_mask = extract_edge_gt(
            mask, self.ignore_index, self.connectivity, self.dilation_radius)
        self.n_valid += valid_mask.sum().item()
        self.n_edge  += edge_gt.sum().item()
        self.n_batches += 1

    def counts_tensor(self, device, dtype=torch.float64) -> torch.Tensor:
        """(2,) tensor [n_valid, n_edge] — dùng cho dist.all_reduce(SUM) giữa
        các rank DDP (mỗi rank tích luỹ trên batch của riêng mình trước)."""
        return torch.tensor([self.n_valid, self.n_edge], dtype=dtype, device=device)

    def load_counts_tensor(self, t: torch.Tensor):
        """Nạp lại [n_valid, n_edge] đã all_reduce SUM — GHI ĐÈ, không cộng
        dồn thêm (gọi 1 lần duy nhất ngay sau all_reduce)."""
        self.n_valid = float(t[0].item())
        self.n_edge  = float(t[1].item())

    def result(self, w_max: float = 20.0) -> dict:
        n_valid = max(self.n_valid, 0.0)
        n_edge  = max(self.n_edge, 0.0)
        n_nonedge = n_valid - n_edge
        r_edge = n_edge / max(n_valid, 1.0)
        return {
            'N_valid':         n_valid,
            'N_edge':          n_edge,
            'N_nonedge':       n_nonedge,
            'r_edge':          r_edge,
            'r_nonedge':       1.0 - r_edge,
            'pos_weight_raw':  n_nonedge / max(n_edge, 1.0),
            'pos_weight':      compute_pos_weight(n_edge, n_nonedge, w_max),
            'connectivity':    self.connectivity,
            'dilation_radius': self.dilation_radius,
        }


def compute_edge_ratio_stats(mask_batches, ignore_index: int = 255,
                              connectivity: int = 4, dilation_radius: int = 0,
                              w_max: float = 20.0, max_batches: int = None) -> dict:
    """Chạy qua 1 iterable batch (mask tensor trực tiếp, hoặc (images, masks)
    tuple như DataLoader của OpenEarthMapDataset trả về), cộng dồn N_valid/
    N_edge, trả về dict thống kê đầy đủ. Dùng bởi Tools/measure_edge_ratio.py
    (đo 1 lần, "chính thức", 1 tiến trình, không DDP)."""
    acc = EdgeStatsAccumulator(ignore_index, connectivity, dilation_radius)
    for i, batch in enumerate(mask_batches):
        if max_batches is not None and i >= max_batches:
            break
        mask = batch[1] if isinstance(batch, (tuple, list)) else batch
        acc.update(mask)
    result = acc.result(w_max)
    result['n_batches'] = acc.n_batches
    return result
