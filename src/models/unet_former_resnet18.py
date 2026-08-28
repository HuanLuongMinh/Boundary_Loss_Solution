"""
UNetFormer baseline — encoder ResNet-18 (pretrained), decoder GLTB
(Global-Local Transformer Block) — kiến trúc PAPER-FAITHFUL, đúng cấu hình gốc
của UNetFormer (Wang et al., "UNetFormer: A UNet-like Transformer for
Efficient Semantic Segmentation of Remote Sensing Urban Scene Imagery").

Đây là baseline cho đề tài "Dynamic Boundary-Aware Loss via Adaptive
BCE-Affinity Coupling" (xem idead_research.md): baseline = L_region
(CrossEntropy + Dice trên Fused Feature của decoder), CHƯA có bất kỳ thành
phần boundary loss / affinity / DAPCN / aux head nào — đúng với dòng
"Baseline (L_region)" trong Bảng 1 của đề cương.

Vì sao ResNet-18 (không phải ResNeXt101_32x16d như các track UNetFormer cũ
trong repo)?
    Paper gốc UNetFormer chọn ResNet-18 làm encoder mặc định — lý do được nêu
    rõ trong bài: "we select the pre-trained ResNet18 as the encoder here to
    extract multi-scale semantic features with significantly low
    computational cost" (ablation so sánh với ViT-Tiny/Swin-Tiny/CoaT-Mini
    cho thấy ResNet18 tối ưu về hiệu năng thời gian thực mà vẫn giữ độ chính
    xác cạnh tranh). Backbone nhẹ này còn đặc biệt phù hợp với protocol thực
    nghiệm 2 giai đoạn của đề tài (18 runs: 8 runs Phase 1 + 10 runs Phase 2
    5-fold CV) trên hạ tầng Kaggle 2×T4 hạn chế — xem phân tích chi tiết
    trong tài liệu phân tích dataset đi kèm.

Encoder options hỗ trợ (channels giống hệt nhau: [64, 128, 256, 512], stride
4/8/16/32 — đã verify bằng timm.create_model(..., features_only=True)):
    - 'resnet18.fb_swsl_ig1b_ft_in1k'  (mặc định — tương đương 'swsl_resnet18'
      dùng trong code gốc GeoSeg/UNetFormer, pretrain bán giám sát trên ~1 tỷ
      ảnh Instagram rồi fine-tune ImageNet — cùng họ SWSL với
      'resnext101_32x16d.fb_swsl_ig1b_ft_in1k' đã dùng ở các track khác trong
      repo này, đảm bảo nhất quán về "chất lượng pretrain" khi so sánh giữa
      các track).
    - 'resnet18.tv_in1k'  (ResNet-18 pretrain ImageNet chuẩn torchvision, dùng
      nếu muốn khớp đúng nghĩa đen "pretrained ResNet18" của paper mà không
      phụ thuộc thêm giả định SWSL).

Decoder = đúng 3 khối GLTB (Global-Local Transformer Block) phân cấp + WF
(Weighted Fusion, skip connection có trọng số học được) + FeatureRefinementHead
(spatial attention + channel attention) — không có DAPCN/boundary loss/
prototype/contrastive learning/dynamic anchor point. Output cuối của decoder
(sau FeatureRefinementHead, trước segmentation_head) là "Fused Feature" theo
đúng thuật ngữ trong yêu cầu — CE+Dice được tính trên logits suy ra từ Fused
Feature này (xem forward(), tham số return_fused_feature).

Interface giữ giống các model khác trong project:
    - class UNetFormer(encoder_name, num_classes, pretrained, ...)
    - build_model(cfg) -> UNetFormer
    - forward(x) -> logits ở resolution gốc của input
    - forward(x, return_fused_feature=True) -> (logits, fused_feature)
nên src/train_unet_former_resnet18.py dùng được pattern train giống các
script train_*.py hiện có, và fused_feature có sẵn cho các nghiên cứu sau này
cần gắn thêm boundary/affinity loss lên đúng feature map này mà không phải
sửa lại kiến trúc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from einops import rearrange
from timm.layers import DropPath, trunc_normal_

# ResNet-18 (basic-block) — 4 stage feature maps, stride 4/8/16/32.
# Đã verify bằng timm.create_model(name, features_only=True, out_indices=(1,2,3,4)):
#   channels = [64, 128, 256, 512], reduction = [4, 8, 16, 32]  (cả 2 biến thể)
_ENCODER_CHANNELS = {
    'resnet18.fb_swsl_ig1b_ft_in1k': [64, 128, 256, 512],
    'resnet18.tv_in1k':              [64, 128, 256, 512],
}


# ───────────────────────── Basic conv blocks ──────────────────────────────────
class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(),
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


# ───────────────────────── Global-Local Attention (lõi của GLTB) ──────────────
class GlobalLocalAttention(nn.Module):
    """Nhánh attention của GLTB: window-based multi-head self-attention (nhánh
    global) cộng với 2 conv song song 3x3 + 1x1 (nhánh local), theo đúng mô tả
    trong paper UNetFormer gốc. Relative position bias học được trong mỗi
    window (giống Swin) để mã hoá vị trí tương đối giữa các pixel cục bộ."""

    def __init__(self, dim=256, num_heads=16, qkv_bias=False, window_size=8,
                 relative_pos_embedding=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,
                                   padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1,
                                   padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding
        if relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        pad_w = (ps - W % ps) % ps
        pad_h = (ps - H % ps) % ps
        if pad_w or pad_h:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x

    def pad_out(self, x):
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def forward(self, x):
        B, C, H, W = x.shape
        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        _, _, Hp, Wp = x.shape
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws,
            qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        if self.relative_pos_embedding:
            rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            rpb = rpb.permute(2, 0, 1).contiguous()
            dots = dots + rpb.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v
        attn = rearrange(
            attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads, d=C // self.num_heads, hh=Hp // self.ws, ww=Wp // self.ws,
            ws1=self.ws, ws2=self.ws)
        attn = attn[:, :, :H, :W]

        # Khuếch tán attention ra ngoài biên window (attn_x/attn_y) để giảm
        # hiệu ứng "block artifact" giữa các window liền kề — đặc trưng riêng
        # của UNetFormer so với Swin attention thuần.
        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))
        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]
        return out


class GLTB(nn.Module):
    """Global-Local Transformer Block — đơn vị attention chính của decoder
    UNetFormer. Cấu trúc chuẩn Transformer block: LayerNorm -> Attention ->
    residual -> LayerNorm -> MLP -> residual, nhưng norm dùng BatchNorm2d và
    attention là GlobalLocalAttention (global window-attention + local conv)
    thay vì self-attention thuần, để phù hợp với dữ liệu ảnh dày đặc
    (dense prediction) thay vì chuỗi token."""

    def __init__(self, dim=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                         window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    """Weighted Fusion: upsample decoder feature rồi cộng có trọng số học
    được (softmax-normalized, giống BiFPN) với skip connection từ encoder."""

    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[2:], mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        return self.post_conv(x)


class FeatureRefinementHead(nn.Module):
    """Stage nông nhất (stride 4) của decoder: hợp nhất WF + spatial
    attention (depth-wise conv + sigmoid) + channel attention (global avg
    pool + 2 conv 1x1 + sigmoid), cộng shortcut — sinh ra "Fused Feature"
    cuối cùng trước segmentation_head."""

    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1,
                      groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, decode_channels // 16, kernel_size=1),
            nn.ReLU6(),
            Conv(decode_channels // 16, decode_channels, kernel_size=1),
            nn.Sigmoid())
        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[2:], mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        return self.act(x)


# ───────────────────────── UNetFormer (full model) ────────────────────────────
class UNetFormer(nn.Module):
    """UNetFormer baseline — encoder ResNet-18 (pretrained), decoder GLTB.

    Pipeline: encoder (4 stage, stride 4/8/16/32)
              -> pre_conv4 + GLTB (stride 32, "b4")
              -> WF(stride16 skip) + GLTB ("b3")
              -> WF(stride8 skip)  + GLTB ("b2")
              -> FeatureRefinementHead(stride4 skip)  ==> Fused Feature
              -> dropout -> upsample về input size -> classifier (1x1 conv)
              => logits (B, num_classes, H, W)

    KHÔNG có DAPCN extras: không boundary loss, không prototype/contrastive
    learning, không dynamic anchor point, không aux head phụ (paper gốc có
    thêm nhánh aux CE 0.4x trên stride-4 feature khi training — baseline này
    cố tình bỏ để khớp đúng "L_region" thuần của đề cương nghiên cứu, không
    trộn thêm tín hiệu supervision khác).
    """

    def __init__(self, encoder_name: str = 'resnet18.fb_swsl_ig1b_ft_in1k',
                 num_classes: int = 9, pretrained: bool = True, decode_channels: int = 64,
                 window_size: int = 8, num_heads: int = 8, mlp_ratio: float = 4.0,
                 drop_path_rate: float = 0.1, dropout_ratio: float = 0.1):
        super().__init__()
        assert encoder_name in _ENCODER_CHANNELS, \
            f"encoder_name must be one of {list(_ENCODER_CHANNELS)}"
        enc_ch = _ENCODER_CHANNELS[encoder_name]

        # ── Encoder ────────────────────────────────────────────────────────
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
        )

        # ── UNetFormer decoder (3x GLTB + WF + FeatureRefinementHead) ───────
        self.pre_conv4 = ConvBN(enc_ch[3], decode_channels, kernel_size=1)
        self.b4 = GLTB(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                       mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        self.wf3 = WF(in_channels=enc_ch[2], decode_channels=decode_channels)
        self.b3 = GLTB(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                       mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        self.wf2 = WF(in_channels=enc_ch[1], decode_channels=decode_channels)
        self.b2 = GLTB(dim=decode_channels, num_heads=num_heads, window_size=window_size,
                       mlp_ratio=mlp_ratio, drop_path=drop_path_rate)

        self.frh = FeatureRefinementHead(in_channels=enc_ch[0], decode_channels=decode_channels)

        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        self.classifier = nn.Conv2d(decode_channels, num_classes, kernel_size=1)

    def forward(self, x, return_fused_feature: bool = False):
        input_size = x.shape[2:]
        res1, res2, res3, res4 = self.encoder(x)   # strides 4, 8, 16, 32

        d = self.pre_conv4(res4)   # (B, dc, H/32, W/32)
        d = self.b4(d)
        d = self.wf3(d, res3)      # H/16
        d = self.b3(d)
        d = self.wf2(d, res2)      # H/8
        d = self.b2(d)
        fused_feature = self.frh(d, res1)      # H/4  <-- "Fused Feature"

        d = self.dropout(fused_feature)
        out = F.interpolate(d, size=input_size, mode='bilinear', align_corners=False)
        logits = self.classifier(out)

        if return_fused_feature:
            return logits, fused_feature
        return logits


def build_model(cfg: dict) -> UNetFormer:
    mdl = cfg['MODEL']
    return UNetFormer(
        encoder_name=mdl['ENCODER'],
        num_classes=cfg['TRAIN']['NUM_CLASSES'],
        pretrained=mdl['PRETRAINED'],
        decode_channels=mdl.get('DECODE_CHANNELS', 64),
        window_size=mdl.get('WINDOW_SIZE', 8),
        num_heads=mdl.get('NUM_HEADS', 8),
        mlp_ratio=mdl.get('MLP_RATIO', 4.0),
        drop_path_rate=mdl.get('DROP_PATH_RATE', 0.1),
        dropout_ratio=mdl.get('DROPOUT_RATIO', 0.1),
    )
