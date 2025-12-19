import math
import torch
from torch import nn
import timm


# =========================
# SMALL UTILS (NEW)
# =========================
def temporal_diff(x):
    # x: (B, T, D)
    dx = x[:, 1:] - x[:, :-1]
    dx = torch.cat([torch.zeros_like(dx[:, :1]), dx], dim=1)
    return dx


class TemporalConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        # (B, T, D)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class AttnPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))

    def forward(self, x):
        # x: (B, T, D)
        attn = torch.softmax(torch.einsum('d,btd->bt', self.q, x), dim=1)
        return torch.einsum('bt,btd->bd', attn, x)


# =========================
# MASK STEM
# =========================
class ConvStem1to3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


# =========================
# IMAGE ENCODER (UNCHANGED)
# =========================
class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, freeze=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


# =========================
# TS ENCODER (UNCHANGED)
# =========================
class TS_Encoder(nn.Module):
    def __init__(self, ts_feat_dim, ts_embed_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ts_feat_dim, ts_embed_dim),
            nn.GELU(),
            nn.LayerNorm(ts_embed_dim),
        )
        self.out_dim = ts_embed_dim

    def forward(self, x):
        return self.proj(x)


# =========================
# POSITIONAL ENCODING (UNCHANGED)
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =========================
# CROSS-MODAL FUSION (UNCHANGED)
# =========================
class CrossModalFusion(nn.Module):
    def __init__(self, sky_dim, flow_dim, mask_dim, ts_dim, fused_dim):
        super().__init__()
        self.sky_proj = nn.Linear(sky_dim, ts_dim)
        self.flow_proj = nn.Linear(flow_dim, ts_dim)
        self.mask_proj = nn.Linear(mask_dim, ts_dim)

        self.attn_sky = nn.MultiheadAttention(ts_dim, 4, batch_first=True)
        self.attn_flow = nn.MultiheadAttention(ts_dim, 4, batch_first=True)
        self.attn_mask = nn.MultiheadAttention(ts_dim, 4, batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(ts_dim * 1, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim)
        )

    def forward(self, sky, flow, mask, ts):
        sky = self.sky_proj(sky)
        flow = self.flow_proj(flow)
        mask = self.mask_proj(mask)

        sky_attn, _ = self.attn_sky(ts, sky, sky)
        flow_attn, _ = self.attn_flow(ts, flow, flow)
        mask_attn, _ = self.attn_mask(ts, mask, mask)

        fused = torch.cat([ts], dim=-1)
        return self.proj(fused)


# =========================
# TEMPORAL TRANSFORMER (UNCHANGED)
# =========================
class TemporalTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3)

    def forward(self, x):
        return self.encoder(x)


# =========================
# MULTIMODAL FORECASTER (MINIMAL CHANGES)
# =========================
class MultimodalForecaster(nn.Module):
    def __init__(self, sky_encoder, flow_encoder, mask_encoder,
                 ts_feat_dim, ts_embed_dim=64, fused_dim=128,
                 horizon=25, target_dim=1):
        super().__init__()

        self.sky_encoder = sky_encoder
        self.flow_encoder = flow_encoder
        self.mask_encoder = mask_encoder
        self.mask_stem = ConvStem1to3()

        self.ts_encoder = TS_Encoder(ts_feat_dim, ts_embed_dim)

        # 🔴 CHANGE 1: TS dim doubles due to Δ features
        ts_fused_dim = ts_embed_dim * 2

        self.cross_fusion = CrossModalFusion(
            sky_dim=sky_encoder.out_dim,
            flow_dim=flow_encoder.out_dim,
            mask_dim=mask_encoder.out_dim,
            ts_dim=ts_fused_dim,
            fused_dim=fused_dim
        )

        # 🔴 CHANGE 2: local temporal conv
        self.temp_conv = TemporalConvBlock(fused_dim)

        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(fused_dim)

        # 🔴 CHANGE 3: attention pooling
        self.attn_pool = AttnPool(fused_dim)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, horizon * target_dim)
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, sky_imgs, flow_imgs, mask_imgs, ts):
        B, T_img, C, H, W = sky_imgs.shape

        sky = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)
        flow = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)

        mask_1ch = mask_imgs.view(B*T_img, 1, H, W)
        mask_rgb = self.mask_stem(mask_1ch)
        mean = torch.tensor([0.485, 0.456, 0.406], device=mask_rgb.device).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225], device=mask_rgb.device).view(1,3,1,1)
        mask_rgb = (mask_rgb - mean) / std
        mask = self.mask_encoder(mask_rgb).view(B, T_img, -1)

        ts = self.ts_encoder(ts)
        ts_delta = temporal_diff(ts)
        ts = torch.cat([ts, ts_delta], dim=-1)

        fused = self.cross_fusion(sky, flow, mask, ts)
        fused = fused + self.temp_conv(fused)

        fused = self.pos_enc(fused)
        fused = self.temporal_tf(fused)

        context = self.attn_pool(fused)
        out = self.head(context)
        return out.view(B, self.horizon, self.target_dim)


# =========================
# TEST
# =========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sky_enc = ImageEncoder("convnextv2_tiny", True)
    flow_enc = ImageEncoder("resnet18", True)
    mask_enc = ImageEncoder("resnet18", True)

    model = MultimodalForecaster(
        sky_enc, flow_enc, mask_enc,
        ts_feat_dim=5
    ).to(device)

    B, T_img, T_ts = 2, 5, 30
    sky = torch.randn(B, T_img, 3, 224, 224).to(device)
    flow = torch.randn(B, T_img, 3, 224, 224).to(device)
    mask = torch.randn(B, T_img, 1, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 5).to(device)

    y = model(sky, flow, mask, ts)
    print("Output shape:", y.shape)
