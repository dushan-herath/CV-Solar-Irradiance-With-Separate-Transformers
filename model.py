import math
import torch
from torch import nn
import torch.nn.functional as F
import timm

# =========================
# SMALL UTILS
# =========================
def temporal_diff(x):
    dx = x[:, 1:] - x[:, :-1]
    dx = torch.cat([torch.zeros_like(dx[:, :1]), dx], dim=1)
    return dx


def causal_mask(T, device):
    return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()


# =========================
# TEMPORAL CONV
# =========================
class TemporalConvBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


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
# IMAGE ENCODER
# =========================
class ImageEncoder(nn.Module):
    def __init__(self, model_name, pretrained=True, freeze=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool="avg"
        )
        self.out_dim = self.backbone.num_features

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


# =========================
# TS ENCODER
# =========================
class TS_Encoder(nn.Module):
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =========================
# CROSS-MODAL FUSION (IMAGES → TS)
# =========================
class CrossModalFusion(nn.Module):
    def __init__(self, img_dim, ts_dim, out_dim):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, ts_dim)

        self.attn = nn.MultiheadAttention(
            ts_dim, num_heads=4, batch_first=True
        )

        self.proj = nn.Sequential(
            nn.Linear(ts_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, ts, img):
        img = self.img_proj(img)
        attn, _ = self.attn(ts, img, img)
        return self.proj(torch.cat([ts, attn], dim=-1))


# =========================
# TEMPORAL TRANSFORMER (CAUSAL)
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
        mask = causal_mask(x.size(1), x.device)
        return self.encoder(x, mask)


# =========================
# MULTIMODAL FORECASTER
# =========================
class MultimodalForecaster(nn.Module):
    def __init__(
        self,
        sky_encoder,
        flow_encoder,
        mask_encoder,
        ts_feat_dim,
        ts_embed_dim=64,
        fused_dim=128,
        horizon=25,
        target_dim=1
    ):
        super().__init__()

        self.sky_encoder = sky_encoder
        self.flow_encoder = flow_encoder
        self.mask_encoder = mask_encoder
        self.mask_stem = ConvStem1to3()

        # TS + ΔTS
        self.ts_encoder = TS_Encoder(ts_feat_dim * 2, ts_embed_dim)

        img_dim = (
            sky_encoder.out_dim +
            flow_encoder.out_dim +
            mask_encoder.out_dim
        )

        self.cross_fusion = CrossModalFusion(
            img_dim=img_dim,
            ts_dim=ts_embed_dim,
            out_dim=fused_dim
        )

        self.temp_conv = TemporalConvBlock(fused_dim)
        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(fused_dim)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Linear(fused_dim // 2, horizon * target_dim)
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, sky, flow, mask, ts):
        """
        sky, flow, mask: (B, T_img, C, H, W)
        ts:              (B, T_ts, F), T_ts > T_img
        """

        B, T_img = sky.shape[:2]
        T_ts = ts.size(1)

        # =====================
        # Encode images
        # =====================
        def enc_img(enc, x):
            B, T, C, H, W = x.shape
            return enc(x.view(B*T, C, H, W)).view(B, T, -1)

        sky = enc_img(self.sky_encoder, sky)
        flow = enc_img(self.flow_encoder, flow)

        mask = mask.view(B*T_img, 1, mask.size(-2), mask.size(-1))
        mask = self.mask_stem(mask)
        mask = self.mask_encoder(mask).view(B, T_img, -1)

        img = torch.cat([sky, flow, mask], dim=-1)

        # =====================
        # TS encoding (long)
        # =====================
        ts_delta = temporal_diff(ts)
        ts = self.ts_encoder(torch.cat([ts, ts_delta], dim=-1))

        # =====================
        # Align images to last TS steps
        # =====================
        ts_img = ts[:, -T_img:]

        fused = self.cross_fusion(ts_img, img)
        fused = fused + self.temp_conv(fused)

        fused = self.pos_enc(fused)
        fused = self.temporal_tf(fused)

        # =====================
        # Causal forecast
        # =====================
        context = fused[:, -1]
        out = self.head(context)

        return out.view(B, self.horizon, self.target_dim)


# =========================
# TEST
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
