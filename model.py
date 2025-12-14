import math
import torch
from torch import nn
import timm


# =========================
# 1 → 3 CHANNEL STEM (MASK)
# =========================
class ConvStem1to3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)   # (B*T, 1, H, W)


# =========================
# IMAGE TEMPORAL LSTM (NEW)
# =========================
class ImageTemporalLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, num_layers=1, dropout=0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        """
        x: (B, T_img, D)
        """
        out, _ = self.lstm(x)
        return out


# =========================
# IMAGE ENCODER (UNCHANGED)
# =========================
class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True, freeze=True, unfreeze_last=0):
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

            if unfreeze_last > 0:
                self._unfreeze_last_layers(unfreeze_last)

    def _unfreeze_last_layers(self, n):
        backbone_type = self.backbone.__class__.__name__.lower()

        if "resnet" in backbone_type:
            layers = [self.backbone.layer1, self.backbone.layer2,
                      self.backbone.layer3, self.backbone.layer4]
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        elif "convnext" in backbone_type:
            stages = self.backbone.stages
            for stage in stages[-n:]:
                for p in stage.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# =========================
# TIME SERIES ENCODER
# =========================
class TS_Encoder(nn.Module):
    def __init__(self, ts_feat_dim, ts_embed_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(ts_feat_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, ts_embed_dim),
            nn.GELU(),
            nn.LayerNorm(ts_embed_dim),
            nn.Dropout(dropout),
        )
        self.out_dim = ts_embed_dim

    def forward(self, x):
        return self.proj(x)


# =========================
# POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =========================
# CROSS-MODAL FUSION
# =========================
class CrossModalFusion(nn.Module):
    def __init__(self, sky_dim, flow_dim, mask_dim, ts_dim, fused_dim, n_heads=4):
        super().__init__()
        self.sky_proj = nn.Linear(sky_dim, ts_dim)
        self.flow_proj = nn.Linear(flow_dim, ts_dim)
        self.mask_proj = nn.Linear(mask_dim, ts_dim)

        self.attn_sky = nn.MultiheadAttention(ts_dim, n_heads, batch_first=True)
        self.attn_flow = nn.MultiheadAttention(ts_dim, n_heads, batch_first=True)
        self.attn_mask = nn.MultiheadAttention(ts_dim, n_heads, batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(ts_dim * 2, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim)
        )

    def forward(self, sky_feats, flow_feats, mask_feats, ts_feats):
        sky = self.sky_proj(sky_feats)
        flow = self.flow_proj(flow_feats)
        mask = self.mask_proj(mask_feats)

        sky_attn, _ = self.attn_sky(ts_feats, sky, sky)
        flow_attn, _ = self.attn_flow(ts_feats, flow, flow)
        mask_attn, _ = self.attn_mask(ts_feats, mask, mask)

        fused = torch.cat([ts_feats, sky_attn], dim=-1)
        return self.proj(fused)


# =========================
# TEMPORAL TRANSFORMER
# =========================
class TemporalTransformer(nn.Module):
    def __init__(self, d_model, nhead=8, num_layers=3):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x):
        return self.encoder(x)


# =========================
# MULTIMODAL FORECASTER
# =========================
class MultimodalForecaster(nn.Module):
    def __init__(self, sky_encoder, flow_encoder, mask_encoder,
                 ts_feat_dim, ts_embed_dim=64,
                 fused_dim=128, horizon=25, target_dim=1):

        super().__init__()
        self.sky_encoder = sky_encoder
        self.flow_encoder = flow_encoder
        self.mask_encoder = mask_encoder
        self.mask_stem = ConvStem1to3()

        # 🔹 NEW: Image LSTMs
        self.sky_lstm = ImageTemporalLSTM(self.sky_encoder.out_dim)
        self.flow_lstm = ImageTemporalLSTM(self.flow_encoder.out_dim)
        self.mask_lstm = ImageTemporalLSTM(self.mask_encoder.out_dim)

        self.ts_encoder = TS_Encoder(ts_feat_dim, ts_embed_dim)

        self.cross_fusion = CrossModalFusion(
            self.sky_lstm.out_dim,
            self.flow_lstm.out_dim,
            self.mask_lstm.out_dim,
            ts_embed_dim,
            fused_dim
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, fused_dim))
        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(fused_dim)

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.LayerNorm(fused_dim // 2),
            nn.Linear(fused_dim // 2, horizon * target_dim)
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, sky_imgs, flow_imgs, mask_imgs, ts):
        B, T_img, C, H, W = sky_imgs.shape

        # SKY
        sky = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)
        sky = self.sky_lstm(sky)

        # FLOW
        flow = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)
        flow = self.flow_lstm(flow)

        # MASK
        mask = self.mask_stem(mask_imgs.view(B*T_img, 1, H, W))
        mask = self.mask_encoder(mask).view(B, T_img, -1)
        mask = self.mask_lstm(mask)

        # TS
        ts_feats = self.ts_encoder(ts)

        fused = self.cross_fusion(sky, flow, mask, ts_feats)

        cls = self.cls_token.repeat(B, 1, 1)
        fused = torch.cat([cls, fused], dim=1)

        fused = self.pos_enc(fused)
        out = self.temporal_tf(fused)

        context = out[:, 0]
        out = self.head(context)
        return out.view(B, self.horizon, self.target_dim)


# =========================
# TEST
# =========================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sky_enc = ImageEncoder("convnextv2_tiny", pretrained=True)
    flow_enc = ImageEncoder("resnet18", pretrained=True)
    mask_enc = ImageEncoder("resnet18", pretrained=True)

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
    print("Output:", y.shape)
