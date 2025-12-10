import math
import torch
from torch import nn
import timm
import random


# =========================
# IMAGE ENCODER (unchanged)
# =========================
class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True,
                 freeze: bool = True, unfreeze_last: int = 0):
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

    def _unfreeze_last_layers(self, n: int):
        backbone_type = self.backbone.__class__.__name__.lower()

        if "swin" in backbone_type:
            layers = [self.backbone.layers[0], self.backbone.layers[1],
                      self.backbone.layers[2], self.backbone.layers[3]]
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        elif "resnet" in backbone_type:
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

        elif "efficientnet" in backbone_type:
            blocks = list(self.backbone.blocks)
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "conv_head"):
                for p in self.backbone.conv_head.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "bn2"):
                for p in self.backbone.bn2.parameters():
                    p.requires_grad = True

        elif "mobilenetv3" in backbone_type:
            blocks = list(self.backbone.blocks)
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "conv_stem"):
                for p in self.backbone.conv_stem.parameters():
                    p.requires_grad = True

        else:
            print(f"Unfreeze last layers: please customize for backbone {backbone_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# =========================
# TIME SERIES ENCODER (unchanged)
# =========================
class TS_Encoder(nn.Module):
    def __init__(self, ts_feat_dim: int, ts_embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# =========================
# SINUSOIDAL POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1)]


# =========================
# CROSS-MODAL FUSION (unchanged)
# =========================
class CrossModalFusion(nn.Module):
    def __init__(self, sky_dim, flow_dim, mask_dim, ts_dim, fused_dim, n_heads=4, dropout=0.1):
        super().__init__()

        self.sky_proj = nn.Linear(sky_dim, ts_dim)
        self.flow_proj = nn.Linear(flow_dim, ts_dim)
        self.mask_proj = nn.Linear(mask_dim, ts_dim)

        self.attn_sky = nn.MultiheadAttention(embed_dim=ts_dim, num_heads=n_heads, batch_first=True)
        self.attn_flow = nn.MultiheadAttention(embed_dim=ts_dim, num_heads=n_heads, batch_first=True)
        self.attn_mask = nn.MultiheadAttention(embed_dim=ts_dim, num_heads=n_heads, batch_first=True)

        self.proj = nn.Sequential(
            nn.Linear(ts_dim * 4, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout)
        )

    def forward(self, sky_feats, flow_feats, mask_feats, ts_feats):

        sky_feats_proj = self.sky_proj(sky_feats)
        flow_feats_proj = self.flow_proj(flow_feats)
        mask_feats_proj = self.mask_proj(mask_feats)

        sky_attn, _ = self.attn_sky(ts_feats, sky_feats_proj, sky_feats_proj)
        flow_attn, _ = self.attn_flow(ts_feats, flow_feats_proj, flow_feats_proj)
        mask_attn, _ = self.attn_mask(ts_feats, mask_feats_proj, mask_feats_proj)

        fused = torch.cat([ts_feats, sky_attn, flow_attn, mask_attn], dim=-1)
        fused = self.proj(fused)

        return fused


# =========================
# TEMPORAL TRANSFORMER
# =========================
class TemporalTransformer(nn.Module):
    def __init__(self, d_model: int, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor):
        return self.transformer(x)


# =========================
# MULTIMODAL FORECASTER WITH RESIDUAL (OPTION 2)
# =========================
class MultimodalForecaster(nn.Module):
    def __init__(self, sky_encoder, flow_encoder, mask_encoder, ts_feat_dim, ts_embed_dim=64,
                 fused_dim=128, horizon=25, target_dim=1):
        super().__init__()
        self.sky_encoder = sky_encoder
        self.flow_encoder = flow_encoder
        self.mask_encoder = mask_encoder
        self.ts_encoder = TS_Encoder(ts_feat_dim=ts_feat_dim, ts_embed_dim=ts_embed_dim)

        self.cross_fusion = CrossModalFusion(
            sky_dim=self.sky_encoder.out_dim,
            flow_dim=self.flow_encoder.out_dim,
            mask_dim=self.mask_encoder.out_dim,
            ts_dim=ts_embed_dim,
            fused_dim=fused_dim
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, fused_dim))

        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(d_model=fused_dim)

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

        sky_feats = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)
        flow_feats = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)

        mask_imgs_rgb = mask_imgs.repeat(1, 1, 3, 1, 1)
        mask_feats = self.mask_encoder(mask_imgs_rgb.view(B*T_img, 3, H, W)).view(B, T_img, -1)

        ts_feats = self.ts_encoder(ts)

        fused_feats = self.cross_fusion(sky_feats, flow_feats, mask_feats, ts_feats)

        cls = self.cls_token.repeat(B, 1, 1)
        fused_feats = torch.cat([cls, fused_feats], dim=1)

        fused_feats = self.pos_enc(fused_feats)

        # 🔥🔥 OPTION 2 RESIDUAL CONNECTION HERE
        temporal_raw = self.temporal_tf(fused_feats)
        temporal_out = temporal_raw + fused_feats   # <--- RESIDUAL SKIP

        context = temporal_out[:, 0]

        out = self.head(context)
        out = out.view(B, self.horizon, self.target_dim)

        return out



# =========================
# TEST SCRIPT
# =========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sky_enc = ImageEncoder(model_name='convnextv2_tiny', pretrained=True, freeze=True)
    flow_enc = ImageEncoder(model_name='resnet18', pretrained=True, freeze=True)
    mask_enc = ImageEncoder(model_name='resnet18', pretrained=True, freeze=True)

    model = MultimodalForecaster(
        sky_encoder=sky_enc,
        flow_encoder=flow_enc,
        mask_encoder=mask_enc,
        ts_feat_dim=5,
        ts_embed_dim=64,
        fused_dim=128,
        horizon=25,
        target_dim=1
    ).to(device)

    B, T_img, T_ts = 2, 5, 30
    sky_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    flow_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    mask_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 5).to(device)

    preds = model(sky_imgs, flow_imgs, mask_imgs, ts)
    print("preds.shape:", preds.shape)
