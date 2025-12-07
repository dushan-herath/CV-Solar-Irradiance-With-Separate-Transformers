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

        # Create backbone with no head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        self.out_dim = self.backbone.num_features

        # Freeze backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

            if unfreeze_last > 0:
                self._unfreeze_last_layers(unfreeze_last)

    def _unfreeze_last_layers(self, n: int):
        """
        Supports: Swin, ResNet, ConvNeXt, EfficientNet, MobileNetV3
        Falls back to your existing warning.
        """
        backbone_type = self.backbone.__class__.__name__.lower()

        # ============================
        # SWIN TRANSFORMER
        # ============================
        if "swin" in backbone_type:
            layers = [self.backbone.layers[0], self.backbone.layers[1],
                      self.backbone.layers[2], self.backbone.layers[3]]
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        # ============================
        # RESNET FAMILY
        # ============================
        elif "resnet" in backbone_type:
            layers = [self.backbone.layer1, self.backbone.layer2,
                      self.backbone.layer3, self.backbone.layer4]
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        # ============================
        # CONVNEXT FAMILY
        # ============================
        elif "convnext" in backbone_type:
            stages = self.backbone.stages
            for stage in stages[-n:]:
                for p in stage.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        # ============================
        # EFFICIENTNET FAMILY
        # ============================
        elif "efficientnet" in backbone_type:
            blocks = list(self.backbone.blocks)
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True

            # Unfreeze final conv/bn if present
            if hasattr(self.backbone, "conv_head"):
                for p in self.backbone.conv_head.parameters():
                    p.requires_grad = True
            if hasattr(self.backbone, "bn2"):
                for p in self.backbone.bn2.parameters():
                    p.requires_grad = True

        # ============================
        # MOBILENETV3 FAMILY
        # ============================
        elif "mobilenetv3" in backbone_type:
            blocks = list(self.backbone.blocks)
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True

            if hasattr(self.backbone, "conv_stem"):
                for p in self.backbone.conv_stem.parameters():
                    p.requires_grad = True

        # ============================
        # FALLBACK
        # ============================
        else:
            print(f"Unfreeze last layers: please customize for backbone {backbone_type}")

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #    return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)  # (B, C, H, W) for CNNs
        if feat.dim() == 4:
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)  # (B, C, 1, 1)
            feat = feat.view(feat.size(0), -1)                        # (B, C)
        return feat


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
# SINUSOIDAL POSITIONAL ENCODING (Vaswani et al.)
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # Create a [max_len, d_model] matrix of positional encodings
        position = torch.arange(max_len).unsqueeze(1)               # (T, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )                                                           # (d_model/2)

        pe = torch.zeros(max_len, d_model)                          # (T, D)
        pe[:, 0::2] = torch.sin(position * div_term)                # even dims
        pe[:, 1::2] = torch.cos(position * div_term)                # odd dims

        # Register as buffer so it moves with model but is not trainable
        self.register_buffer("pe", pe.unsqueeze(0))                 # (1, T, D)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, D)
        returns x + positional encodings for first T positions
        """
        return x + self.pe[:, : x.size(1)]


# =========================
# CROSS-MODAL FUSION (now accepts mask branch)
# =========================
class CrossModalFusion(nn.Module):
    def __init__(self, sky_dim, flow_dim, mask_dim, ts_dim, fused_dim, n_heads=4, dropout=0.1):
        super().__init__()
        # Project image features to match TS dimension
        self.sky_proj = nn.Linear(sky_dim, ts_dim)
        self.flow_proj = nn.Linear(flow_dim, ts_dim)
        self.mask_proj = nn.Linear(mask_dim, ts_dim)

        self.attn_sky = nn.MultiheadAttention(embed_dim=ts_dim, num_heads=n_heads, batch_first=True)
        self.attn_flow = nn.MultiheadAttention(embed_dim=ts_dim, num_heads=n_heads, batch_first=True)
        self.attn_mask = nn.MultiheadAttention(embed_dim=ts_dim, num_heads=n_heads, batch_first=True)

        # Now concatenating ts + sky_attn + flow_attn + mask_attn => ts_dim * 4
        self.proj = nn.Sequential(
            nn.Linear(ts_dim * 4, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout)
        )

    def forward(self, sky_feats, flow_feats, mask_feats, ts_feats):
        """
        sky_feats:  (B, T_img, sky_dim)
        flow_feats: (B, T_img, flow_dim)
        mask_feats: (B, T_img, mask_dim)
        ts_feats:   (B, T_ts, ts_dim)   <-- may have different temporal length
        """
        sky_feats_proj = self.sky_proj(sky_feats)     # (B, T_img, ts_dim)
        flow_feats_proj = self.flow_proj(flow_feats)  # (B, T_img, ts_dim)
        mask_feats_proj = self.mask_proj(mask_feats)  # (B, T_img, ts_dim)

        # Cross-attend: query=ts_feats, key/value=image_feats_proj
        sky_attn, _ = self.attn_sky(query=ts_feats, key=sky_feats_proj, value=sky_feats_proj)
        flow_attn, _ = self.attn_flow(query=ts_feats, key=flow_feats_proj, value=flow_feats_proj)
        mask_attn, _ = self.attn_mask(query=ts_feats, key=mask_feats_proj, value=mask_feats_proj)

        fused = torch.cat([ts_feats, sky_attn, flow_attn, mask_attn], dim=-1)  # (B, T_ts, ts_dim*4)
        fused = self.proj(fused)  # (B, T_ts, fused_dim)
        return fused


# =========================
# TEMPORAL TRANSFORMER AFTER FUSION
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
# MULTIMODAL FORECASTER (with mask branch)
# =========================
class MultimodalForecaster(nn.Module):
    def __init__(self, sky_encoder, flow_encoder, mask_encoder, ts_feat_dim, ts_embed_dim=64,
                 fused_dim=128, horizon=25, target_dim=1):
        super().__init__()
        self.sky_encoder = sky_encoder
        self.flow_encoder = flow_encoder
        self.mask_encoder = mask_encoder
        self.ts_encoder = TS_Encoder(ts_feat_dim=ts_feat_dim, ts_embed_dim=ts_embed_dim)

        # Fusion & temporal modeling
        self.cross_fusion = CrossModalFusion(
            sky_dim=self.sky_encoder.out_dim,
            flow_dim=self.flow_encoder.out_dim,
            mask_dim=self.mask_encoder.out_dim,
            ts_dim=ts_embed_dim,
            fused_dim=fused_dim
        )

        # CLS token for temporal pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, fused_dim))

        self.pos_enc = PositionalEncoding(fused_dim)
        self.temporal_tf = TemporalTransformer(d_model=fused_dim)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.LayerNorm(fused_dim // 2),
            nn.Linear(fused_dim // 2, horizon * target_dim)
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, sky_imgs, flow_imgs, mask_imgs, ts):
        """
        sky_imgs, flow_imgs, mask_imgs: (B, T_img, C, H, W)
        ts: (B, T_ts, ts_feat_dim)
        """
        B, T_img, C, H, W = sky_imgs.shape

        # Encode images (flatten time -> batch)
        sky_feats = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)
        flow_feats = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)

        # Convert masks to RGB expected by encoder
        mask_imgs_rgb = mask_imgs.repeat(1, 1, 3, 1, 1)
        mask_feats = self.mask_encoder(mask_imgs_rgb.view(B*T_img, 3, H, W)).view(B, T_img, -1)

        # Encode TS (B, T_ts, ts_embed_dim)
        ts_feats = self.ts_encoder(ts)

        # Cross-modal fusion -> (B, T_ts, fused_dim)
        fused_feats = self.cross_fusion(sky_feats, flow_feats, mask_feats, ts_feats)

        # --------- ADD CLS TOKEN BEFORE POSITIONAL ENCODING ---------
        cls = self.cls_token.repeat(B, 1, 1)             # (B, 1, D)
        fused_feats = torch.cat([cls, fused_feats], dim=1)  # (B, 1 + T_ts, D)
        # ------------------------------------------------------------

        # Positional encoding + transformer
        fused_feats = self.pos_enc(fused_feats)
        temporal_out = self.temporal_tf(fused_feats)     # (B, 1 + T_ts, D)

        # --------- CLS POOLING ---------
        context = temporal_out[:, 0]  # take CLS token
        # --------------------------------

        # Regression head
        out = self.head(context)
        out = out.view(B, self.horizon, self.target_dim)
        return out


# =========================
# TEST SCRIPT
# =========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create encoders (minimal: freeze backbones)
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

    # Fake data
    B, T_img, T_ts = 2, 5, 30
    sky_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    flow_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    mask_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)  # cloud masks as 3-channel image inputs
    ts = torch.randn(B, T_ts, 5).to(device)

    preds = model(sky_imgs, flow_imgs, mask_imgs, ts)
    print("preds.shape:", preds.shape)  # [B, horizon, target_dim]
