import math
import torch
from torch import nn
import timm
import random


# =========================
# IMAGE ENCODER (Updated: Spatial Preservation)
# =========================
class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True,
                 freeze: bool = True, unfreeze_last: int = 0):
        super().__init__()

        # CRITICAL CHANGE 1: global_pool='' ensures we get (B, C, H, W) 
        # instead of a pooled vector (B, C).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='' 
        )

        # Automatically determine feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            self.feat_dim = feats.shape[1] # Channels (C)
            self.h = feats.shape[2]        # Height (H)
            self.w = feats.shape[3]        # Width (W)

        self.out_dim = self.feat_dim

        # CRITICAL CHANGE 2: Spatial Positional Embeddings
        # We need to tell the transformer which patch is "Top-Left" vs "Center"
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.feat_dim, self.h, self.w))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # Freeze backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last > 0:
                self._unfreeze_last_layers(unfreeze_last)

    def _unfreeze_last_layers(self, n: int):
        # (Your original unfreeze logic remains here - omitted for brevity)
        # Ensure you include your original implementation here if needed
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, H, W)
        Output: (B, Spatial_Sequence_Len, C) -> e.g., (B, 49, C) for ResNet18
        """
        x = self.backbone(x) # (B, C, H, W)
        
        # Add spatial position info
        x = x + self.spatial_pos_embed
        
        # Flatten spatial dims: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        return x


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
# MULTIMODAL FORECASTER (Updated Reshaping)
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

        # 1. ENCODE IMAGES (Spatial)
        # Input: (B * T_img, C, H, W)
        # Output: (B * T_img, Spatial_Patches, C)  <-- e.g. 49 patches
        sky_feats = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W))
        flow_feats = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W))
        
        mask_imgs_rgb = mask_imgs.repeat(1, 1, 3, 1, 1)
        mask_feats = self.mask_encoder(mask_imgs_rgb.view(B*T_img, 3, H, W))

        # 2. RESHAPE FOR ATTENTION
        # We merge Time and Space dimensions for the Key/Value sequence
        # Shape: (B, T_img * Spatial_Patches, C)
        sky_feats = sky_feats.view(B, -1, sky_feats.shape[-1])
        flow_feats = flow_feats.view(B, -1, flow_feats.shape[-1])
        mask_feats = mask_feats.view(B, -1, mask_feats.shape[-1])

        # 3. ENCODE TS (Query)
        ts_feats = self.ts_encoder(ts)

        # 4. CROSS FUSION
        # Query: TS (B, T_ts, D)
        # Key: All Image Patches (B, T_img * 49, D)
        fused_feats = self.cross_fusion(sky_feats, flow_feats, mask_feats, ts_feats)

        # 5. TEMPORAL TRANSFORMER
        cls = self.cls_token.repeat(B, 1, 1)
        fused_feats = torch.cat([cls, fused_feats], dim=1)
        fused_feats = self.pos_enc(fused_feats)
        temporal_out = self.temporal_tf(fused_feats)
        
        # 6. HEAD
        context = temporal_out[:, 0]
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
