import math
import torch
from torch import nn
import timm
import random

# =========================
# 1. IMAGE ENCODER (Spatial Features + Positional Embeddings)
# =========================
class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True,
                 freeze: bool = True, unfreeze_last: int = 0):
        super().__init__()

        # CRITICAL: global_pool='' keeps the spatial grid (H, W)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='' 
        )

        # Automatically determine feature dimensions (C, H, W)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feats = self.backbone(dummy)
            self.feat_dim = feats.shape[1] # Channels (C)
            self.h = feats.shape[2]        # Height (H)
            self.w = feats.shape[3]        # Width (W)

        self.out_dim = self.feat_dim

        # CRITICAL: Spatial Positional Embeddings
        # Learnable vector for each position in the 7x7 grid
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.feat_dim, self.h, self.w))
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

        # Freeze backbone
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last > 0:
                self._unfreeze_last_layers(unfreeze_last)

    def _unfreeze_last_layers(self, n: int):
        # Basic unfreeze logic (simplified for brevity)
        # You can paste your specific backbone-dependent logic here if needed
        params = list(self.backbone.parameters())
        for p in params[-n:]:
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, 3, H, W)
        Output: (B, Spatial_Sequence_Len, C) -> e.g., (B, 49, C)
        """
        x = self.backbone(x) # (B, C, H, W)
        
        # Add spatial position info (broadcasts over batch)
        x = x + self.spatial_pos_embed
        
        # Flatten spatial dims: (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        return x


# =========================
# 2. TIME SERIES ENCODER
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
# 3. POSITIONAL ENCODING
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        return x + self.pe[:, : x.size(1)]


# =========================
# 4. CROSS-MODAL FUSION
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

        # Output projection
        self.proj = nn.Sequential(
            nn.Linear(ts_dim * 4, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout)
        )

    def forward(self, sky_feats, flow_feats, mask_feats, ts_feats):
        # Project all visual features to TS dimension
        sky_feats_proj = self.sky_proj(sky_feats)   
        flow_feats_proj = self.flow_proj(flow_feats) 
        mask_feats_proj = self.mask_proj(mask_feats) 

        # Cross-Attend: TS (Query) looks at ALL spatial patches (Key/Value)
        sky_attn, _ = self.attn_sky(query=ts_feats, key=sky_feats_proj, value=sky_feats_proj)
        flow_attn, _ = self.attn_flow(query=ts_feats, key=flow_feats_proj, value=flow_feats_proj)
        mask_attn, _ = self.attn_mask(query=ts_feats, key=mask_feats_proj, value=mask_feats_proj)

        fused = torch.cat([ts_feats, sky_attn, flow_attn, mask_attn], dim=-1)
        fused = self.proj(fused)
        return fused



# =========================
# 5. TEMPORAL TRANSFORMER
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
# 6. MULTIMODAL FORECASTER (Main Model)
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
        # Inputs: (B, T_img, C, H, W)
        # We flatten batch and time to feed into CNN: (B*T_img, C, H, W)
        # Output from encoder: (B*T_img, Spatial_Patches, C)  <-- e.g. 49 patches per image
        sky_feats = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W))
        flow_feats = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W))
        
        mask_imgs_rgb = mask_imgs.repeat(1, 1, 3, 1, 1)
        mask_feats = self.mask_encoder(mask_imgs_rgb.view(B*T_img, 3, H, W))

        # 2. RESHAPE FOR ATTENTION (FIXED)
        # We need to merge Time and Space dimensions for the Key/Value sequence
        # Target Shape: (B, T_img * Spatial_Patches, C)
        # We use .reshape() because the encoder output might be non-contiguous due to transpose
        sky_feats = sky_feats.reshape(B, -1, sky_feats.shape[-1])
        flow_feats = flow_feats.reshape(B, -1, flow_feats.shape[-1])
        mask_feats = mask_feats.reshape(B, -1, mask_feats.shape[-1])

        # 3. ENCODE TS (Query)
        # Shape: (B, T_ts, D)
        ts_feats = self.ts_encoder(ts)

        # 4. CROSS FUSION
        # Query: TS (B, T_ts, D)
        # Key: All Image Patches from all timesteps (B, Total_Patches, D)
        fused_feats = self.cross_fusion(sky_feats, flow_feats, mask_feats, ts_feats)

        # 5. TEMPORAL TRANSFORMER
        # Add CLS token
        cls = self.cls_token.repeat(B, 1, 1)
        fused_feats = torch.cat([cls, fused_feats], dim=1)
        
        # Positional Encoding + Transformer
        fused_feats = self.pos_enc(fused_feats)
        temporal_out = self.temporal_tf(fused_feats)
        
        # 6. HEAD (Predict from CLS token)
        context = temporal_out[:, 0]
        out = self.head(context)
        out = out.view(B, self.horizon, self.target_dim)
        return out


# =========================
# TEST SCRIPT
# =========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create encoders 
    # global_pool='' is now hardcoded inside ImageEncoder, so we don't need to pass it
    sky_enc = ImageEncoder(model_name='convnextv2_tiny', pretrained=True, freeze=True)
    flow_enc = ImageEncoder(model_name='resnet18', pretrained=True, freeze=True)
    mask_enc = ImageEncoder(model_name='resnet18', pretrained=True, freeze=True)

    print(f"Sky Encoder Output (H, W): ({sky_enc.h}, {sky_enc.w})")
    print(f"Total Spatial Patches per Image: {sky_enc.h * sky_enc.w}")

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

    # Fake data for testing
    B, T_img, T_ts = 2, 5, 30
    sky_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    flow_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    mask_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 5).to(device)

    print("Running forward pass...")
    try:
        preds = model(sky_imgs, flow_imgs, mask_imgs, ts)
        print("Success!")
        print("preds.shape:", preds.shape)  # Expected: [B, horizon, target_dim]
    except Exception as e:
        print(f"Error: {e}")