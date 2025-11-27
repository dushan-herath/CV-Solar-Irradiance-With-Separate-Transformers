import math
import torch
from torch import nn
import timm
import random


# =========================================
# IMAGE ENCODER
# =========================================
class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = 'vit_base_patch16_224', pretrained: bool = True,
                 freeze: bool = True, unfreeze_last: int = 0):
        """
        Args:
            model_name: timm model name
            pretrained: load pretrained weights
            freeze: freeze backbone parameters
            unfreeze_last: number of last layers/stages to unfreeze
        """
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
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
            for p in self.backbone.norm.parameters():
                p.requires_grad = True

        elif "resnet" in backbone_type:
            layers = [self.backbone.layer1, self.backbone.layer2,
                    self.backbone.layer3, self.backbone.layer4]
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad = True

        elif "convnext" in backbone_type:  
            # ConvNeXtV1/V2 have 4 stages
            stages = self.backbone.stages
            for stage in stages[-n:]:
                for p in stage.parameters():
                    p.requires_grad = True

            # Also unfreeze final norm/head if present
            if hasattr(self.backbone, "norm"):
                for p in self.backbone.norm.parameters():
                    p.requires_grad = True

        else:
            print(f"Unfreeze last layers: please customize for backbone {backbone_type}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)



# =========================================
# TIME SERIES ENCODER
# =========================================
class TS_Encoder(nn.Module):
    def __init__(self, ts_feat_dim: int, ts_embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        """
        Args:
            ts_feat_dim: number of input TS features per time step
            ts_embed_dim: output embedding dimension
            hidden_dim: hidden dimension of intermediate layer
            dropout: dropout rate
        """
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
        """
        Args:
            x: (B, T, F) - batch, time steps, features
        Returns:
            (B, T, ts_embed_dim)
        """
        return self.proj(x)


# =========================================
# LEARNABLE POSITIONAL EMBEDDING (replaces old PositionalEncoding)
# =========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # Learnable parameter instead of sinusoidal constant buffer
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)

    def forward(self, x: torch.Tensor):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pos_emb[:, :T, :]



# =========================================
# FUSION MODULE
# =========================================
""""
class GatedFusion(nn.Module):
    def __init__(self, img_dim, ts_dim, fused_dim, dropout: float = 0.1):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, fused_dim),
            nn.LayerNorm(fused_dim)
        )

        self.ts_proj = nn.Sequential(
            nn.Linear(ts_dim, fused_dim),
            nn.LayerNorm(fused_dim)
        )

        self.gate = nn.Sequential(
            nn.Linear(fused_dim*2, fused_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_feats, ts_feats):
        # Align lengths (use last T_img TS steps)
        ts_last = ts_feats[:, -img_feats.shape[1]:, :]
        
        img_proj = self.img_proj(img_feats)
        ts_proj = self.ts_proj(ts_last)

         # -------------------------------
        # L2 magnitude matching (new)
        # -------------------------------
        #norm_img = img_proj.norm(dim=-1, keepdim=True)
        #norm_ts  = ts_proj.norm(dim=-1, keepdim=True)
        #scale = (norm_ts + 1e-6) / (norm_img + 1e-6)
        #img_proj = img_proj * scale.detach()  

        if random.random() < 0.005:
            print(f"img_feats norm: {img_proj.norm(dim=-1).mean().item():.3f}, "
              f"ts_feats norm: {ts_proj.norm(dim=-1).mean().item():.3f}")
            
        gate = self.gate(torch.cat([img_proj, ts_proj], dim=-1))

        fused = gate * img_proj + (1 - gate) * ts_proj
        return self.dropout(fused)

"""

class GatedFusion(nn.Module):
    def __init__(self, img_dim, ts_dim, fused_dim, dropout=0.1):
        super().__init__()
        
        self.img_proj = nn.Linear(img_dim, fused_dim)
        self.ts_proj  = nn.Linear(ts_dim, fused_dim)
        
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim*2, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout)
        )

    def forward(self, img, ts):
        # Align lengths: take last T_img TS steps
        ts_last = ts[:, -img.shape[1]:, :]  # ts_last: [B, T_img, ts_dim]
        img_f = self.img_proj(img)
        ts_f  = self.ts_proj(ts_last)

        fused = self.fuse(torch.cat([img_f, ts_f], dim=-1))
        return fused


# =========================================
# TEMPORAL TRANSFORMER
# =========================================
class FusionTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 3,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        return x


# =========================================
# MULTIMODAL FORECASTER WITH BRANCH TRANSFORMERS
# =========================================
class MultimodalForecasterWithBranchTransformers(nn.Module):
    def __init__(
        self,
        sky_encoder: ImageEncoder,
        flow_encoder: ImageEncoder,
        ts_feat_dim: int,
        img_embed_dim: int = None,
        ts_embed_dim: int = 128,
        fused_dim: int = 256,
        branch_d_model: int = 128,
        branch_num_layers: int = 2,
        branch_nhead: int = 4,
        branch_ff_dim: int = 256,
        fusion_dropout: float = 0.2,
        horizon: int = 25,
        target_dim: int = 3
    ):
        super().__init__()
        self.sky_encoder = sky_encoder
        self.flow_encoder = flow_encoder
        self.ts_encoder = TS_Encoder(ts_feat_dim=ts_feat_dim, ts_embed_dim=ts_embed_dim)

        # Positional encodings
        self.sky_pos_enc = PositionalEncoding(self.sky_encoder.out_dim)
        self.flow_pos_enc = PositionalEncoding(self.flow_encoder.out_dim)
        self.ts_pos_enc = PositionalEncoding(ts_embed_dim)

        # Branch transformers
        self.sky_transformer = FusionTransformer(
            input_dim=self.sky_encoder.out_dim,
            d_model=branch_d_model,
            nhead=branch_nhead,
            num_layers=branch_num_layers,
            dim_feedforward=branch_ff_dim,
            dropout=fusion_dropout
        )
        self.flow_transformer = FusionTransformer(
            input_dim=self.flow_encoder.out_dim,
            d_model=branch_d_model,
            nhead=branch_nhead,
            num_layers=branch_num_layers,
            dim_feedforward=branch_ff_dim,
            dropout=fusion_dropout
        )
        self.ts_transformer = FusionTransformer(
            input_dim=ts_embed_dim,
            d_model=branch_d_model,
            nhead=branch_nhead,
            num_layers=branch_num_layers,
            dim_feedforward=branch_ff_dim,
            dropout=fusion_dropout
        )

        # Fusion module
        self.fusion = GatedFusion(img_dim=branch_d_model*2, ts_dim=branch_d_model, fused_dim=fused_dim, dropout=fusion_dropout)

        self.horizon = horizon
        self.target_dim = target_dim

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(fusion_dropout),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.LayerNorm(fused_dim // 2),
            nn.Dropout(fusion_dropout),
            nn.Linear(fused_dim // 2, horizon * target_dim)
        )

    def forward(self, sky_imgs, flow_imgs, ts):
        B, T_img, C, H, W = sky_imgs.shape

        # Encode images
        sky_feats = self.sky_encoder(sky_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)
        sky_feats = self.sky_pos_enc(sky_feats)
        sky_feats = self.sky_transformer(sky_feats)

        flow_feats = self.flow_encoder(flow_imgs.view(B*T_img, C, H, W)).view(B, T_img, -1)
        flow_feats = self.flow_pos_enc(flow_feats)
        flow_feats = self.flow_transformer(flow_feats)

        # Encode time-series
        ts_feats = self.ts_encoder(ts)
        ts_feats = self.ts_pos_enc(ts_feats)
        ts_feats = self.ts_transformer(ts_feats)

        # Fuse modalities
        fused_feats = self.fusion(torch.cat([sky_feats, flow_feats], dim=-1), ts_feats)

        # Skip temporal transformer and attention pooling
        context = fused_feats.mean(dim=1)  # simple average over time dimension

        # Regression
        out = self.head(context)
        out = out.view(B, self.horizon, self.target_dim)
        return out



# =========================================
# TEST SCRIPT FOR BRANCH-LEVEL TRANSFORMERS
# =========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Image encoders using ConvNeXtV2-Tiny and ResNet18
    sky_enc = ImageEncoder(model_name='convnextv2_tiny', pretrained=True, freeze=True)
    flow_enc = ImageEncoder(model_name='resnet18', pretrained=True, freeze=True)

    # Model
    model = MultimodalForecasterWithBranchTransformers(
        sky_encoder=sky_enc,
        flow_encoder=flow_enc,
        ts_feat_dim=5,           # number of TS features
        ts_embed_dim=64,         # TS embedding dim
        fused_dim=128,           # fused feature dim
        branch_d_model=128,      # transformer hidden dim per branch
        branch_num_layers=2,     # transformer layers per branch
        branch_nhead=4,          # transformer heads
        branch_ff_dim=256,       # feedforward dim in transformers
        fusion_dropout=0.2,
        horizon=25,
        target_dim=1
    ).to(device)

    # Dummy input
    B, T_img, T_ts = 2, 5, 30
    sky_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    flow_imgs = torch.randn(B, T_img, 3, 224, 224).to(device)
    ts = torch.randn(B, T_ts, 5).to(device)

    # Forward pass
    preds = model(sky_imgs, flow_imgs, ts)
    print("preds.shape:", preds.shape)  # Expected: [B, horizon, target_dim]
