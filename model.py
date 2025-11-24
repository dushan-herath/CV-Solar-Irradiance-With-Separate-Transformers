# model.py
import torch
import torch.nn as nn
from typing import Optional

# Try importing modern weights enum where available (torchvision >= 0.13).
# If it's not available, fall back to the old `pretrained=True` style.
try:
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    _HAS_WEIGHTS_ENUM = True
except Exception:
    from torchvision.models.video import r2plus1d_18
    _HAS_WEIGHTS_ENUM = False


# --------------------------------------------------
# Video Encoder using TorchVision R(2+1)D-18
# --------------------------------------------------
class VideoR2Plus1DEncoder(nn.Module):
    """
    Wrapper around torchvision's r2plus1d_18 video model.
    Input expected: (B, T, C, H, W)
    Torchvision expects: (B, C, T, H, W) -> we permute accordingly.
    """
    def __init__(self, pretrained: bool = True, freeze: bool = True, unfreeze_last: int = 0):
        """
        Args:
            pretrained: load Kinetics-pretrained weights when available
            freeze: set all backbone params requires_grad=False
            unfreeze_last: number of last ResNet stages to unfreeze (0..4)
        """
        super().__init__()

        if _HAS_WEIGHTS_ENUM and pretrained:
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            self.backbone = r2plus1d_18(weights=weights)
        else:
            # older torchvision accepts pretrained bool
            self.backbone = r2plus1d_18(pretrained=pretrained)

        # remove classification head, keep feature extractor
        # r2plus1d_18 has `fc` classifier similar to ResNet
        self.backbone.fc = nn.Identity()
        self.out_dim = 512  # r2plus1d_18 final feature dim

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if unfreeze_last > 0:
            # ResNet-like stages are layer1..layer4
            stages = [self.backbone.layer1, self.backbone.layer2,
                      self.backbone.layer3, self.backbone.layer4]
            n = min(unfreeze_last, len(stages))
            for stage in stages[-n:]:
                for p in stage.parameters():
                    p.requires_grad = True

            # also unfreeze any final norm / head if present (defensive)
            if hasattr(self.backbone, "bn2"):
                for p in self.backbone.bn2.parameters():
                    p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            features: (B, out_dim)
        """
        # permute to (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        feats = self.backbone(x)  # (B, out_dim) because fc is Identity
        return feats


# --------------------------------------------------
# Time-Series Transformer Encoder
# --------------------------------------------------
class TSTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)           # (B, T, d_model)
        x = self.encoder(x)              # (B, T, d_model)
        x = x.permute(0, 2, 1)           # (B, d_model, T)
        x = self.pool(x).squeeze(-1)     # (B, d_model)
        return x


# --------------------------------------------------
# Fusion + Regression Head
# --------------------------------------------------
class FusionRegressor(nn.Module):
    def __init__(self, img_dim: int, ts_dim: int, hidden: int = 256,
                 horizon: int = 25, target_dim: int = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim * 2 + ts_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, horizon * target_dim),
        )
        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, img1: torch.Tensor, img2: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([img1, img2, ts], dim=-1)
        out = self.fc(fused)
        return out.view(-1, self.horizon, self.target_dim)


# --------------------------------------------------
# Final Multimodal Forecasting Model
# --------------------------------------------------
class MultimodalForecaster(nn.Module):
    def __init__(self, ts_feat_dim: int, horizon: int = 25, target_dim: int = 1,
                 freeze_img: bool = True, r2_pretrained: bool = True, unfreeze_last: int = 0):
        """
        Args:
            ts_feat_dim: number of time-series features per timestep
            horizon: forecast horizon
            target_dim: number of target variables
            freeze_img: freeze R(2+1)D backbones
            r2_pretrained: use Kinetics pretrained weights for r2plus1d_18
            unfreeze_last: unfreeze last N ResNet stages after freezing
        """
        super().__init__()

        # Sky and Flow video encoders (same architecture)
        self.sky_encoder = VideoR2Plus1DEncoder(pretrained=r2_pretrained, freeze=freeze_img, unfreeze_last=unfreeze_last)
        self.flow_encoder = VideoR2Plus1DEncoder(pretrained=r2_pretrained, freeze=freeze_img, unfreeze_last=unfreeze_last)

        # TS encoder
        self.ts_encoder = TSTransformer(input_dim=ts_feat_dim, d_model=128, nhead=4, num_layers=2)

        # regressor
        self.regressor = FusionRegressor(
            img_dim=self.sky_encoder.out_dim,
            ts_dim=self.ts_encoder.out_dim,
            horizon=horizon,
            target_dim=target_dim,
        )

    def forward(self, sky_imgs: torch.Tensor, flow_imgs: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """
        sky_imgs: (B, T_img, C, H, W)
        flow_imgs: (B, T_img, C, H, W)
        ts: (B, T_ts, F)
        returns: (B, horizon, target_dim)
        """
        sky_feat = self.sky_encoder(sky_imgs)   # (B, 512)
        flow_feat = self.flow_encoder(flow_imgs) # (B, 512)
        ts_feat = self.ts_encoder(ts)           # (B, d_model)

        return self.regressor(sky_feat, flow_feat, ts_feat)
