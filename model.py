import torch
import torch.nn as nn
from torchvision.models.video import mc3_18  # Alternative video encoder


# --------------------------------------------------
# Video Encoder using TorchVision MC3_18
# --------------------------------------------------
class VideoResNetEncoder(nn.Module):
    def __init__(self, model_name="mc3_18", pretrained=True, freeze=True):
        super().__init__()
        if model_name == "mc3_18":
            self.backbone = mc3_18(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Remove classifier head
        self.backbone.fc = nn.Identity()
        self.out_dim = 512  # MC3_18 output feature dimension

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x: (B, T, C, H, W)
        # torchvision expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.backbone(x)  # (B, out_dim)


# --------------------------------------------------
# Time-Series Transformer Encoder
# --------------------------------------------------
class TSTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2, dropout=0.1):
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

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)
        x = self.encoder(x)  # (B, T, d_model)
        x = x.permute(0, 2, 1)  # (B, d_model, T)
        x = self.pool(x).squeeze(-1)
        return x  # (B, d_model)


# --------------------------------------------------
# Fusion + Regression Head
# --------------------------------------------------
class FusionRegressor(nn.Module):
    def __init__(self, img_dim, ts_dim, hidden=256, horizon=25, target_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(img_dim * 2 + ts_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.LayerNorm(hidden // 2),
            nn.Linear(hidden // 2, horizon * target_dim),
        )

        self.horizon = horizon
        self.target_dim = target_dim

    def forward(self, img1, img2, ts):
        fused = torch.cat([img1, img2, ts], dim=-1)
        out = self.fc(fused)
        return out.view(-1, self.horizon, self.target_dim)


# --------------------------------------------------
# Final Multimodal Forecasting Model
# --------------------------------------------------
class MultimodalForecaster(nn.Module):
    def __init__(self, ts_feat_dim, horizon=25, target_dim=1, freeze_img=True):
        super().__init__()

        # Replace VideoSwinEncoder with VideoResNetEncoder
        self.sky_encoder = VideoResNetEncoder(freeze=freeze_img)
        self.flow_encoder = VideoResNetEncoder(freeze=freeze_img)

        self.ts_encoder = TSTransformer(
            input_dim=ts_feat_dim, d_model=128, nhead=4, num_layers=2
        )

        self.regressor = FusionRegressor(
            img_dim=self.sky_encoder.out_dim,
            ts_dim=self.ts_encoder.out_dim,
            horizon=horizon,
            target_dim=target_dim,
        )

    def forward(self, sky_imgs, flow_imgs, ts):
        sky_feat = self.sky_encoder(sky_imgs)
        flow_feat = self.flow_encoder(flow_imgs)
        ts_feat = self.ts_encoder(ts)
        return self.regressor(sky_feat, flow_feat, ts_feat)
