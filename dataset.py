import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class IrradianceForecastDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        split: str = "train",
        val_ratio: float = 0.25,
        img_seq_len: int = 5,
        ts_seq_len: int = 30,
        horizon: int = 25,
        feature_cols=None,
        target_cols=None,
        img_size: int = 224,
        time_col: str = "timestamp",
        normalization_stats: dict = None,
    ):
        df = pd.read_csv(csv_path)
        n = len(df)
        split_idx = int(n * (1 - val_ratio))

        if split == "train":
            self.df = df.iloc[:split_idx].reset_index(drop=True)
        elif split == "val":
            self.df = df.iloc[split_idx:].reset_index(drop=True)
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.split = split
        self.img_seq_len = img_seq_len
        self.ts_seq_len = ts_seq_len
        self.horizon = horizon
        self.img_size = img_size
        self.time_col = time_col

        self.feature_cols = feature_cols or ["ghi", "dni", "dhi", "temp", "pressure"]
        self.target_cols = target_cols or ["ghi"]

        self.sky_col = "image_path_sky"
        self.flow_col = "image_path_optical_flow"
        self.mask_col = "image_path_cloud_mask"   # <-- NEW

        self.max_lookback = max(img_seq_len, ts_seq_len)

        # Timestamp
        if self.time_col in self.df.columns:
            self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])

        # Normalization
        if split == "train":
            mean = self.df[self.feature_cols].mean()
            std = self.df[self.feature_cols].std()
            self.normalization_stats = {"mean": mean, "std": std}
        else:
            if normalization_stats is None:
                raise ValueError("Validation split requires normalization_stats")
            self.normalization_stats = normalization_stats
            mean = normalization_stats["mean"]
            std = normalization_stats["std"]

        self.df[self.feature_cols] = (self.df[self.feature_cols] - mean) / std

        # Transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Mask transform (no normalization)
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()   # gives 0 or 1
        ])

        print(f"\nDataset initialized ({split.upper()}): {len(self)} samples")
        print(f"Image seq length: {img_seq_len}, TS seq length: {ts_seq_len}, Horizon: {horizon}")

    def __len__(self):
        return len(self.df) - self.max_lookback - self.horizon

    def __getitem__(self, idx):
        img_window = self.df.iloc[idx + self.ts_seq_len - self.img_seq_len : idx + self.ts_seq_len]
        ts_window = self.df.iloc[idx : idx + self.ts_seq_len]
        target_window = self.df.iloc[idx + self.ts_seq_len : idx + self.ts_seq_len + self.horizon]

        # ---- Sky images ----
        sky_seq = torch.stack([
            self.img_transform(Image.open(p).convert("RGB"))
            for p in img_window[self.sky_col].values
        ])

        # ---- Optical flow images ----
        flow_seq = torch.stack([
            self.img_transform(Image.open(p).convert("RGB"))
            for p in img_window[self.flow_col].values
        ])

        # ---- Binary cloud masks ----
        mask_seq = torch.stack([
            self.mask_transform(Image.open(p).convert("L"))  # grayscale
            for p in img_window[self.mask_col].values
        ])

        # ---- Time series ----
        ts_seq = torch.tensor(ts_window[self.feature_cols].values, dtype=torch.float32)

        # ---- Target ----
        target_seq = torch.tensor(target_window[self.target_cols].values, dtype=torch.float32)

        # RETURN MASK AS WELL
        return sky_seq, flow_seq, mask_seq, ts_seq, target_seq
