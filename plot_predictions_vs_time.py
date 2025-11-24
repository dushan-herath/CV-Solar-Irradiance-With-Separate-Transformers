"""
Visualization script for multimodal solar irradiance forecasting.
Plots actual vs predicted irradiance (GHI, DNI, DHI) over time (samples)
for a selected forecast horizon, using denormalized units.
Each target is shown in a separate chart.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import IrradianceForecastDataset
from model import ImageEncoder, MultimodalForecaster


@torch.no_grad()
def get_predictions(model, loader, device, mean_targets, std_targets):
    """Run model inference and return denormalized predictions and targets."""
    model.eval()
    all_preds, all_targets = [], []

    for img_seq, ts_seq, targets, *_ in tqdm(loader, desc="Predicting", leave=False):
        img_seq, ts_seq = img_seq.to(device), ts_seq.to(device)
        preds = model(img_seq, ts_seq)

        if preds.shape[1] != targets.shape[1]:
            preds = preds[:, :targets.shape[1], :]

        all_preds.append(preds.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    preds_denorm = preds * std_targets + mean_targets
    targets_denorm = targets * std_targets + mean_targets

    return preds_denorm, targets_denorm


def plot_predictions_vs_time_separate(preds, targets, target_names, horizon=1, num_samples=500):
    """
    Plot actual vs predicted irradiance for each target (separate charts).
    horizon: which forecast horizon to visualize (1-based index)
    num_samples: number of samples to display
    """
    t_idx = horizon - 1  # adjust to 0-based
    num_samples = min(num_samples, preds.shape[0])

    for i, name in enumerate(target_names):
        plt.figure(figsize=(10, 5))
        plt.plot(targets[:num_samples, t_idx, i], label=f"Actual {name.upper()}", linewidth=2)
        plt.plot(preds[:num_samples, t_idx, i], linestyle="--", label=f"Predicted {name.upper()}", linewidth=2)
        plt.xlabel("Sample Index (Time)")
        plt.ylabel(f"{name.upper()} (W/m²)")
        plt.title(f"Actual vs Predicted {name.upper()} @ Horizon={horizon}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = f"pred_vs_actual_{name.lower()}_h{horizon}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"📈 Saved -> {save_path}")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CSV_PATH = "processed_dataset.csv"
    IMG_SEQ_LEN = 5
    TS_SEQ_LEN = 30
    MAX_HORIZON = 25
    TARGET_DIM = 3
    BATCH_SIZE = 8

    print(f"🔍 Generating prediction plots on {DEVICE} using best_model.pth")

    # --- Load normalization stats ---
    if not os.path.exists("norm_stats.json"):
        raise FileNotFoundError("❌ norm_stats.json not found.")
    full_norm_stats = json.load(open("norm_stats.json"))
    full_mean = pd.Series(full_norm_stats["mean"])
    full_std = pd.Series(full_norm_stats["std"])

    normalization_stats = {"mean": full_mean, "std": full_std}
    TARGET_NAMES = ["ghi", "dni", "dhi"]

    mean_targets = np.array([full_mean[n] for n in TARGET_NAMES]).reshape(1, 1, TARGET_DIM)
    std_targets = np.array([full_std[n] for n in TARGET_NAMES]).reshape(1, 1, TARGET_DIM)

    # --- Dataset setup ---
    val_ds = IrradianceForecastDataset(
        csv_path=CSV_PATH,
        split="val",
        img_seq_len=IMG_SEQ_LEN,
        ts_seq_len=TS_SEQ_LEN,
        horizon=MAX_HORIZON,
        normalization_stats=normalization_stats
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, pin_memory=True)
    print(f"🧾 Dataset initialized (VAL): {len(val_ds)} samples, horizon={MAX_HORIZON}")

    # --- Model setup ---
    img_encoder = ImageEncoder(model_name="vit_small_patch16_224", pretrained=False, freeze=True)
    model = MultimodalForecaster(
        img_encoder=img_encoder,
        ts_feat_dim=len(full_mean),
        horizon=MAX_HORIZON,
        target_dim=TARGET_DIM,
        d_model=256,
        num_layers=3
    ).to(DEVICE)

    if not os.path.exists("best_model.pth"):
        raise FileNotFoundError("❌ best_model.pth not found.")
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    print("✅ Loaded best_model.pth")

    # --- Run inference ---
    preds_denorm, targets_denorm = get_predictions(model, val_loader, DEVICE, mean_targets, std_targets)

    # --- Plot variations for selected horizons ---
    horizons_to_plot = [1, 5, 10, 25]  # adjust as desired
    for h in horizons_to_plot:
        plot_predictions_vs_time_separate(preds_denorm, targets_denorm, TARGET_NAMES, horizon=h, num_samples=400)

    print("\n✅ Individual prediction vs actual plots saved successfully!")
