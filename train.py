import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from dataset import IrradianceForecastDataset
from model import MultimodalForecaster  # <- uses R(2+1)D-18

# --- Training / Validation Functions ---
def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for sky_seq, flow_seq, ts_seq, targets in tqdm(loader, leave=False):
        sky_seq, flow_seq, ts_seq, targets = map(lambda x: x.to(device, non_blocking=True), (sky_seq, flow_seq, ts_seq, targets))
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            preds = model(sky_seq, flow_seq, ts_seq)
            loss = criterion(preds, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sky_seq, flow_seq, ts_seq, targets in tqdm(loader, leave=False):
            sky_seq, flow_seq, ts_seq, targets = map(lambda x: x.to(device, non_blocking=True), (sky_seq, flow_seq, ts_seq, targets))
            preds = model(sky_seq, flow_seq, ts_seq)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


def plot_losses(train_losses, val_losses, save_path="training_curve.png"):
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curve: {save_path}")


# --- Main ---
if __name__ == "__main__":
    CSV_PATH = "processed_dataset_cropped_full.csv"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 2
    NUM_EPOCHS = 25
    IMG_SEQ_LEN = 16
    TS_SEQ_LEN = 30
    HORIZON = 25
    TARGET_DIM = 1

    # --- Dataset ---
    train_ds = IrradianceForecastDataset(CSV_PATH, split="train", img_seq_len=IMG_SEQ_LEN, ts_seq_len=TS_SEQ_LEN, horizon=HORIZON)
    val_ds = IrradianceForecastDataset(CSV_PATH, split="val", img_seq_len=IMG_SEQ_LEN, ts_seq_len=TS_SEQ_LEN, horizon=HORIZON, normalization_stats=train_ds.normalization_stats)

    # Save normalization stats
    with open("norm_stats.json", "w") as f:
        json.dump({"mean": train_ds.normalization_stats["mean"].to_dict(),
                   "std": train_ds.normalization_stats["std"].to_dict()}, f, indent=4)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- Model ---
    model = MultimodalForecaster(ts_feat_dim=len(train_ds.feature_cols), horizon=HORIZON, target_dim=TARGET_DIM, freeze_img=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(DEVICE.type=="cuda"))

    # --- Training Loop ---
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model updated!")

    plot_losses(train_losses, val_losses)
    print(f"\nTraining complete! Best val loss: {best_val_loss:.5f}")
