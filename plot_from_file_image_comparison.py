import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# USER-DEFINED PARAMETERS
# ==========================================
SAMPLE_START = 300
SAMPLE_END = 2250
HORIZONS_TO_PLOT = [1, 5, 10, 25]
SAVE_DIR = "plots_new"

# ----------------------------------------------------
# Plot 1: ERROR vs HORIZON (multi-model, shared graph)
# ----------------------------------------------------
def plot_error_vs_horizon_multi(models_data, metric_key, metric_name, target_names, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    for target_idx, target_name in enumerate(target_names):
        plt.figure(figsize=(8, 5))

        for model_label, data in models_data.items():
            errors = data[metric_key]
            horizons = np.arange(1, errors.shape[0] + 1)

            plt.plot(
                horizons,
                errors[:, target_idx],
                marker='o',
                label=model_label
            )

        plt.xlabel("Forecast Horizon")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} vs Forecast Horizon — {target_name.upper()}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{metric_key}_{target_name.lower()}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved -> {save_path}")

# ----------------------------------------------------
# Plot 2: ACTUAL vs PREDICTED — separate per model
# ----------------------------------------------------
def plot_predictions_vs_time_separate(models_data, target_names,
                                      horizon=1,
                                      start_idx=SAMPLE_START,
                                      end_idx=SAMPLE_END,
                                      save_dir=SAVE_DIR):

    os.makedirs(save_dir, exist_ok=True)
    t_idx = horizon - 1

    # Consistent colors
    colors = {
        "Actual": "black",
        "No optical flow": "tab:blue",
        "Optical flow": "tab:red",
    }

    # Use any model for targets
    sample_model = next(iter(models_data.values()))
    targets = sample_model["targets"]

    # Clip slice bounds
    end_idx = end_idx or targets.shape[0]
    end_idx = min(end_idx, targets.shape[0])
    idx_range = slice(start_idx, end_idx)

    for target_idx, name in enumerate(target_names):

        for model_label, data in models_data.items():

            preds = data["preds"]

            plt.figure(figsize=(10, 5))

            # Actual
            plt.plot(
                targets[idx_range, t_idx, target_idx],
                label=f"Actual {name.upper()}",
                linewidth=2,
                color=colors["Actual"]
            )

            # This model’s prediction
            plt.plot(
                preds[idx_range, t_idx, target_idx],
                linestyle="--",
                linewidth=2,
                label=f"{model_label} (Pred)",
                color=colors.get(model_label, "tab:gray")
            )

            plt.xlabel("Sample Index (Time)")
            plt.ylabel(f"{name.upper()} (W/m²)")
            plt.title(f"{name.upper()} — Horizon={horizon} — {model_label}")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()

            clean_label = model_label.lower().replace(" ", "_")
            save_path = os.path.join(
                save_dir,
                f"pred_{name.lower()}_h{horizon}_{clean_label}.png"
            )
            plt.savefig(save_path, dpi=150)
            plt.close()

            print(f"Saved -> {save_path}")

# ----------------------------------------------------
# Main script — loads models, creates all plots
# ----------------------------------------------------
def main():
    model_files = {
        "No optical flow": "forecast_results_cropped.npz",
        "Optical flow": "forecast_results.npz",
    }

    models_data = {}

    print("Loading model outputs...\n")
    for label, path in model_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")

        print(f"Loading {label}: {path}")
        data = np.load(path, allow_pickle=True)

        models_data[label] = {
            "preds": data["preds"],
            "targets": data["targets"],
            "mse": data["mse"],
            "mae": data["mae"],
            "rmse": data["rmse"],
            "target_names": data["target_names"]
        }

    print("\nAll files loaded successfully!\n")

    target_names = models_data["No optical flow"]["target_names"]

    # Error vs horizon
    print("Generating multi-model error vs horizon plots...")
    plot_error_vs_horizon_multi(models_data, "mse", "MSE", target_names)
    plot_error_vs_horizon_multi(models_data, "mae", "MAE", target_names)
    plot_error_vs_horizon_multi(models_data, "rmse", "RMSE", target_names)

    # Prediction plots
    print("\nGenerating per-model prediction plots...")
    for h in HORIZONS_TO_PLOT:
        plot_predictions_vs_time_separate(
            models_data,
            target_names,
            horizon=h,
            start_idx=SAMPLE_START,
            end_idx=SAMPLE_END,
        )

    print("\nAll plots saved in 'plots/' directory.")

if __name__ == "__main__":
    main()
