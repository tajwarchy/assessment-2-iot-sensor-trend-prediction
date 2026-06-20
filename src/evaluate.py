"""
Phase 5: Final Evaluation & Visualization

Input:  data/processed/mote_<id>_test.csv (Phase 3 output, untouched until now)
        models/mote_<id>_model.joblib
        models/mote_<id>_feature_columns.joblib
        models/mote_<id>_metadata.joblib
Output: outputs/phase5_actual_vs_predicted.png
        outputs/phase5_residuals.png
        outputs/phase5_feature_importance.png
        outputs/phase5_metrics.json

This is the first and only time the held-out test set is used. Final
absolute-temperature predictions are reconstructed as:
    predicted = current_temperature + predicted_delta
matching the delta-target reformulation from Phase 4.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_files():
    test_files = list(PROC_DIR.glob("mote_*_test.csv"))
    if not test_files:
        raise FileNotFoundError("No mote_*_test.csv found. Run Phase 3 first.")
    test_path = test_files[0]
    mote_id = test_path.stem.split("_")[1]

    model_path = MODEL_DIR / f"mote_{mote_id}_model.joblib"
    cols_path = MODEL_DIR / f"mote_{mote_id}_feature_columns.joblib"
    meta_path = MODEL_DIR / f"mote_{mote_id}_metadata.joblib"
    for p in (model_path, cols_path, meta_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run Phase 4 first.")
    return test_path, model_path, cols_path, meta_path, mote_id


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    test_path, model_path, cols_path, meta_path, mote_id = find_files()
    print(f"Loading test set: {test_path}")
    print(f"Loading model: {model_path}")

    df = pd.read_csv(test_path, parse_dates=["timestamp"]).set_index("timestamp")
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    meta = joblib.load(meta_path)
    current_col = meta["current_value_col"]

    X_test = df[feature_cols]
    y_test = df["target"]

    delta_preds = model.predict(X_test)
    abs_preds = X_test[current_col].values + delta_preds
    baseline_preds = X_test[current_col].values

    model_rmse = rmse(y_test, abs_preds)
    model_mae = mean_absolute_error(y_test, abs_preds)
    baseline_rmse = rmse(y_test, baseline_preds)
    baseline_mae = mean_absolute_error(y_test, baseline_preds)

    print("\n--- Final Test Set Evaluation (held out, never seen until now) ---")
    print(f"Model:    RMSE={model_rmse:.3f} C   MAE={model_mae:.3f} C")
    print(f"Baseline: RMSE={baseline_rmse:.3f} C   MAE={baseline_mae:.3f} C")
    rmse_improvement = (1 - model_rmse / baseline_rmse) * 100
    mae_improvement = (1 - model_mae / baseline_mae) * 100
    print(f"Improvement vs. persistence baseline: {rmse_improvement:+.1f}% RMSE, {mae_improvement:+.1f}% MAE")

    metrics = {
        "mote_id": mote_id,
        "test_rows": len(df),
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_mae": baseline_mae,
        "rmse_improvement_pct": rmse_improvement,
        "mae_improvement_pct": mae_improvement,
    }
    metrics_path = OUT_DIR / "phase5_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # --- Plot 1: Actual vs predicted trend ---
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df.index, y_test, label="Actual", color="steelblue", linewidth=1)
    ax.plot(df.index, abs_preds, label="Predicted (model)", color="darkorange", linewidth=1)
    ax.plot(
        df.index, baseline_preds, label="Baseline (persistence)",
        color="gray", linewidth=0.7, linestyle="--", alpha=0.6,
    )
    ax.set_title(f"Mote {mote_id} — Actual vs Predicted Temperature (30-min horizon), Test Set")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase5_actual_vs_predicted.png", dpi=120)
    print(f"Saved {OUT_DIR / 'phase5_actual_vs_predicted.png'}")
    plt.close(fig)

    # --- Plot 2: Residuals over time ---
    residuals = y_test.values - abs_preds
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, residuals, color="firebrick", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Mote {mote_id} — Model Residuals (Actual - Predicted)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Residual (C)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase5_residuals.png", dpi=120)
    print(f"Saved {OUT_DIR / 'phase5_residuals.png'}")
    plt.close(fig)

    # --- Plot 3: Feature importance ---
    importances = (
        pd.Series(model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    importances.iloc[::-1].plot(kind="barh", ax=ax, color="teal")
    ax.set_title(f"Mote {mote_id} — Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "phase5_feature_importance.png", dpi=120)
    print(f"Saved {OUT_DIR / 'phase5_feature_importance.png'}")
    plt.close(fig)


if __name__ == "__main__":
    main()