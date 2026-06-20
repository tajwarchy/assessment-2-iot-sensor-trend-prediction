"""
Phase 4: Model Training & Time-Series Cross-Validation

Input:  data/processed/mote_<id>_train.csv (Phase 3 output)
Output: models/mote_<id>_model.joblib
        models/mote_<id>_feature_columns.joblib
        models/mote_<id>_metadata.joblib

MODELING DECISION: PREDICTING DELTA, NOT ABSOLUTE VALUE
---------------------------------------------------------
An initial version of this script trained the model to predict the future
absolute temperature directly. That underperformed a naive "no change"
persistence baseline (CV RMSE 1.047 vs baseline RMSE 0.808). Cause:
indoor lab temperature is highly autocorrelated over a 30-minute horizon,
so persistence is already a very strong predictor. Tree ensembles
approximate relationships with piecewise-constant leaf averages, which
tends to pull predictions toward the training mean -- a poor fit for a
near-identity relationship like short-horizon persistence.

Fix: reformulate the target as the DELTA (change from the current reading
to the future reading) instead of the absolute future value. This removes
the dominant, trivially-predictable persistence component from the
learning target, leaving only the smaller, genuinely uncertain residual
signal for the model to learn. At inference time:
    predicted_temperature = current_temperature + predicted_delta
This is a standard technique for highly autocorrelated time series.

Walk-forward TimeSeriesSplit cross-validation remains the core overfitting
guard: each fold trains only on an expanding window of the past and
validates on the next, unseen chunk -- never shuffled.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

PROC_DIR = Path("data/processed")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

N_SPLITS = 5
PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 8, "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": 12, "min_samples_leaf": 5},
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 2},
]
RANDOM_STATE = 42
CURRENT_VALUE_COL = "temperature"


def find_train_file() -> Path:
    candidates = list(PROC_DIR.glob("mote_*_train.csv"))
    if not candidates:
        raise FileNotFoundError("No mote_*_train.csv found. Run Phase 3 first.")
    return candidates[0]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def walk_forward_eval(params: dict, X: pd.DataFrame, y_abs: pd.Series, n_splits=N_SPLITS) -> dict:
    """Each fold trains on an expanding past window, validates on the next
    chunk. Model is trained on the DELTA target; predictions are
    reconstructed to absolute temperature for a fair comparison against
    the persistence baseline."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    y_delta = y_abs - X[CURRENT_VALUE_COL]

    rmses, maes = [], []
    baseline_rmses, baseline_maes = [], []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_delta_train = y_delta.iloc[train_idx]
        y_abs_val = y_abs.iloc[val_idx]

        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        model.fit(X_train, y_delta_train)
        delta_preds = model.predict(X_val)
        abs_preds = X_val[CURRENT_VALUE_COL].values + delta_preds

        rmses.append(rmse(y_abs_val, abs_preds))
        maes.append(mean_absolute_error(y_abs_val, abs_preds))

        # Persistence baseline: predict zero change
        baseline_preds = X_val[CURRENT_VALUE_COL].values
        baseline_rmses.append(rmse(y_abs_val, baseline_preds))
        baseline_maes.append(mean_absolute_error(y_abs_val, baseline_preds))

    return {
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "baseline_rmse": float(np.mean(baseline_rmses)),
        "baseline_mae": float(np.mean(baseline_maes)),
    }


def main():
    train_path = find_train_file()
    mote_id = train_path.stem.split("_")[1]
    print(f"Loading {train_path}")

    df = pd.read_csv(train_path, parse_dates=["timestamp"]).set_index("timestamp")
    feature_cols = [c for c in df.columns if c not in ("segment_id", "target")]
    X, y_abs = df[feature_cols], df["target"]
    print(f"Training rows: {len(df):,}, features: {len(feature_cols)}")
    print(f"Target reformulated as delta from '{CURRENT_VALUE_COL}' (see module docstring)\n")

    print(f"Running {N_SPLITS}-fold walk-forward cross-validation across {len(PARAM_GRID)} configs...")
    results = []
    for params in PARAM_GRID:
        scores = walk_forward_eval(params, X, y_abs)
        scores["params"] = params
        results.append(scores)
        print(
            f"  {params} -> RMSE={scores['rmse']:.3f}  MAE={scores['mae']:.3f}  "
            f"(baseline RMSE={scores['baseline_rmse']:.3f})"
        )

    best = min(results, key=lambda r: r["rmse"])
    print(f"\nBest config: {best['params']} (CV RMSE={best['rmse']:.3f}, CV MAE={best['mae']:.3f})")
    improvement = (1 - best["rmse"] / best["baseline_rmse"]) * 100
    print(f"Improvement over persistence baseline: {improvement:.1f}% lower RMSE")

    print("\nRefitting best model (on delta target) using the full training set...")
    y_delta_full = y_abs - X[CURRENT_VALUE_COL]
    final_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **best["params"])
    final_model.fit(X, y_delta_full)

    model_path = MODEL_DIR / f"mote_{mote_id}_model.joblib"
    cols_path = MODEL_DIR / f"mote_{mote_id}_feature_columns.joblib"
    meta_path = MODEL_DIR / f"mote_{mote_id}_metadata.joblib"
    joblib.dump(final_model, model_path)
    joblib.dump(feature_cols, cols_path)
    joblib.dump({"target_type": "delta", "current_value_col": CURRENT_VALUE_COL}, meta_path)
    print(f"Saved model to {model_path}")
    print(f"Saved feature column list to {cols_path}")
    print(f"Saved metadata (delta reconstruction info) to {meta_path}")


if __name__ == "__main__":
    main()