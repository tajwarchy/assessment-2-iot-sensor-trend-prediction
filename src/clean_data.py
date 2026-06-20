"""
Phase 2: Data Cleaning & Anomaly Handling

Pipeline (see message accompanying this script for full justification):
  1. Domain-bound filtering   -> removes physically impossible values
  2. Statistical spike check  -> rolling-MAD outlier detection on temperature
  3. Resample to regular grid -> exposes missing timestamps explicitly
  4. Bounded interpolation    -> fills short dropouts (<=30 min), drops long ones
  5. Light smoothing          -> rolling median removes residual hardware jitter

Input:  data/processed/mote_<id>_raw.csv   (from Phase 1)
Output: data/processed/mote_<id>_cleaned.csv
        outputs/phase2_cleaning_comparison.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Physically plausible ranges for an indoor lab deployment
DOMAIN_BOUNDS = {
    "temperature": (-5, 45),    # Celsius; known fault spikes reach 100+
    "humidity": (0, 100),       # relative humidity %, can't be negative
    "light": (0, 2000),         # lux; indoor lab, generous upper margin
    "voltage": (1.8, 3.3),      # lithium-ion cell range with safety margin
}

RESAMPLE_FREQ = "1min"
MAX_GAP_MINUTES = 30            # longer dropouts are left as NaN, then dropped
SMOOTH_WINDOW = 5               # minutes, for rolling-median smoothing
MAD_WINDOW = 61                 # ~61 minutes, for spike detection
MAD_THRESHOLD = 6.0             # modified z-score threshold


def find_raw_file() -> Path:
    candidates = list(PROC_DIR.glob("mote_*_raw.csv"))
    if not candidates:
        raise FileNotFoundError("No mote_*_raw.csv found in data/processed/. Run Phase 1 first.")
    return candidates[0]


def apply_domain_bounds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    report = {}
    for col, (lo, hi) in DOMAIN_BOUNDS.items():
        bad = ~df[col].between(lo, hi)
        report[col] = int(bad.sum())
        df.loc[bad, col] = np.nan
    print("Domain-bound violations removed:", report)
    return df


def detect_statistical_spikes(series: pd.Series, window=MAD_WINDOW, threshold=MAD_THRESHOLD) -> pd.Series:
    """Rolling-median + MAD based outlier mask (robust to existing outliers,
    unlike mean/std). Flags points far from their local neighborhood even if
    they're within the domain bounds."""
    rolling_median = series.rolling(window, center=True, min_periods=window // 3).median()
    abs_dev = (series - rolling_median).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=window // 3).median()
    mad_scaled = (mad * 1.4826).replace(0, np.nan)  # 1.4826 ~ consistency constant for normal dist.
    modified_z = abs_dev / mad_scaled
    return modified_z > threshold


def main():
    raw_path = find_raw_file()
    mote_id = raw_path.stem.split("_")[1]
    print(f"Loading {raw_path}")

    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp")
    original_count = len(df)
    print(f"Starting rows: {original_count:,}")

    # --- Layer 1: domain bounds ---
    df = apply_domain_bounds(df)

    # --- Layer 2: statistical spike detection (temperature) ---
    spike_mask = detect_statistical_spikes(df["temperature"])
    print(f"Statistical spikes flagged (temperature): {int(spike_mask.sum())}")
    df.loc[spike_mask, "temperature"] = np.nan

    # --- Layer 3: resample to a regular grid ---
    df = df.set_index("timestamp")
    resampled = df.resample(RESAMPLE_FREQ).mean(numeric_only=True)
    missing_bins = resampled["temperature"].isna().sum()
    print(f"Regular {RESAMPLE_FREQ} grid: {len(resampled):,} bins, {missing_bins:,} empty after cleaning")

    # --- Layer 4: bounded interpolation ---
    max_gap_periods = MAX_GAP_MINUTES  # 1 row per minute at this frequency
    interpolated = resampled.interpolate(method="linear", limit=max_gap_periods, limit_direction="both")
    remaining_na = interpolated["temperature"].isna().sum()
    print(f"Remaining gaps after bounded interpolation (dropped, >{MAX_GAP_MINUTES} min): {remaining_na:,}")
    cleaned = interpolated.dropna(subset=["temperature", "humidity", "light", "voltage"])

    # --- Layer 5: light smoothing ---
    smoothed = cleaned.copy()
    for col in ["temperature", "humidity", "light", "voltage"]:
        smoothed[col] = cleaned[col].rolling(SMOOTH_WINDOW, center=True, min_periods=1).median()

    final_count = len(smoothed)
    print(f"\nFinal cleaned rows: {final_count:,} ({final_count / original_count * 100:.1f}% of original raw rows)")

    out_path = PROC_DIR / f"mote_{mote_id}_cleaned.csv"
    smoothed.to_csv(out_path)
    print(f"Saved cleaned dataset to {out_path}")

    # --- Comparison plot: raw vs cleaned temperature ---
    raw_for_plot = pd.read_csv(raw_path, parse_dates=["timestamp"]).set_index("timestamp")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(raw_for_plot.index, raw_for_plot["temperature"], color="lightcoral", linewidth=0.5, label="Raw")
    ax.plot(smoothed.index, smoothed["temperature"], color="steelblue", linewidth=0.8, label="Cleaned")
    ax.set_title(f"Mote {mote_id} — Temperature: Raw vs Cleaned")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Temperature (C)")
    ax.legend()
    plt.tight_layout()
    fig_path = OUT_DIR / "phase2_cleaning_comparison.png"
    plt.savefig(fig_path, dpi=120)
    print(f"Saved comparison plot to {fig_path}")


if __name__ == "__main__":
    main()