"""
Phase 3: Feature Engineering

Input:  data/processed/mote_<id>_cleaned.csv (Phase 2 output)
Output: data/processed/mote_<id>_train.csv
        data/processed/mote_<id>_test.csv

Builds lag features, rolling-window statistics, and cyclical time features
to predict temperature HORIZON_MINUTES ahead.

Phase 2 dropped dropouts longer than 30 minutes rather than fabricating
them, so the cleaned series has a handful of real time discontinuities.
Computing lag/rolling features naively across those boundaries would leak
stale values from hours/days earlier into the first rows after each gap.
To prevent this, continuous segments are detected from elapsed time
between rows, and every lag/rolling/target feature is computed within
each segment independently. Segment warm-up/tail rows without full
history are then dropped.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROC_DIR = Path("data/processed")

RESAMPLE_FREQ_MINUTES = 1
HORIZON_MINUTES = 30           # how far ahead we predict temperature
LAGS = [1, 5, 15, 30]          # minutes
ROLLING_WINDOWS = [5, 15, 30]  # minutes
TEST_FRACTION = 0.2


def find_cleaned_file() -> Path:
    candidates = list(PROC_DIR.glob("mote_*_cleaned.csv"))
    if not candidates:
        raise FileNotFoundError("No mote_*_cleaned.csv found. Run Phase 2 first.")
    return candidates[0]


def add_segment_ids(df: pd.DataFrame) -> pd.DataFrame:
    """A new segment starts whenever the gap to the previous row is larger
    than the expected resample interval — i.e. wherever Phase 2 dropped a
    long dropout."""
    df = df.copy()
    expected = pd.Timedelta(minutes=RESAMPLE_FREQ_MINUTES)
    time_diff = df.index.to_series().diff()
    new_segment = (time_diff > expected) | time_diff.isna()
    df["segment_id"] = new_segment.cumsum()
    return df


def add_lag_features(df: pd.DataFrame, target_col="temperature") -> pd.DataFrame:
    df = df.copy()
    for lag in LAGS:
        df[f"{target_col}_lag_{lag}m"] = df.groupby("segment_id")[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, columns=("temperature", "humidity", "light", "voltage")) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        grouped = df.groupby("segment_id")[col]
        for w in ROLLING_WINDOWS:
            df[f"{col}_roll_mean_{w}m"] = grouped.transform(lambda s, w=w: s.rolling(w, min_periods=w).mean())
            df[f"{col}_roll_std_{w}m"] = grouped.transform(lambda s, w=w: s.rolling(w, min_periods=w).std())
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = df.index.hour + df.index.minute / 60
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def add_target(df: pd.DataFrame, target_col="temperature", horizon=HORIZON_MINUTES) -> pd.DataFrame:
    df = df.copy()
    df["target"] = df.groupby("segment_id")[target_col].shift(-horizon)
    return df


def main():
    cleaned_path = find_cleaned_file()
    mote_id = cleaned_path.stem.split("_")[1]
    print(f"Loading {cleaned_path}")
    df = pd.read_csv(cleaned_path, parse_dates=["timestamp"]).set_index("timestamp")
    print(f"Cleaned rows: {len(df):,}")

    df = add_segment_ids(df)
    n_segments = df["segment_id"].nunique()
    print(f"Detected {n_segments} continuous time segments (boundaries = dropped long gaps)")

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_time_features(df)
    df = add_target(df, horizon=HORIZON_MINUTES)

    before = len(df)
    df = df.dropna()
    print(
        f"Dropped {before - len(df):,} rows lacking full lag/rolling/target history "
        f"(segment warm-ups and tail ends) -> {len(df):,} usable rows"
    )

    split_idx = int(len(df) * (1 - TEST_FRACTION))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df):,} rows ({train_df.index.min()} -> {train_df.index.max()})")
    print(f"Test:  {len(test_df):,} rows ({test_df.index.min()} -> {test_df.index.max()})")

    train_path = PROC_DIR / f"mote_{mote_id}_train.csv"
    test_path = PROC_DIR / f"mote_{mote_id}_test.csv"
    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    print(f"Saved {train_path}")
    print(f"Saved {test_path}")

    feature_cols = [c for c in df.columns if c not in ("segment_id", "target")]
    print(f"\nTotal feature columns: {len(feature_cols)} (excludes segment_id and target)")


if __name__ == "__main__":
    main()