"""
Phase 1: Data Acquisition & Initial Exploration
Dataset: Intel Berkeley Research Lab Sensor Network Data
Source: https://db.csail.mit.edu/labdata/labdata.html

54 Mica2Dot wireless sensor motes deployed in a research lab, logging
temperature, humidity, light, and voltage every ~31 seconds over 36 days
(Feb 28 - Apr 5, 2004). The dataset has well-documented real-world IoT
issues: dropped readings, missing timestamps, and faulty spikes caused
by low-voltage sensor malfunction (temps registering >100C).

This script:
  1. Downloads the raw log + mote location file
  2. Parses the space-separated log (skipping malformed lines)
  3. Picks the mote with the most complete readings as our focus sensor
  4. Saves that mote's raw time series to data/processed/
  5. Plots a raw overview and prints a data-quality summary
"""

from pathlib import Path
import requests
import pandas as pd
import matplotlib.pyplot as plt

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
OUT_DIR = Path("outputs")
for d in (RAW_DIR, PROC_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://db.csail.mit.edu/labdata/data.txt.gz"
LOCS_URL = "https://db.csail.mit.edu/labdata/mote_locs.txt"

COLUMNS = ["date", "time", "epoch", "moteid", "temperature", "humidity", "light", "voltage"]


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest}")
        return
    print(f"Downloading {url} ...")
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def load_raw_data(path: Path) -> pd.DataFrame:
    """Parse the whitespace-separated sensor log, skipping malformed lines
    (a known quirk of this dataset — some rows have extra/missing fields)."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
        compression="gzip",
        on_bad_lines="skip",
        engine="python",
    )
    return df


def main():
    download_file(DATA_URL, RAW_DIR / "data.txt.gz")
    download_file(LOCS_URL, RAW_DIR / "mote_locs.txt")

    print("\nLoading raw sensor log (~2.3M rows, may take a minute)...")
    df = load_raw_data(RAW_DIR / "data.txt.gz")
    print(f"Loaded {len(df):,} raw readings across {df['moteid'].nunique()} motes")

    # Build a proper datetime column; drop rows where parsing failed
    df["timestamp"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce")
    df = df.dropna(subset=["timestamp", "moteid"])
    df["moteid"] = df["moteid"].astype(int)

    # Pick the mote with the most readings -> the most complete series to work with
    counts = df["moteid"].value_counts()
    best_mote = int(counts.index[0])
    print(f"\nSelected mote {best_mote} ({counts.iloc[0]:,} readings) as the focus sensor")

    mote_df = df[df["moteid"] == best_mote].sort_values("timestamp")
    mote_df = mote_df[["timestamp", "temperature", "humidity", "light", "voltage"]]
    mote_df = mote_df.drop_duplicates(subset="timestamp")

    out_path = PROC_DIR / f"mote_{best_mote}_raw.csv"
    mote_df.to_csv(out_path, index=False)
    print(f"Saved selected mote's raw series to {out_path}")

    # Quick overview plot
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for ax, col in zip(axes, ["temperature", "humidity", "light", "voltage"]):
        ax.plot(mote_df["timestamp"], mote_df[col], linewidth=0.5)
        ax.set_ylabel(col)
    axes[0].set_title(f"Mote {best_mote} — raw sensor readings (Intel Lab Data)")
    axes[-1].set_xlabel("Timestamp")
    plt.tight_layout()
    fig_path = OUT_DIR / "phase1_raw_overview.png"
    plt.savefig(fig_path, dpi=120)
    print(f"Saved overview plot to {fig_path}")

    # Data quality summary — confirms the anomalies Phase 2 will need to handle
    expected = pd.date_range(mote_df["timestamp"].min(), mote_df["timestamp"].max(), freq="31s")
    coverage = len(mote_df) / len(expected) * 100
    print("\n--- Data quality summary ---")
    print(f"Time range: {mote_df['timestamp'].min()} -> {mote_df['timestamp'].max()}")
    print(f"Expected readings at ~31s interval: {len(expected):,}")
    print(f"Actual readings: {len(mote_df):,} ({coverage:.1f}% coverage -> dropouts confirmed)")
    print(
        f"Temperature range: {mote_df['temperature'].min():.1f} to "
        f"{mote_df['temperature'].max():.1f} C (values far outside ~0-40C are sensor faults)"
    )
    print(f"Humidity range: {mote_df['humidity'].min():.1f} to {mote_df['humidity'].max():.1f} %")
    print(f"Voltage range: {mote_df['voltage'].min():.2f} to {mote_df['voltage'].max():.2f} V")


if __name__ == "__main__":
    main()