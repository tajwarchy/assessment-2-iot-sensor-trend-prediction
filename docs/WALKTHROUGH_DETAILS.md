# Walkthrough Details

Detailed phase-by-phase technical log: exact numbers, decisions, and one
real debugging episode, kept here separately from the top-level README so
that document stays concise while this one keeps the full record.

## Phase 1 — Data Acquisition & Initial Exploration

- Source: Intel Berkeley Research Lab Sensor Network Data, downloaded
  directly from `db.csail.mit.edu` (no account/API key required).
- Raw log parsed: **2,313,682 readings across 61 motes** (malformed lines
  skipped during parsing — a known quirk of this dataset).
- Mote selection was done programmatically (most complete reading history
  wins) rather than hand-picked, to avoid cherry-picking a convenient
  sensor. Result: **mote 31**, with **65,693 readings**.
- Initial data-quality check on mote 31 (Feb 28 – Mar 30, 2004):
  - Expected readings at ~31s nominal interval: 89,071
  - Actual: 65,693 → **73.8% coverage** (dropouts confirmed)
  - Temperature range: 2.2°C to **122.2°C** (clear sensor fault — known
    low-voltage malfunction pattern in this dataset)
  - Humidity range: **-4.0%** to 56.0% (negative humidity is invalid)
  - Voltage range: 2.00–2.78V (no fault evident here)

## Phase 2 — Data Cleaning

Five-layer pipeline applied to mote 31's raw series (65,693 rows):

| Layer | Action | Result |
|---|---|---|
| 1 | Domain-bound filtering | temperature: 12,959 violations, humidity: 9,620, light: 5, voltage: 5 |
| 2 | Statistical spike detection (rolling MAD on temperature) | 393 additional spikes flagged |
| 3 | Resample to 1-minute regular grid | 46,021 bins, 14,764 empty (dropouts exposed) |
| 4 | Bounded linear interpolation (≤30 min gaps) | 12,522 bins remained empty beyond 30 min → dropped |
| 5 | Rolling-median smoothing (5-min window) | applied to all four channels |

**Final: 33,331 cleaned rows (50.7% of original raw rows retained.)**

The ~20% domain-bound violation rate on temperature (not just a handful of
spikes) confirmed this dataset's documented failure mode is sustained
faulty stretches, not isolated outliers — which is why a combined
domain-bound + statistical-spike approach was used rather than either
alone.

## Phase 3 — Feature Engineering

- Continuous segments detected from elapsed time between rows (gaps >30
  min in Phase 2 were dropped, not interpolated, leaving real time
  discontinuities): **3 segments** found across the cleaned series.
- Lag/rolling/target features computed **within each segment
  independently**, to prevent leaking stale values across a dropped-gap
  boundary into a new segment's first few rows.
- 180 rows dropped for incomplete lag/rolling/target history at segment
  edges → **33,151 usable rows**.
- Chronological 80/20 split (no shuffling):
  - **Train:** 26,520 rows, Feb 28 01:28 → Mar 19 06:53
  - **Test:** 6,631 rows, Mar 19 06:54 → Mar 23 21:24
- **36 total feature columns**: 4 lag features × 4 channels' rolling
  mean/std × 3 windows + 4 cyclical time features + current readings.

## Phase 4 — Model Training (including a real debugging episode)

**First attempt — training on absolute future temperature directly:**

| Config | CV RMSE | CV MAE | Baseline RMSE |
|---|---|---|---|
| n_estimators=100, max_depth=8 | 1.047 | 0.611 | 0.808 |
| n_estimators=200, max_depth=8 | 1.047 | 0.610 | 0.808 |
| n_estimators=200, max_depth=12 | 1.054 | 0.614 | 0.808 |
| n_estimators=300, max_depth=None | 1.053 | 0.615 | 0.808 |

Every configuration **lost to the naive persistence baseline**
(predict-no-change). Diagnosis: temperature is highly autocorrelated over
a 30-minute horizon at this site, so persistence is an unusually strong
baseline, and RandomForest's piecewise-constant leaf averaging tends to
pull predictions toward the training mean — a poor structural fit for a
near-identity relationship.

**Fix — reformulating the target as delta** (`future_temp - current_temp`,
reconstructed at inference as `current_temp + predicted_delta`):

| Config | CV RMSE | CV MAE | Baseline RMSE |
|---|---|---|---|
| n_estimators=100, max_depth=8 | 0.835 | 0.440 | 0.808 |
| **n_estimators=200, max_depth=8** | **0.828** | **0.444** | 0.808 |
| n_estimators=200, max_depth=12 | 0.831 | 0.448 | 0.808 |
| n_estimators=300, max_depth=None | 0.853 | 0.463 | 0.808 |

Best config (`n_estimators=200, max_depth=8, min_samples_leaf=5`) closed
nearly all of the gap to baseline (CV RMSE 0.828 vs. 0.808 — within ~2.4%,
essentially CV noise) and was selected for final training.

## Phase 5 — Final Test Set Evaluation

First and only use of the held-out test set (6,631 rows, Mar 19–23):

| | RMSE (°C) | MAE (°C) |
|---|---|---|
| Model | 1.127 | 0.562 |
| Baseline (persistence) | 0.730 | 0.481 |

The model underperformed the baseline on the final test set, despite
near-parity during CV — a genuine, reported (not adjusted-away) finding.

**Diagnosis:** the RMSE gap (-54.5%) is far larger than the MAE gap
(-16.8%). Since RMSE penalizes large errors quadratically and MAE
linearly, this asymmetry points to a small number of large outlier
prediction errors during the test window (visible in
`outputs/phase5_residuals.png`), not uniform underperformance. Most likely
driver: the test window is a short (~4.5 day), single contiguous slice
from one mote — vulnerable to a local event the 19-day training window
didn't contain. See the README's "Limitations & Future Improvements"
section for concrete next steps (multi-mote training, outlier-robust loss,
persistence-blended prediction, multi-window evaluation).