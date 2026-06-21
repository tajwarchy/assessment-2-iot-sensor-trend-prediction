# IoT Sensor Data Trend Prediction

End-to-end machine learning pipeline for forecasting future readings from a
real, noisy IoT sensor network — from raw data ingest through cleaning,
feature engineering, model training, and evaluation.

## 1. Industrial Context & Objective

**Dataset:** [Intel Berkeley Research Lab Sensor Network Data](https://db.csail.mit.edu/labdata/labdata.html)
54 Mica2Dot wireless sensor motes were deployed throughout a research lab
to continuously monitor environmental conditions — temperature, humidity,
light, and battery voltage — every ~31 seconds over 36 days (Feb 28 – Apr 5,
2004). This is representative of real industrial/field IoT deployments
(smart buildings, precision agriculture, environmental/aquaculture
monitoring networks), and includes the genuine noise those systems produce:
dropped transmissions, missing timestamps, and faulty spikes from
low-voltage sensor malfunction — not synthetically injected.

**Target variable:** temperature, predicted 30 minutes ahead, for the mote
with the most complete reading history in the network (mote 31, chosen
programmatically rather than hand-picked).

## 2. Project Structure

```
.
├── data/
│   ├── raw/                  # downloaded dataset (gitignored)
│   └── processed/            # cleaned + feature-engineered CSVs
├── models/                   # trained model, feature list, metadata
├── outputs/                  # plots and metrics
├── src/
│   ├── acquire_and_explore.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

## 3. Pipeline

| Phase | Script | Purpose |
|---|---|---|
| 1 | `acquire_and_explore.py` | Download dataset, select most complete mote, initial exploration |
| 2 | `clean_data.py` | Five-layer anomaly cleaning |
| 3 | `feature_engineering.py` | Lag/rolling/time features, chronological train/test split |
| 4 | `train_model.py` | Walk-forward CV model selection and training |
| 5 | `evaluate.py` | Final held-out test set evaluation and visualization |

Run in order: `python src/acquire_and_explore.py` → `clean_data.py` →
`feature_engineering.py` → `train_model.py` → `evaluate.py`.

## 4. Data Cleaning Methodology

Raw data had **73.8% timestamp coverage** (26.2% dropouts), **12,959
temperature readings** and **9,620 humidity readings** outside physically
plausible bounds (faulty spikes up to 122.2°C, humidity down to -4%), and
**393 additional statistical outliers** caught by a secondary check. Five
layers were applied, in order:

1. **Domain-bound filtering** — flags physically impossible values (temp
   outside −5 to 45°C, humidity outside 0–100%, etc.), targeting the known
   low-voltage sensor fault mode directly.
2. **Statistical spike detection** — rolling-median + MAD (Median Absolute
   Deviation) outlier detection on temperature; robust to existing
   outliers, catches moderate noise spikes that stay within domain bounds.
3. **Resampling to a regular 1-minute grid** — exposes missing timestamps
   explicitly as NaN.
4. **Bounded linear interpolation** — fills dropouts up to 30 minutes;
   longer outages are dropped rather than fabricated, preserving the
   integrity of genuine system downtime.
5. **Rolling-median smoothing** (5-minute window) — removes residual
   hardware jitter while preserving the underlying trend.

Result: **33,331 clean rows** retained (50.7% of raw readings) across
**3 continuous time segments**.

## 5. Feature Engineering

Phase 2 deliberately drops gaps longer than 30 minutes rather than
fabricating them, which leaves real time discontinuities in the cleaned
series. Naively computing lag/rolling features across those boundaries
would leak stale values from hours or days earlier into the first rows of
a new segment. To prevent this, continuous segments are detected from
elapsed time between rows, and every feature below is computed **within
each segment independently**:

- **Lag features** at 1, 5, 15, 30 minutes — captures immediate momentum.
- **Rolling mean/std** at 5, 15, 30-minute windows, on temperature,
  humidity, light, and voltage (voltage is documented to correlate with
  temperature in this dataset).
- **Cyclical time features** — hour-of-day and day-of-week via sin/cos
  encoding, capturing HVAC/occupancy cycles without an artificial
  discontinuity at midnight.
- **Prediction horizon:** 30 minutes ahead.
- **Chronological 80/20 train/test split** (no shuffling) — train: Feb 28
  – Mar 19, test: Mar 19 – Mar 23. 36 total feature columns, 26,520
  training rows, 6,631 test rows.

## 6. Model Architecture & Overfitting Guards

**Model:** RandomForestRegressor, hyperparameters selected via 5-fold
**walk-forward `TimeSeriesSplit`** cross-validation — each fold trains
only on an expanding window of the past and validates on the next,
unseen chunk, never shuffled. Tree complexity is bounded
(`max_depth`, `min_samples_leaf`) to limit memorization of training noise.

**Key modeling decision — predicting delta, not absolute value:** an
initial version trained the model on the raw future temperature directly,
which underperformed a naive "no change" persistence baseline. Indoor lab
temperature is highly autocorrelated over a 30-minute horizon, and tree
ensembles approximate via piecewise-constant leaf averages, which pulls
predictions toward the training mean — a poor fit for a near-identity
relationship. Reformulating the target as the **delta** (change from
current to future reading) removed the dominant, trivially-predictable
persistence component, letting the model focus capacity on the smaller,
genuinely uncertain residual. Final predictions are reconstructed as
`current_temperature + predicted_delta`.

A **persistence baseline** (assume no change) is scored on identical CV
folds and the test set throughout, for a fair, consistent comparison.

## 7. Results

**Cross-validation (training period, best config:
`n_estimators=200, max_depth=8, min_samples_leaf=5`):**

| | RMSE (°C) | MAE (°C) |
|---|---|---|
| Model | 0.828 | 0.444 |
| Baseline (persistence) | 0.808 | — |

**Final held-out test set (Mar 19 – Mar 23, never touched until evaluation):**

| | RMSE (°C) | MAE (°C) |
|---|---|---|
| Model | 1.127 | 0.562 |
| Baseline (persistence) | 0.730 | 0.481 |

See `outputs/phase5_actual_vs_predicted.png`, `outputs/phase5_residuals.png`,
and `outputs/phase5_feature_importance.png`.

### Discussion: CV vs. test set discrepancy

The model performed roughly on par with the baseline during cross-validation
but underperformed it on the final test set, with a much larger relative
gap in RMSE (-54.5%) than in MAE (-16.8%). Because RMSE penalizes large
errors quadratically while MAE treats them linearly, this asymmetry points
to a small number of large outlier prediction errors during the test
window — visible in `phase5_residuals.png` — rather than the model being
uniformly worse across the board. A likely contributor is the short test
window (~4.5 days, a single contiguous slice of one mote's history),
which leaves the evaluation vulnerable to a local event the training
period didn't contain. This is being reported directly rather than
adjusted away, since it's a genuine and instructive limitation of
evaluating a single-mote, single-window pipeline under time pressure.

## 8. Limitations & Future Improvements

- **Single mote, single train/test window:** results may not generalize
  across the sensor network. Training on multiple motes jointly (with
  mote ID as a feature, or one model per mote evaluated on multiple test
  windows) would give a more reliable performance estimate.
- **Outlier sensitivity:** the test-set RMSE gap suggests a few large
  errors dominate the score. A quantile/Huber loss objective, or
  clipping predicted deltas to a plausible physical range, would likely
  reduce sensitivity to those outliers.
- **Persistence-blended prediction:** given how strong the baseline is at
  this horizon, a weighted blend of the model's prediction and the
  persistence baseline (rather than the model alone) could be a robust,
  low-risk improvement to try next.
- **Longer/rolling evaluation:** evaluating across multiple held-out
  windows (rather than one) would better separate genuine model skill
  from window-specific noise.

## 9. Setup & How to Run

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python src/acquire_and_explore.py
python src/clean_data.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate.py
```

## 10. Demo Video

https://www.youtube.com/watch?v=zplfL_sEVbE