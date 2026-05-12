# ChaiIntel — Model Selection & Validation

_Generated artefact summarising the methodology used by the ChaiIntel forecasting engine. Suitable for inclusion in a project report._

---

## 1. Problem

Forecast monthly auction prices (¢/kg) for five Kenyan tea grades — **BP1, PF1, DUST1, FNGS 1/2, DUST 1/2** — using historical data from the Mombasa Tea Auction. The system must produce a 12-month price forecast for each grade and quantify forecast uncertainty.

## 2. Candidate models

Four models were considered per grade:

| Model | Role | Rationale |
|---|---|---|
| **Naïve** (last-value carry-forward) | Reference baseline | Any production model must beat this to be useful. Excluded from selection by design. |
| **Linear Regression** | Interpretable benchmark | Engineered features: time index, month, quarter, sin/cos month, lags 1–3, rolling mean. |
| **SARIMAX(1,1,1)** | Classical time-series | Industry standard for univariate price series; differencing handles non-stationarity. |
| **Random Forest** | Ensemble ML | Captures non-linear interactions between lag and seasonal features without manual specification. |

## 3. Model selection methodology

The single best model is chosen using a **two-stage rank-based aggregation**:

1. For each tea grade, run **3-fold walk-forward time-series cross-validation** (`sklearn.model_selection.TimeSeriesSplit`) on the three eligible models (Naïve is excluded — it is a reference baseline only).
2. Compute **MAPE** on each CV fold and rank the three models for that grade (1 = best, 3 = worst).
3. For each candidate model, compute the **mean rank across all five grades**.
4. The model with the **lowest mean rank** is selected and used uniformly for every grade.
5. Ties are broken by mean MAPE, then mean RMSE.

### Why rank-based aggregation?

Mean MAPE across grades can be dominated by a single grade with high absolute prices or high volatility. Ranks are scale-free and give every grade an equal vote in the decision.

### Chosen model

For the current dataset the selected model is **Random Forest** (with `n_estimators=100`, `max_depth=4`, `min_samples_leaf=2`, `random_state=42`). See the dashboard's "Why Random Forest?" panel for the live mean-rank table.

## 4. Feature engineering

### Autoregressive / seasonal features (used by Linear Regression and Random Forest)

For each grade `g`:

- `time_idx` — global trend
- `month`, `quarter` — calendar seasonality
- `month_sin`, `month_cos` — cyclical seasonality (sin/cos transform of month)
- `lag_1`, `lag_2`, `lag_3` — recent autoregressive signal
- `rolling_mean` — 3-month rolling mean shifted by one step (prevents leakage)

### Exogenous regressors (Random Forest only)

Extracted from the KTDA monthly auction PDFs and an external FX feed:

- `vol_lag_1` — the previous month's **sales volume in packages (Pkgs)** for the grade. Lag-1 avoids leaking "this month's" volume into "this month's" price.
- `usd_kes` — monthly **USD→KES exchange rate** (current month). FX is known at the start of the month so using the current value is safe.

The extractor reads the totals row at the bottom of each monthly PDF (e.g. `4,360 259 36,960 280 … 65,160 4,583,566.00 273`) which contains `[Pkgs, Avg]` columns for each grade plus the all-grades totals. Daily PDFs in the dataset are image-only (no embedded text) and are currently skipped — OCR would be required to use them.

Why keep exog features RF-only? Linear Regression and SARIMAX implementations stay tied to the autoregressive feature set so the comparison table is apples-to-apples between the classical baselines; Random Forest is the natural home for non-linear interactions between price, volume and FX.

Rows with NaN lags are dropped before fitting.

## 5. Hold-out validation

Independent of the CV used for selection, the chosen model is validated by a **chronological 80/20 holdout** per grade:

1. The series is split chronologically: first 80% → train, last 20% → test.
2. The model is **re-fitted on the training partition only**.
3. The model forecasts the held-out test horizon.
4. The same window is forecast by the Naïve baseline for comparison.

### Metrics reported

| Metric | Formula | Interpretation |
|---|---|---|
| **MAE** | $\frac{1}{n}\sum |y_i - \hat{y}_i|$ | Average error in ¢/kg. |
| **RMSE** | $\sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$ | Penalises large errors more heavily. |
| **MAPE** | $\frac{100}{n}\sum |y_i - \hat{y}_i|/y_i$ | Scale-free; comparable across grades. |
| **Bias** | $\frac{1}{n}\sum (y_i - \hat{y}_i)$ | Sign reveals systematic under-forecasting (+) or over-forecasting (−). |
| **Directional accuracy** | % of times $\text{sign}(\hat{y}_i - y_{i-1}) = \text{sign}(y_i - y_{i-1})$ | Captures whether the model gets the up/down move right. |
| **Improvement vs Naïve** | $(\text{MAPE}_{\text{naive}} - \text{MAPE}_{\text{model}})/\text{MAPE}_{\text{naive}}$ | Justifies using the model in production. |

The dashboard's **Validation** tab shows per-grade predicted-vs-actual line charts on the held-out window so visual diagnostics are immediately available.

## 6. Forecast uncertainty bands

The dashboard renders heuristic 80% intervals: $\hat{y} \pm 1.28 \cdot \sigma_{\text{hist}}$, clipped to plausible ranges. These are **not formal prediction intervals** — they are bounded estimates derived from the historical price standard deviation, useful for visual indication but not for statistical inference. A future iteration could replace them with bootstrap residual intervals (RF / Linear Regression) or `SARIMAX.get_forecast().conf_int()` for the SARIMAX model.

## 7. Limitations & next steps

- The dataset currently contains only the 9 monthly KTDA summary PDFs that have extractable text (post-aggregation: 8 unique months after deduping by `auction_no`). The 15 daily-sale PDFs in `salesreport/` were printed via Microsoft Print-to-PDF and contain no embedded text or images — OCR would be required to ingest them. Adding those would roughly triple the observation count.
- With ~8 observations the naive last-value baseline is hard to beat (it implicitly assumes prices are flat over short horizons, which is true on this thin dataset). More data, not better models, is the highest-leverage improvement.
- SARIMAX requires ≥8 observations after differencing and is therefore frequently unavailable; it drops out of the ranking on this dataset.
- The RF recursive forecasting loop applies a small trend-damping term to pull long-horizon forecasts toward the recent mean. A direct multi-output regressor per horizon would be more principled.
- Uncertainty bands shown on the dashboard are heuristic ($\hat{y} \pm 1.28\sigma_{\text{hist}}$), not statistical prediction intervals. Bootstrap residual intervals (for RF / LR) or `SARIMAX.get_forecast().conf_int()` are the obvious next steps.
- USD→KES exchange rates are taken from a manually-maintained monthly lookup (with a free-tier API fallback). For a production deployment a daily fetch from a tier-one rate source (CBK, OANDA) would be advisable.
- SARIMAX order is hard-coded to `(1,1,1)`; an `auto_arima` search would likely improve its standing once more data is available.

---

_File generated by the ChaiIntel dashboard. The methodology described here matches the code in `analytics/rfutils.py`._
