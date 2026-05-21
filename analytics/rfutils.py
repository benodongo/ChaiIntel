"""
rfutils.py – ChaiIntel forecasting engine
==========================================
Implements and compares four models per tea grade:
  1. Naïve (last-value carry-forward)       – academic baseline
  2. Linear Regression with time features   – interpretable benchmark
  3. SARIMAX                                – industry-standard time-series
  4. Random Forest                          – ensemble ML model

Cross-validation uses time-series walk-forward splits (no data leakage).
Metrics: MAE, RMSE, MAPE, R².
"""

import os
import numpy as np
import pandas as pd
import logging
import io
import base64
import warnings

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMAX_AVAILABLE = True
except ImportError:
    SARIMAX_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings('ignore')
logging.getLogger('statsmodels').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRADES = ['BP1', 'PF1', 'DUST1', 'FNGS_1_2', 'DUST_1_2']
GRADE_LABELS = {
    'BP1': 'BP1',
    'PF1': 'PF1',
    'DUST1': 'DUST1',
    'FNGS_1_2': 'FNGS 1/2',
    'DUST_1_2': 'DUST 1/2',
}
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'tea_auction_data.csv')

# Optional exogenous columns extracted from the auction PDFs / FX data.
# - <grade>_pkgs  : monthly sales volume for that grade (Pkgs)
# - usd_kes       : USD→KES exchange rate for that month
# These are kept in the loaded DataFrame and consumed by Random Forest as
# extra predictors when `use_exog=True`. They are NOT used by Linear
# Regression, SARIMAX or the Naïve baseline so the model-comparison table
# stays apples-to-apples between the classical candidates.
VOLUME_COLS = [f'{g}_pkgs' for g in GRADES]
FX_COL      = 'usd_kes'
EXOG_COLS   = VOLUME_COLS + [FX_COL]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_historical_data():
    """
    Load tea auction data from CSV.
    Columns: date, auction_no, BP1, PF1, DUST1, FNGS_1_2, DUST_1_2,
             optional <grade>_pkgs sales-volume columns, optional usd_kes.
    Returns a DataFrame sorted by date with internal grade column names
    plus the exogenous regressor columns where available.

    Handles both DD/MM/YYYY and YYYY-MM-DD date formats automatically.
    Aggregates multiple auctions in the same calendar month by taking
    the mean so the model always receives one row per month.
    """
    df = pd.read_csv(DATA_PATH)

    # Robust date parsing. Try ISO 8601 first (YYYY-MM-DD as written by the
    # PDF extractor); fall back to day-first for legacy DD/MM/YYYY rows.
    parsed = pd.to_datetime(df['date'], format='ISO8601', errors='coerce')
    if parsed.isna().any():
        fallback = pd.to_datetime(df.loc[parsed.isna(), 'date'],
                                  dayfirst=True, errors='coerce')
        parsed.loc[parsed.isna()] = fallback
    df['date'] = parsed
    df = df.dropna(subset=['date'])

    # Snap all dates to month-start so grouping works correctly
    df['date'] = df['date'].values.astype('datetime64[M]').astype('datetime64[ns]')

    # Aggregate: if multiple auctions fall in the same month, average
    # numeric columns. We keep prices, volumes, and FX if present.
    keep_cols = [c for c in GRADES + VOLUME_COLS + [FX_COL] if c in df.columns]
    df = df.groupby('date')[keep_cols].mean().reset_index()

    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_grade_display_name(grade):
    return GRADE_LABELS.get(grade, grade)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_features(df, grade, lag_periods=None, *, include_exog=False):
    """
    Build a feature matrix for time-series regression.
    Features:
      - time index (global trend)
      - month (seasonality proxy)
      - quarter
      - lag_1, lag_2, lag_3  (autoregressive)
      - rolling_mean_3        (local trend smoothing)

    When include_exog=True and the columns are present in `df` AND contain at
    least one non-NaN value, also adds:
      - vol_lag_1   : previous month's sales volume for this grade
      - usd_kes     : current month's USD→KES exchange rate (forward-filled)
    Exog columns that are entirely NaN (e.g. when the FX feed is unavailable)
    are silently dropped to avoid wiping out every row in ``dropna()``.
    These extra features are used only by the Random Forest model so that
    the comparison table stays fair between the classical candidates.
    """
    if lag_periods is None:
        lag_periods = [1, 2, 3]

    cols = ['date', grade]
    vol_col = f'{grade}_pkgs'
    use_vol = False
    use_fx  = False
    if include_exog:
        if vol_col in df.columns and df[vol_col].notna().any():
            cols.append(vol_col)
            use_vol = True
        if FX_COL in df.columns and df[FX_COL].notna().any():
            cols.append(FX_COL)
            use_fx = True

    out = df[cols].copy()
    out['time_idx'] = np.arange(len(out))
    out['month'] = out['date'].dt.month
    out['quarter'] = out['date'].dt.quarter
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)

    for lag in lag_periods:
        out[f'lag_{lag}'] = out[grade].shift(lag)

    n_roll = min(3, len(df) // 2)
    if n_roll >= 2:
        out['rolling_mean'] = out[grade].rolling(n_roll).mean().shift(1)
    else:
        out['rolling_mean'] = out[grade].shift(1)

    if use_vol:
        # Use lag-1 volume to avoid leakage of "this month's" volume
        # into the model when predicting "this month's" price.
        out['vol_lag_1'] = out[vol_col].shift(1)
        out = out.drop(columns=[vol_col])
    if use_fx:
        # FX is generally known at the start of the month, so we can
        # safely use the current value. Forward-fill any gaps.
        out[FX_COL] = out[FX_COL].ffill()

    out = out.dropna().reset_index(drop=True)
    return out


def feature_cols(df, grade):
    return [c for c in df.columns if c not in ['date', grade]]


def contiguous_tail(df, grade, max_gap_months=6):
    """
    Return the latest contiguous monthly slice of ``df`` for ``grade``.

    Walks backwards from the last observation and stops as soon as it hits
    a month-to-month gap larger than ``max_gap_months``. This protects the
    time-series models from a degenerate series like
    ``[May 2022, Apr 2024, May 2024, ...]`` — where the 23-month gap would
    otherwise pollute lag features and force SARIMAX to forward-fill two
    years of phantom values.

    The naive baseline is unaffected; only the lag/AR/seasonal models call
    this helper.
    """
    clean = df[['date', grade]].dropna(subset=[grade]).reset_index(drop=True)
    if len(clean) <= 1:
        return df.reset_index(drop=True)

    # Walk from the end backwards; cut as soon as we find a long gap.
    dates = pd.to_datetime(clean['date']).reset_index(drop=True)
    cut = 0
    for i in range(len(dates) - 1, 0, -1):
        delta = (dates.iloc[i].year - dates.iloc[i - 1].year) * 12 \
              + (dates.iloc[i].month - dates.iloc[i - 1].month)
        if delta > max_gap_months:
            cut = i
            break

    if cut == 0:
        return df.reset_index(drop=True)

    tail_dates = set(dates.iloc[cut:].dt.strftime('%Y-%m').tolist())
    out = df[pd.to_datetime(df['date']).dt.strftime('%Y-%m').isin(tail_dates)]
    return out.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Model 1 – Naïve baseline
# ---------------------------------------------------------------------------

def naive_forecast(df, grade, periods=12):
    """Carry the last observed value forward (random-walk baseline)."""
    last_val = df[grade].iloc[-1]
    last_date = pd.Timestamp(df['date'].iloc[-1])   # ensure Timestamp, not str
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=periods, freq='MS')
    hist = pd.DataFrame({'date': df['date'], f'forecast_{grade}': df[grade]})
    fut  = pd.DataFrame({'date': future_dates,
                         f'forecast_{grade}': [last_val] * periods})
    return pd.concat([hist, fut], ignore_index=True)


def naive_cv_metrics(df, grade, n_splits=3):
    prices = df[grade].values
    n = len(prices)
    n_splits = min(n_splits, n - 2)
    # sklearn's TimeSeriesSplit requires at least 2 folds.
    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes = [], [], []
    for train_idx, test_idx in tscv.split(prices):
        y_test = prices[test_idx]
        y_pred = np.full_like(y_test, fill_value=prices[train_idx[-1]], dtype=float)
        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mapes.append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

    return {
        'model': 'Naïve',
        'mae': float(np.mean(maes)),
        'rmse': float(np.mean(rmses)),
        'mape': float(np.mean(mapes)),
        'r2': None,
    }

# ---------------------------------------------------------------------------
# Model 2 – Linear Regression
# ---------------------------------------------------------------------------

def linear_forecast(df, grade, periods=12):
    train = contiguous_tail(df, grade)
    feat_df = create_features(train, grade)
    if len(feat_df) < 4:
        return naive_forecast(df, grade, periods)

    X = feat_df[feature_cols(feat_df, grade)].values
    y = feat_df[grade].values
    model = LinearRegression()
    model.fit(X, y)

    last_date = pd.Timestamp(train['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=periods, freq='MS')

    # Build future feature rows iteratively
    all_prices = list(train[grade].values)
    forecasts = []
    lags_used = [1, 2, 3]

    for i, fd in enumerate(future_dates):
        t = len(train) + i
        month = fd.month
        quarter = fd.quarter
        m_sin = np.sin(2 * np.pi * month / 12)
        m_cos = np.cos(2 * np.pi * month / 12)

        lv = [all_prices[-l] if l <= len(all_prices) else all_prices[0]
              for l in lags_used]
        roll = np.mean(all_prices[-3:]) if len(all_prices) >= 3 else all_prices[-1]

        row = np.array([[t, month, quarter, m_sin, m_cos] + lv + [roll]])
        pred = model.predict(row)[0]
        pred = float(np.clip(pred, train[grade].min() * 0.7, train[grade].max() * 1.4))
        forecasts.append(pred)
        all_prices.append(pred)

    # Keep the FULL history in the chart so the May-2022 row is still visible;
    # only the model was trained on the contiguous tail.
    hist = pd.DataFrame({'date': df['date'], f'forecast_{grade}': df[grade]})
    fut  = pd.DataFrame({'date': future_dates, f'forecast_{grade}': forecasts})
    return pd.concat([hist, fut], ignore_index=True)


def linear_cv_metrics(df, grade, n_splits=3):
    train = contiguous_tail(df, grade)
    feat_df = create_features(train, grade)
    if len(feat_df) < 4:
        return None

    X = feat_df[feature_cols(feat_df, grade)].values
    y = feat_df[grade].values
    n_splits = min(n_splits, len(feat_df) - 2)
    # sklearn's TimeSeriesSplit requires at least 2 folds.
    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes, r2s = [], [], [], []
    for train_idx, test_idx in tscv.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        mapes.append(np.mean(np.abs((y[test_idx] - pred) / y[test_idx])) * 100)
        if len(y[test_idx]) > 1:
            r2s.append(r2_score(y[test_idx], pred))

    return {
        'model': 'Linear Regression',
        'mae': float(np.mean(maes)),
        'rmse': float(np.mean(rmses)),
        'mape': float(np.mean(mapes)),
        'r2': float(np.mean(r2s)) if r2s else None,
    }

# ---------------------------------------------------------------------------
# Model 3 – SARIMAX
# ---------------------------------------------------------------------------

def sarimax_forecast(df, grade, periods=12):
    if not SARIMAX_AVAILABLE:
        return linear_forecast(df, grade, periods)

    train = contiguous_tail(df, grade)
    if len(train) < 6:
        return linear_forecast(df, grade, periods)

    try:
        # Build a true month-start indexed series. The tail is already
        # contiguous (no gap > 6 months) so any small gaps are safely
        # forward-filled without distorting the regression.
        series = (train.set_index('date')[grade]
                       .asfreq('MS')
                       .ffill())
        model = SARIMAX(series, order=(1, 1, 1), trend='c',
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fcast = fit.forecast(steps=periods)

        last_date = pd.Timestamp(train['date'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=periods, freq='MS')

        hist = pd.DataFrame({'date': df['date'], f'forecast_{grade}': df[grade]})
        fut  = pd.DataFrame({'date': future_dates,
                             f'forecast_{grade}': fcast.values.clip(
                                 train[grade].min() * 0.7, train[grade].max() * 1.4)})
        return pd.concat([hist, fut], ignore_index=True)
    except Exception:
        return linear_forecast(df, grade, periods)


def sarimax_cv_metrics(df, grade, n_splits=3):
    if not SARIMAX_AVAILABLE:
        return None

    train = contiguous_tail(df, grade)
    if len(train) < 8:
        return None

    prices = train[grade].values
    dates  = pd.to_datetime(train['date']).values
    n_splits = min(n_splits, len(prices) - 4)
    # sklearn's TimeSeriesSplit requires at least 2 folds.
    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes, r2s = [], [], [], []
    for train_idx, test_idx in tscv.split(prices):
        if len(train_idx) < 4:
            continue
        try:
            idx = (pd.DatetimeIndex(dates[train_idx])
                     .to_period('M').to_timestamp())
            train_s = pd.Series(prices[train_idx], index=idx).asfreq('MS').ffill()
            model = SARIMAX(train_s, order=(1, 1, 1), trend='c',
                            enforce_stationarity=False, enforce_invertibility=False)
            fit   = model.fit(disp=False)
            pred  = fit.forecast(steps=len(test_idx)).values
            y_t   = prices[test_idx]
            maes.append(mean_absolute_error(y_t, pred))
            rmses.append(np.sqrt(mean_squared_error(y_t, pred)))
            mapes.append(np.mean(np.abs((y_t - pred) / y_t)) * 100)
            if len(y_t) > 1:
                r2s.append(r2_score(y_t, pred))
        except Exception:
            continue

    if not maes:
        return None

    return {
        'model': 'SARIMAX',
        'mae': float(np.mean(maes)),
        'rmse': float(np.mean(rmses)),
        'mape': float(np.mean(mapes)),
        'r2': float(np.mean(r2s)) if r2s else None,
    }

# ---------------------------------------------------------------------------
# Model 4 – Random Forest
# ---------------------------------------------------------------------------
# When True, Random Forest sees the volume + FX features in addition to the
# autoregressive/seasonal ones. Other models are unaffected.
RF_USE_EXOG = True


def rf_forecast(df, grade, periods=12):
    use_exog = RF_USE_EXOG
    train = contiguous_tail(df, grade)
    feat_df = create_features(train, grade, include_exog=use_exog)
    if len(feat_df) < 4:
        return naive_forecast(df, grade, periods)

    fcols = feature_cols(feat_df, grade)
    X = feat_df[fcols].values
    y = feat_df[grade].values

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=2,
        random_state=42
    )
    model.fit(X, y)

    last_date = pd.Timestamp(train['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=periods, freq='MS')

    all_prices = list(train[grade].values)
    forecasts  = []
    lags_used  = [1, 2, 3]

    # Carry-forward values for the optional exog regressors. We don't have
    # future volume / FX, so the most-recent-known value is used — a sensible
    # neutral assumption that lets the trained model still query its trees.
    vol_col = f'{grade}_pkgs'
    last_vol = float(train[vol_col].dropna().iloc[-1]) if (use_exog and vol_col in train.columns and train[vol_col].notna().any()) else None
    last_fx  = float(train[FX_COL].dropna().iloc[-1])  if (use_exog and FX_COL  in train.columns and train[FX_COL].notna().any())  else None

    for i, fd in enumerate(future_dates):
        t     = len(train) + i
        month = fd.month
        quarter = fd.quarter
        m_sin = np.sin(2 * np.pi * month / 12)
        m_cos = np.cos(2 * np.pi * month / 12)

        lv   = [all_prices[-l] if l <= len(all_prices) else all_prices[0]
                for l in lags_used]
        roll = np.mean(all_prices[-3:]) if len(all_prices) >= 3 else all_prices[-1]

        feat_row = [t, month, quarter, m_sin, m_cos] + lv + [roll]
        # Order MUST match the order returned by feature_cols(feat_df, grade).
        # feature_cols just drops 'date' and the target grade column; the
        # remaining order matches the order in which we appended to `out`
        # inside create_features. We mirror that here.
        if use_exog:
            if 'vol_lag_1' in fcols and last_vol is not None:
                feat_row.append(last_vol)
            if FX_COL in fcols and last_fx is not None:
                feat_row.append(last_fx)

        # Safety check: feature length must match training X width.
        if len(feat_row) != X.shape[1]:
            # Should never happen; fall back gracefully.
            feat_row = feat_row[: X.shape[1]] + [0.0] * max(0, X.shape[1] - len(feat_row))

        row  = np.array([feat_row])
        pred = model.predict(row)[0]

        # Trend dampening: pull toward recent mean over time
        recent_avg = np.mean(all_prices[-3:])
        damp = min(0.5, (i + 1) * 0.08)
        pred = pred * (1 - damp) + recent_avg * damp
        pred = float(np.clip(pred, train[grade].min() * 0.7, train[grade].max() * 1.4))
        forecasts.append(pred)
        all_prices.append(pred)

    hist = pd.DataFrame({'date': df['date'], f'forecast_{grade}': df[grade]})
    fut  = pd.DataFrame({'date': future_dates, f'forecast_{grade}': forecasts})
    return pd.concat([hist, fut], ignore_index=True)


def rf_cv_metrics(df, grade, n_splits=3):
    train = contiguous_tail(df, grade)
    feat_df = create_features(train, grade, include_exog=RF_USE_EXOG)
    if len(feat_df) < 4:
        return None

    fcols  = feature_cols(feat_df, grade)
    X = feat_df[fcols].values
    y = feat_df[grade].values
    n_splits = min(n_splits, len(feat_df) - 2)
    # sklearn's TimeSeriesSplit requires at least 2 folds.
    if n_splits < 2:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes, r2s = [], [], [], []
    all_actuals, all_preds = [], []

    for train_idx, test_idx in tscv.split(X):
        model = RandomForestRegressor(n_estimators=100, max_depth=4,
                                      min_samples_leaf=2, random_state=42)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        maes.append(mean_absolute_error(y[test_idx], pred))
        rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        mapes.append(np.mean(np.abs((y[test_idx] - pred) / y[test_idx])) * 100)
        if len(y[test_idx]) > 1:
            r2s.append(r2_score(y[test_idx], pred))
        all_actuals.extend(y[test_idx])
        all_preds.extend(pred)

    # Feature importance (train on full dataset)
    full_model = RandomForestRegressor(n_estimators=100, max_depth=4,
                                       min_samples_leaf=2, random_state=42)
    full_model.fit(X, y)
    importances = dict(zip(fcols, full_model.feature_importances_.tolist()))

    return {
        'model': 'Random Forest',
        'mae': float(np.mean(maes)),
        'rmse': float(np.mean(rmses)),
        'mape': float(np.mean(mapes)),
        'r2': float(np.mean(r2s)) if r2s else None,
        'feature_importances': importances,
        'actuals': [float(v) for v in all_actuals],
        'preds':   [float(v) for v in all_preds],
    }

# ---------------------------------------------------------------------------
# Confidence intervals (prediction interval via historical residuals)
# ---------------------------------------------------------------------------

def add_confidence_intervals(df, forecasts, grade):
    """
    Approximate 80% prediction interval using historical price std.
    Note: these are heuristic bounds, not rigorous statistical intervals.
    """
    fc_col = f'forecast_{grade}'
    if fc_col not in forecasts.columns:
        return forecasts

    historical_std = df[grade].std()
    z = 1.28  # ~80% coverage

    forecasts[f'{fc_col}_lower'] = (
        forecasts[fc_col] - z * historical_std).clip(lower=df[grade].min() * 0.6)
    forecasts[f'{fc_col}_upper'] = forecasts[fc_col] + z * historical_std
    return forecasts

# ---------------------------------------------------------------------------
# Forecast-function registry
# ---------------------------------------------------------------------------
# Maps the human-readable model name (used in evaluation tables) back to the
# function that produces a forecast for a single grade. Lets us pick ONE model
# globally and apply it uniformly across all grades.
FORECAST_FUNCS = {
    'Naïve':             naive_forecast,
    'Linear Regression': linear_forecast,
    'SARIMAX':           sarimax_forecast,
    'Random Forest':     rf_forecast,
}

# Models excluded from global selection. Naïve is kept in the comparison
# tables as a *baseline* but must never be "chosen" — by convention any
# production model must outperform naïve carry-forward.
GLOBAL_SELECTION_EXCLUDE = {'Naïve'}

# ---------------------------------------------------------------------------
# Best-model selection
# ---------------------------------------------------------------------------

def select_best_model(grade_metrics):
    """
    Pick the model with lowest MAPE for a SINGLE grade.
    Used inside the per-grade evaluation table.
    Falls back to Random Forest if metrics are missing.
    """
    candidates = {k: v for k, v in grade_metrics.items() if v and v.get('mape') is not None}
    if not candidates:
        return 'Random Forest'
    return min(candidates, key=lambda k: candidates[k]['mape'])


def select_global_model(evaluation_results):
    """
    Pick ONE model used for every grade. Two-stage ranking:
      1. For each grade, rank the four models by MAPE (1 = best).
      2. Average each model's rank across all grades.
      3. Lowest average rank wins; ties broken by mean MAPE, then mean RMSE.

    Using mean-rank instead of mean-MAPE avoids letting a single grade with
    huge prices distort the choice (rank is scale-free).

    Returns:
      {
        'name': str,                # chosen model name
        'mean_rank': float,         # average rank across grades (1=best)
        'mean_mape': float,         # mean MAPE across grades for chosen model
        'mean_rmse': float,
        'mean_mae':  float,
        'ranking':  list[dict],     # full ranking table, used for justification UI
      }
    """
    # Collect each model's metrics across grades
    by_model = {name: [] for name in FORECAST_FUNCS.keys() if name not in GLOBAL_SELECTION_EXCLUDE}
    rank_acc = {name: [] for name in FORECAST_FUNCS.keys() if name not in GLOBAL_SELECTION_EXCLUDE}

    for grade, ev in evaluation_results.items():
        metrics = ev.get('metrics', {})
        # Rank only the eligible models (Naïve is reference-only)
        available = [(m, metrics[m]['mape'])
                     for m in metrics
                     if m not in GLOBAL_SELECTION_EXCLUDE
                     and metrics[m] and metrics[m].get('mape') is not None]
        available.sort(key=lambda t: t[1])
        for rank, (m, _mape) in enumerate(available, start=1):
            rank_acc[m].append(rank)
        for m, mdata in metrics.items():
            if m in GLOBAL_SELECTION_EXCLUDE:
                continue
            if mdata and mdata.get('mape') is not None:
                by_model[m].append(mdata)

    ranking = []
    for name in by_model.keys():
        ranks = rank_acc[name]
        rows = by_model[name]
        if not rows:
            continue
        ranking.append({
            'name': name,
            'mean_rank': float(np.mean(ranks)) if ranks else float('inf'),
            'mean_mape': float(np.mean([r['mape'] for r in rows])),
            'mean_rmse': float(np.mean([r['rmse'] for r in rows])),
            'mean_mae':  float(np.mean([r['mae']  for r in rows])),
            'grades_evaluated': len(rows),
        })

    if not ranking:
        return {
            'name': 'Random Forest',
            'mean_rank': float('nan'),
            'mean_mape': float('nan'),
            'mean_rmse': float('nan'),
            'mean_mae':  float('nan'),
            'ranking': [],
        }

    ranking.sort(key=lambda r: (r['mean_rank'], r['mean_mape'], r['mean_rmse']))
    winner = ranking[0]
    return {
        'name':      winner['name'],
        'mean_rank': winner['mean_rank'],
        'mean_mape': winner['mean_mape'],
        'mean_rmse': winner['mean_rmse'],
        'mean_mae':  winner['mean_mae'],
        'ranking':   ranking,
    }


# ---------------------------------------------------------------------------
# Holdout validation diagnostics
# ---------------------------------------------------------------------------

def _holdout_split(df, grade, holdout_frac=0.2, min_test=2):
    """Return (train_df, test_df) split chronologically. Drops NaNs in grade."""
    clean = df[['date', grade]].dropna().reset_index(drop=True)
    n = len(clean)
    n_test = max(min_test, int(round(n * holdout_frac)))
    n_test = min(n_test, max(1, n - 4))  # always leave at least 4 train rows
    if n - n_test < 4 or n_test < 1:
        return None, None
    return clean.iloc[:n - n_test].reset_index(drop=True), clean.iloc[n - n_test:].reset_index(drop=True)


def _directional_accuracy(actual, predicted):
    """% of time forecast direction (up/down vs previous actual) matches reality."""
    if len(actual) < 2:
        return None
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    a_dir = np.sign(np.diff(actual))
    p_dir = np.sign(predicted[1:] - actual[:-1])
    match = (a_dir == p_dir) & (a_dir != 0)
    return float(match.mean() * 100) if len(a_dir) else None


def validate_model(df, model_name):
    """
    Holdout validation for the chosen global model across all grades.

    For each grade:
      - chronological 80/20 split,
      - refit chosen model on train,
      - forecast the test horizon,
      - compute MAE / RMSE / MAPE / bias / directional accuracy,
      - compare against Naïve baseline on the same window.

    Returns:
      {
        'model': model_name,
        'per_grade': {grade: {...}},   # detail per grade incl. residuals
        'summary': {                   # aggregate across grades
          'mean_mape': float, 'mean_rmse': float, 'mean_bias': float,
          'mean_directional': float, 'beats_naive_count': int,
          'grades_evaluated': int,
        }
      }
    """
    forecast_fn = FORECAST_FUNCS.get(model_name, rf_forecast)
    per_grade = {}

    for grade in GRADES:
        if grade not in df.columns:
            continue
        train, test = _holdout_split(df, grade)
        if train is None or test is None or len(test) == 0:
            continue

        # Forecast horizon = length of held-out tail
        try:
            fc = forecast_fn(train, grade, periods=len(test))
        except Exception as e:
            per_grade[grade] = {'error': str(e)}
            continue

        # The forecast frame appends future rows AFTER the train tail.
        # Extract just the future (test) horizon.
        fc_future = fc[fc['date'] > train['date'].iloc[-1]].head(len(test))
        fc_col = f'forecast_{grade}'
        if fc_col not in fc_future.columns or len(fc_future) == 0:
            continue

        y_true = test[grade].to_numpy(dtype=float)
        y_pred = fc_future[fc_col].to_numpy(dtype=float)[: len(y_true)]
        if len(y_pred) != len(y_true) or len(y_true) == 0:
            continue

        residuals = y_true - y_pred
        mae   = float(mean_absolute_error(y_true, y_pred))
        rmse  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mape  = float(np.mean(np.abs(residuals / y_true)) * 100)
        bias  = float(residuals.mean())
        d_acc = _directional_accuracy(y_true, y_pred)

        # Naïve baseline on the same window
        naive_pred = np.full_like(y_true, train[grade].iloc[-1], dtype=float)
        naive_mape = float(np.mean(np.abs((y_true - naive_pred) / y_true)) * 100)

        per_grade[grade] = {
            'display_name':       get_grade_display_name(grade),
            'train_size':         int(len(train)),
            'test_size':          int(len(y_true)),
            'mae':                mae,
            'rmse':               rmse,
            'mape':               mape,
            'bias':               bias,
            'directional_acc':    d_acc,
            'naive_mape':         naive_mape,
            'beats_naive':        mape < naive_mape,
            'improvement_vs_naive_pct': float((naive_mape - mape) / naive_mape * 100) if naive_mape > 0 else None,
            # Series for plotting in the UI
            'dates':              [str(d.date()) for d in test['date']],
            'actuals':            [float(v) for v in y_true],
            'predictions':        [float(v) for v in y_pred],
            'residuals':          [float(v) for v in residuals],
        }

    valid = [g for g in per_grade.values() if 'error' not in g]
    if not valid:
        summary = {}
    else:
        summary = {
            'mean_mape':         float(np.mean([g['mape'] for g in valid])),
            'mean_rmse':         float(np.mean([g['rmse'] for g in valid])),
            'mean_bias':         float(np.mean([g['bias'] for g in valid])),
            'mean_directional':  float(np.mean([g['directional_acc'] for g in valid if g['directional_acc'] is not None])) if any(g['directional_acc'] is not None for g in valid) else None,
            'beats_naive_count': int(sum(1 for g in valid if g['beats_naive'])),
            'grades_evaluated':  len(valid),
        }

    return {
        'model':    model_name,
        'per_grade': per_grade,
        'summary':  summary,
    }


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def forecast_prices(df=None, periods=12, evaluate=False, model_name=None):
    """
    Main entry point called by views.py.

    By default (model_name=None) this function:
      1. Runs the evaluation across all four models,
      2. Picks ONE global winner via `select_global_model`,
      3. Uses that single model for every grade.

    Pass `model_name` explicitly to force a specific model (e.g. 'Random Forest').

    Returns:
      - If evaluate=True: dict of evaluation results per grade
      - Otherwise: merged DataFrame with forecast columns for each grade
    """
    if df is None:
        df = load_historical_data()

    if evaluate:
        return get_model_evaluation(df)

    # Decide which single model to use for every grade
    if model_name is None:
        eval_results = get_model_evaluation(df)
        selection = select_global_model(eval_results)
        model_name = selection['name']

    forecast_fn = FORECAST_FUNCS.get(model_name, rf_forecast)

    forecasts = pd.DataFrame()
    for grade in GRADES:
        if grade not in df.columns:
            continue
        try:
            fc = forecast_fn(df, grade, periods)
            fc = add_confidence_intervals(df, fc, grade)
        except Exception as e:
            print(f"Forecast error for {grade} with {model_name}: {e}")
            fc = naive_forecast(df, grade, periods)

        forecasts = fc if forecasts.empty else pd.merge(forecasts, fc, on='date', how='outer')

    return forecasts


def get_model_evaluation(df=None):
    """
    Run all four models across all grades and return comparison metrics.
    Kept verbose so the dashboard can JUSTIFY the choice of a single model.
    """
    if df is None:
        df = load_historical_data()

    results = {}
    for grade in GRADES:
        if grade not in df.columns:
            continue

        naive_m  = naive_cv_metrics(df, grade)
        linear_m = linear_cv_metrics(df, grade)
        sarimx_m = sarimax_cv_metrics(df, grade)
        rf_m     = rf_cv_metrics(df, grade)

        grade_metrics = {
            'Naïve':             naive_m,
            'Linear Regression': linear_m,
            'SARIMAX':           sarimx_m,
            'Random Forest':     rf_m,
        }
        best = select_best_model(grade_metrics)

        results[grade] = {
            'display_name': get_grade_display_name(grade),
            'metrics': grade_metrics,
            'best_model': best,
            'feature_importances': rf_m.get('feature_importances') if rf_m else None,
            'data_points': int(df[grade].notna().sum()),
        }

    return results


def get_feature_importance_chart(grade, importances):
    """
    Generate a base64 PNG bar chart of Random Forest feature importances.
    """
    if not importances:
        return None

    labels = list(importances.keys())
    values = list(importances.values())
    sorted_pairs = sorted(zip(values, labels), reverse=True)
    values, labels = zip(*sorted_pairs)

    friendly = {
        'lag_1': 'Lag 1 month', 'lag_2': 'Lag 2 months', 'lag_3': 'Lag 3 months',
        'rolling_mean': 'Rolling avg (3m)', 'month': 'Month',
        'quarter': 'Quarter', 'time_idx': 'Time trend',
        'month_sin': 'Seasonality (sin)', 'month_cos': 'Seasonality (cos)',
    }
    labels = [friendly.get(l, l) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#2D6A4F' if v == max(values) else '#74C69D' for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor='none', height=0.6)
    ax.set_xlabel('Importance', fontsize=9)
    ax.set_title(f'{get_grade_display_name(grade)} – Feature Importance', fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
