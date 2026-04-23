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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_historical_data():
    """
    Load tea auction data from CSV.
    Columns: date, auction_no, BP1, PF1, DUST1, FNGS_1_2, DUST_1_2
    Returns a DataFrame sorted by date with internal grade column names.

    Handles both DD/MM/YYYY and YYYY-MM-DD date formats automatically.
    Aggregates multiple auctions in the same calendar month by taking
    the mean price so the model always receives one row per month.
    """
    df = pd.read_csv(DATA_PATH)

    # Robust date parsing: handles DD/MM/YYYY and YYYY-MM-DD
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')

    # Drop rows where date could not be parsed
    df = df.dropna(subset=['date'])

    # Snap all dates to month-start so grouping works correctly
    df['date'] = df['date'].values.astype('datetime64[M]').astype('datetime64[ns]')

    # Aggregate: if multiple auctions fall in the same month, take the mean
    grade_cols = [c for c in GRADES if c in df.columns]
    df = df.groupby('date')[grade_cols].mean().reset_index()

    df = df.sort_values('date').reset_index(drop=True)
    return df


def get_grade_display_name(grade):
    return GRADE_LABELS.get(grade, grade)

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_features(df, grade, lag_periods=None):
    """
    Build a feature matrix for time-series regression.
    Features:
      - time index (global trend)
      - month (seasonality proxy)
      - quarter
      - lag_1, lag_2, lag_3  (autoregressive)
      - rolling_mean_3        (local trend smoothing)
    """
    if lag_periods is None:
        lag_periods = [1, 2, 3]

    out = df[['date', grade]].copy()
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

    out = out.dropna().reset_index(drop=True)
    return out


def feature_cols(df, grade):
    return [c for c in df.columns if c not in ['date', grade]]

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
    if n_splits < 1:
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
    feat_df = create_features(df, grade)
    if len(feat_df) < 4:
        return naive_forecast(df, grade, periods)

    X = feat_df[feature_cols(feat_df, grade)].values
    y = feat_df[grade].values
    model = LinearRegression()
    model.fit(X, y)

    last_date = pd.Timestamp(df['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=periods, freq='MS')

    # Build future feature rows iteratively
    all_prices = list(df[grade].values)
    forecasts = []
    lags_used = [1, 2, 3]

    for i, fd in enumerate(future_dates):
        t = len(df) + i
        month = fd.month
        quarter = fd.quarter
        m_sin = np.sin(2 * np.pi * month / 12)
        m_cos = np.cos(2 * np.pi * month / 12)

        lv = [all_prices[-l] if l <= len(all_prices) else all_prices[0]
              for l in lags_used]
        roll = np.mean(all_prices[-3:]) if len(all_prices) >= 3 else all_prices[-1]

        row = np.array([[t, month, quarter, m_sin, m_cos] + lv + [roll]])
        pred = model.predict(row)[0]
        pred = float(np.clip(pred, df[grade].min() * 0.7, df[grade].max() * 1.4))
        forecasts.append(pred)
        all_prices.append(pred)

    hist = pd.DataFrame({'date': df['date'], f'forecast_{grade}': df[grade]})
    fut  = pd.DataFrame({'date': future_dates, f'forecast_{grade}': forecasts})
    return pd.concat([hist, fut], ignore_index=True)


def linear_cv_metrics(df, grade, n_splits=3):
    feat_df = create_features(df, grade)
    if len(feat_df) < 4:
        return None

    X = feat_df[feature_cols(feat_df, grade)].values
    y = feat_df[grade].values
    n_splits = min(n_splits, len(feat_df) - 2)
    if n_splits < 1:
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
    if not SARIMAX_AVAILABLE or len(df) < 6:
        return linear_forecast(df, grade, periods)

    try:
        series = df.set_index('date')[grade].asfreq('MS').fillna(method='ffill')
        # Simple ARIMA(1,1,1) — justified by small dataset size
        model = SARIMAX(series, order=(1, 1, 1), trend='c',
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        fcast = fit.forecast(steps=periods)

        last_date = pd.Timestamp(df['date'].iloc[-1])
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                     periods=periods, freq='MS')

        hist = pd.DataFrame({'date': df['date'], f'forecast_{grade}': df[grade]})
        fut  = pd.DataFrame({'date': future_dates,
                             f'forecast_{grade}': fcast.values.clip(
                                 df[grade].min() * 0.7, df[grade].max() * 1.4)})
        return pd.concat([hist, fut], ignore_index=True)
    except Exception:
        return linear_forecast(df, grade, periods)


def sarimax_cv_metrics(df, grade, n_splits=3):
    if not SARIMAX_AVAILABLE or len(df) < 8:
        return None

    prices = df[grade].values
    dates  = df['date'].values
    n_splits = min(n_splits, len(prices) - 4)
    if n_splits < 1:
        return None

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, mapes, r2s = [], [], [], []
    for train_idx, test_idx in tscv.split(prices):
        if len(train_idx) < 4:
            continue
        try:
            train_s = pd.Series(prices[train_idx],
                                index=pd.DatetimeIndex(dates[train_idx]).to_period('M').to_timestamp())
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

def rf_forecast(df, grade, periods=12):
    feat_df = create_features(df, grade)
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

    last_date = pd.Timestamp(df['date'].iloc[-1])
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                                 periods=periods, freq='MS')

    all_prices = list(df[grade].values)
    forecasts  = []
    lags_used  = [1, 2, 3]

    for i, fd in enumerate(future_dates):
        t     = len(df) + i
        month = fd.month
        quarter = fd.quarter
        m_sin = np.sin(2 * np.pi * month / 12)
        m_cos = np.cos(2 * np.pi * month / 12)

        lv   = [all_prices[-l] if l <= len(all_prices) else all_prices[0]
                for l in lags_used]
        roll = np.mean(all_prices[-3:]) if len(all_prices) >= 3 else all_prices[-1]

        row  = np.array([[t, month, quarter, m_sin, m_cos] + lv + [roll]])
        pred = model.predict(row)[0]
        # Trend dampening: pull toward recent mean over time
        recent_avg = np.mean(all_prices[-3:])
        damp = min(0.5, (i + 1) * 0.08)
        pred = pred * (1 - damp) + recent_avg * damp
        pred = float(np.clip(pred, df[grade].min() * 0.7, df[grade].max() * 1.4))
        forecasts.append(pred)
        all_prices.append(pred)

    hist = pd.DataFrame({'date': df['date'], f'forecast_{grade}': df[grade]})
    fut  = pd.DataFrame({'date': future_dates, f'forecast_{grade}': forecasts})
    return pd.concat([hist, fut], ignore_index=True)


def rf_cv_metrics(df, grade, n_splits=3):
    feat_df = create_features(df, grade)
    if len(feat_df) < 4:
        return None

    fcols  = feature_cols(feat_df, grade)
    X = feat_df[fcols].values
    y = feat_df[grade].values
    n_splits = min(n_splits, len(feat_df) - 2)
    if n_splits < 1:
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
# Best-model selection
# ---------------------------------------------------------------------------

def select_best_model(grade_metrics):
    """
    Pick the model with lowest MAPE. Falls back to RF if metrics missing.
    Returns the model name string.
    """
    candidates = {k: v for k, v in grade_metrics.items() if v and v.get('mape') is not None}
    if not candidates:
        return 'Random Forest'
    return min(candidates, key=lambda k: candidates[k]['mape'])


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def forecast_prices(df=None, periods=12, evaluate=False):
    """
    Main entry point called by views.py.

    Returns:
      - If evaluate=True: dict of evaluation results per grade
      - Otherwise: merged DataFrame with forecast columns for each grade
    """
    if df is None:
        df = load_historical_data()

    if evaluate:
        return get_model_evaluation(df)

    forecasts = pd.DataFrame()
    for grade in GRADES:
        if grade not in df.columns:
            continue
        try:
            fc = rf_forecast(df, grade, periods)
            fc = add_confidence_intervals(df, fc, grade)
        except Exception as e:
            print(f"Forecast error for {grade}: {e}")
            fc = naive_forecast(df, grade, periods)

        forecasts = fc if forecasts.empty else pd.merge(forecasts, fc, on='date', how='outer')

    return forecasts


def get_model_evaluation(df=None):
    """
    Run all four models across all grades and return comparison metrics.
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
