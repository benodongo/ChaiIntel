import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import io
import base64

# Suppress unnecessary warnings
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- Historical Data Loader ---
def load_historical_data():
    """Load historical tea auction data and ensure it's sorted chronologically."""
    data = {
        'Month': ['May-22', 'Jun-24', 'May-24', 'Aug-24', 'Dec-24', 'Jan-25', 'Apr-25', 'May-25','Jun-25', 'Jul-25'],
        'Auction_No': ['2022/20', '2024/26', '2024/22', '2024/35', '2024/51', '2025/01', '2025/16', '2025/21', '2025/23','2025/29'],
        'BP1': [265, 256, 259, 287, 268, 277, 201, 176, 246, 216],
        'PF1': [260, 280, 280, 291, 277, 281, 239, 228, 256, 245],
        'DUST1': [269, 272, 276, 284, 258, 255, 250, 229, 268, 264],
        'FNGS 1/2': [150, 132, 130, 129, 129, 151, 140, 126, 141, 127],
        'DUST 1/2': [177, 148, 162, 149, 144, 144, 134, 126, 131, 130],
    }
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['Month'], format='%b-%y')
    
    # Sort by date
    df = df[['date', 'BP1', 'PF1', 'DUST1', 'FNGS 1/2', 'DUST 1/2']].sort_values('date')
    return df

def calculate_price_bounds(historical_prices):
    """Calculate reasonable price bounds based on historical data."""
    mean_price = np.mean(historical_prices)
    std_price = np.std(historical_prices)
    min_price = np.min(historical_prices)
    
    # Set floor as 60% of historical minimum or mean - 2*std, whichever is higher
    floor_price = max(min_price * 0.6, mean_price - 2 * std_price, 50)  # Absolute minimum of 50
    
    # Set ceiling as 150% of historical maximum
    ceiling_price = np.max(historical_prices) * 1.5
    
    return floor_price, ceiling_price

def enhance_data_with_smoothing(df, grade):
    """Apply exponential smoothing to reduce volatility in limited data."""
    prices = df[grade].dropna()
    if len(prices) < 3:
        return df
    
    # Apply exponential smoothing with alpha=0.3 for moderate smoothing
    smoothed = prices.ewm(alpha=0.3).mean()
    
    # Create enhanced dataset with both original and smoothed values
    enhanced_df = df.copy()
    enhanced_df[f'{grade}_smoothed'] = smoothed
    
    return enhanced_df

def create_time_features(df):
    """Create time-based features for Random Forest."""
    df = df.copy()
    df['time'] = np.arange(len(df))
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    return df

def forecast_with_random_forest(historical_df, target_col, periods=12):
    """
    Forecast using Random Forest with safeguards against unrealistic predictions.
    Maintains same interface as Prophet version for easy swapping.
    """
    # Calculate bounds first
    floor_price, ceiling_price = calculate_price_bounds(historical_df[target_col])
    
    # Prepare features
    df = historical_df[['date', target_col]].copy()
    df = create_time_features(df)
    
    # Create lag features (using fewer lags for small dataset)
    for lag in [1, 2]:  # Reduced from [1, 2, 3] to avoid overfitting
        df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    # Add rolling statistics
    df['rolling_avg_2'] = df[target_col].rolling(2).mean().shift(1)
    df = df.dropna()
    
    if len(df) < 2:  # Need at least 2 samples to train
        return create_fallback_forecast(historical_df, target_col, periods)
    
    # Prepare data
    X = df.drop(columns=['date', target_col])
    y = df[target_col]
    
    # Configure Random Forest (simplified for small dataset)
    model = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=3,      # Shallower trees
        random_state=42,
        min_samples_leaf=2  # Prevent overfitting
    )
    model.fit(X, y)
    
    # Create future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    
    # Initialize with last known values
    last_values = df[target_col].values[-2:]  # Only need last 2 values
    
    forecasts = []
    for i in range(periods):
        # Prepare future feature vector
        X_future = pd.DataFrame({
            'time': [len(df) + i],
            'month': [(last_date.month + i) % 12 or 12],  # Handle month wrap-around
            'quarter': [(last_date.month + i - 1) // 3 + 1],
            'lag_1': [last_values[0] if i == 0 else forecasts[i-1]],
            'lag_2': [last_values[1] if i == 0 else (last_values[0] if i == 1 else forecasts[i-2])],
            'rolling_avg_2': [np.mean(last_values)] if i == 0 else [
                np.mean([forecasts[i-1], (last_values[0] if i == 1 else forecasts[i-2])])
            ]
        }, index=[0])
        
        # Predict
        pred = model.predict(X_future)[0]
        
        
        # Apply safeguards
        pred = np.clip(pred, floor_price, ceiling_price)
        
        # Apply trend dampening
        last_actual = historical_df[target_col].iloc[-1]
        recent_avg = historical_df[target_col].tail(3).mean()
        dampening_factor = min(0.5, (i+1)*0.1)  # Increase dampening over time
        pred = pred * (1 - dampening_factor) + recent_avg * dampening_factor
        
        # Final safety check
        min_viable_price = max(floor_price, last_actual * 0.5)
        pred = max(pred, min_viable_price)
        
        forecasts.append(pred)
    
    # Combine historical and forecast data
    all_dates = list(historical_df['date']) + list(future_dates)
    all_values = list(historical_df[target_col]) + forecasts
    
    return pd.DataFrame({
        'date': all_dates,
        f'forecast_{target_col}': all_values
    })

def create_fallback_forecast(df, grade, periods):
    """
    Fallback forecasting method using simple trend analysis.
    (Unchanged from original implementation)
    """
    prices = df[grade].dropna()
    dates = df.loc[df[grade].notna(), 'date']
    
    if len(prices) == 0:
        # If no data, use market average
        base_price = 200
    elif len(prices) == 1:
        base_price = prices.iloc[0]
    else:
        # Calculate simple trend
        base_price = prices.iloc[-1]
        if len(prices) >= 2:
            trend = (prices.iloc[-1] - prices.iloc[0]) / len(prices)
            # Dampen extreme trends
            trend = np.clip(trend, -5, 5)
        else:
            trend = 0
    
    # Generate future dates
    last_date = df['date'].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=periods, freq='M')
    
    # Create conservative forecast
    forecast_prices = []
    for i, future_date in enumerate(future_dates):
        # Apply dampened trend with noise reduction
        forecast_price = base_price + (trend * (i + 1) * 0.5)  # Reduce trend impact
        
        # Add small random variation but keep it conservative
        variation = np.random.normal(0, base_price * 0.02)  # 2% variation
        forecast_price += variation
        
        # Ensure minimum price
        forecast_price = max(forecast_price, base_price * 0.7, 50)
        forecast_prices.append(forecast_price)
    
    # Combine historical and forecast data
    all_dates = list(df['date']) + list(future_dates)
    all_forecasts = [None] * len(df) + forecast_prices
    
    return pd.DataFrame({
        'date': all_dates,
        f'forecast_{grade}': all_forecasts
    })

def add_confidence_intervals(df, forecasts):
    """
    Add confidence intervals to provide uncertainty estimates.
    (Unchanged from original implementation)
    """
    grades = ['BP1', 'PF1', 'DUST1', 'FNGS 1/2', 'DUST 1/2']
    
    for grade in grades:
        if f'forecast_{grade}' in forecasts.columns:
            forecast_col = f'forecast_{grade}'
            # Calculate confidence intervals based on historical volatility
            historical_prices = df[grade].dropna()
            
            if len(historical_prices) > 1:
                volatility = historical_prices.std()
                
                # Add confidence intervals
                forecasts[f'{forecast_col}_lower'] = forecasts[forecast_col] - (1.96 * volatility * 0.5)
                forecasts[f'{forecast_col}_upper'] = forecasts[forecast_col] + (1.96 * volatility * 0.5)
                
                # Ensure lower bound is never negative
                forecasts[f'{forecast_col}_lower'] = forecasts[f'{forecast_col}_lower'].clip(lower=50)
    
    return forecasts

def forecast_prices_robust(df, periods=12):
    """
    Main forecasting function using Random Forest.
    Maintains same interface as original Prophet version.
    """
    grades = ['BP1', 'PF1', 'DUST1', 'FNGS 1/2', 'DUST 1/2']
    forecasts = pd.DataFrame()
    
    for grade in grades:
        try:
            # Use Random Forest forecasting
            forecast = forecast_with_random_forest(df, grade, periods)
            
            # Merge forecasts
            if forecasts.empty:
                forecasts = forecast
            else:
                forecasts = pd.merge(forecasts, forecast, on='date', how='outer')
                
        except Exception as e:
            print(f"Error forecasting {grade}: {e}")
            # Use fallback method
            forecast = create_fallback_forecast(df, grade, periods)
            if forecasts.empty:
                forecasts = forecast
            else:
                forecasts = pd.merge(forecasts, forecast, on='date', how='outer')
    
    return forecasts

def forecast_prices(df=None, periods=12, evaluate=False):
    """
    Wrapper function that combines forecasting with confidence intervals.
    (Unchanged from original implementation)
    """
    if df is None:
        df = load_historical_data()
    
    if evaluate:
        return get_model_evaluation(df)
    # Use the robust forecasting method
    forecasts = forecast_prices_robust(df, periods)
    
    # Add confidence intervals
    forecasts = add_confidence_intervals(df, forecasts)
    
    return forecasts

# Add these new functions to rfutils.py
def evaluate_model_performance(df, grade, plot=True):
    """
    Evaluate Random Forest model performance for a specific grade
    Returns evaluation metrics and optionally a plot image
    """
    # Prepare features
    temp_df = df[['date', grade]].copy()
    temp_df = create_time_features(temp_df)
    
    # Create lag features
    for lag in [1, 2]:
        temp_df[f'lag_{lag}'] = temp_df[grade].shift(lag)
    temp_df['rolling_avg_2'] = temp_df[grade].rolling(2).mean().shift(1)
    temp_df = temp_df.dropna()
    
    if len(temp_df) < 3:
        return None  # Not enough data for proper evaluation
    
    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=min(3, len(temp_df)-2))
    metrics = {'mae': [], 'rmse': [], 'mape': []}
    actuals, preds, dates = [], [], []
    
    for train_idx, test_idx in tscv.split(temp_df):
        train = temp_df.iloc[train_idx]
        test = temp_df.iloc[test_idx]
        
        X_train = train.drop(columns=['date', grade])
        y_train = train[grade]
        X_test = test.drop(columns=['date', grade])
        y_test = test[grade]
        
        model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        # Store results
        actuals.extend(y_test)
        preds.extend(pred)
        dates.extend(test['date'])
        
        # Calculate metrics
        metrics['mae'].append(mean_absolute_error(y_test, pred))
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, pred)))
        metrics['mape'].append(np.mean(np.abs((y_test - pred) / y_test)) * 100)
    
    # Aggregate metrics
    results = {
        'grade': grade,
        'mae': np.mean(metrics['mae']),
        'rmse': np.mean(metrics['rmse']),
        'mape': np.mean(metrics['mape']),
        'last_actual': actuals[-1],
        'last_pred': preds[-1],
        'error_pct': abs(actuals[-1] - preds[-1]) / actuals[-1] * 100
    }
    
    if plot:
        plot_url = generate_evaluation_plot(dates, actuals, preds, grade)
        results['plot_url'] = plot_url
    
    return results

def generate_evaluation_plot(dates, actuals, preds, grade):
    """Generate base64 encoded plot for HTML embedding"""
    plt.figure(figsize=(10, 4))
    plt.plot(dates, actuals, 'o-', label='Actual')
    plt.plot(dates, preds, 'x--', label='Predicted')
    plt.title(f'{grade} Model Evaluation')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Save to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    
    # Encode as base64
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{plot_data}"

def get_model_evaluation(df=None):
    """
    Evaluate all models and return results
    Compatible with Django view (returns dict of results)
    """
    if df is None:
        df = load_historical_data()
    
    evaluation_results = {}
    grades = ['BP1', 'PF1', 'DUST1', 'FNGS 1/2', 'DUST 1/2']
    
    for grade in grades:
        try:
            evaluation_results[grade] = evaluate_model_performance(df, grade)
        except Exception as e:
            print(f"Error evaluating {grade}: {e}")
            evaluation_results[grade] = None
    
    return evaluation_results

