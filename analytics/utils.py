import numpy as np
import pandas as pd
from prophet import Prophet
import logging

# Suppress Prophet warnings for cleaner output
logging.getLogger('prophet').setLevel(logging.WARNING)

# --- Historical Data Loader ---
def load_historical_data():
    """Load historical tea auction data and ensure it's sorted chronologically."""
    data = {
        'Month': ['May-22', 'Jun-24', 'May-24', 'Aug-24', 'Dec-24', 'Jan-25', 'Apr-25', 'May-25','Jun-25'],
        'Auction_No': ['2022/20', '2024/26', '2024/22', '2024/35', '2024/51', '2025/01', '2025/16', '2025/21', '2025/23'],
        'BP1': [265, 256, 259, 287, 268, 277, 201, 176, 246],
        'PF1': [260, 280, 280, 291, 277, 281, 239, 228, 256],
        'DUST1': [269, 272, 276, 284, 258, 255, 250, 229, 268],
        'FNGS 1/2': [150, 132, 130, 129, 129, 151, 140, 126, 141],
        'DUST 1/2': [177, 148, 162, 149, 144, 144, 134, 126,131],
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

def forecast_prices_robust(df, periods=12):
    """
    Robust forecasting with multiple safeguards against negative/zero predictions.
    """
    grades = ['BP1', 'PF1', 'DUST1', 'FNGS 1/2', 'DUST 1/2']
    forecasts = pd.DataFrame()
    
    for grade in grades:
        try:
            # Prepare data and calculate bounds
            prophet_df = df[['date', grade]].rename(columns={'date': 'ds', grade: 'y'}).dropna()
            
            if len(prophet_df) < 3:
                print(f"Warning: Insufficient data for {grade}, using fallback method")
                forecast = create_fallback_forecast(df, grade, periods)
            else:
                # Calculate price bounds
                floor_price, ceiling_price = calculate_price_bounds(prophet_df['y'])
                
                # Configure Prophet with custom parameters for limited data
                model = Prophet(
                    yearly_seasonality=False,  # Disable with limited data
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode='additive',
                    changepoint_prior_scale=0.1,  # Reduce overfitting
                    seasonality_prior_scale=0.1,
                    interval_width=0.8
                )
                
                # Add custom seasonality if we have enough data points
                if len(prophet_df) >= 6:
                    model.add_seasonality(name='custom_trend', period=365.25/4, fourier_order=2)
                
                model.fit(prophet_df)
                
                # Create future dataframe
                future = model.make_future_dataframe(periods=periods, freq='M')
                forecast_df = model.predict(future)
                
                # Apply multiple layers of protection against unrealistic values
                
                # 1. Clip to calculated bounds
                forecast_df['yhat'] = forecast_df['yhat'].clip(lower=floor_price, upper=ceiling_price)
                
                # 2. Apply trend dampening for future predictions
                last_actual = prophet_df['y'].iloc[-1]
                future_mask = forecast_df['ds'] > prophet_df['ds'].max()
                
                if future_mask.any():
                    # Gradually converge forecast towards recent average
                    recent_avg = prophet_df['y'].tail(3).mean()
                    future_forecasts = forecast_df.loc[future_mask, 'yhat']
                    
                    # Apply dampening factor that increases over time
                    dampening_factors = np.linspace(0.1, 0.5, len(future_forecasts))
                    dampened_forecasts = (future_forecasts * (1 - dampening_factors) + 
                                        recent_avg * dampening_factors)
                    
                    forecast_df.loc[future_mask, 'yhat'] = dampened_forecasts
                
                # 3. Final safety check - ensure minimum viable price
                min_viable_price = max(floor_price, last_actual * 0.5)
                forecast_df['yhat'] = forecast_df['yhat'].clip(lower=min_viable_price)
                
                forecast = forecast_df[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': f'forecast_{grade}'})
            
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

def create_fallback_forecast(df, grade, periods):
    """
    Fallback forecasting method using simple trend analysis.
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
    forecasts = []
    for i, future_date in enumerate(future_dates):
        # Apply dampened trend with noise reduction
        forecast_price = base_price + (trend * (i + 1) * 0.5)  # Reduce trend impact
        
        # Add small random variation but keep it conservative
        variation = np.random.normal(0, base_price * 0.02)  # 2% variation
        forecast_price += variation
        
        # Ensure minimum price
        forecast_price = max(forecast_price, base_price * 0.7, 50)
        forecasts.append(forecast_price)
    
    # Combine historical and forecast data
    all_dates = list(df['date']) + list(future_dates)
    all_forecasts = [None] * len(df) + forecasts
    
    return pd.DataFrame({
        'date': all_dates,
        f'forecast_{grade}': all_forecasts
    })

def add_confidence_intervals(df, forecasts):
    """
    Add confidence intervals to provide uncertainty estimates.
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

# --- Main forecasting function with all improvements ---
def forecast_prices(df, periods=12):
    """
    Main forecasting function that combines all robust forecasting techniques.
    """
    # Use the robust forecasting method
    forecasts = forecast_prices_robust(df, periods)
    
    # Add confidence intervals
    forecasts = add_confidence_intervals(df, forecasts)
    
    return forecasts