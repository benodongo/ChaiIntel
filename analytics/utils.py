import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# --- Simulated Data Generators ---
def simulate_weather_impact(n=12):
    """Simulate monthly weather impacts as percentage changes."""
    return np.round(np.random.normal(0, 2, n), 2)


def simulate_regulatory_changes(n=12):
    """Simulate monthly regulatory changes."""
    options = ['None', 'Mild Restriction', 'Major Policy Shift']
    return [random.choice(options) for _ in range(n)]


def simulate_competitor_behavior(n=5):
    """Simulate competitor pricing behavior."""
    return [{'country': f'Competitor {i+1}', 'price': round(random.uniform(2.5, 3.5), 2)} for i in range(n)]


def generate_data():
    """Generate simulated export data for each tea grade with realistic fluctuations."""
    dates = pd.date_range(start='2020-01-01', end=datetime.today())
    num_dates = len(dates)
    
    # Current prices for each grade
    current_prices = {
        'BP': 160,
        'PF': 155,
        'FNGS': 120,
        'DUST': 100,
        'BMF': 80
    }
    
    data = {'date': dates}
    
    # Simulate historical data for each grade with seasonal fluctuations
    for grade, price in current_prices.items():
        # Base trend from 80% to 120% of current price to allow for ups and downs
        base_trend = np.linspace(price * 0.8, price * 1.2, num_dates)
        
        # Add seasonal patterns (quarterly and yearly cycles)
        time_period = np.linspace(0, 10, num_dates)
        seasonal = (
            np.sin(time_period * 2 * np.pi) * 10 +  # Yearly cycle
            np.sin(time_period * 8 * np.pi) * 5 +   # Quarterly cycle
            np.random.normal(0, 8, num_dates)       # Random noise
        )
        
        # Combine components
        historical = base_trend + seasonal
        data[grade] = np.round(historical, 2)
    
    # Other simulated columns with more realistic fluctuations
    time_period = np.linspace(0, 10, num_dates)
    seasonal_pattern = np.sin(time_period * 2 * np.pi) * 0.2 + 0.8  # Oscillates between 60% and 100%
    
    data.update({
        'domestic_production': (np.linspace(5000, 8000, num_dates) * 
                               seasonal_pattern + 
                               np.random.normal(0, 300, num_dates)),
        'production_cost': (np.linspace(80, 120, num_dates) + 
                          np.sin(time_period * 4 * np.pi) * 15 +
                          np.random.normal(0, 8, num_dates)),
        'global_demand': (np.cos(np.linspace(0, 15, num_dates)) * 30 + 70 + 
                         np.random.normal(0, 15, num_dates))
    })
    
    return pd.DataFrame(data)

# --- Forecasting ---
def forecast_prices(df, periods=365):
    """Forecast prices for each tea grade using Prophet."""
    grades = ['BP', 'PF', 'FNGS', 'DUST', 'BMF']
    forecasts = pd.DataFrame()
    
    for grade in grades:
        # Prepare data for Prophet
        prophet_df = df[['date', grade]].rename(columns={'date': 'ds', grade: 'y'})
        
        # Train model
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
        model.fit(prophet_df)
        
        # Generate future dates
        future = model.make_future_dataframe(periods=periods)
        grade_forecast = model.predict(future)
        
        # Extract relevant forecast columns
        forecast = grade_forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': f'forecast_{grade}'})
        
        # Merge forecasts
        if forecasts.empty:
            forecasts = forecast
        else:
            forecasts = pd.merge(forecasts, forecast, on='date', how='outer')
    
    return forecasts


# --- Clustering ---
def cluster_markets(market_data):
    """Cluster markets based on economic indicators."""
    features = market_data[['gdp', 'population', 'tea_consumption', 'import_duty', 'distance_km', 'political_risk', 'currency_volatility']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    market_data['cluster'] = kmeans.fit_predict(scaled_features)
    return market_data


# --- Price Calculation ---
def calculate_optimal_price(production_cost, competitor_price, demand_factor):
    """Calculate optimal selling price."""
    margin = 0.25
    demand_multiplier = 1 + (demand_factor - 0.5)
    competition_factor = np.clip(competitor_price / production_cost, 0.8, 1.2)
    return production_cost * (1 + margin) * demand_multiplier * competition_factor


# --- Country Forecasts ---
def generate_country_forecasts(base_forecast, countries):
    """Generate adjusted forecasts per country."""
    country_forecasts = {}
    for country in countries:
        offset = np.random.uniform(-10, 10)
        multiplier = np.random.uniform(0.9, 1.1)
        forecast = base_forecast.copy()
        forecast['forecast_price'] = (forecast['forecast_price'] + offset) * multiplier
        country_forecasts[country] = forecast
    return country_forecasts


# --- Pricing Strategies ---
def simulate_pricing_strategies(production_cost, demand_curve_params, strategies):
    """Simulate profitability under different pricing markups."""
    a, b = demand_curve_params
    results = {}
    for markup in strategies:
        price = production_cost * (1 + markup)
        demand = max(a - b * price, 0)
        profit = (price - production_cost) * demand
        results[markup] = profit
    return results


# --- Buyer Behavior Prediction ---
def predict_buyer_behavior(market_data):
    """Predict tea consumption based on GDP."""
    X = market_data[['gdp']]
    y = market_data['tea_consumption']
    model = LinearRegression()
    model.fit(X, y)
    market_data['predicted_tea_consumption'] = model.predict(X)
    return market_data


# --- Policy Simulation ---
def simulate_policy_impact(export_data, reserve_price_increase=5, payment_speed_bonus=0.02):
    """Simulate policy impact on export prices."""
    adjusted = export_data.copy()
    adjusted['export_price'] += reserve_price_increase
    adjusted['export_price'] *= (1 + payment_speed_bonus)
    return adjusted


# --- Price Forecast Generation ---
def generate_price_forecast():
    """Generate price forecasts for 12 months."""
    base_date = datetime.today()
    dates = [base_date + timedelta(weeks=4 * i) for i in range(12)]
    prices = [round(2.5 + 0.05 * i + random.uniform(-0.1, 0.1), 2) for i in range(12)]
    return [d.strftime('%Y-%m') for d in dates], prices


# --- Country Datasets ---
def generate_country_datasets():
    """Generate export price datasets for multiple countries."""
    countries = ['UK', 'Pakistan', 'Egypt']
    labels, _ = generate_price_forecast()
    datasets = []
    for country in countries:
        data = [round(2.4 + 0.03 * i + random.uniform(-0.05, 0.05), 2) for i in range(12)]
        datasets.append({'label': country, 'data': data, 'borderWidth': 2})
    return datasets


# --- Dynamic Pricing Simulation ---
def simulate_dynamic_pricing():
    """Simulate profit for different dynamic pricing models."""
    models = ['Cost Plus', 'AI Optimized', 'Competitor Based']
    profits = [random.randint(50000, 90000), random.randint(80000, 120000), random.randint(60000, 95000)]
    return models, profits


# --- Policy Chart Data ---
def simulate_policy_chart_data():
    """Simulate policy impact chart values."""
    policies = ['Export Tax Relief', 'Auction Reform']
    values = [random.uniform(1, 5), random.uniform(2, 6)]
    return policies, [round(v, 2) for v in values]


# --- Market Opportunities Simulation ---
def simulate_market_opportunities():
    """Simulate market opportunities for different countries."""
    countries = ['India', 'Germany', 'China']
    return [{
        'country': c,
        'gdp': round(random.uniform(1000, 3000), 2),
        'population': round(random.uniform(50, 1500), 1),
        'tea_consumption': round(random.uniform(0.2, 1.5), 2),
        'import_duty': random.randint(5, 30),
        'market_potential': random.choice(['High', 'Medium', 'Low'])
    } for c in countries]


# --- Recommendation Engine ---
def recommend_sale_timing(forecast_prices):
    """Recommend best month to sell based on forecast prices."""
    max_price = max(forecast_prices)
    best_month = forecast_prices.index(max_price)
    best_month_name = (datetime.today() + timedelta(weeks=4 * best_month)).strftime('%B')
    return f"Month {best_month + 1} ({best_month_name})"


# --- Benchmarking ---
def benchmark_against_traditional(forecast_prices):
    """Compare AI prices against traditional pricing."""
    traditional_prices = [p - random.uniform(0.05, 0.2) for p in forecast_prices]
    accuracy = [round(100 - abs(f - t) / t * 100, 2) for f, t in zip(forecast_prices, traditional_prices)]
    return traditional_prices, accuracy


# --- Risk Assessment ---
def simulate_risk_assessment(n=12):
    """Simulate risk levels."""
    return [round(random.uniform(0, 1), 2) for _ in range(n)]


# --- Competitor Modeling ---
def model_competitor_behavior():
    """Model competitor behavior and average price."""
    competitors = simulate_competitor_behavior()
    average_price = np.mean([c['price'] for c in competitors])
    return competitors, round(average_price, 2)


# --- Scenario Planning ---
def generate_scenario_comparison():
    """Generate values for different future scenarios."""
    scenarios = ['Optimistic', 'Baseline', 'Pessimistic']
    values = [round(random.uniform(3.0, 4.0), 2), round(random.uniform(2.5, 3.5), 2), round(random.uniform(2.0, 3.0), 2)]
    return scenarios, values
