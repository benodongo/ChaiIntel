from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
import pandas as pd
import json
import numpy as np  # Important: numpy is needed
from .utils import (
    generate_data, forecast_prices, cluster_markets,
    calculate_optimal_price, generate_country_forecasts,
    simulate_pricing_strategies, predict_buyer_behavior,
    simulate_policy_impact, generate_price_forecast, generate_country_datasets,
    simulate_dynamic_pricing, simulate_weather_impact, simulate_regulatory_changes,
    simulate_market_opportunities, recommend_sale_timing, benchmark_against_traditional,
    simulate_risk_assessment, model_competitor_behavior, generate_scenario_comparison
)

def upload_csv(request):
    if request.method == 'POST' and request.FILES.get('csv_file'):
        csv_file = request.FILES['csv_file']
        if not csv_file.name.endswith('.csv'):
            return render(request, 'analytics/dashboard.html', {'error': 'Please upload a CSV file.'})

        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        uploaded_file_path = fs.url(filename)
        request.session['uploaded_csv'] = uploaded_file_path
        return redirect('chaiintel_dashboard')

    return redirect('chaiintel_dashboard')

def load_data(request):
    if request.session.get('uploaded_csv'):
        uploaded_csv_path = request.session['uploaded_csv']
        try:
            data = pd.read_csv('.' + uploaded_csv_path)
            if not {'date', 'export_price'}.issubset(data.columns):
                raise ValueError("Uploaded CSV missing required columns.")
            data['date'] = pd.to_datetime(data['date'])
            return data
        except Exception as e:
            print(f"CSV loading error: {e}")
            return generate_data()
    else:
        return generate_data()

def safe_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def dashboard(request):
    export_data = load_data(request)
    base_forecast = forecast_prices(export_data)
    #base_forecast = base_forecast[['ds', 'yhat']].rename(columns={'ds': 'date', 'yhat': 'forecast_price'})
    bp_forecast = base_forecast[['date', 'forecast_BP']].rename(columns={'forecast_BP': 'forecast_price'})
    merged = pd.merge(export_data, base_forecast, on='date', how='outer').sort_values('date')

    market_data = pd.DataFrame({
        'country': ['Pakistan', 'Egypt', 'UK', 'UAE', 'USA', 'Russia'],
        'gdp': [348.7, 435.6, 3131, 501.3, 22675, 1833],
        'population': [242, 109, 68, 10, 332, 146],
        'tea_consumption': [1.8, 0.9, 1.2, 0.7, 0.4, 0.6],
        'import_duty': [10.5, 15.0, 6.5, 8.2, 5.0, 12.5],
        'distance_km': [4300, 3200, 6700, 3800, 12500, 7100],
        'political_risk': [0.6, 0.7, 0.2, 0.4, 0.1, 0.5],
        'currency_volatility': [0.3, 0.5, 0.2, 0.4, 0.1, 0.6],
    })
    # Prepare datasets for each grade's chart
    grades = ['BP', 'PF', 'FNGS', 'DUST', 'BMF']
    colors = ['#4CAF50', '#36A2EB', '#FF6384', '#FFCE56', '#9966FF']
    grade_datasets = []

    for idx, grade in enumerate(grades):
        # Actual and forecast data
        actual_prices = merged[grade].fillna(0).tolist()
        grade_forecast_prices  = merged[f'forecast_{grade}'].fillna(0).tolist()
        
        # Add datasets for actual and forecast
        grade_datasets.append({
            'label': f'{grade} Actual',
            'data': actual_prices,
            'borderColor': colors[idx],
            'borderDash': [5, 5],
            'fill': False
        })
        grade_datasets.append({
            'label': f'{grade} Forecast',
            'data': grade_forecast_prices,
            'borderColor': colors[idx],
            'fill': False
        })


    clustered_markets = cluster_markets(market_data)
    countries = list(market_data['country'])

    country_forecasts_raw = generate_country_forecasts(bp_forecast, countries)

    kenya_cost = 85
    competitor_price = 110
    demand_factor = 0.8
    optimal_price = calculate_optimal_price(kenya_cost, competitor_price, demand_factor)
    pricing_strategies = simulate_pricing_strategies(kenya_cost, (1500, 5), strategies=[0.1, 0.2, 0.3, 0.4])

    buyer_predictions = predict_buyer_behavior(market_data)
    #policy_simulation = simulate_policy_impact(export_data)
    avg_prices = {c: forecasts['forecast_price'].mean() for c, forecasts in country_forecasts_raw.items()}
    best_market = min(avg_prices, key=avg_prices.get)
    highest_growth_market = max(avg_prices, key=avg_prices.get)

    country_datasets = []
    colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40']
    for idx, country in enumerate(countries):
        dataset = {
            'label': country,
            'data': list(country_forecasts_raw[country]['forecast_price'].fillna(0)),
            'borderColor': colors[idx % len(colors)],
            'fill': False,
            'tension': 0.4
        }
        country_datasets.append(dataset)

    # Advanced Simulations
    months, forecast_prices_line = generate_price_forecast()
    traditional_prices, accuracy = benchmark_against_traditional(forecast_prices_line)
    sale_recommendation = recommend_sale_timing(forecast_prices_line)
    weather_impact = simulate_weather_impact()
    regulation_changes = simulate_regulatory_changes()
    dynamic_models, dynamic_profits = simulate_dynamic_pricing()
    risks = simulate_risk_assessment()
    competitors, avg_competitor_price = model_competitor_behavior()
    buyer_model_data = predict_buyer_behavior(market_data).to_dict('records')
    opportunities = simulate_market_opportunities()
    scenario_labels, scenario_values = generate_scenario_comparison()

    # SAFETY: Convert numpy arrays before JSON serialization
    forecast_prices_line = safe_to_list(forecast_prices_line)
    traditional_prices = safe_to_list(traditional_prices)
    weather_impact = safe_to_list(weather_impact)
    dynamic_models = safe_to_list(dynamic_models)
    dynamic_profits = safe_to_list(dynamic_profits)
    risks = safe_to_list(risks)
    scenario_values = safe_to_list(scenario_values)

    context = {
        'export_dates': json.dumps(list(merged['date'].dt.strftime('%Y-%m-%d'))),
        'grade_datasets': json.dumps(grade_datasets),
        #'export_prices': json.dumps(list(merged['export_price'].fillna(0))),
       # 'forecast_prices': json.dumps(list(merged['forecast_price'].fillna(0))),
        'markets': clustered_markets.to_dict('records'),
        'optimal_price': round(optimal_price, 2),
        'countries': countries,
        'country_datasets': json.dumps(country_datasets),
        'best_market': best_market,
        'highest_growth_market': highest_growth_market,
        'pricing_strategies': pricing_strategies,
        'buyer_predictions': buyer_predictions.to_dict('records'),
        #'policy_simulation_prices': json.dumps(list(policy_simulation['export_price'].fillna(0))),
        'dynamic_pricing_labels': json.dumps([f"{int(k * 100)}%" for k in pricing_strategies.keys()]),
        'dynamic_pricing_values': json.dumps(list(pricing_strategies.values())),
        #'policy_labels': json.dumps(list(policy_simulation['date'].dt.strftime('%Y-%m-%d'))),
        #'policy_values': json.dumps(list(policy_simulation['export_price'].fillna(0))),
        # Advanced
        'months': json.dumps(months),
        'forecast_prices_line': json.dumps(forecast_prices_line),
        'traditional_prices': json.dumps(traditional_prices),
        'forecast_accuracy': json.dumps(accuracy),
        'sale_recommendation': sale_recommendation,
        'weather_impact': json.dumps(weather_impact),
        'regulation_changes': regulation_changes,
        'dynamic_models': json.dumps(dynamic_models),
        'dynamic_profits': json.dumps(dynamic_profits),
        'risks': json.dumps(risks),
        'competitors': competitors,
        'avg_competitor_price': avg_competitor_price,
        'buyer_model_data': buyer_model_data,
        'market_opportunities': opportunities,
        'scenario_labels': json.dumps(scenario_labels),
        'scenario_values': json.dumps(scenario_values),
    }

    return render(request, 'analytics/dashboard.html', context)
