from django.shortcuts import render
import pandas as pd
import json
import numpy as np
from .utils import load_historical_data, forecast_prices

def dashboard(request):
    # Load and prepare data
    export_data = load_historical_data()
    base_forecast = forecast_prices(export_data)
    
    # Merge actual and forecast data
    merged = pd.merge(export_data, base_forecast, on='date', how='outer').sort_values('date')
    
    grades = ['BP1', 'PF1', 'DUST1', 'FNGS 1/2', 'DUST 1/2']
    colors = ['#4CAF50', '#36A2EB', '#FF6384', '#FFCE56', '#9966FF']
    grade_datasets = []
    
    for idx, grade in enumerate(grades):
        # Actual prices - keep None for missing values
        actual_prices = [val if not pd.isna(val) else None for val in merged[grade]]
        
        # Forecast values - ensure they're never zero or negative
        forecast_col = f'forecast_{grade}'
        if forecast_col in merged.columns:
            forecast_values = []
            for val in merged[forecast_col]:
                if pd.isna(val):
                    forecast_values.append(None)
                else:
                    # Ensure minimum viable price
                    forecast_values.append(max(val, 50))  # Minimum 50 units
        else:
            forecast_values = [None] * len(merged)
        
        # Add actual price line
        grade_datasets.append({
            'label': f'{grade} Actual',
            'data': actual_prices,
            'borderColor': colors[idx],
            'backgroundColor': colors[idx] + '20',  # Add transparency
            'borderWidth': 3,
            'pointRadius': 5,
            'pointHoverRadius': 7,
            'fill': False,
            'tension': 0.1
        })
        
        # Add forecast line
        grade_datasets.append({
            'label': f'{grade} Forecast',
            'data': forecast_values,
            'borderColor': colors[idx],
            'backgroundColor': colors[idx] + '10',
            'borderDash': [8, 4],
            'borderWidth': 2,
            'pointRadius': 3,
            'pointHoverRadius': 5,
            'fill': False,
            'tension': 0.2
        })
        
        # Add confidence intervals if available
        lower_col = f'forecast_{grade}_lower'
        upper_col = f'forecast_{grade}_upper'
        
        if lower_col in merged.columns and upper_col in merged.columns:
            # Lower confidence bound
            lower_values = [val if not pd.isna(val) else None for val in merged[lower_col]]
            upper_values = [val if not pd.isna(val) else None for val in merged[upper_col]]
            
            grade_datasets.append({
                'label': f'{grade} Confidence Range',
                'data': upper_values,
                'borderColor': colors[idx] + '40',
                'backgroundColor': colors[idx] + '15',
                'borderWidth': 1,
                'pointRadius': 0,
                'fill': '+1',  # Fill to previous dataset (lower bound)
                'tension': 0.2
            })
            
            grade_datasets.append({
                'label': f'{grade} Lower Bound',
                'data': lower_values,
                'borderColor': colors[idx] + '40',
                'backgroundColor': colors[idx] + '15',
                'borderWidth': 1,
                'pointRadius': 0,
                'fill': False,
                'tension': 0.2
            })
    
    # Calculate summary statistics for the template
    summary_stats = []
    for grade in grades:
        actual_data = merged[grade].dropna()
        forecast_col = f'forecast_{grade}'
        
        if len(actual_data) > 0:
            current_price = actual_data.iloc[-1] if len(actual_data) > 0 else 0
            
            # Get next month's forecast
            future_forecasts = merged[merged['date'] > export_data['date'].max()][forecast_col].dropna()
            next_forecast = future_forecasts.iloc[0] if len(future_forecasts) > 0 else current_price
            
            # Calculate trend
            if len(actual_data) >= 2:
                trend = ((actual_data.iloc[-1] - actual_data.iloc[-2]) / actual_data.iloc[-2]) * 100
            else:
                trend = 0
            
            summary_stats.append({
                'grade': grade,
                'current_price': round(current_price, 2),
                'next_forecast': round(max(next_forecast, 50), 2),  # Ensure minimum
                'trend': round(trend, 1),
                'min_historical': round(actual_data.min(), 2),
                'max_historical': round(actual_data.max(), 2),
                'avg_historical': round(actual_data.mean(), 2)
            })

    # Create combined dataset (actual prices only) - THIS WAS MISINDENTED BEFORE
    combined_data  = []
    for idx, grade in enumerate(grades):
        # Actual prices
        actual_prices = [val if not pd.isna(val) else None for val in merged[grade]]
        combined_data.append({
            'label': f'{grade} Actual',
            'data': actual_prices,
            'borderColor': colors[idx],
            'backgroundColor': colors[idx] + '20',
            'borderWidth': 2,
            'pointRadius': 3,
            'pointHoverRadius': 5,
            'fill': False,
            'tension': 0.1
        })
         # Forecast values
        forecast_col = f'forecast_{grade}'
        if forecast_col in merged.columns:
            forecast_values = [max(val, 50) if not pd.isna(val) else None for val in merged[forecast_col]]
            combined_data.append({
                'label': f'{grade} Forecast',
                'data': forecast_values,
                'borderColor': colors[idx],
                'backgroundColor': colors[idx] + '10',
                'borderDash': [8, 4],  # Dashed line for forecast
                'borderWidth': 2,
                'pointRadius': 3,
                'pointHoverRadius': 5,
                'fill': False,
                'tension': 0.2
            })
    
    context = {
        'export_dates': json.dumps(list(merged['date'].dt.strftime('%Y-%m-%d'))),
        'grade_datasets': json.dumps(grade_datasets),
        'combined_datasets': json.dumps(combined_data),  # Now properly included
        'grades': grades,
        'summary_stats': summary_stats,
        'total_data_points': len(export_data),
        'forecast_months': 12
    }
    
    return render(request, 'analytics/dashboard.html', context)

def api_forecast_single_grade(request, grade):
    """API endpoint to get forecast for a single grade."""
    if grade not in ['BP1', 'PF1', 'DUST1', 'FNGS_1_2', 'DUST_1_2']:
        return JsonResponse({'error': 'Invalid grade'}, status=400)
    
    # Convert URL-safe grade name back
    grade_name = grade.replace('_', ' ').replace('FNGS 1 2', 'FNGS 1/2').replace('DUST 1 2', 'DUST 1/2')
    
    export_data = load_historical_data()
    forecasts = forecast_prices(export_data, periods=6)  # 6-month forecast for API
    
    merged = pd.merge(export_data, forecasts, on='date', how='outer').sort_values('date')
    
    forecast_col = f'forecast_{grade_name}'
    if forecast_col not in merged.columns:
        return JsonResponse({'error': 'Forecast not available'}, status=404)
    
    # Prepare response data
    response_data = {
        'grade': grade_name,
        'historical': [],
        'forecast': []
    }
    
    for _, row in merged.iterrows():
        date_str = row['date'].strftime('%Y-%m-%d')
        
        if not pd.isna(row[grade_name]):
            response_data['historical'].append({
                'date': date_str,
                'price': round(row[grade_name], 2)
            })
        
        if not pd.isna(row[forecast_col]):
            response_data['forecast'].append({
                'date': date_str,
                'price': round(max(row[forecast_col], 50), 2)  # Ensure minimum
            })
    
    return JsonResponse(response_data)