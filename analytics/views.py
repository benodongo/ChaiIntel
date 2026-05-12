from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import json
import numpy as np
from .rfutils import (
    load_historical_data, forecast_prices, get_model_evaluation,
    get_feature_importance_chart, select_global_model, validate_model,
    GRADES, GRADE_LABELS,
)


GRADE_COLORS = {
    'BP1':      '#2D6A4F',
    'PF1':      '#52B788',
    'DUST1':    '#F4A261',
    'FNGS_1_2': '#E76F51',
    'DUST_1_2': '#8338EC',
}


def dashboard(request):
    export_data        = load_historical_data()
    evaluation_results = get_model_evaluation(export_data)

    # Pick ONE global model used uniformly for every grade.
    selection          = select_global_model(evaluation_results)
    chosen_model       = selection['name']

    base_forecast      = forecast_prices(export_data, model_name=chosen_model)
    validation         = validate_model(export_data, chosen_model)

    merged = pd.merge(export_data, base_forecast, on='date', how='outer').sort_values('date')

    # ── Chart datasets ──────────────────────────────────────────────────────
    grade_datasets = []
    for grade, color in GRADE_COLORS.items():
        if grade not in merged.columns:
            continue

        actual_prices = [
            float(v) if not pd.isna(v) else None for v in merged[grade]
        ]
        grade_datasets.append({
            'label': f'{GRADE_LABELS[grade]} Actual',
            'data': actual_prices,
            'borderColor': color,
            'backgroundColor': color + '22',
            'borderWidth': 2.5,
            'pointRadius': 4,
            'pointHoverRadius': 7,
            'fill': False,
            'tension': 0.2,
        })

        fc_col = f'forecast_{grade}'
        if fc_col in merged.columns:
            fc_vals = [
                float(v) if not pd.isna(v) else None for v in merged[fc_col]
            ]
            grade_datasets.append({
                'label': f'{GRADE_LABELS[grade]} Forecast',
                'data': fc_vals,
                'borderColor': color,
                'backgroundColor': color + '11',
                'borderDash': [8, 4],
                'borderWidth': 1.8,
                'pointRadius': 2,
                'pointHoverRadius': 5,
                'fill': False,
                'tension': 0.2,
            })

        lower_col = f'{fc_col}_lower'
        upper_col = f'{fc_col}_upper'
        if lower_col in merged.columns and upper_col in merged.columns:
            grade_datasets.append({
                'label': f'{GRADE_LABELS[grade]} Upper',
                'data': [float(v) if not pd.isna(v) else None for v in merged[upper_col]],
                'borderColor': color + '40',
                'backgroundColor': color + '14',
                'borderWidth': 0,
                'pointRadius': 0,
                'fill': '+1',
                'tension': 0.2,
            })
            grade_datasets.append({
                'label': f'{GRADE_LABELS[grade]} Lower',
                'data': [float(v) if not pd.isna(v) else None for v in merged[lower_col]],
                'borderColor': color + '40',
                'backgroundColor': color + '14',
                'borderWidth': 0,
                'pointRadius': 0,
                'fill': False,
                'tension': 0.2,
            })

    # ── Summary stats cards ─────────────────────────────────────────────────
    summary_stats = []
    for grade in GRADES:
        if grade not in merged.columns:
            continue
        actual = merged[grade].dropna()
        if len(actual) == 0:
            continue

        fc_col = f'forecast_{grade}'
        future_fc = merged[merged['date'] > export_data['date'].max()][fc_col].dropna()
        next_fc   = float(future_fc.iloc[0]) if len(future_fc) > 0 else float(actual.iloc[-1])

        trend = 0.0
        if len(actual) >= 2:
            trend = (float(actual.iloc[-1]) - float(actual.iloc[-2])) / float(actual.iloc[-2]) * 100

        # Best model name for this grade
        best_model = 'RF'
        if grade in evaluation_results:
            best_model = evaluation_results[grade].get('best_model', 'RF')

        summary_stats.append({
            'grade':           GRADE_LABELS[grade],
            'grade_key':       grade,
            'color':           GRADE_COLORS.get(grade, '#333'),
            'current_price':   round(float(actual.iloc[-1]), 2),
            'next_forecast':   round(next_fc, 2),
            'trend':           round(trend, 1),
            'min_historical':  round(float(actual.min()), 2),
            'max_historical':  round(float(actual.max()), 2),
            'avg_historical':  round(float(actual.mean()), 2),
            'best_model':      best_model,
            'data_points':     int(actual.notna().count()),
        })

    # ── Model comparison table data ─────────────────────────────────────────
    model_comparison = []
    for grade in GRADES:
        if grade not in evaluation_results:
            continue
        ev = evaluation_results[grade]
        row = {
            'grade': GRADE_LABELS[grade],
            'grade_key': grade,
            'color': GRADE_COLORS.get(grade, '#2D6A4F'),
            'models': [],
        }
        for model_name, metrics in ev['metrics'].items():
            if metrics:
                row['models'].append({
                    'name':   model_name,
                    'mae':    round(metrics['mae'], 2),
                    'rmse':   round(metrics['rmse'], 2),
                    'mape':   round(metrics['mape'], 2),
                    'r2':     round(metrics['r2'], 3) if metrics.get('r2') is not None else '—',
                    'best':   model_name == ev['best_model'],
                    'chosen': model_name == chosen_model,
                })
        model_comparison.append(row)

    # ── Global model ranking (justification table) ─────────────────────────
    global_ranking = []
    for r in selection.get('ranking', []):
        global_ranking.append({
            'name':      r['name'],
            'mean_rank': round(r['mean_rank'], 2),
            'mean_mape': round(r['mean_mape'], 2),
            'mean_rmse': round(r['mean_rmse'], 2),
            'mean_mae':  round(r['mean_mae'],  2),
            'chosen':    r['name'] == chosen_model,
        })

    # ── Validation cards ───────────────────────────────────────────────────
    validation_rows = []
    for grade in GRADES:
        v = validation.get('per_grade', {}).get(grade)
        if not v or 'error' in v:
            continue
        validation_rows.append({
            'grade':         GRADE_LABELS[grade],
            'grade_key':     grade,
            'color':         GRADE_COLORS.get(grade, '#2D6A4F'),
            'train_size':    v['train_size'],
            'test_size':     v['test_size'],
            'mae':           round(v['mae'], 2),
            'rmse':          round(v['rmse'], 2),
            'mape':          round(v['mape'], 2),
            'bias':          round(v['bias'], 2),
            'directional':   round(v['directional_acc'], 1) if v.get('directional_acc') is not None else '—',
            'naive_mape':    round(v['naive_mape'], 2),
            'beats_naive':   v['beats_naive'],
            'improvement':   round(v['improvement_vs_naive_pct'], 1) if v.get('improvement_vs_naive_pct') is not None else None,
            'dates':         v['dates'],
            'actuals':       v['actuals'],
            'predictions':   v['predictions'],
        })

    validation_summary = validation.get('summary') or {}
    if validation_summary:
        validation_summary = {
            'mean_mape':         round(validation_summary['mean_mape'], 2),
            'mean_rmse':         round(validation_summary['mean_rmse'], 2),
            'mean_bias':         round(validation_summary['mean_bias'], 2),
            'mean_directional':  round(validation_summary['mean_directional'], 1) if validation_summary.get('mean_directional') is not None else None,
            'beats_naive_count': validation_summary['beats_naive_count'],
            'grades_evaluated':  validation_summary['grades_evaluated'],
        }

    # ── Feature importance charts (only meaningful when RF is chosen) ──────
    feature_charts = {}
    if chosen_model == 'Random Forest':
        for grade in GRADES:
            if grade not in evaluation_results:
                continue
            imps = evaluation_results[grade].get('feature_importances')
            if imps:
                feature_charts[grade] = get_feature_importance_chart(grade, imps)

    context = {
        'export_dates':       json.dumps(list(merged['date'].dt.strftime('%Y-%m-%d'))),
        'grade_datasets':     json.dumps(grade_datasets),
        'grades':             GRADES,
        'grade_labels':       GRADE_LABELS,
        'grade_colors':       GRADE_COLORS,
        'summary_stats':      summary_stats,
        'total_data_points':  len(export_data),
        'forecast_months':    12,
        'evaluation_results': evaluation_results,
        'model_comparison':   model_comparison,
        'feature_charts':     feature_charts,
        # New: single chosen model + justification + validation
        'chosen_model':       chosen_model,
        'global_ranking':     global_ranking,
        'selection_stats': {
            'mean_rank': round(selection['mean_rank'], 2) if selection.get('mean_rank') is not None else None,
            'mean_mape': round(selection['mean_mape'], 2) if selection.get('mean_mape') is not None else None,
            'mean_rmse': round(selection['mean_rmse'], 2) if selection.get('mean_rmse') is not None else None,
        },
        'validation_rows':    validation_rows,
        'validation_summary': validation_summary,
        'validation_payload': json.dumps({
            'model': chosen_model,
            'rows':  [
                {'grade_key': r['grade_key'], 'grade': r['grade'], 'color': r['color'],
                 'dates': r['dates'], 'actuals': r['actuals'], 'predictions': r['predictions']}
                for r in validation_rows
            ],
        }),
    }

    return render(request, 'analytics/dashboard.html', context)
