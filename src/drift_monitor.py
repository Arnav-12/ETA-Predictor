"""
Drift simulation: compare training distribution vs a simulated production window.
Generates an HTML drift report with per-feature drift scores.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


NUMERICAL_COLS = [
    'haversine_distance_km', 'traffic_index', 'num_items',
    'order_value_inr', 'rider_experience_days', 'concurrent_orders_at_store',
]
CATEGORICAL_COLS = ['weather_condition', 'store_type', 'is_peak_hour']


def simulate_production_data(n=500):
    """Create a production-like window with injected distribution shift."""
    df = pd.read_csv('data/raw/delivery_data.csv', parse_dates=['order_timestamp'])
    prod = df.tail(n).copy()

    # Shift traffic_index upward
    prod['traffic_index'] = np.clip(prod['traffic_index'] + np.random.uniform(0.1, 0.25, n), 0, 1)

    # Shift distance slightly higher
    prod['haversine_distance_km'] = prod['haversine_distance_km'] * np.random.uniform(1.05, 1.15, n)

    # Increase rain frequency
    rain_mask = np.random.random(n) < 0.3
    prod.loc[rain_mask, 'weather_condition'] = 'rain'

    return prod


def check_numerical_drift(reference, production, col):
    """Compute drift as mean shift in units of reference std."""
    ref_mean = reference[col].mean()
    prod_mean = production[col].mean()
    ref_std = reference[col].std()
    shift = abs(prod_mean - ref_mean) / max(ref_std, 1e-6)
    drift_score = min(shift, 1.0)
    return ref_mean, prod_mean, shift, drift_score


def check_categorical_drift(reference, production, col):
    """Total variation distance between reference and production distributions."""
    ref_dist = reference[col].value_counts(normalize=True)
    prod_dist = production[col].value_counts(normalize=True)
    all_cats = sorted(set(ref_dist.index) | set(prod_dist.index))
    tvd = sum(abs(ref_dist.get(c, 0) - prod_dist.get(c, 0)) for c in all_cats)
    drift_score = min(tvd / 2, 1.0)
    return tvd, drift_score


def build_html_report(rows, drifted):
    """Build a self-contained HTML drift report."""
    rows_html = ""
    for r in rows:
        color = "#ffcccc" if r['status'] == "DRIFTED" else "#ccffcc"
        rows_html += (
            f"<tr style='background:{color}'>"
            f"<td>{r['feature']}</td>"
            f"<td>{r.get('ref', '-')}</td>"
            f"<td>{r.get('prod', '-')}</td>"
            f"<td>{r['shift']}</td>"
            f"<td>{r['score']}</td>"
            f"<td>{r['status']}</td></tr>\n"
        )

    return f"""<!DOCTYPE html>
<html>
<head><title>Data Drift Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 40px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
  th {{ background: #1a1a2e; color: #fff; }}
  h1 {{ color: #1a1a2e; }}
</style>
</head>
<body>
<h1>Data Drift Report</h1>
<table>
<tr><th>Feature</th><th>Ref Mean / Dist</th><th>Prod Mean / Dist</th>
    <th>Shift</th><th>Drift Score</th><th>Status</th></tr>
{rows_html}
</table>
<h2>Summary</h2>
<p>Features checked: {len(rows)}</p>
<p>Drifted features: {', '.join(drifted) if drifted else 'None'}</p>
</body></html>"""


def main():
    print("Loading reference data...")
    df = pd.read_csv('data/raw/delivery_data.csv', parse_dates=['order_timestamp'])

    n_train = int(len(df) * 0.70)
    reference = df.head(n_train).copy()

    print("Simulating production data with drift...")
    production = simulate_production_data(n=500)

    try:
        from evidently.report import Report
        from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

        print("Generating Evidently drift report...")
        report = Report(metrics=[
            DatasetDriftMetric(),
            *[ColumnDriftMetric(column_name=c) for c in NUMERICAL_COLS],
            *[ColumnDriftMetric(column_name=c) for c in CATEGORICAL_COLS],
        ])
        report.run(reference_data=reference, current_data=production)

        os.makedirs('reports', exist_ok=True)
        report.save_html('reports/drift_report.html')
        print("Saved: reports/drift_report.html")

        result = report.as_dict()
        print("\n" + "=" * 60)
        print("DRIFT MONITORING SUMMARY")
        print("=" * 60)
        for metric in result.get('metrics', []):
            res = metric.get('result', {})
            if 'drift_score' in res:
                col = metric.get('metric', {}).get('column_name', '?')
                status = "DRIFTED" if res.get('drift_detected') else "OK"
                print(f"  {col:<35} score={res['drift_score']:.4f}  [{status}]")
        print("=" * 60)
        return

    except Exception:
        print("Evidently unavailable, using manual drift detection...")

    os.makedirs('reports', exist_ok=True)
    report_rows = []
    drifted = []

    print("\n" + "=" * 60)
    print("DRIFT MONITORING SUMMARY")
    print("=" * 60)

    for col in NUMERICAL_COLS:
        ref_mean, prod_mean, shift, score = check_numerical_drift(reference, production, col)
        status = "DRIFTED" if shift > 0.1 else "OK"
        print(f"  {col:<35} ref={ref_mean:.4f}  prod={prod_mean:.4f}  "
              f"shift={shift:.3f}  score={score:.4f}  [{status}]")
        report_rows.append({
            'feature': col, 'ref': f'{ref_mean:.4f}', 'prod': f'{prod_mean:.4f}',
            'shift': f'{shift:.4f}', 'score': f'{score:.4f}', 'status': status,
        })
        if status == "DRIFTED":
            drifted.append(col)

    for col in CATEGORICAL_COLS:
        tvd, score = check_categorical_drift(reference, production, col)
        status = "DRIFTED" if tvd > 0.2 else "OK"
        print(f"  {col:<35} tvd={tvd:.4f}  score={score:.4f}  [{status}]")
        report_rows.append({
            'feature': col, 'ref': '-', 'prod': '-',
            'shift': f'{tvd:.4f}', 'score': f'{score:.4f}', 'status': status,
        })
        if status == "DRIFTED":
            drifted.append(col)

    print(f"\nDrifted features: {len(drifted)} / {len(report_rows)}")
    if drifted:
        print(f"  {', '.join(drifted)}")

    html = build_html_report(report_rows, drifted)
    with open('reports/drift_report.html', 'w') as f:
        f.write(html)
    print("Saved: reports/drift_report.html")
    print("=" * 60)


if __name__ == '__main__':
    main()
