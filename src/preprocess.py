"""
Preprocess data: read csvs, compute q/q growth(%), align quarterly panel.

Usage:
python -m src.preprocess --data-dir data --out-dir results/data
"""

import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np

def read_quarterly_gdp(path):
    # expects columns: ['period','gdp_level'] or ['period','gdp_qoq']
    df = pd.read_csv(path, parse_dates=['period'], dayfirst=True)
    # Allow either level or growth; if level present compute q/q growth
    if 'gdp_level' in df.columns:
        df = df.sort_values('period')
        df['gdp_qoq'] = df['gdp_level'].pct_change(periods=1) * 100.0
    return df[['period','gdp_qoq']].dropna().reset_index(drop=True)

def read_quarterly_components(path):
    # expects a CSV with columns: period, gfcf, private_cons, gov, manuf, construction, net_exports_pp
    df = pd.read_csv(path, parse_dates=['period'])
    return df

def aggregate_monthly_to_quarterly(monthly_df, method='mean'):
    # monthly_df: index monthly, value series
    q = monthly_df.resample('Q').mean()
    return q

def build_panel(gdp_csv, components_csv, monthly_indicator_files, out_dir):
    gdp = read_quarterly_gdp(gdp_csv)
    comps = read_quarterly_components(components_csv)
    panel = pd.merge(gdp, comps, on='period', how='inner')
    # Load monthly indicators and aggregate
    for fp in monthly_indicator_files:
        if fp.exists():
            name = fp.stem
            m = pd.read_csv(fp, parse_dates=['period'])
            m = m.set_index('period').sort_index()
            q = aggregate_monthly_to_quarterly(m)
            panel = panel.merge(q.rename(columns={q.columns[0]: name}).reset_index().rename(columns={'period':'period'}), on='period', how='left')
    os.makedirs(out_dir, exist_ok=True)
    panel.to_csv(Path(out_dir)/'panel.csv', index=False)
    print(f'Panel saved to {Path(out_dir)/"panel.csv"}')
    return panel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default='results/data')
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    # expected files (user should supply)
    gdp_csv = data_dir / 'mospi_quarterly_gdp.csv'
    comps_csv = data_dir / 'quarterly_components.csv'
    monthly_files = [data_dir / f for f in ['ip.csv','pmi.csv','gst.csv','electricity.csv','credit.csv']]
    build_panel(gdp_csv, comps_csv, monthly_files, args.out_dir)
