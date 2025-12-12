"""
Extract PCA factors from standardized monthly indicators aggregated to quarterly.
Usage:
python -m src.factor_extract --in-dir results/data --out-dir results/factors
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def extract_factors(panel_csv, var_names, n_factors=1, out_dir='results/factors'):
    panel = pd.read_csv(panel_csv, parse_dates=['period'])
    panel = panel.sort_values('period').set_index('period')
    X = panel[var_names].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_factors)
    F = pca.fit_transform(Xs)
    factors = pd.DataFrame(F, index=X.index, columns=[f'factor{i+1}' for i in range(n_factors)])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    factors.to_csv(out/'factors.csv')
    # save loadings
    loadings = pd.DataFrame(pca.components_.T, index=var_names, columns=[f'factor{i+1}' for i in range(n_factors)])
    loadings.to_csv(out/'loadings.csv')
    print('Factors and loadings saved to', out)
    return factors, loadings

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', default='results/data')
    parser.add_argument('--out-dir', default='results/factors')
    parser.add_argument('--n-factors', default=1, type=int)
    args = parser.parse_args()
    panel_csv = Path(args.in_dir)/'panel.csv'
    # variables present in panel that represent monthly-aggregated indicators
    var_names = ['ip','pmi','gst','electricity','credit']
    extract_factors(panel_csv, var_names, n_factors=args.n_factors, out_dir=args.out_dir)
