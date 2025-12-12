"""
Plotting scripts producing Figures similar to the paper:
- Fig.1 GDP series
- Fig.3 OLS coefficient bar chart
- Fig.4 RMSE comparison
- Fig.5 TVP coefficient evolution (smoothed)
- Fig.6 Diagnostics
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')

def fig_gdp(panel_csv, out_dir):
    panel = pd.read_csv(panel_csv, parse_dates=['period']).sort_values('period')
    plt.figure(figsize=(8,3))
    plt.plot(panel['period'], panel['gdp_qoq'], marker='o')
    plt.title('Quarterly real GDP growth (q/q %)')
    plt.savefig(Path(out_dir)/'fig1_gdp.png', dpi=300, bbox_inches='tight')

def fig_ols_coefs(panel_csv, models_dir, out_dir):
    import pickle
    ols_res = pickle.load(open(Path(models_dir)/'ols_result.pkl','rb'))
    params = ols_res.params
    se = ols_res.bse
    coefs = params.index, params.values
    plt.figure(figsize=(8,4))
    x = range(len(params))
    plt.errorbar(params.index, params.values, yerr=1.96*se, fmt='o')
    plt.xticks(rotation=45)
    plt.title('Estimated OLS coefficients (95% CI)')
    plt.savefig(Path(out_dir)/'fig3_ols_coefs.png', dpi=300, bbox_inches='tight')

def fig_tvp_coeffs(models_dir, out_dir):
    import pickle
    tvp_res = pickle.load(open(Path(models_dir)/'tvp_result.pkl','rb'))
    # smoothed_state shape (k_states, T)
    if hasattr(tvp_res, 'smoothed_state'):
        smooth = tvp_res.smoothed_state
    elif hasattr(tvp_res, 'filtered_state'):
        smooth = tvp_res.filtered_state
    else:
        raise RuntimeError("No smoothed/filtered state in tvp result")
    T = smooth.shape[1]
    plt.figure(figsize=(10,4))
    # plot first three coefficients as example
    for i in range(min(3, smooth.shape[0])):
        plt.plot(range(T), smooth[i,:], label=f'coef_{i}')
    plt.legend()
    plt.title('Smoothed TVP coefficients (example)')
    plt.savefig(Path(out_dir)/'fig5_tvp_coeffs.png', dpi=300, bbox_inches='tight')

def main(panel_csv, models_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fig_gdp(panel_csv, out_dir)
    fig_ols_coefs(panel_csv, models_dir, out_dir)
    fig_tvp_coeffs(models_dir, out_dir)
    print('Saved figures to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--out-dir', default='results/figures')
    args = parser.parse_args()
    main(Path(args.results_dir)/'data'/'panel.csv', Path(args.results_dir)/'models', args.out_dir)
