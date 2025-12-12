"""
Rolling-window forecast evaluation, RMSE/MAE computation, and Diebold-Mariano test.
Saves metrics to CSV in out_dir.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import medcouple

def rmse(y, yhat):
    return np.sqrt(np.mean((y-yhat)**2))

def rolling_nowcast(y, model_predict_func, window=24):
    # model_predict_func(t) should return one-step ahead forecast for observation at t
    T = len(y)
    preds = np.full(T, np.nan)
    for t in range(window, T):
        preds[t] = model_predict_func(t)
    return preds

def diebold_mariano(e1, e2, h=1, alternative='two-sided'):
    # e1,e2 are loss series (e.g., squared errors)
    d = e1 - e2
    mean_d = np.nanmean(d)
    # Newey-West style long-run var (lag = h-1)
    lag = max(1, h-1)
    cov = sm.tsa.stattools.acf(d, nlags=lag, fft=False)
    # compute DM stat (simple approximate)
    denom = np.nanvar(d) / np.sum(~np.isnan(d))
    dm_stat = mean_d / np.sqrt(denom)
    # p-value using normal approx
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p

def evaluate_models(panel_csv, models_dir, out_dir, window=24):
    panel = pd.read_csv(panel_csv, parse_dates=['period']).sort_values('period')
    y = panel['gdp_qoq'].values
    # Load OLS and TVP fitted objects
    import pickle
    ols_res = pickle.load(open(Path(models_dir)/'ols_result.pkl','rb'))
    tvp_res = pickle.load(open(Path(models_dir)/'tvp_result.pkl','rb'))
    # create simple predict functions:
    def ols_predict(t):
        # use fitted ols parameters and available X up to t-1
        X = panel.iloc[:t].copy()
        Xmat = sm.add_constant(X[['gfcf','private_cons','gov','manuf','construction','net_exports']].iloc[-1]).values
        return float(np.dot(ols_res.params, Xmat))
    # For tvp, use smoothed state at t-1 to predict t (one-step)
    # tvp_res is a statespace result object; use tvp_res.filtered_state or smoothed_state if present
    try:
        smoothed = tvp_res.filtered_state
    except Exception:
        smoothed = None

    def tvp_predict(t):
        # pick most recent smoothed beta at t-1 if available else use tvp_res.params
        Xrow = panel.iloc[t-1][['gfcf','private_cons','gov','manuf','construction','net_exports','factor1']].values
        if hasattr(tvp_res, 'smoothed_state'):
            beta = tvp_res.smoothed_state[:, t-1]
        elif hasattr(tvp_res, 'filtered_state'):
            beta = tvp_res.filtered_state[:, t-1]
        else:
            # fallback: use zeros
            beta = np.zeros(len(Xrow))
        return float(np.dot(beta, Xrow))

    # rolling preds
    preds_ols = rolling_nowcast(y, ols_predict, window=window)
    preds_tvp = rolling_nowcast(y, tvp_predict, window=window)

    # compute metrics on overlapping out-of-sample segment
    mask = ~np.isnan(preds_ols) & ~np.isnan(preds_tvp)
    y_oos = y[mask]
    p_ols = preds_ols[mask]
    p_tvp = preds_tvp[mask]

    metrics = {
        'rmse_ols': rmse(y_oos, p_ols),
        'rmse_tvp': rmse(y_oos, p_tvp),
        'mae_ols': np.mean(np.abs(y_oos-p_ols)),
        'mae_tvp': np.mean(np.abs(y_oos-p_tvp)),
    }
    # DM test on squared errors
    e1 = (y_oos - p_ols)**2
    e2 = (y_oos - p_tvp)**2
    dm_stat, dm_p = diebold_mariano(e1, e2)
    metrics.update({'dm_stat': dm_stat, 'dm_p': dm_p})
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out/'forecast_metrics.csv', index=False)
    print('Saved metrics to', out/'forecast_metrics.csv')
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='results/data')
    parser.add_argument('--models-dir', default='results/models')
    parser.add_argument('--out-dir', default='results/metrics')
    parser.add_argument('--window', default=24, type=int)
    args = parser.parse_args()
    panel_csv = Path(args.data_dir)/'panel.csv'
    evaluate_models(panel_csv, args.models_dir, args.out_dir, window=args.window)
