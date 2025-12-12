"""
Models: OLSBaseline and TVPRegression (MAximum-likelihood Kalman-based TVP).

Usage as a script for fitting:
python -m src.models --mode fit --data-dir results/data --factor-dir results/factors --out-dir results/models
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel

class OLSBaseline:
    def __init__(self, y, X):
        self.y = y
        self.X = sm.add_constant(X)
    def fit(self):
        mod = sm.OLS(self.y, self.X).fit(cov_type='HC1')
        return mod

class TVPRegression(MLEModel):
    """
    Time-Varying Parameter regression via state-space MLEModel.
    State vector is coefficients (k x 1). State equation: beta_t = beta_{t-1} + u_t.
    Measurement: y_t = x_t' beta_t + eps_t.
    Parameters to estimate: sigma2_eps (obs var) and diag(Q) (process noise) optionally.
    For stability we parameterize process noise as vector of variances.
    """
    def __init__(self, endog, exog, k_states=None, **kwargs):
        # exog: (T x k) array without constant. We include intercept as exog column if desired.
        endog = np.asarray(endog)
        exog = np.asarray(exog)
        k = exog.shape[1]
        super().__init__(endog, k_states=k, **kwargs)
        self.exog = exog
        self.k_exog = k
        # state design matrix changes over time: Z_t = exog[t,:] row
        # We'll supply time-varying design in 'design' method via self['design']
        # initialise state_transition as identity; state_cov will be diagonal from params
        self['state_intercept'] = np.zeros(self.k_states)
        self.initialize_approximate_diffuse()
    def start_params(self):
        # starting parameters: log(obs_var) and log(process variances vector)
        start = np.r_[np.log(np.var(self.endog)), np.log(np.repeat(0.01, self.k_exog))]
        return start
    def transform_params(self, params):
        # params: [log_sigma2, log_q1, log_q2, ...]
        sigma2 = np.exp(params[0])
        qs = np.exp(params[1:1+self.k_exog])
        return sigma2, qs
    def untransform_params(self, params_trans):
        sigma2, qs = params_trans
        return np.r_[np.log(sigma2), np.log(qs)]
    def update(self, params, **kwargs):
        sigma2, qs = self.transform_params(params)
        # update observation covariance
        self.ssm['obs_cov'] = np.array([[sigma2]])
        # update state covariance Q (k x k)
        Q = np.diag(qs)
        self.ssm['state_cov'] = Q
        # update design matrix time-varying
        design = self.exog.reshape(self.nobs, 1, self.k_exog)
        self.ssm['design'] = design  # shape (nobs, k_endog, k_states)
    def loglike(self, params, **kwargs):
        return super().loglike(params, **kwargs)

    def fit_mle(self, disp=False, maxiter=200):
        res = super().fit(disp=disp, maxiter=maxiter)
        return res

def prepare_X(panel_df):
    # build regressor matrix Xt for each quarter
    # order: [gfcf, private_cons, gov, manuf, construction, net_exports, factor1]
    exog_names = ['gfcf','private_cons','gov','manuf','construction','net_exports','factor1']
    X = panel_df[exog_names].fillna(method='ffill').values
    return X, exog_names

def fit_models(panel_csv, factors_csv, out_dir):
    panel = pd.read_csv(panel_csv, parse_dates=['period']).sort_values('period')
    factors = pd.read_csv(factors_csv, index_col=0, parse_dates=True)
    # merge factor into panel on period
    panel = panel.set_index('period').join(factors, how='left').reset_index()
    # dependent variable
    y = panel['gdp_qoq'].values
    X, exog_names = prepare_X(panel)
    # Fit OLS baseline
    ols = OLSBaseline(y, X[:, :6])  # exclude factor for OLS baseline (option)
    ols_res = ols.fit()
    # Fit TVP with factor augmented X (includes factor)
    tvp = TVPRegression(y, X, k_states=X.shape[1])
    tvp_res = tvp.fit_mle(disp=False)
    tvp_smooth = tvp_res.smooth()  # attaches smoother results
    # Save results
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(out/'ols_result.pkl','wb') as f:
        pickle.dump(ols_res, f)
    with open(out/'tvp_result.pkl','wb') as f:
        pickle.dump(tvp_res, f)
    print('Saved model results to', out)
    return ols_res, tvp_res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='fit')
    parser.add_argument('--data-dir', default='results/data')
    parser.add_argument('--factor-dir', default='results/factors')
    parser.add_argument('--out-dir', default='results/models')
    args = parser.parse_args()
    panel_csv = Path(args.data_dir)/'panel.csv'
    factors_csv = Path(args.factor_dir)/'factors.csv'
    fit_models(panel_csv, factors_csv, args.out_dir)
