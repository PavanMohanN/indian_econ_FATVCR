<img width="1919" height="1079" alt="image" src="https://github.com/user-attachments/assets/75e21403-203a-469f-b44b-86e7bc644da2" />

FA-TVCR Nowcast pipeline
========================

Reproduce the analyses described in the manuscript (FA-TVCR nowcasting India, Q1-2018→Q1-2025).

Quick start:
1. Create virtualenv and install requirements:
   python -m venv venv; source venv/bin/activate
   pip install -r requirements.txt

2. Place raw data CSV/xlsx in `data/` or configure `src/data_fetch.py` to download automatically.

3. Run end-to-end:
   bash run_all.sh
   or open `examples/example_run.ipynb` and run cells.

Outputs:
- figures saved in `results/` (created by scripts)
- model objects saved to `results/models/`
- summary tables in CSV

See code docstrings for per-script options.
$ ./tree-md .

fa_tvcr_nowcast/                <- repo root
├─ README.md
├─ requirements.txt
├─ run_all.sh                   <- convenience runner
├─ src/
│  ├─ __init__.py
│  ├─ data_fetch.py            # download MoSPI & indicator files (placeholders)
│  ├─ preprocess.py           # construct quarter index, compute q/q, align panel
│  ├─ factor_extract.py       # PCA for high-frequency panel -> factor series
│  ├─ models.py               # OLS & FA-TVCR (Kalman MLEModel subclass)
│  ├─ forecast_eval.py        # rolling forecasts, RMSE/MAE, DM test, bootstraps
│  ├─ diagnostics.py          # residual tests, VIF, ACF plots
│  └─ plots.py                # figure routines to reproduce paper figures

