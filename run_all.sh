#!/usr/bin/env bash
set -e
python -m src.preprocess --data-dir data --out-dir results/data
python -m src.factor_extract --in-dir results/data --out-dir results/factors
python -m src.models --mode fit --data-dir results/data --factor-dir results/factors --out-dir results/models
python -m src.forecast_eval --models-dir results/models --out-dir results/metrics
python -m src.plots --results-dir results --out-dir results/figures
echo "Finished. Figures and metrics in results/"
