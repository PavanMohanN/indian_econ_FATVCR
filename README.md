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
