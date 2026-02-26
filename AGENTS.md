# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python ML/DL project: "Predicting the Unpredictable: Deep Learning for Bitcoin Price Dynamics" — NYU Capstone, Group 36. Implements 8 models for hourly Bitcoin price prediction. See `README.md` for full details.

### Dataset

`filtered_df.csv` is **not committed** (gitignored — too large). Generate synthetic data for testing:

```bash
python3 data_collection.py
```

This creates a 2000-row synthetic dataset sufficient for running all models. For the real dataset (~68k rows), see the sourcing documentation in `README.md` and `data_collection.py`.

### Running models

- Individual models: `python3 xgboost_model.py`, `python3 softs_model.py`, etc.
- All models: `python3 run_all_models.py` (slow — runs all 8)
- For quick validation, **XGBoost** or **LightGBM** are fastest (~2-3s each)
- Deep learning models are slower on CPU. Use smaller params for testing:
  - SOFTS is especially slow due to 4D tensor ops; use `seq_length=20` for quick tests
- EDA: `python3 eda_analysis.py`
- Feature selection: `python3 feature_selection.py`

### Matplotlib backend

Set `MPLBACKEND=Agg` or `matplotlib.use('Agg')` before importing pyplot in headless environments. The existing scripts already handle this.

### No test framework / no linter

No automated tests, no linting config, no CI/CD. Validation: run model scripts and check metrics output. Syntax check: `python3 -m py_compile <file.py>`.

### PyTorch

CPU-only is sufficient. Install via: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
