# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Python ML/DL project for Bitcoin price prediction (8 models). See `README.md` for full details.

### Dataset

The project requires `filtered_df.csv` (hourly Bitcoin data with ~70 features) which is **not committed to the repo**. To run any model, you must generate a synthetic dataset first:

```bash
python3 -c "
import numpy as np, pandas as pd
from datetime import datetime, timedelta
np.random.seed(42)
n=2000; start=datetime(2016,11,1)
dates=[start+timedelta(hours=i) for i in range(n)]
bp=700+np.cumsum(np.random.randn(n)*5); bp=np.abs(bp)+500
d={'date':dates,'listing_close':bp,'target_nexthour':np.roll(bp,-1),
   'target_log_return':np.log(np.roll(bp,-1)/bp),
   'percentage_return':(np.roll(bp,-1)-bp)/bp*100}
for w in [2,6,12,24]: d[f'ma_{w}']=pd.Series(bp).rolling(w,min_periods=1).mean().values
nas=5000+np.cumsum(np.random.randn(n)*2)
d.update({'NasClose/Last':nas,'NasOpen':nas+np.random.randn(n)*5,'NasHigh':nas+np.abs(np.random.randn(n)*10),'NasLow':nas-np.abs(np.random.randn(n)*10)})
gold=1300+np.cumsum(np.random.randn(n)*0.5)
d.update({'GClose/Last':gold,'GOpen':gold+np.random.randn(n)*2,'GHigh':gold+np.abs(np.random.randn(n)*5),'GLow':gold-np.abs(np.random.randn(n)*5),'GVolume':np.random.randint(1000,50000,n).astype(float)})
vix=15+np.random.randn(n)*3
d.update({'CLOSE':np.abs(vix),'OPEN':np.abs(vix+np.random.randn(n)*0.5),'HIGH':np.abs(vix+np.abs(np.random.randn(n))),'LOW':np.abs(vix-np.abs(np.random.randn(n)))})
for src in ['twitter','reddit','bitcointalk']:
    o=np.random.rand(n)*0.5+0.3; neg=np.random.rand(n)*0.3
    d[f'{src}_optimistic']=o; d[f'{src}_negative']=neg
    for lag in [1,5,12,24]: d[f'{src}_optimistic_lag{lag}']=np.roll(o,lag); d[f'{src}_negative_lag{lag}']=np.roll(neg,lag)
for i in range(10): d[f'feature_{i}']=np.random.randn(n)*10
df=pd.DataFrame(d); df.loc[df.index[-1],'target_nexthour']=df.loc[df.index[-1],'listing_close']+np.random.randn()*5
df.to_csv('filtered_df.csv',index=False); print(f'Generated: {df.shape}')
"
```

### Running models

- All models: `python3 run_all_models.py` (slow — runs all 8 models)
- Individual models: `python3 xgboost_model.py`, `python3 lightgbm_model.py`, etc.
- For quick validation, XGBoost or LightGBM are fastest (~2s each)
- Deep learning models (iTransformer, SE-GRN, CNN-LSTM, SOFTS, Times-FM) take longer, especially on CPU

### Matplotlib backend

When running in headless environments, set `matplotlib.use('Agg')` before importing pyplot, or use `MPLBACKEND=Agg` env var. Otherwise scripts that call `plt.show()` will fail.

### No test framework / no linter

This project has no automated tests, no linting configuration, and no CI/CD. Validation is done by running model scripts and checking metrics output. Syntax checking: `python3 -m py_compile <file.py>`.

### PyTorch

PyTorch CPU-only is sufficient. Install via `pip install torch --index-url https://download.pytorch.org/whl/cpu` to save disk space.
