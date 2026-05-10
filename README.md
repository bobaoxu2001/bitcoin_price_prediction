# Bitcoin Price Dynamics: A Forecasting & Model-Comparison Study

> A forecasting and model-comparison study for Bitcoin price dynamics using macro-market and social-sentiment features.
>
> **NYU Capstone — Group 36** · Sam Lai · Zexuan Yang · Yichao Yang · Ao Xu · 2024

---

## Executive Summary

This project investigates how well classical statistical models, gradient-boosted trees, and modern deep-learning architectures forecast hourly Bitcoin price dynamics when augmented with macro-market features (Nasdaq, Gold, VIX) and social sentiment (Twitter/X, Reddit, Bitcointalk). We evaluate eight models — ARIMA, XGBoost, LightGBM, SE-GRN, iTransformer, Times-FM, SOFTS, and CNN-LSTM — on ~68k hourly observations from Nov 2016 to Aug 2024.

The goal is *not* to claim a tradable price-prediction edge. The goal is to compare model families on a noisy, high-volatility asset, study how macro and sentiment features behave, and document the limits of pure price forecasting — which motivates the regime/behavior-focused work in my newer [Digital Asset Market Behavior Intelligence Platform](https://github.com/bobaoxu2001/Digital-Asset-Market-Behavior-Intelligence-Platform).

---

## Why This Project Matters

- **Bitcoin price formation is multi-factor.** Crypto markets do not move in isolation; they respond to equity-market risk appetite (Nasdaq), safe-haven flows (Gold), implied volatility (VIX), and crowd sentiment. Modeling these together is more honest than treating BTC as a univariate series.
- **Model choice has real cost.** Bigger and newer is not always better. We show large foundation-style models can underperform a tuned LightGBM on this task — a useful reminder for any quant or ML team picking architectures under deadline.
- **It exposes the limits of price-only modeling.** Even the best model here only narrows error; none give a stable edge once you account for noise, regime shifts, and the absence of execution costs. That negative result is the single most useful finding for a markets role.

---

## Key Findings

1. **Tree-based models are very strong baselines.** LightGBM achieved the lowest MAE (≈254) and XGBoost was close behind. Tuned gradient boosting on hand-engineered features remains a hard benchmark to beat on hourly BTC.
2. **SOFTS gave the lowest MSE** among tested models (≈183.7k), suggesting it controlled large errors better than the boosted trees, even though its average error (MAE) was higher.
3. **Larger ≠ better.** Times-FM and CNN-LSTM had the worst MSE despite higher capacity; capacity without inductive bias for this data hurts.
4. **Sentiment features add information, not magic.** Lagged Reddit/Bitcointalk/Twitter sentiment showed up in feature-importance rankings but did not change the relative model ordering or remove residual noise.
5. **Macro features matter.** Nasdaq Close/Open and short moving averages (`ma_2`, `ma_6`) consistently rank as top predictors, supporting the well-known crypto–tech-equity correlation.

---

## Model Comparison Summary

| Category | Model | MAE | MSE |
|----------|-------|-----|-----|
| Baseline | **LightGBM** | **254.37** | 258,918.80 |
| Baseline | XGBoost | 258.53 | 271,929.61 |
| Baseline | ARIMA | 1,602.19 | 4,847,913.23 |
| Proposed | **SOFTS** | 304.23 | **183,679.47** |
| Proposed | iTransformer | 1,948.00 | 8,123,500.00 |
| Proposed | CNN-LSTM | 1,941.32 | 6,730,000.00 |
| Proposed | SE-GRN | 2,377.06 | 883,273.00 |
| Proposed | Times-FM | 2,672.28 | 3,600,000.00 |

**Headline result:** *LightGBM achieved the lowest MAE, while SOFTS achieved the lowest MSE, suggesting different trade-offs between average error and large-error control.* No single model dominates on both metrics — the right choice depends on whether the downstream use cares more about typical error or tail-error containment.

---

## How This Connects to Digital Asset Market Behavior Analysis

For a digital-asset market-behavior role, this project is best read as foundational research, not a strategy. It demonstrates working knowledge of:

- **Price-movement modeling** on hourly BTC across linear, tree, and neural architectures.
- **Volatility & correlation analysis** — ACF/PACF, time-varying NASDAQ–BTC correlation (notable break during COVID-19), heteroscedasticity.
- **Macro risk proxy engineering** — Nasdaq, Gold, VIX aligned to a crypto-native frequency.
- **Sentiment feature engineering** — multi-platform optimistic/negative scores with 1h/5h/12h/24h lags.
- **Disciplined model comparison** — chronological splits (no leakage), MAE *and* MSE reported, walk-forward where applicable.
- **Honest negative results** — documenting that pure price prediction is fragile motivates the shift toward regime classification, event studies, and behavior analytics.

---

## Relevance to Digital Asset Market Behavior & Strategy

Mapping this project to a digital-asset market-behavior & strategy analyst remit:

| Analyst lens | What this repo demonstrates |
|---|---|
| Price-movement analysis | Hourly BTC modeling across 8 architectures with disciplined splits |
| Volatility analysis | ACF/PACF, time-varying volatility, heteroscedasticity, structural-break inspection |
| Sentiment feature engineering | Multi-platform sentiment with multi-lag features |
| Macro risk proxies | VIX (fear), Nasdaq (risk appetite), Gold (safe-haven) integrated into the feature set |
| Model comparison | Honest MAE/MSE table; explicit "best on what metric" framing |
| Limits of prediction | Negative result: pure price prediction is fragile — motivates regime/behavior framing |

---

## Relationship to the Digital Asset Market Behavior Intelligence Platform

This is a **companion / earlier research project**. My main project for digital-asset market-behavior work is the
[Digital Asset Market Behavior Intelligence Platform](https://github.com/bobaoxu2001/Digital-Asset-Market-Behavior-Intelligence-Platform), which deliberately moves *past* price prediction and covers the pieces missing here:

| Missing here | Covered in the newer platform |
|---|---|
| No on-chain activity / wallet clustering | Native on-chain metrics and flow analysis |
| No DeFi liquidity view | DEX / liquidity-pool depth and TVL signals |
| No event studies | Pre/post-event abnormal-return analysis |
| No regime classification | Volatility/correlation regime tagging |
| Pure-prediction framing | Behavior-, regime-, and strategy-insight framing |
| No exchange flow data | CEX flow and stablecoin proxies |

Read this repo as the "what I learned trying to predict BTC directly" prerequisite to that platform.

---

## Limitations

- **Bitcoin price prediction is intrinsically noisy.** Reported MAE/MSE are point-forecast accuracy on a single test window; they are *not* a trading-strategy P&L.
- **No transaction costs, slippage, fees, or borrow costs** are modeled. Translating any of these errors into a strategy would require execution modeling.
- **Sentiment data is noisy and platform-dependent.** Bot activity, sampling differences across Twitter / Reddit / Bitcointalk, and labeler variance (VADER vs. CryptoBERT) all bias the signal; we did not adversarially audit the sentiment pipeline.
- **No on-chain or exchange-flow data.** Wallet clustering, miner flows, stablecoin supply, CEX inflows/outflows are absent — exactly the data a behavior-focused analyst would want.
- **One test regime.** Test period is Jan–Aug 2024; results may not generalize to later regimes.
- **Historical performance does not guarantee future performance.**
- **This project is research and educational.** It is **not financial advice**, and nothing here should be used as a basis for live trading.

---

## How to Run

```bash
# 1. Install
pip install -r requirements.txt
# CPU-only PyTorch:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Get data
# `filtered_df.csv` is not committed (~68k rows). Either:
#   (a) Build it from sources (see Dataset Overview below + data_collection.py), or
#   (b) Generate a small synthetic dataset for code testing:
python3 data_collection.py     # writes a synthetic filtered_df.csv

# 3. Run a single model (fastest first)
python3 lightgbm_model.py
python3 xgboost_model.py

# 4. Run everything (slow on CPU)
python3 run_all_models.py
```

Outputs land in `figures/` and `results/` (gitignored).

---

## Project Structure

```
bitcoin_price_prediction/
├── README.md                    # This file
├── AGENTS.md                    # Dev environment notes
├── capstone1006_Group36.pdf     # Original capstone report
├── requirements.txt
├── docs/
│   ├── project_summary.md       # 1-page overview
│   ├── model_results.md         # Full results table + interpretation
│   └── interview_talking_points.md
├── data_collection.py           # Source pipeline + synthetic-data fallback
├── data_preprocessing.py        # Loading, cleaning, scaling, sequence creation
├── eda_analysis.py              # Correlation, ACF/PACF, volatility analysis
├── feature_selection.py         # LightGBM/XGBoost feature importance
├── visualization.py             # Plotting utilities
├── run_all_models.py            # Master script
├── arima_model.py               # Statistical baseline
├── xgboost_model.py             # Boosted-tree baseline
├── lightgbm_model.py            # Boosted-tree baseline (best MAE)
├── se_grn_model.py              # Squeeze-Excitation GRU
├── itransformer_model.py        # Inverted Transformer
├── times_fm_model.py            # Times-FM-style decoder
├── softs_model.py               # Series-cOre Fused TS (best MSE)
└── cnn_lstm_model.py            # CNN + BiLSTM + attention
```

---

## Dataset Overview

**Time range:** 2016-11-01 → 2024-08-22 (~68,000 hourly rows).

**Sources & features:**

| Group | Source | Features |
|---|---|---|
| Target | CryptoCompare / CoinGecko | `listing_close` (hourly BTC close) |
| Macro | Yahoo Finance ^IXIC | NASDAQ OHLC (daily → hourly forward-fill) |
| Macro | Yahoo Finance GC=F | Gold OHLCV (daily → hourly forward-fill) |
| Macro | Yahoo Finance ^VIX | VIX OHLC (daily → hourly forward-fill) |
| Sentiment | Twitter/X (snscrape / API) | `twitter_optimistic`, `twitter_negative` + 1h/5h/12h/24h lags |
| Sentiment | Reddit (PRAW) | `reddit_*` + same lags |
| Sentiment | Bitcointalk (scrape) | `bitcointalk_*` + same lags |
| Engineered | — | `target_nexthour`, `target_log_return`, `percentage_return`, `ma_2/6/12/24` |

**Preprocessing:** chronological train/test split (no shuffling), forward-fill then backward-fill for gaps, StandardScaler. See `data_preprocessing.py`.

---

## Models Implemented

### Baselines
| Model | File | Note |
|---|---|---|
| ARIMA | `arima_model.py` | `pmdarima` auto-order, walk-forward |
| XGBoost | `xgboost_model.py` | Early stopping, L1/L2 regularization |
| LightGBM | `lightgbm_model.py` | Histogram-based, leaf-wise growth — **best MAE** |

### Proposed deep models
| Model | File | Note | Reference |
|---|---|---|---|
| SE-GRN | `se_grn_model.py` | GRU + Squeeze-Excitation + attention | Zhang et al. |
| iTransformer | `itransformer_model.py` | Inverted Transformer for multivariate TS | Liu et al., 2023 |
| Times-FM | `times_fm_model.py` | Decoder-only patch tokenization, FM-inspired | Das et al., 2024 |
| SOFTS | `softs_model.py` | Series-cOre Fused TS — **best MSE** | Han et al., 2024 |
| CNN-LSTM | `cnn_lstm_model.py` | Multi-scale CNN + BiLSTM + attention | Shi et al., 2015 |

---

## Training Setup

- Train: 62,809 rows (pre-2024-01-01) · Test: 2024-01-01 → 2024-08-22
- Sequence length 100h (~4 days), batch 32, early stopping patience 15
- 10% of train held out for validation

---

## Future Work

- Update test window beyond 2024-08 to assess regime stability.
- Replace the pure price target with regime / volatility-bucket classification.
- Add on-chain features (active addresses, exchange netflow, stablecoin supply).
- Move from accuracy metrics to strategy metrics (Sharpe, max drawdown) with realistic cost modeling.

These directions are pursued in the [Digital Asset Market Behavior Intelligence Platform](https://github.com/bobaoxu2001/Digital-Asset-Market-Behavior-Intelligence-Platform).

---

## References

1. David Lee Kuo Chuen. *Handbook of Digital Currency.* Academic Press, 2015.
2. A. Das, W. Kong, R. Sen, Y. Zhou. "A decoder-only foundation model for time-series forecasting." *ICML 2024.*
3. L. Han, X.-Y. Chen, H.-J. Ye, D.-C. Zhan. "SOFTS: Efficient multivariate time series forecasting with series-core fusion." *NeurIPS 2024.*
4. Yong Liu et al. "iTransformer: Inverted transformers are effective for time series forecasting." *arXiv:2310.06625*, 2023.
5. Satoshi Nakamoto. "Bitcoin: A peer-to-peer electronic cash system." 2008.
6. Xingjian Shi et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." *NeurIPS*, 2015.
7. Jiawei Zhang, Limeng Cui, Fisher B. Gouza. "SeGen: Sample-ensemble genetic evolutional network model." *arXiv:1803.08631*, 2018.

---

## Authors

| Name | Email |
|---|---|
| Sam Lai | jl12560@nyu.edu |
| Zexuan Yang | zy3035@nyu.edu |
| Yichao Yang | yy5020@nyu.edu |
| Ao Xu | ax2183@nyu.edu |

> **Disclaimer:** Research and educational project. Not financial advice. Not a trading strategy.
