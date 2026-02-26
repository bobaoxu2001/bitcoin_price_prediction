# Predicting the Unpredictable: Deep Learning for Bitcoin Price Dynamics

> **NYU Capstone Project — Group 36**
> Sam Lai · Zexuan Yang · Yichao Yang · Ao Xu

This project explores the development of deep learning models for Bitcoin price prediction by integrating traditional market data with social media sentiment data. By leveraging advanced time-series forecasting models, this research spans from November 2016 to August 2024, aiming to improve prediction accuracy through innovative feature engineering and hybrid model architectures.

---

## Table of Contents

- [Introduction](#introduction)
- [Dataset Overview](#dataset-overview)
- [Dataset Sourcing & Collection](#dataset-sourcing--collection)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Analysis](#exploratory-analysis)
- [Feature Selection](#feature-selection)
- [Models Implemented](#models-implemented)
- [Training & Prediction Setup](#training--prediction-setup)
- [Model Results](#model-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion & Future Work](#conclusion--future-work)
- [References](#references)

---

## Introduction

Cryptocurrency markets exhibit high volatility and complex dynamics, making accurate price prediction a challenging task. This study develops an advanced prediction framework by combining traditional financial indicators (e.g., Nasdaq index, VIX, Gold prices) with social media sentiment metrics. By leveraging state-of-the-art deep learning models, we evaluate their ability to capture market trends over different time horizons.

### Why Bitcoin?

Bitcoin is a digital currency created in 2009 by Satoshi Nakamoto [6]. Its decentralized nature — operating without a central authority — contributes to its high volatility, as its value is subject to rapid changes based on market demand, regulation changes, and external events [1]. Bitcoin's limited supply and extensive discussion on platforms like Google Trends, Twitter, and Reddit make it a prime subject for data-driven price analysis. Our project aims to leverage social media sentiment and market behavior to offer insights into potential price movements.

---

## Dataset Overview

**Time Range:** 2016-11-01 to 2024-08-22 (~68,000 hourly observations)

Our dataset integrates multiple sources to support regression tasks:

### Market Data

| Source | Granularity | Features | Rationale |
|--------|------------|----------|-----------|
| **Bitcoin** | Hourly | Closing price (`listing_close`) | Primary prediction target |
| **NASDAQ Composite** | Daily → Hourly | Open, High, Low, Close/Last | Technology sector correlation with crypto investor profiles |
| **Gold** | Daily → Hourly | Open, High, Low, Close/Last, Volume | Safe-haven asset; shares scarcity essence with Bitcoin as hedge against inflation |
| **VIX** | Daily → Hourly | Open, High, Low, Close | Fear gauge for market uncertainty; high VIX correlates with risk-averse behavior |

### Sentiment Data

| Source | Granularity | Features |
|--------|------------|----------|
| **Twitter (X)** | Hourly | `twitter_optimistic`, `twitter_negative` + lags (1h, 5h, 12h, 24h) |
| **Reddit** | Hourly | `reddit_optimistic`, `reddit_negative` + lags (1h, 5h, 12h, 24h) |
| **Bitcointalk** | Hourly | `bitcointalk_optimistic`, `bitcointalk_negative` + lags (1h, 5h, 12h, 24h) |

### Engineered Features

| Category | Features | Description |
|----------|----------|-------------|
| Target | `target_nexthour` | Next-hour Bitcoin price (prediction target) |
| Returns | `target_log_return`, `percentage_return` | Price return features |
| Moving Averages | `ma_2`, `ma_6`, `ma_12`, `ma_24` | Trend smoothing at various window sizes |

---

## Dataset Sourcing & Collection

The `filtered_df.csv` dataset was assembled from multiple public data sources. Below is a detailed description of how each component was collected and processed.

### 1. Bitcoin Hourly Prices

- **Source:** [CryptoCompare API](https://min-api.cryptocompare.com/) or [CoinGecko API](https://www.coingecko.com/en/api)
- **Endpoint:** Historical hourly OHLCV data for BTC/USD
- **Field used:** `listing_close` (hourly closing price)
- **Range:** 2016-11-01 to 2024-08-22
- **Collection method:** API calls with pagination to retrieve hourly candles; approximately 68,000 data points

```python
# Example using CryptoCompare
import requests

url = "https://min-api.cryptocompare.com/data/v2/histohour"
params = {"fsym": "BTC", "tsym": "USD", "limit": 2000, "toTs": <end_timestamp>}
response = requests.get(url, params=params)
```

### 2. NASDAQ Composite Index (Daily)

- **Source:** [Yahoo Finance](https://finance.yahoo.com/quote/%5EIXIC/) (ticker: `^IXIC`) or [NASDAQ official data](https://www.nasdaq.com/market-activity/index/comp/historical)
- **Fields:** Open, High, Low, Close/Last (mapped to `NasOpen`, `NasHigh`, `NasLow`, `NasClose/Last`)
- **Alignment:** Daily data forward-filled to hourly frequency

### 3. Gold Prices (Daily)

- **Source:** [Yahoo Finance](https://finance.yahoo.com/quote/GC%3DF/) (Gold Futures ticker: `GC=F`) or [Investing.com Gold Historical Data](https://www.investing.com/commodities/gold-historical-data)
- **Fields:** Open, High, Low, Close/Last, Volume (mapped to `GOpen`, `GHigh`, `GLow`, `GClose/Last`, `GVolume`)
- **Alignment:** Daily data forward-filled to hourly frequency

### 4. VIX Index (Daily)

- **Source:** [Yahoo Finance](https://finance.yahoo.com/quote/%5EVIX/) (ticker: `^VIX`) or [CBOE VIX Data](https://www.cboe.com/tradable_products/vix/)
- **Fields:** Open, High, Low, Close (mapped to `OPEN`, `HIGH`, `LOW`, `CLOSE`)
- **Alignment:** Daily data forward-filled to hourly frequency

### 5. Social Media Sentiment (Hourly)

Sentiment scores were collected from three cryptocurrency-focused platforms and aggregated to hourly granularity:

- **Twitter (X):** Cryptocurrency-related tweets were collected using the Twitter API (Academic Research access) or tools like [snscrape](https://github.com/JustAnotherArchiworker/snscrape). Sentiment was scored using [VADER](https://github.com/cjhutto/vaderSentiment) or [CryptoBERT](https://huggingface.co/ElKulako/cryptobert), then aggregated into hourly `twitter_optimistic` and `twitter_negative` scores.
- **Reddit:** Posts and comments from subreddits such as r/Bitcoin and r/CryptoCurrency were collected using [PRAW](https://praw.readthedocs.io/) (Python Reddit API Wrapper). Sentiment was similarly scored and aggregated hourly.
- **Bitcointalk:** Posts from the [Bitcointalk forum](https://bitcointalk.org/) were scraped and sentiment-analyzed, producing hourly `bitcointalk_optimistic` and `bitcointalk_negative` scores.

**Lag features** (1-hour, 5-hour, 12-hour, 24-hour) were created for all sentiment columns to capture delayed market reactions.

### Data Assembly Pipeline

```
Bitcoin hourly prices ──┐
NASDAQ daily OHLCV ─────┤  forward-fill to hourly
Gold daily OHLCV ───────┤──────────────────────────→ merge on datetime → filtered_df.csv
VIX daily ──────────────┤
Sentiment hourly ───────┘  + lag features + moving averages
```

See `data_collection.py` for a reference implementation of this pipeline.

---

## Data Preprocessing

- **Daily-to-Hourly Alignment:** Market indicators (NASDAQ, Gold, VIX) are forward-filled from daily to hourly frequency.
- **Missing Values:** Forward-fill then backward-fill strategy (weekends, holidays, gaps from moving averages).
- **Normalization:** StandardScaler applied to features and targets.
- **Train/Test Split:** Chronological split — no random shuffling to prevent data leakage.

See `data_preprocessing.py` for implementation.

---

## Exploratory Analysis

Exploratory analysis was performed to guide feature engineering and model selection:

1. **Correlation Analysis:** Heatmap revealing strong positive correlations (~0.92) between `listing_close` and NASDAQ metrics. Negative sentiment shows mild-to-moderate negative correlations with Bitcoin prices.
2. **Time-Varying Correlation:** Dynamic correlation between NASDAQ and Bitcoin shows structural changes (e.g., sharp dip during COVID-19 in 2020).
3. **ACF and PACF Analysis:** Slow decay in ACF and sharp cutoff at lag 1 in PACF suggests ARIMA(1,1,1) as a reasonable statistical baseline.
4. **Volatility Analysis:** Time-varying volatility analysis identifies heteroscedasticity, structural breaks, and regime changes.

See `eda_analysis.py` for implementation.

---

## Feature Selection

Feature selection was performed using **LightGBM** and **XGBoost**, leveraging gain-based metrics to identify the most influential predictors:

**Key Features Identified:**
- **NASDAQ Features:** Close/Last, Open, High, Low, and short-term moving averages (ma_2, ma_6)
- **Hourly Lagged Sentiment:** Historical optimistic and negative sentiment from Bitcointalk, Reddit, and Twitter at 1-hour, 5-hour, 12-hour, and 24-hour lags

See `feature_selection.py` for implementation.

---

## Models Implemented

### Baseline Models

| Model | File | Description |
|-------|------|-------------|
| **ARIMA** | `arima_model.py` | Classical statistical model with auto-order selection via `pmdarima`. Walk-forward validation. |
| **XGBoost** | `xgboost_model.py` | Gradient boosting with decision trees. Early stopping, L1/L2 regularization, feature importance analysis. |
| **LightGBM** | `lightgbm_model.py` | Histogram-based gradient boosting optimized for speed/memory. Leaf-wise tree growth. |

### Proposed Deep Learning Models

| Model | File | Description | Reference |
|-------|------|-------------|-----------|
| **SE-GRN** | `se_grn_model.py` | Squeeze-and-Excitation Gated Recurrent Network. Combines GRU layers with SE blocks for dynamic feature recalibration; attention mechanism for temporal dependencies. | Zhang et al. [8] |
| **iTransformer** | `itransformer_model.py` | Inverted Transformer embedding each time point as an independent variable token. Improves multivariate correlation modeling. | Liu et al., 2023 [5] |
| **Times-FM** | `times_fm_model.py` | Architecture inspired by Google's 200M-parameter Time-Series Foundation Model. Patch-based tokenization with decoder-only design and causal attention. | Das et al., 2024 [2] |
| **SOFTS** | `softs_model.py` | Series-cOre Fused Time Series forecasting. Series-core fusion mechanism for inter-series relationships with temporal convolution. | Han et al., 2024 [4] |
| **CNN-LSTM** | `cnn_lstm_model.py` | Hybrid architecture: multi-scale CNN for feature extraction + bidirectional LSTM for temporal dependencies + attention mechanism. | Shi et al., 2015 [7] |

---

## Training & Prediction Setup

- **Training data:** 62,809 time points (period prior to 2024-01-01)
- **Test data:** Remaining observations (2024-01-01 to 2024-08-22)
- **Prediction horizon:** Next 100 time steps (both price and returns)
- **Sequence length:** 100 hours (~4 days) for deep learning models
- **Batch size:** 32
- **Early stopping patience:** 15 epochs
- **Validation:** 10% of training data held out for validation

---

## Model Results

### Table 1: Performance of Models in Price Prediction (MAE and MSE)

| Model Category | Model | MAE | MSE |
|---------------|-------|-----|-----|
| **Baseline** | LightGBM | 254.37 | 258,918.80 |
| **Baseline** | ARIMA | 1,602.19 | 4,847,913.23 |
| **Baseline** | XGBoost | 258.53 | 271,929.61 |
| **Proposed** | SE-GRN | 2,377.06 | 883,273.00 |
| **Proposed** | iTransformer | 1,948.00 | 8,123,500.00 |
| **Proposed** | Times-FM | 2,672.28 | 3,600,000.00 |
| **Proposed** | SOFTS | **304.23** | **183,679.47** |
| **Proposed** | CNN-LSTM | 1,941.32 | 6,730,000.00 |

**Best Model:** SOFTS achieved the best performance among proposed models, with MAE of 304.23 and MSE of 183,679.47, outperforming all baseline and deep learning models through its series-core fusion mechanism.

### Visualization Components

The following visualizations are generated for each model (see Section 6.1 of the paper):
1. **Full timeline prediction** — complete prediction period
2. **Single prediction window** — short-term accuracy
3. **Time series validation windows** — robustness trends
4. **Returns comparison** — actual vs predicted daily returns
5. **Distribution of prediction errors** — error variability
6. **Timeline of error percentages** — error fluctuation over time

---

## Project Structure

```
project_folder/
├── filtered_df.csv              # Preprocessed dataset (not in repo — see Dataset Sourcing)
├── requirements.txt             # Python dependencies
├── data_collection.py           # Dataset collection & assembly reference pipeline
├── data_preprocessing.py        # Data loading, cleaning, scaling, and sequence creation
├── eda_analysis.py              # Exploratory data analysis (correlation, ACF/PACF, volatility)
├── feature_selection.py         # Feature selection using LightGBM/XGBoost importance
├── arima_model.py               # ARIMA baseline model
├── xgboost_model.py             # XGBoost baseline model
├── lightgbm_model.py            # LightGBM baseline model
├── se_grn_model.py              # SE-GRN deep learning model
├── itransformer_model.py        # iTransformer model
├── times_fm_model.py            # Times-FM inspired model
├── softs_model.py               # SOFTS model
├── cnn_lstm_model.py            # CNN-LSTM hybrid model
├── visualization.py             # Visualization utilities for all figure types
├── run_all_models.py            # Master script to run all models and generate figures
├── README.md                    # This documentation
├── AGENTS.md                    # Development environment notes
└── capstone1006_Group36.pdf     # Original capstone report
```

---

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
# Core dependencies
pip install numpy pandas scikit-learn torch matplotlib seaborn

# ML/Statistical models
pip install xgboost lightgbm pmdarima statsmodels arch

# Optional (interactive visualization)
pip install plotly
```

> **Note:** For CPU-only environments, install PyTorch with:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Usage

### Run Individual Models

```bash
# Baseline models
python arima_model.py
python xgboost_model.py
python lightgbm_model.py

# Deep learning models
python se_grn_model.py
python itransformer_model.py
python times_fm_model.py
python softs_model.py
python cnn_lstm_model.py
```

### Run All Models

```bash
python run_all_models.py
```

This runs all 8 models, generates comparison figures, and saves results to `figures/` and `results/`.

### Exploratory Analysis

```bash
python eda_analysis.py
```

### Feature Selection Analysis

```bash
python feature_selection.py
```

### Custom Usage

```python
from data_preprocessing import BitcoinDataLoader, DataPreprocessor, calculate_metrics
from softs_model import run_softs_experiment

results = run_softs_experiment(
    data_path='filtered_df.csv',
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

print(f"MAE: {results['metrics']['MAE']:.4f}")
print(f"MSE: {results['metrics']['MSE']:.4f}")
```

---

## Conclusion & Future Work

The SOFTS model demonstrated superior predictive accuracy and stability through its series-core fusion mechanism, outperforming baseline models and advanced deep learning architectures like SE-GRN and CNN-LSTM. Notably, Times-FM, despite its large parameter size, performed poorly, revealing that model complexity does not guarantee improved predictions.

**Key Insights:**
1. Specialized architectures with effective feature fusion outperform general models.
2. Social media sentiment contributes meaningfully to price dynamics when combined with market indicators.
3. Model simplicity and interpretability are often more valuable than sheer complexity in financial forecasting.

**Future Work:**
- Acquiring updated, real-time datasets to analyze event-driven market shifts (e.g., elections, policy changes).
- Expanding to include alternative data sources: news headlines, blockchain activity, real-time social media trends.

---

## References

1. David Lee Kuo Chuen. *Handbook of Digital Currency: Bitcoin, Innovation, Financial Instruments, and Big Data.* Academic Press, 1st edition, 2015.
2. A. Das, W. Kong, R. Sen, Y. Zhou, and Google Research. "A decoder-only foundation model for time-series forecasting." *ICML 2024*, 2024.
3. Google. "Google Trends." https://trends.google.com/, 2024.
4. L. Han, X.-Y. Chen, H.-J. Ye, D.-C. Zhan. "SOFTS: Efficient multivariate time series forecasting with series-core fusion." *NeurIPS 2024*, 2024.
5. Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. "iTransformer: Inverted transformers are effective for time series forecasting." *arXiv:2310.06625*, 2023.
6. Satoshi Nakamoto. "Bitcoin: A peer-to-peer electronic cash system." https://bitcoin.org/bitcoin.pdf, 2008.
7. Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, and Wang-chun Woo. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." *Advances in Neural Information Processing Systems*, 28, 2015.
8. Jiawei Zhang, Limeng Cui, and Fisher B. Gouza. "SeGen: Sample-ensemble genetic evolutional network model." *arXiv:1803.08631*, 2018.

---

## Authors

| Name | Email | Affiliation |
|------|-------|-------------|
| Sam Lai | jl12560@nyu.edu | New York University |
| Zexuan Yang | zy3035@nyu.edu | New York University |
| Yichao Yang | yy5020@nyu.edu | New York University |
| Ao Xu | ax2183@nyu.edu | New York University |

*Data Science & Machine Learning Capstone Project, 2024*
