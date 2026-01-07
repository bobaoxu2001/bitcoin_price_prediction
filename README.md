# Bitcoin Price Prediction: A Comprehensive Time Series Forecasting Study

This project implements multiple machine learning and deep learning models for Bitcoin price prediction, combining traditional market indicators with social media sentiment data.

## Project Overview

**Objective:** Predict hourly Bitcoin prices using multimodal data including market indicators, cross-asset correlations, and social media sentiment.

**Time Range:** November 2016 - August 2024 (~68,000 hourly observations)

**Key Innovation:** Integration of sentiment data from multiple social platforms (Twitter, Reddit, Bitcointalk) with traditional financial indicators (NASDAQ, Gold, VIX).

## Dataset Description

### Features (70 total)
| Category | Features | Description |
|----------|----------|-------------|
| Target | `target_nexthour` | Next hour Bitcoin price (prediction target) |
| Price | `listing_close` | Current Bitcoin hourly closing price |
| Returns | `target_log_return`, `percentage_return` | Price returns |
| Moving Averages | `ma_2`, `ma_6`, `ma_12`, `ma_24` | Various window sizes |
| NASDAQ | `NasClose/Last`, `NasOpen`, `NasHigh`, `NasLow` | Stock market correlation |
| Gold | `GClose/Last`, `GOpen`, `GHigh`, `GLow`, `GVolume` | Safe-haven asset |
| VIX | `CLOSE`, `OPEN`, `HIGH`, `LOW` | Market volatility index |
| Twitter Sentiment | `twitter_optimistic`, `twitter_negative` + lags | Social sentiment |
| Reddit Sentiment | `reddit_optimistic`, `reddit_negative` + lags | Community sentiment |
| Bitcointalk Sentiment | `bitcointalk_optimistic`, `bitcointalk_negative` + lags | Crypto community |

### Data Preprocessing
- **Daily-to-Hourly Alignment:** Market indicators (NASDAQ, Gold, VIX) are forward-filled from daily to hourly frequency
- **Missing Values:** Forward-fill then backward-fill strategy
- **Normalization:** StandardScaler for features and targets

## Models Implemented

### Baseline Models

#### 1. ARIMA (`arima_model.py`)
Classical statistical time series model using autoregression, differencing, and moving averages.
- Auto-order selection using `pmdarima`
- Walk-forward validation

#### 2. XGBoost (`xgboost_model.py`)
Gradient boosting with decision trees.
- Early stopping with validation set
- Feature importance analysis
- L1/L2 regularization

#### 3. LightGBM (`lightgbm_model.py`)
Optimized gradient boosting with histogram-based learning.
- Leaf-wise tree growth
- Feature importance (gain-based)
- Lower memory footprint

### Advanced Deep Learning Models

#### 4. SE-GRN (`se_grn_model.py`)
Squeeze-and-Excitation Gated Recurrent Network.
- **SE Block:** Channel-wise feature recalibration
- **GRU Layers:** Temporal pattern learning
- **Attention Mechanism:** Focus on important timesteps

#### 5. iTransformer (`itransformer_model.py`)
Inverted Transformer for multivariate time series.
- **Feature-wise Attention:** Attention across variables instead of time
- **Better Multivariate Correlation:** Captures cross-feature dependencies
- Based on Liu et al., 2023

#### 6. Times-FM Inspired (`times_fm_model.py`)
Architecture inspired by Google's Time Series Foundation Model.
- **Patch-based Tokenization:** Divide sequences into patches
- **Decoder-only Design:** Causal attention for forecasting
- Based on Das et al., 2024

#### 7. SOFTS (`softs_model.py`)
Series-cOre Fused Time Series forecasting.
- **Series-Core Fusion:** Captures global cross-series patterns
- **Temporal Convolution:** Local pattern extraction
- **Adaptive Aggregation:** Dynamic feature weighting

#### 8. CNN-LSTM (`cnn_lstm_model.py`)
Hybrid convolutional-recurrent architecture.
- **Multi-scale CNN:** Feature extraction at multiple scales
- **Bidirectional LSTM:** Long-range dependencies
- **Attention:** Temporal focus mechanism

## Project Structure

```
project_folder/
├── filtered_df.csv          # Preprocessed dataset
├── data_preprocessing.py    # Data loading and preprocessing utilities
├── arima_model.py          # ARIMA baseline
├── xgboost_model.py        # XGBoost baseline
├── lightgbm_model.py       # LightGBM baseline
├── se_grn_model.py         # SE-GRN deep learning model
├── itransformer_model.py   # iTransformer model
├── times_fm_model.py       # Times-FM inspired model
├── softs_model.py          # SOFTS model
├── cnn_lstm_model.py       # CNN-LSTM hybrid model
├── visualization.py        # Comprehensive visualization functions
├── run_all_models.py       # Script to run all models and generate figures
└── README.md               # This documentation
```

## Installation

### Requirements
```bash
pip install numpy pandas scikit-learn torch matplotlib seaborn
pip install xgboost lightgbm pmdarima statsmodels arch
```

### Optional (for interactive visualization)
```bash
pip install plotly
```

## Reproducing Report Figures

All figures from the capstone report can be reproduced using the `visualization.py` module:

```python
from visualization import generate_all_figures, plot_model_comparison

# After running a model experiment
from itransformer_model import run_itransformer_experiment

results = run_itransformer_experiment()

# Generate all figures for this model
generate_all_figures(results, 'iTransformer', output_dir='figures')
```

### Available Figure Types
- **Full Timeline Prediction**: Complete actual vs predicted time series
- **Last 7/30 Days Analysis**: Zoomed-in recent predictions
- **From 2024 Onwards**: Analysis of 2024 data
- **Error Distribution**: Histogram of prediction errors
- **Returns Comparison**: Actual vs predicted returns
- **Scatter Plot**: Actual vs predicted values
- **Training Curves**: Loss progression during training
- **Feature Importance**: For tree-based models

## Usage

Each model file is self-contained and can be run independently:

```bash
# Run baseline models
python arima_model.py
python xgboost_model.py
python lightgbm_model.py

# Run advanced models
python se_grn_model.py
python itransformer_model.py
python times_fm_model.py
python softs_model.py
python cnn_lstm_model.py

# Run ALL models and generate all figures
python run_all_models.py
```

### Custom Usage Example

```python
from data_preprocessing import BitcoinDataLoader, DataPreprocessor, calculate_metrics
from itransformer_model import iTransformer, run_itransformer_experiment

# Run with custom parameters
results = run_itransformer_experiment(
    data_path='filtered_df.csv',
    train_ratio=0.8,
    seq_length=100,
    d_model=64,
    n_heads=4,
    n_layers=3,
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

# Access results
print(f"MSE: {results['metrics']['MSE']:.4f}")
print(f"MAE: {results['metrics']['MAE']:.4f}")
```

## Evaluation Metrics

All models are evaluated using:

| Metric | Description |
|--------|-------------|
| **MSE** | Mean Squared Error |
| **MAE** | Mean Absolute Error |
| **RMSE** | Root Mean Squared Error |
| **MAPE** | Mean Absolute Percentage Error |
| **Direction Accuracy** | % of correct direction predictions |

## Key Design Decisions

### 1. Time Series Split
- Chronological train/test split to prevent data leakage
- No random shuffling of time series data
- Walk-forward validation for robust evaluation

### 2. Sequence Length
- Default: 100 hours (~4 days of hourly data)
- Captures intraday patterns and short-term trends

### 3. Feature Engineering
- Lag features for sentiment (1, 5, 12, 24 hours)
- Moving averages for trend smoothing
- Log returns for stationarity

### 4. Regularization
- Dropout in neural networks
- Early stopping based on validation loss
- Weight decay in optimizers

## Technical Highlights

### For Hiring Managers
This project demonstrates:

1. **Software Engineering Best Practices**
   - Clean, modular code structure
   - Comprehensive documentation
   - Consistent coding style (PEP 8)
   - Proper error handling

2. **Machine Learning Expertise**
   - Understanding of time series forecasting challenges
   - Proper validation strategies (no data leakage)
   - Feature engineering for financial data
   - Model selection and comparison

3. **Deep Learning Proficiency**
   - Implementation of state-of-the-art architectures
   - Understanding of attention mechanisms
   - Experience with PyTorch
   - Hyperparameter tuning

4. **Domain Knowledge**
   - Financial time series understanding
   - Sentiment analysis integration
   - Multi-modal data handling

## References

1. Liu, Y., et al. (2023). "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting"
2. Das, A., et al. (2024). "A Decoder-Only Foundation Model for Time-Series Forecasting" (Times-FM)
3. Han, L., et al. (2024). "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion"

## Authors

**Group 36**
- Sam Lai
- Zexuan Yang
- Yichao Yang
- Ao Xu

Data Science & Machine Learning Capstone Project  
2024

---

*This project was developed as a capstone project demonstrating expertise in time series forecasting and machine learning model implementation.*

