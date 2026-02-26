"""
Data Collection Pipeline for Bitcoin Price Prediction

This module documents and implements the data collection and assembly pipeline
used to create the `filtered_df.csv` dataset. The dataset integrates:
  - Bitcoin hourly closing prices
  - NASDAQ Composite daily OHLCV (forward-filled to hourly)
  - Gold daily OHLCV (forward-filled to hourly)
  - VIX daily index (forward-filled to hourly)
  - Social media sentiment (Twitter, Reddit, Bitcointalk) at hourly granularity

Data Sources:
  - Bitcoin: CryptoCompare Historical Hourly OHLCV API
  - NASDAQ: Yahoo Finance (^IXIC) or NASDAQ official historical data
  - Gold: Yahoo Finance (GC=F) or Investing.com gold historical data
  - VIX: Yahoo Finance (^VIX) or CBOE VIX historical data
  - Sentiment: Twitter API / snscrape, Reddit PRAW, Bitcointalk scraping
    with VADER / CryptoBERT sentiment scoring

Time Range: 2016-11-01 to 2024-08-22 (~68,000 hourly observations)

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - NYU Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import warnings
import os

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Section 1: Bitcoin Hourly Prices
# ---------------------------------------------------------------------------

def fetch_bitcoin_hourly(
    start_date: str = '2016-11-01',
    end_date: str = '2024-08-22',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch Bitcoin hourly closing prices from CryptoCompare API.

    The CryptoCompare histohour endpoint returns up to 2000 hourly candles
    per request. Multiple paginated requests are needed for the full range.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        save_path: Optional path to save intermediate CSV

    Returns:
        DataFrame with columns ['date', 'listing_close']

    API Reference:
        https://min-api.cryptocompare.com/documentation?key=Historical&cat=dataHistoHour
    """
    try:
        import requests
    except ImportError:
        raise ImportError("Install requests: pip install requests")

    base_url = "https://min-api.cryptocompare.com/data/v2/histohour"
    end_ts = int(pd.Timestamp(end_date).timestamp())
    start_ts = int(pd.Timestamp(start_date).timestamp())

    all_data = []
    current_ts = end_ts

    while current_ts > start_ts:
        params = {
            "fsym": "BTC",
            "tsym": "USD",
            "limit": 2000,
            "toTs": current_ts
        }
        resp = requests.get(base_url, params=params)
        data = resp.json()

        if data.get("Response") != "Success":
            print(f"API error: {data.get('Message', 'Unknown error')}")
            break

        records = data["Data"]["Data"]
        all_data.extend(records)
        current_ts = records[0]["time"] - 1

        if records[0]["time"] <= start_ts:
            break

    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df = df.rename(columns={'close': 'listing_close'})
    df = df[['date', 'listing_close']].drop_duplicates('date').sort_values('date')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Bitcoin hourly data saved to {save_path}: {len(df)} rows")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section 2: NASDAQ Composite (Daily)
# ---------------------------------------------------------------------------

def fetch_nasdaq_daily(
    start_date: str = '2016-11-01',
    end_date: str = '2024-08-22',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch NASDAQ Composite daily OHLCV data.

    Primary source: Yahoo Finance (ticker ^IXIC).
    Alternative: Download CSV from https://www.nasdaq.com/market-activity/index/comp/historical

    Args:
        start_date: Start date string
        end_date: End date string
        save_path: Optional path to save CSV

    Returns:
        DataFrame with columns ['date', 'NasOpen', 'NasHigh', 'NasLow', 'NasClose/Last']
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("^IXIC")
        hist = ticker.history(start=start_date, end=end_date)
        df = hist.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'NasOpen',
            'High': 'NasHigh',
            'Low': 'NasLow',
            'Close': 'NasClose/Last'
        })
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df[['date', 'NasOpen', 'NasHigh', 'NasLow', 'NasClose/Last']]
    except ImportError:
        print("yfinance not installed. Download NASDAQ data manually from Yahoo Finance.")
        print("Ticker: ^IXIC, Period: {} to {}".format(start_date, end_date))
        print("Rename columns: Open→NasOpen, High→NasHigh, Low→NasLow, Close→NasClose/Last")
        return pd.DataFrame()

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"NASDAQ data saved to {save_path}: {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Section 3: Gold Prices (Daily)
# ---------------------------------------------------------------------------

def fetch_gold_daily(
    start_date: str = '2016-11-01',
    end_date: str = '2024-08-22',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch Gold futures daily OHLCV data.

    Primary source: Yahoo Finance (ticker GC=F).
    Alternative: Investing.com gold historical data.

    Returns:
        DataFrame with columns ['date', 'GOpen', 'GHigh', 'GLow', 'GClose/Last', 'GVolume']
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("GC=F")
        hist = ticker.history(start=start_date, end=end_date)
        df = hist.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'GOpen',
            'High': 'GHigh',
            'Low': 'GLow',
            'Close': 'GClose/Last',
            'Volume': 'GVolume'
        })
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df[['date', 'GOpen', 'GHigh', 'GLow', 'GClose/Last', 'GVolume']]
    except ImportError:
        print("yfinance not installed. Download Gold data manually from Yahoo Finance.")
        print("Ticker: GC=F")
        return pd.DataFrame()

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Gold data saved to {save_path}: {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Section 4: VIX Index (Daily)
# ---------------------------------------------------------------------------

def fetch_vix_daily(
    start_date: str = '2016-11-01',
    end_date: str = '2024-08-22',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch CBOE VIX daily data.

    Primary source: Yahoo Finance (ticker ^VIX).
    Alternative: CBOE VIX data (https://www.cboe.com/tradable_products/vix/).

    Returns:
        DataFrame with columns ['date', 'OPEN', 'HIGH', 'LOW', 'CLOSE']
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(start=start_date, end=end_date)
        df = hist.reset_index()
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'OPEN',
            'High': 'HIGH',
            'Low': 'LOW',
            'Close': 'CLOSE'
        })
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        df = df[['date', 'OPEN', 'HIGH', 'LOW', 'CLOSE']]
    except ImportError:
        print("yfinance not installed. Download VIX data manually from Yahoo Finance.")
        print("Ticker: ^VIX")
        return pd.DataFrame()

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"VIX data saved to {save_path}: {len(df)} rows")

    return df


# ---------------------------------------------------------------------------
# Section 5: Social Media Sentiment (Hourly)
# ---------------------------------------------------------------------------

def collect_sentiment_data(
    start_date: str = '2016-11-01',
    end_date: str = '2024-08-22'
) -> pd.DataFrame:
    """
    Reference implementation for hourly sentiment collection.

    In the original project, sentiment was collected from:
      1. Twitter (X): Using Twitter API (Academic Research) or snscrape
         - Searched for Bitcoin-related keywords
         - Applied VADER / CryptoBERT sentiment analysis
         - Aggregated to hourly optimistic/negative scores

      2. Reddit: Using PRAW (Python Reddit API Wrapper)
         - Subreddits: r/Bitcoin, r/CryptoCurrency, r/btc
         - Scored sentiment per post/comment
         - Aggregated to hourly scores

      3. Bitcointalk: Web scraping of forum posts
         - Focused on Bitcoin Discussion and Speculation boards
         - Sentiment scored and aggregated hourly

    Sentiment scoring used VADER (Valence Aware Dictionary and sEntiment Reasoner)
    with domain-specific adjustments for cryptocurrency terminology.

    Each platform produces two hourly features:
      - {platform}_optimistic: Mean positive sentiment score (0-1)
      - {platform}_negative: Mean negative sentiment score (0-1)

    Returns:
        DataFrame with sentiment columns or empty DataFrame if APIs unavailable
    """
    print("Sentiment collection requires API access:")
    print("  - Twitter: Academic Research API or snscrape")
    print("  - Reddit: PRAW with API credentials")
    print("  - Bitcointalk: Web scraping")
    print()
    print("Install sentiment tools:")
    print("  pip install praw vaderSentiment snscrape")
    print()
    print("Example for Reddit sentiment:")
    print("""
    import praw
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    reddit = praw.Reddit(client_id='...', client_secret='...', user_agent='...')
    analyzer = SentimentIntensityAnalyzer()

    for submission in reddit.subreddit('Bitcoin').new(limit=1000):
        scores = analyzer.polarity_scores(submission.title + ' ' + submission.selftext)
        # Aggregate scores by hour
    """)

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Section 6: Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the merged dataset.

    Creates:
      - Moving averages (2, 6, 12, 24 hour windows)
      - Log returns and percentage returns
      - Target variable (next-hour price)
      - Sentiment lag features (1h, 5h, 12h, 24h)

    Args:
        df: Merged DataFrame with all raw features

    Returns:
        DataFrame with engineered features
    """
    df = df.sort_values('date').reset_index(drop=True)

    price_col = 'listing_close'

    # Target: next hour price
    df['target_nexthour'] = df[price_col].shift(-1)

    # Returns
    df['target_log_return'] = np.log(df['target_nexthour'] / df[price_col])
    df['percentage_return'] = (df['target_nexthour'] - df[price_col]) / df[price_col] * 100

    # Moving averages
    for window in [2, 6, 12, 24]:
        df[f'ma_{window}'] = df[price_col].rolling(window=window, min_periods=1).mean()

    # Sentiment lag features
    sentiment_cols = [
        col for col in df.columns
        if any(p in col for p in ['twitter_', 'reddit_', 'bitcointalk_'])
        and 'lag' not in col
    ]

    for col in sentiment_cols:
        for lag in [1, 5, 12, 24]:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Drop last row (no target) and fill remaining NaN
    df = df.dropna(subset=['target_nexthour'])
    df = df.ffill().bfill()

    return df


# ---------------------------------------------------------------------------
# Section 7: Assembly Pipeline
# ---------------------------------------------------------------------------

def assemble_dataset(
    btc_df: pd.DataFrame,
    nasdaq_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    output_path: str = 'filtered_df.csv'
) -> pd.DataFrame:
    """
    Assemble the complete dataset from individual data sources.

    Daily market data (NASDAQ, Gold, VIX) is merged by date and forward-filled
    to hourly frequency. Sentiment data is merged directly on hourly timestamps.

    Args:
        btc_df: Bitcoin hourly prices
        nasdaq_df: NASDAQ daily OHLCV
        gold_df: Gold daily OHLCV
        vix_df: VIX daily data
        sentiment_df: Hourly sentiment scores
        output_path: Path to save the assembled CSV

    Returns:
        Complete assembled DataFrame
    """
    # Normalize daily data dates to date-only for merging
    for daily_df in [nasdaq_df, gold_df, vix_df]:
        if not daily_df.empty:
            daily_df['date'] = pd.to_datetime(daily_df['date']).dt.normalize()

    # Add date-only column to btc for daily merge
    btc_df = btc_df.copy()
    btc_df['date_only'] = pd.to_datetime(btc_df['date']).dt.normalize()

    # Merge daily data
    merged = btc_df.copy()

    if not nasdaq_df.empty:
        nasdaq_df = nasdaq_df.rename(columns={'date': 'date_only'})
        merged = merged.merge(nasdaq_df, on='date_only', how='left')

    if not gold_df.empty:
        gold_df = gold_df.rename(columns={'date': 'date_only'})
        merged = merged.merge(gold_df, on='date_only', how='left')

    if not vix_df.empty:
        vix_df = vix_df.rename(columns={'date': 'date_only'})
        merged = merged.merge(vix_df, on='date_only', how='left')

    merged = merged.drop(columns=['date_only'])

    # Merge hourly sentiment data
    if not sentiment_df.empty:
        merged = merged.merge(sentiment_df, on='date', how='left')

    # Forward-fill daily data across hours, then backward-fill gaps
    merged = merged.ffill().bfill()

    # Engineer features
    result = engineer_features(merged)

    # Save
    result.to_csv(output_path, index=False)
    print(f"Dataset assembled and saved to {output_path}")
    print(f"Shape: {result.shape[0]} rows × {result.shape[1]} columns")
    print(f"Date range: {result['date'].min()} to {result['date'].max()}")

    return result


# ---------------------------------------------------------------------------
# Section 8: Synthetic Data Generator (for testing without API access)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_samples: int = 2000,
    output_path: str = 'filtered_df.csv'
) -> pd.DataFrame:
    """
    Generate a synthetic dataset matching the schema of filtered_df.csv.

    This is useful for testing model code when the real dataset or API
    access is unavailable. The synthetic data preserves column names and
    data types but does NOT reflect real market dynamics.

    Args:
        n_samples: Number of hourly observations to generate
        output_path: Path to save the synthetic CSV

    Returns:
        Synthetic DataFrame
    """
    np.random.seed(42)

    start_date = datetime(2016, 11, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_samples)]

    base_price = 700 + np.cumsum(np.random.randn(n_samples) * 5)
    base_price = np.abs(base_price) + 500

    data = {
        'date': dates,
        'listing_close': base_price,
        'target_nexthour': np.roll(base_price, -1),
        'target_log_return': np.log(np.roll(base_price, -1) / base_price),
        'percentage_return': (np.roll(base_price, -1) - base_price) / base_price * 100,
    }

    for w in [2, 6, 12, 24]:
        data[f'ma_{w}'] = pd.Series(base_price).rolling(window=w, min_periods=1).mean().values

    # NASDAQ features
    nas_base = 5000 + np.cumsum(np.random.randn(n_samples) * 2)
    data['NasClose/Last'] = nas_base
    data['NasOpen'] = nas_base + np.random.randn(n_samples) * 5
    data['NasHigh'] = nas_base + np.abs(np.random.randn(n_samples) * 10)
    data['NasLow'] = nas_base - np.abs(np.random.randn(n_samples) * 10)

    # Gold features
    gold_base = 1300 + np.cumsum(np.random.randn(n_samples) * 0.5)
    data['GClose/Last'] = gold_base
    data['GOpen'] = gold_base + np.random.randn(n_samples) * 2
    data['GHigh'] = gold_base + np.abs(np.random.randn(n_samples) * 5)
    data['GLow'] = gold_base - np.abs(np.random.randn(n_samples) * 5)
    data['GVolume'] = np.random.randint(1000, 50000, n_samples).astype(float)

    # VIX features
    vix_base = 15 + np.random.randn(n_samples) * 3
    data['CLOSE'] = np.abs(vix_base)
    data['OPEN'] = np.abs(vix_base + np.random.randn(n_samples) * 0.5)
    data['HIGH'] = np.abs(vix_base + np.abs(np.random.randn(n_samples)))
    data['LOW'] = np.abs(vix_base - np.abs(np.random.randn(n_samples)))

    # Sentiment features with lags
    for source in ['twitter', 'reddit', 'bitcointalk']:
        opt = np.random.rand(n_samples) * 0.5 + 0.3
        neg = np.random.rand(n_samples) * 0.3
        data[f'{source}_optimistic'] = opt
        data[f'{source}_negative'] = neg
        for lag in [1, 5, 12, 24]:
            data[f'{source}_optimistic_lag{lag}'] = np.roll(opt, lag)
            data[f'{source}_negative_lag{lag}'] = np.roll(neg, lag)

    df = pd.DataFrame(data)
    df.loc[df.index[-1], 'target_nexthour'] = (
        df.loc[df.index[-1], 'listing_close'] + np.random.randn() * 5
    )

    df.to_csv(output_path, index=False)
    print(f"Synthetic dataset generated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Saved to {output_path}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 60)
    print("BITCOIN PRICE PREDICTION — DATA COLLECTION PIPELINE")
    print("=" * 60)
    print()
    print("This script documents the data collection process.")
    print("To build the real dataset, you need API access to:")
    print("  1. CryptoCompare (Bitcoin hourly prices)")
    print("  2. Yahoo Finance via yfinance (NASDAQ, Gold, VIX)")
    print("  3. Twitter API / snscrape (Twitter sentiment)")
    print("  4. Reddit PRAW (Reddit sentiment)")
    print("  5. Bitcointalk scraper (Bitcointalk sentiment)")
    print()
    print("For testing without API access, generating synthetic data...")
    print()

    df = generate_synthetic_dataset(n_samples=2000, output_path='filtered_df.csv')
    print()
    print(f"Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
