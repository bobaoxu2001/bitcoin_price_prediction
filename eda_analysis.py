"""
Exploratory Data Analysis for Bitcoin Price Prediction

This module implements the exploratory analyses described in Section 3 of the
capstone report, including:
  1. Correlation heatmap (Feature relationships, sentiment impact)
  2. Time-varying correlation (NASDAQ vs Bitcoin, with COVID-19 regime shift)
  3. ACF and PACF analysis (lag structure for ARIMA order selection)
  4. Volatility analysis (heteroscedasticity, structural breaks)

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - NYU Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional, Tuple
import os
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 11


# ---------------------------------------------------------------------------
# 1. Correlation Heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 14)
) -> None:
    """
    Generate the overall correlation heatmap (Figure 1 in the report).

    Reveals strong positive correlations (~0.92) between listing_close and
    NASDAQ metrics, and mild-to-moderate negative correlations with negative
    sentiment metrics.

    Args:
        df: Dataset DataFrame
        save_path: Path to save figure
        figsize: Figure size
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        annot=False,
        ax=ax
    )
    ax.set_title('Overall Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='x', rotation=90, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {save_path}")

    plt.close(fig)


def print_key_correlations(df: pd.DataFrame) -> None:
    """Print key correlations with Bitcoin closing price."""
    numeric_df = df.select_dtypes(include=[np.number])

    if 'listing_close' not in numeric_df.columns:
        print("Column 'listing_close' not found.")
        return

    corr = numeric_df.corr()['listing_close'].drop('listing_close').sort_values(ascending=False)

    print("\nTop 10 Positive Correlations with listing_close:")
    print("-" * 50)
    for feat, val in corr.head(10).items():
        print(f"  {feat:40s} {val:+.4f}")

    print("\nTop 10 Negative Correlations with listing_close:")
    print("-" * 50)
    for feat, val in corr.tail(10).items():
        print(f"  {feat:40s} {val:+.4f}")


# ---------------------------------------------------------------------------
# 2. Time-Varying Correlation
# ---------------------------------------------------------------------------

def plot_time_varying_correlation(
    df: pd.DataFrame,
    col_a: str = 'NasClose/Last',
    col_b: str = 'listing_close',
    window: int = 720,
    save_path: Optional[str] = None
) -> None:
    """
    Plot rolling correlation between two columns over time.

    Demonstrates dynamic correlation, e.g., sharp dip during COVID-19 (2020)
    indicating Bitcoin behaved differently from traditional markets.

    Args:
        df: Dataset DataFrame with 'date' column
        col_a: First column name
        col_b: Second column name
        window: Rolling window size in hours (default 720 = ~30 days)
        save_path: Path to save figure
    """
    if col_a not in df.columns or col_b not in df.columns:
        print(f"Columns {col_a} and/or {col_b} not found.")
        return

    work_df = df[['date', col_a, col_b]].dropna().copy()
    work_df['date'] = pd.to_datetime(work_df['date'])
    work_df = work_df.sort_values('date')

    rolling_corr = work_df[col_a].rolling(window=window).corr(work_df[col_b])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(work_df['date'], rolling_corr, color='steelblue', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    ax.fill_between(work_df['date'], rolling_corr, 0, alpha=0.15, color='steelblue')

    ax.set_title(
        f'Time-Varying Correlation: {col_a} vs {col_b} ({window}h rolling window)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Correlation')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    fig.autofmt_xdate()

    ax.set_ylim(-1, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time-varying correlation plot saved to {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. ACF and PACF Analysis
# ---------------------------------------------------------------------------

def plot_acf_pacf(
    df: pd.DataFrame,
    column: str = 'listing_close',
    lags: int = 50,
    save_path: Optional[str] = None
) -> None:
    """
    Plot ACF and PACF for the given column.

    Based on the paper: slow decay in ACF and sharp cutoff at lag 1 in PACF
    suggests ARIMA(1,1,1) as a reasonable model.

    Args:
        df: Dataset DataFrame
        column: Column to analyze
        lags: Number of lags to display
        save_path: Path to save figure
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf

    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return

    series = df[column].dropna()

    # Differenced series for stationarity
    diff_series = series.diff().dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Raw series ACF/PACF
    plot_acf(series, lags=lags, ax=axes[0, 0], title=f'ACF — {column} (Original)')
    plot_pacf(series, lags=lags, ax=axes[0, 1], title=f'PACF — {column} (Original)', method='ywm')

    # Differenced series ACF/PACF
    plot_acf(diff_series, lags=lags, ax=axes[1, 0], title=f'ACF — {column} (1st Difference)')
    plot_pacf(diff_series, lags=lags, ax=axes[1, 1], title=f'PACF — {column} (1st Difference)', method='ywm')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.suptitle('ACF and PACF Analysis for ARIMA Order Selection', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ACF/PACF plot saved to {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Volatility Analysis
# ---------------------------------------------------------------------------

def plot_volatility_analysis(
    df: pd.DataFrame,
    column: str = 'listing_close',
    window: int = 24,
    save_path: Optional[str] = None
) -> None:
    """
    Analyze and plot volatility over time.

    Demonstrates heteroscedasticity, structural breaks, and regime changes
    as described in Section 3 of the paper.

    Args:
        df: Dataset DataFrame with 'date' column
        column: Price column to analyze
        window: Rolling window for volatility calculation (hours)
        save_path: Path to save figure
    """
    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return

    work_df = df[['date', column]].dropna().copy()
    work_df['date'] = pd.to_datetime(work_df['date'])
    work_df = work_df.sort_values('date')

    # Log returns
    work_df['log_return'] = np.log(work_df[column] / work_df[column].shift(1))
    work_df = work_df.dropna()

    # Rolling volatility (annualized)
    work_df['rolling_vol'] = work_df['log_return'].rolling(window=window).std() * np.sqrt(8760)

    # Rolling volatility with longer window for regime detection
    work_df['rolling_vol_7d'] = work_df['log_return'].rolling(window=168).std() * np.sqrt(8760)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})

    # Price
    axes[0].plot(work_df['date'], work_df[column], color='steelblue', linewidth=0.8)
    axes[0].set_title(f'Bitcoin Price ({column})', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].grid(True, alpha=0.3)

    # Hourly returns
    axes[1].plot(work_df['date'], work_df['log_return'], color='gray', linewidth=0.3, alpha=0.7)
    axes[1].set_title('Hourly Log Returns', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Log Return')
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    # Rolling volatility
    axes[2].plot(work_df['date'], work_df['rolling_vol'], color='orange', linewidth=0.8, alpha=0.7, label=f'{window}h window')
    axes[2].plot(work_df['date'], work_df['rolling_vol_7d'], color='red', linewidth=1.2, label='7-day window')
    axes[2].set_title('Annualized Rolling Volatility', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Volatility')
    axes[2].set_xlabel('Date')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Volatility analysis saved to {save_path}")

    plt.close(fig)


def plot_volatility_distribution(
    df: pd.DataFrame,
    column: str = 'listing_close',
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of returns and volatility (Figure 2 in the report).

    Args:
        df: Dataset DataFrame
        column: Price column
        save_path: Path to save figure
    """
    if column not in df.columns:
        print(f"Column '{column}' not found.")
        return

    returns = np.log(df[column] / df[column].shift(1)).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Returns distribution
    axes[0].hist(returns, bins=100, color='steelblue', edgecolor='black', alpha=0.7, density=True)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=1)
    axes[0].set_title('Distribution of Hourly Log Returns', fontweight='bold')
    axes[0].set_xlabel('Log Return')
    axes[0].set_ylabel('Density')
    axes[0].grid(True, alpha=0.3)

    # QQ plot
    from scipy import stats
    stats.probplot(returns, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Volatility Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Volatility distribution plot saved to {save_path}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Run All EDA
# ---------------------------------------------------------------------------

def run_all_eda(
    data_path: str = 'filtered_df.csv',
    output_dir: str = 'figures/eda'
) -> None:
    """
    Run all exploratory analyses and save figures.

    Args:
        data_path: Path to the dataset CSV
        output_dir: Directory to save output figures
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # 1. Correlation heatmap
    print("\n1. Generating correlation heatmap...")
    plot_correlation_heatmap(df, save_path=os.path.join(output_dir, 'correlation_heatmap.png'))
    print_key_correlations(df)

    # 2. Time-varying correlation
    print("\n2. Generating time-varying correlation plot...")
    if 'NasClose/Last' in df.columns:
        plot_time_varying_correlation(
            df,
            col_a='NasClose/Last',
            col_b='listing_close',
            save_path=os.path.join(output_dir, 'time_varying_correlation.png')
        )
    else:
        print("  Skipped: NasClose/Last column not found")

    # 3. ACF and PACF
    print("\n3. Generating ACF/PACF plots...")
    plot_acf_pacf(
        df,
        column='listing_close',
        lags=50,
        save_path=os.path.join(output_dir, 'acf_pacf.png')
    )

    # 4. Volatility analysis
    print("\n4. Generating volatility analysis...")
    plot_volatility_analysis(
        df,
        column='listing_close',
        save_path=os.path.join(output_dir, 'volatility_analysis.png')
    )
    plot_volatility_distribution(
        df,
        column='listing_close',
        save_path=os.path.join(output_dir, 'volatility_distribution.png')
    )

    print("\n" + "=" * 60)
    print(f"All EDA figures saved to {output_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    run_all_eda()
