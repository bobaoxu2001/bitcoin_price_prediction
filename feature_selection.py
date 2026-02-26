"""
Feature Selection for Bitcoin Price Prediction

This module implements feature selection using LightGBM and XGBoost gain-based
metrics, as described in Section 4 of the capstone report.

Key findings:
  - NASDAQ features (Close/Last, Open, High, Low) and short-term moving averages
    (ma_2, ma_6) are among the most influential predictors.
  - Hourly lagged sentiment from Bitcointalk, Reddit, and Twitter at various
    lag horizons (1h, 5h, 12h, 24h) capture delayed market reactions.

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - NYU Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
import warnings

warnings.filterwarnings('ignore')

from data_preprocessing import BitcoinDataLoader, DataPreprocessor

plt.style.use('seaborn-v0_8-whitegrid')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def xgboost_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05
) -> pd.DataFrame:
    """
    Compute feature importance using XGBoost gain-based metrics.

    Args:
        X_train: Training features
        y_train: Training targets
        feature_names: List of feature names
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate

    Returns:
        DataFrame with feature names, gain importance, and weight importance
    """
    import xgboost as xgb

    X_train = np.nan_to_num(X_train, nan=0.0)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_SEED,
        'verbosity': 0
    }

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        verbose_eval=False
    )

    gain = model.get_score(importance_type='gain')
    weight = model.get_score(importance_type='weight')

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'xgb_gain': [gain.get(f, 0.0) for f in feature_names],
        'xgb_weight': [weight.get(f, 0.0) for f in feature_names]
    })

    return df_importance.sort_values('xgb_gain', ascending=False)


def lightgbm_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    n_estimators: int = 200,
    learning_rate: float = 0.05
) -> pd.DataFrame:
    """
    Compute feature importance using LightGBM gain-based metrics.

    Args:
        X_train: Training features
        y_train: Training targets
        feature_names: List of feature names
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate

    Returns:
        DataFrame with feature names, gain importance, and split importance
    """
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': learning_rate,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': RANDOM_SEED
    }

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
    )

    gain = model.feature_importance(importance_type='gain')
    split = model.feature_importance(importance_type='split')

    df_importance = pd.DataFrame({
        'feature': feature_names,
        'lgb_gain': gain,
        'lgb_split': split
    })

    return df_importance.sort_values('lgb_gain', ascending=False)


def combined_feature_ranking(
    xgb_importance: pd.DataFrame,
    lgb_importance: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine XGBoost and LightGBM importance into a unified ranking.

    Normalizes gain scores from each model and averages them.

    Args:
        xgb_importance: XGBoost feature importance DataFrame
        lgb_importance: LightGBM feature importance DataFrame

    Returns:
        DataFrame with combined normalized importance ranking
    """
    merged = xgb_importance[['feature', 'xgb_gain']].merge(
        lgb_importance[['feature', 'lgb_gain']],
        on='feature',
        how='outer'
    ).fillna(0)

    # Normalize to [0, 1]
    for col in ['xgb_gain', 'lgb_gain']:
        max_val = merged[col].max()
        if max_val > 0:
            merged[f'{col}_norm'] = merged[col] / max_val
        else:
            merged[f'{col}_norm'] = 0.0

    merged['combined_score'] = (merged['xgb_gain_norm'] + merged['lgb_gain_norm']) / 2
    merged = merged.sort_values('combined_score', ascending=False)

    return merged


def plot_feature_importance_comparison(
    combined: pd.DataFrame,
    top_n: int = 25,
    save_path: Optional[str] = None
) -> None:
    """
    Plot side-by-side XGBoost vs LightGBM feature importance.

    Args:
        combined: Combined importance DataFrame
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    top = combined.head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # XGBoost
    axes[0].barh(
        top['feature'][::-1],
        top['xgb_gain_norm'][::-1],
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )
    axes[0].set_title('XGBoost Feature Importance (Gain)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Normalized Gain')
    axes[0].grid(True, alpha=0.3, axis='x')

    # LightGBM
    axes[1].barh(
        top['feature'][::-1],
        top['lgb_gain_norm'][::-1],
        color='coral',
        edgecolor='black',
        linewidth=0.5
    )
    axes[1].set_title('LightGBM Feature Importance (Gain)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Normalized Gain')
    axes[1].grid(True, alpha=0.3, axis='x')

    fig.suptitle('Feature Selection: XGBoost vs LightGBM Gain-Based Metrics',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance comparison saved to {save_path}")

    plt.close(fig)


def plot_combined_ranking(
    combined: pd.DataFrame,
    top_n: int = 25,
    save_path: Optional[str] = None
) -> None:
    """
    Plot combined feature importance ranking.

    Args:
        combined: Combined importance DataFrame
        top_n: Number of top features to show
        save_path: Path to save figure
    """
    top = combined.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['steelblue' if any(k in f for k in ['Nas', 'ma_', 'listing', 'GC', 'Gold', 'VIX', 'CLOSE', 'OPEN', 'HIGH', 'LOW', 'G'])
              else 'coral'
              for f in top['feature']]

    ax.barh(
        top['feature'][::-1],
        top['combined_score'][::-1],
        color=colors[::-1],
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_title('Combined Feature Importance Ranking (XGBoost + LightGBM)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Combined Normalized Score')
    ax.grid(True, alpha=0.3, axis='x')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Market Features'),
        Patch(facecolor='coral', label='Sentiment Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined ranking plot saved to {save_path}")

    plt.close(fig)


def run_feature_selection(
    data_path: str = 'filtered_df.csv',
    output_dir: str = 'figures/feature_selection'
) -> pd.DataFrame:
    """
    Run complete feature selection analysis.

    Args:
        data_path: Path to dataset CSV
        output_dir: Directory to save figures

    Returns:
        Combined feature importance DataFrame
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("FEATURE SELECTION ANALYSIS")
    print("=" * 60)

    # Load data
    loader = BitcoinDataLoader(data_path)
    df = loader.load_data()
    feature_names = loader.get_feature_names()

    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    train_df, test_df = preprocessor.create_train_test_split(df_clean, 0.8)

    X_train, y_train = preprocessor.prepare_features_target(train_df)
    feature_names = list(X_train.columns)
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)

    # XGBoost feature importance
    print("\nComputing XGBoost feature importance...")
    xgb_imp = xgboost_feature_importance(X_train_scaled, y_train_scaled, feature_names)
    print("Top 10 XGBoost features (by gain):")
    for _, row in xgb_imp.head(10).iterrows():
        print(f"  {row['feature']:40s} gain={row['xgb_gain']:.2f}")

    # LightGBM feature importance
    print("\nComputing LightGBM feature importance...")
    lgb_imp = lightgbm_feature_importance(X_train_scaled, y_train_scaled, feature_names)
    print("Top 10 LightGBM features (by gain):")
    for _, row in lgb_imp.head(10).iterrows():
        print(f"  {row['feature']:40s} gain={row['lgb_gain']:.2f}")

    # Combined ranking
    print("\nCombined feature ranking:")
    combined = combined_feature_ranking(xgb_imp, lgb_imp)
    print(f"{'Rank':<5} {'Feature':<40} {'Combined Score':<15}")
    print("-" * 60)
    for i, (_, row) in enumerate(combined.head(20).iterrows(), 1):
        print(f"{i:<5} {row['feature']:<40} {row['combined_score']:.4f}")

    # Plots
    plot_feature_importance_comparison(
        combined, top_n=25,
        save_path=os.path.join(output_dir, 'feature_importance_comparison.png')
    )
    plot_combined_ranking(
        combined, top_n=25,
        save_path=os.path.join(output_dir, 'combined_feature_ranking.png')
    )

    # Save to CSV
    combined.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    print(f"\nFeature importance saved to {output_dir}/feature_importance.csv")

    return combined


if __name__ == '__main__':
    run_feature_selection()
