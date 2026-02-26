"""
Visualization Module for Bitcoin Price Prediction

This module provides comprehensive visualization functions for reproducing
all figures in the capstone report (Section 6.1), including:
  1. Full timeline prediction — complete prediction period
  2. Single prediction window — short-term accuracy
  3. Time series validation windows — robustness trends
  4. Returns comparison — actual vs predicted daily returns
  5. Distribution of prediction errors — error variability
  6. Timeline of error percentages — error fluctuation over time
  7. Model comparison charts
  8. Feature importance plots
  9. Training curves

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
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def plot_full_timeline(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot full timeline prediction vs actual.
    
    Args:
        dates: Array of datetime values
        actuals: Actual price values
        predictions: Predicted price values
        model_name: Name of the model for title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(14, 6))
    
    dates = pd.to_datetime(dates)
    
    plt.plot(dates, actuals, label='Actual', color='blue', linewidth=1.5, alpha=0.8)
    plt.plot(dates, predictions, label='Predicted', color='orange', linewidth=1.5, alpha=0.8)
    
    plt.title(f'{model_name} - Full Timeline Prediction', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_last_n_days(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    n_days: int = 7,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot last N days prediction analysis.
    
    Args:
        dates: Array of datetime values
        actuals: Actual price values
        predictions: Predicted price values
        n_days: Number of days to show
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    # Get last n_days * 24 hours of data
    n_hours = n_days * 24
    
    dates_subset = pd.to_datetime(dates[-n_hours:])
    actuals_subset = actuals[-n_hours:]
    predictions_subset = predictions[-n_hours:]
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(dates_subset, actuals_subset, label='Actual', color='blue', linewidth=1.5)
    plt.plot(dates_subset, predictions_subset, label='Predicted', color='orange', linewidth=1.5)
    
    plt.title(f'{model_name} - Last {n_days} Days Price Prediction Analysis', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_from_2024(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot predictions from 2024-01-01 onwards.
    
    Args:
        dates: Array of datetime values
        actuals: Actual price values
        predictions: Predicted price values
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    dates = pd.to_datetime(dates)
    start_date = pd.Timestamp("2024-01-01")
    
    # Filter data from 2024 onwards
    mask = dates >= start_date
    filtered_dates = dates[mask]
    filtered_actuals = actuals[mask]
    filtered_predictions = predictions[mask]
    
    if len(filtered_dates) == 0:
        print("No data available from 2024 onwards")
        return
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(filtered_dates, filtered_actuals, label='Actual', color='blue', linewidth=1.5)
    plt.plot(filtered_dates, filtered_predictions, label='Predicted', color='orange', linewidth=1.5)
    
    plt.title(f'{model_name} - Prediction from 2024-01-01 Onwards', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.xlim(start_date, filtered_dates.max())
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_error_distribution(
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot prediction error distribution histogram.
    
    Args:
        actuals: Actual price values
        predictions: Predicted price values
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    errors = predictions - actuals
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(errors, bins=50, color='orange', alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, 
                label=f'Mean Error: {np.mean(errors):.2f}')
    
    plt.title(f'{model_name} - Error Distribution', fontsize=16)
    plt.xlabel('Prediction Error (USD)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_returns_comparison(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot actual vs predicted returns comparison.
    
    Args:
        dates: Array of datetime values
        actuals: Actual price values
        predictions: Predicted price values
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    # Calculate returns
    actual_returns = (actuals[1:] - actuals[:-1]) / actuals[:-1]
    predicted_returns = (predictions[1:] - predictions[:-1]) / predictions[:-1]
    dates_returns = pd.to_datetime(dates[1:])
    
    plt.figure(figsize=(14, 6))
    
    plt.plot(dates_returns, actual_returns, label='Actual Returns', color='blue', 
             linewidth=1, alpha=0.7)
    plt.plot(dates_returns, predicted_returns, label='Predicted Returns', color='orange', 
             linewidth=1, alpha=0.7)
    
    plt.title(f'{model_name} - Returns Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Returns', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_prediction_window(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    window_size: int = 100,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot a single prediction window (Section 6.1 — Component 2).

    Focuses on the model's recent short-term accuracy over the specified
    number of time steps.

    Args:
        dates: Array of datetime values
        actuals: Actual price values
        predictions: Predicted price values
        window_size: Number of time steps in the window
        model_name: Name of the model
        save_path: Path to save figure
    """
    n = min(window_size, len(actuals))

    dates_w = pd.to_datetime(dates[-n:])
    actuals_w = actuals[-n:]
    preds_w = predictions[-n:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [3, 1]})

    axes[0].plot(dates_w, actuals_w, label='Actual', color='blue', linewidth=1.5)
    axes[0].plot(dates_w, preds_w, label='Predicted', color='orange', linewidth=1.5)
    axes[0].fill_between(dates_w, actuals_w, preds_w, alpha=0.15, color='orange')
    axes[0].set_title(f'{model_name} — Single Prediction Window (last {n} steps)',
                      fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    error = preds_w - actuals_w
    axes[1].bar(dates_w, error, color=np.where(error >= 0, 'green', 'red'), alpha=0.7, width=0.03)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_title('Prediction Error', fontsize=12)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Error (USD)')
    axes[1].grid(alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close(fig)


def plot_validation_windows(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    n_windows: int = 4,
    window_size: int = 100,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple time series validation windows (Section 6.1 — Component 3).

    Showcases robustness of performance trends across different time windows.

    Args:
        dates: Array of datetime values
        actuals: Actual price values
        predictions: Predicted price values
        n_windows: Number of validation windows to display
        window_size: Size of each window
        model_name: Name of the model
        save_path: Path to save figure
    """
    total = len(actuals)
    if total < n_windows * window_size:
        window_size = max(total // n_windows, 10)

    step = max((total - window_size) // max(n_windows - 1, 1), 1)

    fig, axes = plt.subplots(n_windows, 1, figsize=(14, 3 * n_windows), sharex=False)
    if n_windows == 1:
        axes = [axes]

    for i in range(n_windows):
        start = i * step
        end = min(start + window_size, total)

        d = pd.to_datetime(dates[start:end])
        a = actuals[start:end]
        p = predictions[start:end]

        mae = np.mean(np.abs(a - p))

        axes[i].plot(d, a, label='Actual', color='blue', linewidth=1.2)
        axes[i].plot(d, p, label='Predicted', color='orange', linewidth=1.2)
        axes[i].fill_between(d, a, p, alpha=0.1, color='orange')
        axes[i].set_title(f'Window {i+1} (MAE: {mae:.2f})', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Price')
        axes[i].legend(loc='upper right', fontsize=8)
        axes[i].grid(alpha=0.3)
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    fig.suptitle(f'{model_name} — Time Series Validation Windows',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close(fig)


def plot_error_percentage_timeline(
    dates: np.ndarray,
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot timeline of error percentages (Section 6.1 — Component 6).

    Depicts the fluctuation of percentage errors over time.

    Args:
        dates: Array of datetime values
        actuals: Actual price values
        predictions: Predicted price values
        model_name: Name of the model
        save_path: Path to save figure
    """
    dates_dt = pd.to_datetime(dates)

    non_zero = actuals != 0
    error_pct = np.zeros_like(actuals, dtype=float)
    error_pct[non_zero] = np.abs((predictions[non_zero] - actuals[non_zero]) / actuals[non_zero]) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_dt, error_pct, color='steelblue', linewidth=0.8, alpha=0.7)

    # Rolling average
    window = min(24, len(error_pct) // 5)
    if window > 1:
        rolling_avg = pd.Series(error_pct).rolling(window=window, min_periods=1).mean().values
        ax.plot(dates_dt, rolling_avg, color='red', linewidth=1.5, label=f'{window}-step rolling avg')

    ax.axhline(y=np.mean(error_pct), color='green', linestyle='--', linewidth=1,
               label=f'Mean: {np.mean(error_pct):.2f}%')

    ax.set_title(f'{model_name} — Timeline of Error Percentages', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Absolute Percentage Error (%)')
    ax.legend()
    ax.grid(alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.close(fig)


def plot_scatter(
    actuals: np.ndarray,
    predictions: np.ndarray,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot actual vs predicted scatter plot.
    
    Args:
        actuals: Actual price values
        predictions: Predicted price values
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(8, 8))
    
    plt.scatter(actuals, predictions, alpha=0.3, s=5, c='blue')
    
    # Perfect prediction line
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
             label='Perfect Prediction')
    
    plt.title(f'{model_name} - Actual vs Predicted', fontsize=16)
    plt.xlabel('Actual Price (USD)', fontsize=12)
    plt.ylabel('Predicted Price (USD)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_model_comparison(
    model_results: Dict[str, Dict],
    metric: str = 'MAE',
    save_path: Optional[str] = None
) -> None:
    """
    Plot bar chart comparing different models on a specific metric.
    
    Args:
        model_results: Dictionary with model names as keys and metrics dicts as values
        metric: Metric to compare ('MSE', 'MAE', 'RMSE', 'MAPE', 'Direction_Accuracy')
        save_path: Optional path to save the figure
    """
    model_names = list(model_results.keys())
    metric_values = [model_results[name]['metrics'][metric] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(model_names)))
    bars = plt.bar(model_names, metric_values, color=colors, edgecolor='black')
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(metric_values),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Model Comparison - {metric}', fontsize=16)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    if val_losses:
        plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linewidth=2)
    
    plt.title(f'{model_name} - Training Curves', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 20,
    model_name: str = 'Model',
    save_path: Optional[str] = None
) -> None:
    """
    Plot horizontal bar chart of feature importance.
    
    Args:
        feature_importance: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
    plt.barh(top_features['feature'], top_features['importance'], color=colors)
    
    plt.title(f'{model_name} - Top {top_n} Feature Importance', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.close()


def generate_all_figures(
    results: Dict,
    model_name: str,
    output_dir: str = 'figures'
) -> None:
    """
    Generate all standard figures for a model's results.
    
    Args:
        results: Dictionary containing predictions, actuals, dates, metrics, etc.
        model_name: Name of the model
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    dates = results.get('dates', None)
    actuals = results['actuals']
    predictions = results['predictions']
    
    # Safe model name for filenames
    safe_name = model_name.replace(' ', '_').replace('-', '_').lower()

    print(f"\nGenerating figures for {model_name}...")
    print("=" * 50)

    # --- Report Section 6.1 Figures ---

    # (1) Full timeline prediction
    if dates is not None:
        plot_full_timeline(
            dates, actuals, predictions, model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_full_timeline.png')
        )

    # (2) Single prediction window
    if dates is not None:
        plot_prediction_window(
            dates, actuals, predictions, window_size=100, model_name=model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_prediction_window.png')
        )

    # (3) Time series validation windows
    if dates is not None:
        plot_validation_windows(
            dates, actuals, predictions, n_windows=4, model_name=model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_validation_windows.png')
        )

    # (4) Returns comparison
    if dates is not None:
        plot_returns_comparison(
            dates, actuals, predictions, model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_returns.png')
        )

    # (5) Distribution of prediction errors
    plot_error_distribution(
        actuals, predictions, model_name,
        save_path=os.path.join(output_dir, f'{safe_name}_error_dist.png')
    )

    # (6) Timeline of error percentages
    if dates is not None:
        plot_error_percentage_timeline(
            dates, actuals, predictions, model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_error_pct_timeline.png')
        )

    # --- Additional Figures ---

    # Scatter plot (actual vs predicted)
    plot_scatter(
        actuals, predictions, model_name,
        save_path=os.path.join(output_dir, f'{safe_name}_scatter.png')
    )

    # Period-specific analyses
    if dates is not None and len(dates) > 7 * 24:
        plot_last_n_days(
            dates, actuals, predictions, 7, model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_last_7_days.png')
        )

    if dates is not None and len(dates) > 30 * 24:
        plot_last_n_days(
            dates, actuals, predictions, 30, model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_last_30_days.png')
        )

    if dates is not None:
        plot_from_2024(
            dates, actuals, predictions, model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_from_2024.png')
        )

    # Training curves (if available)
    if 'train_losses' in results:
        plot_training_curves(
            results['train_losses'],
            results.get('val_losses'),
            model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_training_curves.png')
        )

    # Feature importance (if available)
    if 'feature_importance' in results and results['feature_importance'] is not None:
        plot_feature_importance(
            results['feature_importance'], 20, model_name,
            save_path=os.path.join(output_dir, f'{safe_name}_feature_importance.png')
        )

    print(f"\nAll figures saved to {output_dir}/")


def create_metrics_table(
    model_results: Dict[str, Dict]
) -> pd.DataFrame:
    """
    Create a formatted metrics comparison table.
    
    Args:
        model_results: Dictionary with model names as keys
        
    Returns:
        DataFrame with formatted metrics
    """
    rows = []
    for model_name, results in model_results.items():
        metrics = results['metrics']
        rows.append({
            'Model': model_name,
            'MSE': f"{metrics['MSE']:.4f}",
            'MAE': f"{metrics['MAE']:.4f}",
            'RMSE': f"{metrics['RMSE']:.4f}",
            'MAPE (%)': f"{metrics.get('MAPE', 0):.2f}",
            'Direction Acc (%)': f"{metrics.get('Direction_Accuracy', 0) * 100:.2f}"
        })
    
    return pd.DataFrame(rows)


if __name__ == '__main__':
    """
    Example usage of visualization functions.
    """
    print("="*60)
    print("Visualization Module for Bitcoin Price Prediction")
    print("Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu")
    print("="*60)
    
    # Example: Load data and create sample visualizations
    from data_preprocessing import BitcoinDataLoader
    
    loader = BitcoinDataLoader('filtered_df.csv')
    df = loader.load_data()
    
    # Create sample actual/predicted data for demonstration
    dates = df['date'].values[-1000:]
    actuals = df['listing_close'].values[-1000:]
    # Simulate predictions (actual + small noise)
    predictions = actuals + np.random.normal(0, 100, len(actuals))
    
    print("\nGenerating sample visualizations...")
    
    # Generate sample plots
    plot_full_timeline(dates, actuals, predictions, "Sample Model")
    plot_error_distribution(actuals, predictions, "Sample Model")
    plot_scatter(actuals, predictions, "Sample Model")
    
    print("\nVisualization module ready for use!")

