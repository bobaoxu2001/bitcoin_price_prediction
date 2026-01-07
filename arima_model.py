"""
ARIMA Model for Bitcoin Price Prediction

This module implements the ARIMA (AutoRegressive Integrated Moving Average) model
for time series forecasting of Bitcoin prices. ARIMA is a classical statistical
approach that combines autoregression, differencing, and moving average components.

Model Components:
- AR (AutoRegressive): Uses past values to predict future values
- I (Integrated): Differencing to achieve stationarity
- MA (Moving Average): Uses past forecast errors

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Dict, Optional
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import (
    BitcoinDataLoader,
    DataPreprocessor,
    calculate_metrics,
    print_metrics
)

warnings.filterwarnings('ignore')

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class ARIMAPredictor:
    """
    ARIMA-based predictor for Bitcoin price forecasting.
    
    This implementation uses auto_arima for automatic order selection (p, d, q)
    and supports both single-step and multi-step forecasting.
    
    Attributes:
        model: Fitted ARIMA model
        order: Tuple of (p, d, q) parameters
        is_fitted: Whether the model has been fitted
    """
    
    def __init__(
        self,
        order: Optional[Tuple[int, int, int]] = None,
        auto_select: bool = True,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        seasonal: bool = False
    ):
        """
        Initialize the ARIMA predictor.
        
        Args:
            order: Optional tuple of (p, d, q) parameters. If None, uses auto selection.
            auto_select: Whether to automatically select order parameters
            max_p: Maximum p value for auto selection
            max_q: Maximum q value for auto selection  
            max_d: Maximum d value for auto selection
            seasonal: Whether to include seasonal component
        """
        self.order = order
        self.auto_select = auto_select
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.seasonal = seasonal
        self.model = None
        self.is_fitted = False
        
    def fit(self, y: np.ndarray) -> 'ARIMAPredictor':
        """
        Fit the ARIMA model to the target series.
        
        Args:
            y: Target time series array
            
        Returns:
            Self for method chaining
        """
        print("Fitting ARIMA model...")
        
        if self.auto_select:
            # Use auto_arima for automatic order selection
            print("Using auto_arima for order selection...")
            auto_model = auto_arima(
                y,
                start_p=0,
                start_q=0,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                seasonal=self.seasonal,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                trace=False,
                random_state=RANDOM_SEED
            )
            self.order = auto_model.order
            self.model = auto_model
            print(f"Selected order: (p={self.order[0]}, d={self.order[1]}, q={self.order[2]})")
        else:
            # Use specified order
            if self.order is None:
                self.order = (1, 1, 1)  # Default order
                
            self.model = ARIMA(y, order=self.order)
            self.model = self.model.fit()
            print(f"Fitted ARIMA with order: (p={self.order[0]}, d={self.order[1]}, q={self.order[2]})")
        
        self.is_fitted = True
        return self
    
    def predict(self, n_periods: int = 1) -> np.ndarray:
        """
        Generate predictions for future periods.
        
        Args:
            n_periods: Number of periods to forecast
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if hasattr(self.model, 'predict'):
            # pmdarima auto_arima
            predictions = self.model.predict(n_periods=n_periods)
        else:
            # statsmodels ARIMA
            predictions = self.model.forecast(steps=n_periods)
            
        return np.array(predictions)
    
    def update_and_predict(
        self,
        y_new: np.ndarray,
        n_periods: int = 1
    ) -> np.ndarray:
        """
        Update model with new data and make predictions.
        
        Args:
            y_new: New observations to update the model
            n_periods: Number of periods to forecast
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating")
        
        # Update model with new observations
        self.model.update(y_new)
        
        return self.predict(n_periods)
    
    def get_model_summary(self) -> str:
        """Get model summary string."""
        if not self.is_fitted:
            return "Model not fitted"
        
        summary = f"ARIMA{self.order} Model\n"
        summary += f"AIC: {self.model.aic():.4f}\n" if hasattr(self.model, 'aic') else ""
        
        return summary


def walk_forward_arima(
    y_train: np.ndarray,
    y_test: np.ndarray,
    order: Optional[Tuple[int, int, int]] = None,
    auto_select: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Perform walk-forward validation with ARIMA model.
    
    Walk-forward validation retrains the model at each step with all
    available historical data, providing realistic performance estimates.
    
    Args:
        y_train: Training target series
        y_test: Test target series
        order: Optional ARIMA order tuple
        auto_select: Whether to auto-select order on first fit
        
    Returns:
        Tuple of (predictions array, metrics dictionary)
    """
    predictions = []
    history = list(y_train)
    
    print(f"\nWalk-Forward Validation: {len(y_test)} steps")
    print("-" * 40)
    
    # Fit initial model
    predictor = ARIMAPredictor(order=order, auto_select=auto_select)
    predictor.fit(np.array(history))
    
    # Walk forward through test set
    for t, actual in enumerate(y_test):
        # Predict next value
        pred = predictor.predict(n_periods=1)[0]
        predictions.append(pred)
        
        # Update history with actual observation
        history.append(actual)
        
        # Refit model periodically for better performance
        if (t + 1) % 100 == 0:
            print(f"  Step {t+1}/{len(y_test)} completed")
            predictor = ARIMAPredictor(order=predictor.order, auto_select=False)
            predictor.fit(np.array(history))
    
    predictions = np.array(predictions)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, predictions)
    
    return predictions, metrics


def run_arima_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    auto_select_order: bool = True,
    order: Optional[Tuple[int, int, int]] = None
) -> Dict:
    """
    Run complete ARIMA experiment on Bitcoin price data.
    
    Args:
        data_path: Path to the dataset
        train_ratio: Ratio of data to use for training
        auto_select_order: Whether to auto-select ARIMA order
        order: Optional manual order specification
        
    Returns:
        Dictionary containing predictions, metrics, and model info
    """
    print("="*60)
    print("ARIMA Model - Bitcoin Price Prediction")
    print("="*60)
    
    # Load data
    loader = BitcoinDataLoader(data_path)
    df = loader.load_data()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # Split data
    train_df, test_df = preprocessor.create_train_test_split(df_clean, train_ratio)
    
    # Get target series (using listing_close for ARIMA since it's univariate)
    y_train = train_df['listing_close'].values
    y_test = test_df['target_nexthour'].values
    
    # Sample for faster training (ARIMA can be slow on large datasets)
    # Use last portion of training data
    sample_size = min(5000, len(y_train))
    y_train_sample = y_train[-sample_size:]
    
    print(f"\nUsing {sample_size} training samples for ARIMA")
    
    # Run walk-forward validation
    predictions, metrics = walk_forward_arima(
        y_train_sample,
        y_test[:100],  # Limit test size for demonstration
        order=order,
        auto_select=auto_select_order
    )
    
    # Print results
    print_metrics(metrics, "ARIMA")
    
    return {
        'predictions': predictions,
        'actuals': y_test[:100],
        'metrics': metrics,
        'dates': test_df['date'].values[:100]
    }


def visualize_results(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Visualize ARIMA prediction results.
    
    Args:
        results: Dictionary containing predictions and actuals
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(14, 6))
        
        # Plot 1: Time series comparison
        plt.subplot(1, 2, 1)
        plt.plot(results['actuals'], label='Actual', alpha=0.8)
        plt.plot(results['predictions'], label='Predicted', alpha=0.8)
        plt.title('ARIMA: Actual vs Predicted')
        plt.xlabel('Time Step')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(results['actuals'], results['predictions'], alpha=0.5)
        min_val = min(results['actuals'].min(), results['predictions'].min())
        max_val = max(results['actuals'].max(), results['predictions'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        plt.title('ARIMA: Prediction Scatter Plot')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == '__main__':
    """
    Main execution block for ARIMA model training and evaluation.
    
    This script demonstrates:
    1. Data loading and preprocessing
    2. ARIMA model fitting with automatic order selection
    3. Walk-forward validation for realistic performance assessment
    4. Results visualization and metrics reporting
    """
    
    print("\n" + "="*60)
    print("ARIMA MODEL - BITCOIN PRICE PREDICTION")
    print("Baseline Statistical Model")
    print("="*60 + "\n")
    
    # Run experiment
    results = run_arima_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        auto_select_order=True
    )
    
    # Display final metrics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"MSE:  {results['metrics']['MSE']:.4f}")
    print(f"MAE:  {results['metrics']['MAE']:.4f}")
    print(f"RMSE: {results['metrics']['RMSE']:.4f}")
    
    # Visualize if matplotlib is available
    try:
        visualize_results(results)
    except Exception as e:
        print(f"Visualization skipped: {e}")

