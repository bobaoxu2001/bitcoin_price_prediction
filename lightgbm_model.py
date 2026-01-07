"""
LightGBM Model for Bitcoin Price Prediction

This module implements the LightGBM (Light Gradient Boosting Machine) model
for Bitcoin price forecasting. LightGBM uses histogram-based algorithms for
faster training and lower memory usage compared to traditional GBDT methods.

Key Features:
- Leaf-wise tree growth (faster convergence)
- Histogram-based splitting (efficient memory usage)
- Native support for categorical features
- Excellent handling of large datasets

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict, Optional, List
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_preprocessing import (
    BitcoinDataLoader,
    DataPreprocessor,
    calculate_metrics,
    print_metrics
)

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class LightGBMPredictor:
    """
    LightGBM-based predictor for Bitcoin price forecasting.
    
    This implementation uses gradient boosting with leaf-wise tree growth,
    optimized for speed and efficiency on large tabular datasets.
    
    Attributes:
        model: Trained LightGBM model (Booster object)
        params: Model hyperparameters
        feature_importance: Feature importance scores after training
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        min_data_in_leaf: int = 20,
        early_stopping_rounds: int = 10
    ):
        """
        Initialize LightGBM predictor with hyperparameters.
        
        Args:
            n_estimators: Number of boosting iterations
            num_leaves: Maximum number of leaves in one tree
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Boosting learning rate
            feature_fraction: Fraction of features for each iteration
            bagging_fraction: Fraction of data for bagging
            bagging_freq: Frequency for bagging (0 to disable)
            lambda_l1: L1 regularization
            lambda_l2: L2 regularization
            min_data_in_leaf: Minimum number of samples in a leaf
            early_stopping_rounds: Early stopping patience
        """
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        
        self.params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_data_in_leaf': min_data_in_leaf,
            'random_state': RANDOM_SEED,
            'verbose': -1
        }
        
        self.model = None
        self.feature_importance = None
        self.is_fitted = False
        self.feature_names = None
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'LightGBMPredictor':
        """
        Fit the LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets
            feature_names: Optional list of feature names
            
        Returns:
            Self for method chaining
        """
        print("Training LightGBM model...")
        
        self.feature_names = feature_names
        
        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0)
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0)
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, 
            label=y_train,
            feature_name=feature_names
        )
        
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val, 
                label=y_val,
                feature_name=feature_names,
                reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append('eval')
        
        # Configure callbacks
        callbacks = [
            lgb.log_evaluation(period=0)  # Suppress output
        ]
        
        if X_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=True
                )
            )
        
        # Train model
        self.model = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        # Store feature importance
        if feature_names:
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        print(f"LightGBM training completed. Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input features.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        predictions = self.model.predict(
            X, 
            num_iteration=self.model.best_iteration
        )
        
        return predictions
    
    def get_feature_importance(
        self, 
        top_n: int = 20,
        importance_type: str = 'gain'
    ) -> pd.DataFrame:
        """
        Get top N important features.
        
        Args:
            top_n: Number of top features to return
            importance_type: Type of importance ('gain', 'split')
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Model must be fitted to get feature importance")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        if self.is_fitted:
            self.model.save_model(path)
            print(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from file."""
        self.model = lgb.Booster(model_file=path)
        self.is_fitted = True
        print(f"Model loaded from {path}")


def walk_forward_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    train_size: int = 30000,
    test_size: int = 100,
    step_size: int = 100,
    n_windows: int = 10,
    feature_names: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Perform walk-forward validation with LightGBM model.
    
    Walk-forward validation retrains the model at each window,
    providing realistic performance estimates for time series.
    
    Args:
        X: Feature matrix
        y: Target array
        train_size: Size of training window
        test_size: Size of test window (forecast horizon)
        step_size: Step between windows
        n_windows: Number of validation windows
        feature_names: Optional feature names
        
    Returns:
        Tuple of (all predictions, all actuals, metrics dictionary)
    """
    all_predictions = []
    all_actuals = []
    
    print(f"\nWalk-Forward Validation: {n_windows} windows")
    print("-" * 50)
    
    for i in range(n_windows):
        window_start = i * step_size
        train_end = window_start + train_size
        test_end = train_end + test_size
        
        if test_end > len(X):
            break
        
        # Get train and test data for this window
        X_train = X[window_start:train_end]
        y_train = y[window_start:train_end]
        X_test = X[train_end:test_end]
        y_test = y[train_end:test_end]
        
        # Use last 10% of training as validation
        val_size = int(len(X_train) * 0.1)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train[:-val_size]
        y_train_final = y_train[:-val_size]
        
        # Train model
        predictor = LightGBMPredictor(n_estimators=200, learning_rate=0.05)
        predictor.fit(X_train_final, y_train_final, X_val, y_val, feature_names)
        
        # Predict
        predictions = predictor.predict(X_test)
        
        all_predictions.extend(predictions)
        all_actuals.extend(y_test)
        
        print(f"  Window {i+1}/{n_windows} completed")
    
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    # Calculate metrics
    metrics = calculate_metrics(all_actuals, all_predictions)
    
    return all_predictions, all_actuals, metrics


def run_lightgbm_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    n_estimators: int = 100,
    num_leaves: int = 31,
    learning_rate: float = 0.1
) -> Dict:
    """
    Run complete LightGBM experiment on Bitcoin price data.
    
    Args:
        data_path: Path to the dataset
        train_ratio: Ratio of data for training
        n_estimators: Number of boosting iterations
        num_leaves: Maximum number of leaves per tree
        learning_rate: Learning rate
        
    Returns:
        Dictionary containing predictions, metrics, and model info
    """
    print("="*60)
    print("LightGBM Model - Bitcoin Price Prediction")
    print("="*60)
    
    # Load data
    loader = BitcoinDataLoader(data_path)
    df = loader.load_data()
    feature_names = loader.get_feature_names()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # Split data
    train_df, test_df = preprocessor.create_train_test_split(df_clean, train_ratio)
    
    # Prepare features and targets
    X_train, y_train = preprocessor.prepare_features_target(train_df)
    X_test, y_test = preprocessor.prepare_features_target(test_df)
    
    # Get feature names before scaling
    feature_names = list(X_train.columns)
    
    # Scale data
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)
    
    # Create validation split
    val_size = int(len(X_train_scaled) * 0.1)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train_scaled[-val_size:]
    X_train_final = X_train_scaled[:-val_size]
    y_train_final = y_train_scaled[:-val_size]
    
    print(f"\nTraining set: {len(X_train_final)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test_scaled)} samples")
    
    # Initialize and train model
    predictor = LightGBMPredictor(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate
    )
    
    predictor.fit(
        X_train_final, y_train_final,
        X_val, y_val,
        feature_names=feature_names
    )
    
    # Make predictions
    predictions_scaled = predictor.predict(X_test_scaled)
    
    # Inverse transform to original scale
    predictions = preprocessor.inverse_transform_target(predictions_scaled)
    actuals = preprocessor.inverse_transform_target(y_test_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    # Print results
    print_metrics(metrics, "LightGBM")
    
    # Print top features
    print("\nTop 10 Important Features:")
    print("-" * 40)
    importance_df = predictor.get_feature_importance(top_n=10)
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'dates': test_df['date'].values,
        'feature_importance': predictor.feature_importance,
        'model': predictor
    }


def visualize_results(results: Dict, save_path: Optional[str] = None) -> None:
    """
    Visualize LightGBM prediction results.
    
    Args:
        results: Dictionary containing predictions and actuals
        save_path: Optional path to save the figure
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Time series comparison
        ax1 = axes[0, 0]
        ax1.plot(results['actuals'][:500], label='Actual', alpha=0.8)
        ax1.plot(results['predictions'][:500], label='Predicted', alpha=0.8)
        ax1.set_title('LightGBM: Actual vs Predicted (First 500 samples)')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        ax2 = axes[0, 1]
        ax2.scatter(results['actuals'], results['predictions'], alpha=0.3, s=1)
        min_val = min(results['actuals'].min(), results['predictions'].min())
        max_val = max(results['actuals'].max(), results['predictions'].max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        ax2.set_title('LightGBM: Prediction Scatter Plot')
        ax2.set_xlabel('Actual Price')
        ax2.set_ylabel('Predicted Price')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        ax3 = axes[1, 0]
        errors = results['predictions'] - results['actuals']
        ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--', label='Zero Error')
        ax3.set_title('Prediction Error Distribution')
        ax3.set_xlabel('Error (Predicted - Actual)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Feature importance
        ax4 = axes[1, 1]
        if results['feature_importance'] is not None:
            top_features = results['feature_importance'].head(15)
            ax4.barh(top_features['feature'], top_features['importance'])
            ax4.set_title('Top 15 Feature Importance (Gain)')
            ax4.set_xlabel('Importance Score')
            ax4.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == '__main__':
    """
    Main execution block for LightGBM model training and evaluation.
    
    This script demonstrates:
    1. Data loading and preprocessing
    2. LightGBM model training with early stopping
    3. Feature importance analysis
    4. Results visualization and metrics reporting
    """
    
    print("\n" + "="*60)
    print("LIGHTGBM MODEL - BITCOIN PRICE PREDICTION")
    print("Optimized Gradient Boosting Baseline Model")
    print("="*60 + "\n")
    
    # Run experiment
    results = run_lightgbm_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05
    )
    
    # Display final metrics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"MSE:  {results['metrics']['MSE']:.4f}")
    print(f"MAE:  {results['metrics']['MAE']:.4f}")
    print(f"RMSE: {results['metrics']['RMSE']:.4f}")
    if not np.isnan(results['metrics']['Direction_Accuracy']):
        print(f"Direction Accuracy: {results['metrics']['Direction_Accuracy']*100:.2f}%")
    
    # Visualize if matplotlib is available
    try:
        visualize_results(results)
    except Exception as e:
        print(f"Visualization skipped: {e}")

