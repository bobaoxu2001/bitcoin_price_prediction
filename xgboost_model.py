"""
XGBoost Model for Bitcoin Price Prediction

This module implements the XGBoost (eXtreme Gradient Boosting) model for
Bitcoin price forecasting. XGBoost is an optimized gradient boosting library
designed for speed and performance.

Key Features:
- Regularized learning to prevent overfitting
- Parallel tree building for efficiency
- Built-in handling of missing values
- Feature importance analysis

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
import xgboost as xgb
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


class XGBoostPredictor:
    """
    XGBoost-based predictor for Bitcoin price forecasting.
    
    This implementation uses gradient boosting with decision trees,
    optimized for tabular time series regression tasks.
    
    Attributes:
        model: Trained XGBoost model (Booster object)
        params: Model hyperparameters
        feature_importance: Feature importance scores after training
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 10
    ):
        """
        Initialize XGBoost predictor with hyperparameters.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Step size shrinkage (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns per tree
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            early_stopping_rounds: Early stopping patience
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stopping_rounds = early_stopping_rounds
        
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': RANDOM_SEED,
            'verbosity': 0
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
    ) -> 'XGBoostPredictor':
        """
        Fit the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets
            feature_names: Optional list of feature names
            
        Returns:
            Self for method chaining
        """
        print("Training XGBoost model...")
        
        self.feature_names = feature_names
        
        # Handle NaN values
        X_train = np.nan_to_num(X_train, nan=0.0)
        if X_val is not None:
            X_val = np.nan_to_num(X_val, nan=0.0)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        
        evals = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            evals.append((dval, 'eval'))
        
        # Train model with early stopping
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=False
        )
        
        # Store feature importance
        importance = self.model.get_score(importance_type='weight')
        if feature_names:
            self.feature_importance = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False)
        
        self.is_fitted = True
        print(f"XGBoost training completed. Best iteration: {self.model.best_iteration}")
        
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
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        predictions = self.model.predict(dtest)
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top N important features.
        
        Args:
            top_n: Number of top features to return
            
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
        self.model = xgb.Booster()
        self.model.load_model(path)
        self.is_fitted = True
        print(f"Model loaded from {path}")


def create_lag_features(
    X: np.ndarray,
    lags: List[int] = [1, 6, 12, 24]
) -> np.ndarray:
    """
    Create additional lag features for tabular model.
    
    Args:
        X: Original feature matrix
        lags: List of lag periods to create
        
    Returns:
        Feature matrix with additional lag features
    """
    n_samples, n_features = X.shape
    lag_features = []
    
    for lag in lags:
        # Shift features by lag
        lagged = np.zeros((n_samples, n_features))
        lagged[lag:] = X[:-lag]
        lag_features.append(lagged)
    
    # Concatenate original and lag features
    X_with_lags = np.concatenate([X] + lag_features, axis=1)
    
    return X_with_lags


def run_xgboost_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1
) -> Dict:
    """
    Run complete XGBoost experiment on Bitcoin price data.
    
    Args:
        data_path: Path to the dataset
        train_ratio: Ratio of data for training
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Learning rate
        
    Returns:
        Dictionary containing predictions, metrics, and model info
    """
    print("="*60)
    print("XGBoost Model - Bitcoin Price Prediction")
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
    
    # Create validation split from training data
    val_size = int(len(X_train_scaled) * 0.1)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train_scaled[-val_size:]
    X_train_final = X_train_scaled[:-val_size]
    y_train_final = y_train_scaled[:-val_size]
    
    print(f"\nTraining set: {len(X_train_final)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test_scaled)} samples")
    
    # Initialize and train model
    predictor = XGBoostPredictor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )
    
    predictor.fit(
        X_train_final, y_train_final,
        X_val, y_val,
        feature_names=feature_names
    )
    
    # Make predictions
    predictions_scaled = predictor.predict(X_test_scaled)
    
    # Inverse transform predictions to original scale
    predictions = preprocessor.inverse_transform_target(predictions_scaled)
    actuals = preprocessor.inverse_transform_target(y_test_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    # Print results
    print_metrics(metrics, "XGBoost")
    
    # Print top features
    print("\nTop 10 Important Features:")
    print("-" * 40)
    importance_df = predictor.get_feature_importance(top_n=10)
    for _, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.0f}")
    
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
    Visualize XGBoost prediction results.
    
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
        ax1.set_title('XGBoost: Actual vs Predicted (First 500 samples)')
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
        ax2.set_title('XGBoost: Prediction Scatter Plot')
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
            ax4.set_title('Top 15 Feature Importance')
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
    Main execution block for XGBoost model training and evaluation.
    
    This script demonstrates:
    1. Data loading and preprocessing
    2. XGBoost model training with early stopping
    3. Feature importance analysis
    4. Results visualization and metrics reporting
    """
    
    print("\n" + "="*60)
    print("XGBOOST MODEL - BITCOIN PRICE PREDICTION")
    print("Gradient Boosting Baseline Model")
    print("="*60 + "\n")
    
    # Run experiment
    results = run_xgboost_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        n_estimators=200,
        max_depth=6,
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

