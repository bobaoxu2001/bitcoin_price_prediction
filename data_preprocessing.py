"""
Data Preprocessing Module for Bitcoin Price Prediction

This module provides comprehensive data loading, cleaning, and preprocessing
utilities for time series forecasting with multimodal financial data.

Features:
- Bitcoin hourly price data
- NASDAQ, Gold, VIX market indicators (daily -> hourly alignment)
- Social media sentiment scores (Twitter, Reddit, Bitcointalk)

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)


class BitcoinDataLoader:
    """
    Handles loading and initial processing of the Bitcoin prediction dataset.
    
    The dataset contains:
    - Bitcoin hourly closing prices (listing_close)
    - Market indicators: NASDAQ, Gold, VIX (daily, forward-filled to hourly)
    - Sentiment scores: Twitter, Reddit, Bitcointalk optimistic/negative
    - Engineered features: moving averages, lag features, returns
    """
    
    def __init__(self, data_path: str = 'filtered_df.csv'):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the filtered dataset CSV file
        """
        self.data_path = data_path
        self.df = None
        self.feature_columns = None
        self.target_column = 'target_nexthour'
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and parse the dataset with proper date handling.
        
        Returns:
            DataFrame with parsed datetime index
        """
        self.df = pd.read_csv(self.data_path)
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Store feature column names (excluding date and target)
        exclude_cols = ['date', self.target_column]
        self.feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        
        print(f"Loaded dataset: {self.df.shape[0]} samples, {len(self.feature_columns)} features")
        print(f"Date range: {self.df['date'].min()} to {self.df['date'].max()}")
        
        return self.df
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_columns
    
    def get_data_info(self) -> Dict:
        """Return dataset information summary."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        return {
            'n_samples': len(self.df),
            'n_features': len(self.feature_columns),
            'date_range': (self.df['date'].min(), self.df['date'].max()),
            'target_column': self.target_column,
            'missing_values': self.df.isnull().sum().sum()
        }


class DataPreprocessor:
    """
    Comprehensive data preprocessing for time series forecasting.
    
    Handles:
    - Missing value imputation (forward/backward fill for time series)
    - Feature scaling (StandardScaler or MinMaxScaler)
    - Train/test splitting with time series considerations
    - Sequence creation for deep learning models
    """
    
    def __init__(
        self,
        feature_scaler: str = 'standard',
        target_scaler: str = 'standard'
    ):
        """
        Initialize preprocessor with scaling options.
        
        Args:
            feature_scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
            target_scaler: 'standard' for StandardScaler, 'minmax' for MinMaxScaler
        """
        self.feature_scaler = self._get_scaler(feature_scaler)
        self.target_scaler = self._get_scaler(target_scaler)
        self.is_fitted = False
        
    def _get_scaler(self, scaler_type: str):
        """Return appropriate scaler based on type string."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using time series appropriate methods.
        
        Uses forward fill followed by backward fill to maintain temporal consistency.
        
        Args:
            df: Input DataFrame with potential missing values
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Forward fill then backward fill for time series data
        df_clean = df_clean.ffill().bfill()
        
        # Drop any remaining rows with NaN in target column
        if 'target_nexthour' in df_clean.columns:
            df_clean = df_clean.dropna(subset=['target_nexthour'])
        
        return df_clean
    
    def prepare_features_target(
        self,
        df: pd.DataFrame,
        target_column: str = 'target_nexthour'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target from DataFrame.
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple of (features DataFrame, target Series)
        """
        exclude_cols = ['date', target_column]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        return X, y
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scalers and transform data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Tuple of (scaled features array, scaled target array)
        """
        X_scaled = self.feature_scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
        
        self.is_fitted = True
        
        return X_scaled, y_scaled.flatten()
    
    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transform data using fitted scalers.
        
        Args:
            X: Feature DataFrame
            y: Optional target Series
            
        Returns:
            Tuple of (scaled features array, scaled target array or None)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
            
        X_scaled = self.feature_scaler.transform(X)
        
        if y is not None:
            y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1))
            return X_scaled, y_scaled.flatten()
        
        return X_scaled, None
    
    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions to original scale.
        
        Args:
            y_scaled: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform() first.")
            
        return self.target_scaler.inverse_transform(
            y_scaled.reshape(-1, 1)
        ).flatten()
    
    def create_train_test_split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create time-series appropriate train/test split.
        
        For time series, we use chronological split to avoid data leakage.
        
        Args:
            df: Input DataFrame (should be sorted by date)
            train_ratio: Proportion of data for training
            
        Returns:
            Tuple of (train_df, test_df)
        """
        split_idx = int(len(df) * train_ratio)
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Train set: {len(train_df)} samples ({train_df['date'].min()} to {train_df['date'].max()})")
        print(f"Test set: {len(test_df)} samples ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, test_df


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series sequence modeling.
    
    Creates sliding window sequences suitable for RNN, LSTM, Transformer models.
    """
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 100
    ):
        """
        Initialize the dataset.
        
        Args:
            features: Scaled feature array of shape (n_samples, n_features)
            targets: Scaled target array of shape (n_samples,)
            seq_length: Length of input sequences
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length
        
    def __len__(self) -> int:
        return len(self.features) - self.seq_length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sequence-target pair.
        
        Args:
            idx: Index of the sequence start
            
        Returns:
            Tuple of (sequence tensor, target tensor)
        """
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return x, y


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    
    Implements expanding or sliding window validation to properly
    evaluate time series forecasting models without data leakage.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        train_size: int = 30000,
        test_size: int = 100,
        step_size: int = 100
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            n_splits: Number of validation splits
            train_size: Size of training window
            test_size: Size of test window (forecast horizon)
            step_size: Step size between windows
        """
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        
    def split(
        self,
        X: np.ndarray,
        y: np.ndarray = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for walk-forward validation.
        
        Args:
            X: Feature array
            y: Optional target array (not used, for API compatibility)
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(X)
        splits = []
        
        for i in range(self.n_splits):
            train_start = i * self.step_size
            train_end = train_start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break
                
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
            
        return splits


def create_data_loader(
    features: np.ndarray,
    targets: np.ndarray,
    seq_length: int = 100,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Create a PyTorch DataLoader for training.
    
    Args:
        features: Scaled feature array
        targets: Scaled target array
        seq_length: Sequence length for LSTM/Transformer models
        batch_size: Batch size for training
        shuffle: Whether to shuffle data (set False for validation)
        
    Returns:
        PyTorch DataLoader
    """
    dataset = TimeSeriesDataset(features, targets, seq_length)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )
    
    return loader


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics for regression.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing MSE, MAE, RMSE, MAPE, and directional accuracy
    """
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (avoid division by zero)
    non_zero_mask = y_true != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        mape = np.nan
    
    # Directional Accuracy (percentage of correct direction predictions)
    if len(y_true) > 1:
        actual_direction = np.diff(y_true) > 0
        predicted_direction = np.diff(y_pred) > 0
        direction_accuracy = np.mean(actual_direction == predicted_direction)
    else:
        direction_accuracy = np.nan
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Direction_Accuracy': direction_accuracy
    }


def print_metrics(metrics: Dict[str, float], model_name: str = 'Model') -> None:
    """
    Print formatted evaluation metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model for display
    """
    print(f"\n{'='*50}")
    print(f"{model_name} Evaluation Metrics")
    print('='*50)
    for name, value in metrics.items():
        if not np.isnan(value):
            if name == 'MAPE':
                print(f"{name}: {value:.4f}%")
            elif name == 'Direction_Accuracy':
                print(f"{name}: {value:.4f} ({value*100:.2f}%)")
            else:
                print(f"{name}: {value:.4f}")
    print('='*50)


# Convenience function for quick data preparation
def prepare_data(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    seq_length: int = 100,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataPreprocessor, List[str]]:
    """
    Convenience function to prepare data for model training.
    
    Args:
        data_path: Path to the dataset
        train_ratio: Proportion of data for training
        seq_length: Sequence length for deep learning models
        batch_size: Batch size for DataLoader
        
    Returns:
        Tuple of (train_loader, test_loader, preprocessor, feature_names)
    """
    # Load data
    loader = BitcoinDataLoader(data_path)
    df = loader.load_data()
    feature_names = loader.get_feature_names()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # Split
    train_df, test_df = preprocessor.create_train_test_split(df_clean, train_ratio)
    
    # Prepare features and targets
    X_train, y_train = preprocessor.prepare_features_target(train_df)
    X_test, y_test = preprocessor.prepare_features_target(test_df)
    
    # Scale
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)
    
    # Create DataLoaders
    train_loader = create_data_loader(
        X_train_scaled, y_train_scaled,
        seq_length=seq_length,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = create_data_loader(
        X_test_scaled, y_test_scaled,
        seq_length=seq_length,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader, preprocessor, feature_names


if __name__ == '__main__':
    # Example usage
    print("Bitcoin Price Prediction - Data Preprocessing Module")
    print("="*60)
    
    # Load and display data info
    loader = BitcoinDataLoader('filtered_df.csv')
    df = loader.load_data()
    info = loader.get_data_info()
    
    print(f"\nDataset Info:")
    print(f"  Samples: {info['n_samples']}")
    print(f"  Features: {info['n_features']}")
    print(f"  Date Range: {info['date_range'][0]} to {info['date_range'][1]}")
    print(f"  Missing Values: {info['missing_values']}")
    
    # Show sample of feature names
    feature_names = loader.get_feature_names()
    print(f"\nSample Features (first 10):")
    for name in feature_names[:10]:
        print(f"  - {name}")
    
    print(f"\n... and {len(feature_names) - 10} more features")

