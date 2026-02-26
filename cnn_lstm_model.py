"""
CNN-LSTM (Convolutional Neural Network - Long Short-Term Memory) for Bitcoin Price Prediction

Proposed model: Integrates convolutional layers for feature extraction from
raw time-series data (identifying patterns and trends in historical prices)
with LSTM layers to capture temporal dependencies and long-term sequences.
Combines the strengths of CNNs and LSTMs to address the challenges of
volatility and trend variations in Bitcoin price prediction.

Architecture:
1. Multi-scale Conv1D layers with batch normalization
2. Bidirectional LSTM layers
3. Temporal attention mechanism
4. Dense prediction head

Reference:
    Shi et al. (2015) — "Convolutional LSTM Network: A Machine Learning
    Approach for Precipitation Nowcasting" (NeurIPS 2015)

Report Results: MAE 1941.32, MSE 6,730,000.00

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - NYU Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Optional, List
import warnings

from data_preprocessing import (
    BitcoinDataLoader,
    DataPreprocessor,
    create_data_loader,
    calculate_metrics,
    print_metrics
)

warnings.filterwarnings('ignore')

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class CNNFeatureExtractor(nn.Module):
    """
    Multi-scale CNN feature extractor.
    
    Uses parallel convolutions with different kernel sizes to capture
    patterns at multiple temporal scales.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1
    ):
        super(CNNFeatureExtractor, self).__init__()
        
        # Multi-scale convolutions
        self.conv_small = nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=5, padding=2)
        self.conv_large = nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=7, padding=3)
        
        # Stack of conv blocks
        total_channels = hidden_dim
        self.conv_blocks = nn.ModuleList()
        
        for i in range(n_layers):
            self.conv_blocks.append(
                ConvBlock(total_channels, hidden_dim, kernel_size=3, dropout=dropout)
            )
            total_channels = hidden_dim
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.final_norm = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using multi-scale convolutions.
        
        Args:
            x: Input tensor (batch, input_dim, seq_len)
            
        Returns:
            Feature tensor (batch, hidden_dim, reduced_seq_len)
        """
        # Multi-scale feature extraction
        feat_small = F.gelu(self.conv_small(x))
        feat_medium = F.gelu(self.conv_medium(x))
        feat_large = F.gelu(self.conv_large(x))
        
        # Concatenate multi-scale features
        x = torch.cat([feat_small, feat_medium, feat_large], dim=1)
        
        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = self.final_norm(x)
        
        return x


class LSTMSequenceEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for sequential pattern learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(LSTMSequenceEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence with LSTM.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (all hidden states, final hidden state)
        """
        outputs, (hidden, _) = self.lstm(x)
        outputs = self.layer_norm(outputs)
        
        # Combine bidirectional hidden states
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return outputs, hidden


class TemporalAttention(nn.Module):
    """
    Attention mechanism over LSTM outputs.
    """
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs.
        
        Args:
            lstm_outputs: LSTM outputs (batch, seq_len, hidden_dim)
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        # Compute attention scores
        scores = self.attention(lstm_outputs)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum
        context = torch.sum(weights * lstm_outputs, dim=1)
        
        return context, weights.squeeze(-1)


class CNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid Model for Time Series Forecasting.
    
    This architecture combines the strengths of CNNs and LSTMs:
    - CNN layers extract local patterns and features
    - LSTM layers capture long-range temporal dependencies
    - Attention mechanism focuses on important time steps
    
    The model is particularly effective for financial time series
    where both local patterns (e.g., intraday movements) and
    long-term trends are important.
    """
    
    def __init__(
        self,
        input_dim: int,
        cnn_hidden: int = 64,
        lstm_hidden: int = 128,
        n_cnn_layers: int = 3,
        n_lstm_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_dim: Number of input features
            cnn_hidden: CNN hidden dimension
            lstm_hidden: LSTM hidden dimension
            n_cnn_layers: Number of CNN layers
            n_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            bidirectional: Use bidirectional LSTM
        """
        super(CNNLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.num_directions = 2 if bidirectional else 1
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(
            input_dim=input_dim,
            hidden_dim=cnn_hidden,
            n_layers=n_cnn_layers,
            dropout=dropout
        )
        
        # LSTM sequence encoder
        self.lstm = LSTMSequenceEncoder(
            input_dim=cnn_hidden,
            hidden_dim=lstm_hidden,
            n_layers=n_lstm_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Temporal attention
        lstm_output_dim = lstm_hidden * self.num_directions
        self.attention = TemporalAttention(lstm_output_dim)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(lstm_hidden // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of CNN-LSTM.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Predictions (batch, 1)
        """
        batch_size = x.size(0)
        
        # CNN feature extraction: (batch, input_dim, seq_len) -> (batch, cnn_hidden, reduced_len)
        x = x.permute(0, 2, 1)
        cnn_features = self.cnn(x)
        
        # Prepare for LSTM: (batch, reduced_len, cnn_hidden)
        cnn_features = cnn_features.permute(0, 2, 1)
        
        # LSTM encoding
        lstm_outputs, final_hidden = self.lstm(cnn_features)
        
        # Attention over LSTM outputs
        context, _ = self.attention(lstm_outputs)
        
        # Combine attention context with final hidden state
        combined = torch.cat([context, final_hidden], dim=1)
        
        # Prediction
        output = self.predictor(combined)
        
        return output


def train_cnn_lstm(
    model: CNNLSTM,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 10,
    device: torch.device = None
) -> Tuple[CNNLSTM, List[float], List[float]]:
    """
    Train CNN-LSTM model with early stopping.
    
    Args:
        model: CNN-LSTM model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        patience: Early stopping patience
        device: Training device
        
    Returns:
        Tuple of (trained model, train losses, val losses)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training CNN-LSTM on {device}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions.squeeze(), batch_y.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        if val_loader is not None:
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    predictions = model(batch_x)
                    loss = criterion(predictions.squeeze(), batch_y.squeeze())
                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}, Val: {avg_val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train: {avg_train_loss:.6f}")
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_cnn_lstm(
    model: CNNLSTM,
    test_loader: DataLoader,
    preprocessor: DataPreprocessor,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate CNN-LSTM model on test data."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    all_predictions = []
    all_actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            predictions = model(batch_x)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_actuals.extend(batch_y.numpy().flatten())
    
    predictions_scaled = np.array(all_predictions)
    actuals_scaled = np.array(all_actuals)
    
    predictions = preprocessor.inverse_transform_target(predictions_scaled)
    actuals = preprocessor.inverse_transform_target(actuals_scaled)
    
    metrics = calculate_metrics(actuals, predictions)
    
    return predictions, actuals, metrics


def run_cnn_lstm_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    seq_length: int = 100,
    cnn_hidden: int = 64,
    lstm_hidden: int = 128,
    n_cnn_layers: int = 3,
    n_lstm_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict:
    """
    Run complete CNN-LSTM experiment on Bitcoin price data.
    
    Args:
        data_path: Path to the dataset
        train_ratio: Ratio of data for training
        seq_length: Sequence length for input
        cnn_hidden: CNN hidden dimension
        lstm_hidden: LSTM hidden dimension
        n_cnn_layers: Number of CNN layers
        n_lstm_layers: Number of LSTM layers
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Dictionary containing predictions, metrics, and training history
    """
    print("="*60)
    print("CNN-LSTM Model - Bitcoin Price Prediction")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    loader = BitcoinDataLoader(data_path)
    df = loader.load_data()
    feature_names = loader.get_feature_names()
    input_dim = len(feature_names)
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    
    # Split data
    train_df, test_df = preprocessor.create_train_test_split(df_clean, train_ratio)
    
    # Prepare features and targets
    X_train, y_train = preprocessor.prepare_features_target(train_df)
    X_test, y_test = preprocessor.prepare_features_target(test_df)
    
    # Scale data
    X_train_scaled, y_train_scaled = preprocessor.fit_transform(X_train, y_train)
    X_test_scaled, y_test_scaled = preprocessor.transform(X_test, y_test)
    
    # Create validation split
    val_size = int(len(X_train_scaled) * 0.1)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train_scaled[-val_size:]
    X_train_final = X_train_scaled[:-val_size]
    y_train_final = y_train_scaled[:-val_size]
    
    # Create DataLoaders
    train_loader = create_data_loader(X_train_final, y_train_final, seq_length, batch_size, shuffle=True)
    val_loader = create_data_loader(X_val, y_val, seq_length, batch_size, shuffle=False)
    test_loader = create_data_loader(X_test_scaled, y_test_scaled, seq_length, batch_size, shuffle=False)
    
    print(f"\nInput dimension: {input_dim}")
    print(f"Sequence length: {seq_length}")
    print(f"CNN hidden: {cnn_hidden}, LSTM hidden: {lstm_hidden}")
    
    # Initialize model
    model = CNNLSTM(
        input_dim=input_dim,
        cnn_hidden=cnn_hidden,
        lstm_hidden=lstm_hidden,
        n_cnn_layers=n_cnn_layers,
        n_lstm_layers=n_lstm_layers,
        dropout=0.2,
        bidirectional=True
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model, train_losses, val_losses = train_cnn_lstm(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=15,
        device=device
    )
    
    # Evaluate
    predictions, actuals, metrics = evaluate_cnn_lstm(model, test_loader, preprocessor, device)
    
    print_metrics(metrics, "CNN-LSTM")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model
    }


def visualize_results(results: Dict, save_path: Optional[str] = None) -> None:
    """Visualize CNN-LSTM prediction results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Time series
        ax1 = axes[0, 0]
        ax1.plot(results['actuals'][:500], label='Actual', alpha=0.8)
        ax1.plot(results['predictions'][:500], label='Predicted', alpha=0.8)
        ax1.set_title('CNN-LSTM: Actual vs Predicted')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training curves
        ax2 = axes[0, 1]
        ax2.plot(results['train_losses'], label='Train Loss')
        if results['val_losses']:
            ax2.plot(results['val_losses'], label='Val Loss')
        ax2.set_title('Training and Validation Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Scatter
        ax3 = axes[1, 0]
        ax3.scatter(results['actuals'], results['predictions'], alpha=0.3, s=1)
        min_val = min(results['actuals'].min(), results['predictions'].min())
        max_val = max(results['actuals'].max(), results['predictions'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax3.set_title('Prediction Scatter Plot')
        ax3.set_xlabel('Actual Price')
        ax3.set_ylabel('Predicted Price')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Error distribution
        ax4 = axes[1, 1]
        errors = results['predictions'] - results['actuals']
        ax4.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--')
        ax4.set_title('Prediction Error Distribution')
        ax4.set_xlabel('Error')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("CNN-LSTM MODEL - BITCOIN PRICE PREDICTION")
    print("Hybrid Convolutional-Recurrent Architecture")
    print("="*60 + "\n")
    
    results = run_cnn_lstm_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        seq_length=100,
        cnn_hidden=64,
        lstm_hidden=128,
        n_cnn_layers=3,
        n_lstm_layers=2,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"MSE:  {results['metrics']['MSE']:.4f}")
    print(f"MAE:  {results['metrics']['MAE']:.4f}")
    print(f"RMSE: {results['metrics']['RMSE']:.4f}")
    
    # Visualize
    try:
        visualize_results(results)
    except Exception as e:
        print(f"Visualization skipped: {e}")

