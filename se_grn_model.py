"""
SE-GRN (Squeeze-and-Excitation Gated Recurrent Network) for Bitcoin Price Prediction

Proposed model: Combines GRU layers with Squeeze-and-Excitation blocks to
recalibrate features dynamically. Employs attention mechanisms to effectively
capture temporal dependencies in time-series data.

Reference: Zhang, Cui & Gouza (2018) — "SeGen: Sample-ensemble genetic
evolutional network model" [arXiv:1803.08631]

Report Results: MAE 2377.06, MSE 883,273.00

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
    TimeSeriesDataset,
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


class SqueezeExcitationBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for feature recalibration.
    
    The SE block adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.
    
    Architecture:
    1. Squeeze: Global average pooling to get channel descriptor
    2. Excitation: Two FC layers to capture channel-wise dependencies
    3. Scale: Recalibrate original features
    """
    
    def __init__(self, channel: int, reduction: int = 16):
        """
        Initialize SE block.
        
        Args:
            channel: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super(SqueezeExcitationBlock, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE block.
        
        Args:
            x: Input tensor of shape (batch, seq_len, channel)
            
        Returns:
            Recalibrated tensor of same shape
        """
        batch, seq_len, channel = x.size()
        
        # Squeeze: (batch, seq_len, channel) -> (batch, channel)
        y = x.permute(0, 2, 1)  # (batch, channel, seq_len)
        y = self.avg_pool(y).squeeze(-1)  # (batch, channel)
        
        # Excitation: (batch, channel) -> (batch, channel)
        y = self.fc(y)
        
        # Scale: (batch, seq_len, channel) * (batch, 1, channel)
        y = y.unsqueeze(1)  # (batch, 1, channel)
        
        return x * y


class TemporalAttention(nn.Module):
    """
    Temporal Attention mechanism for time series.
    
    Computes attention weights over time steps to focus on
    the most relevant parts of the sequence.
    """
    
    def __init__(self, hidden_dim: int):
        """
        Initialize temporal attention.
        
        Args:
            hidden_dim: Hidden dimension size
        """
        super(TemporalAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention over hidden states.
        
        Args:
            hidden_states: Tensor of shape (batch, seq_len, hidden_dim)
            
        Returns:
            Tuple of (context vector, attention weights)
        """
        # Compute attention scores
        scores = self.attention(hidden_states)  # (batch, seq_len, 1)
        
        # Normalize with softmax
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)
        
        # Compute context vector
        context = torch.sum(weights * hidden_states, dim=1)  # (batch, hidden_dim)
        
        return context, weights.squeeze(-1)


class SEGRN(nn.Module):
    """
    Squeeze-and-Excitation Gated Recurrent Network for time series forecasting.
    
    Architecture:
    1. Input Projection: Projects input features to model dimension
    2. SE Block: Channel-wise feature recalibration
    3. GRU Layers: Temporal pattern learning with gating
    4. Temporal Attention: Focus on important time steps
    5. Output Layer: Final prediction
    
    This model is designed for multivariate time series forecasting
    with complex feature interactions and temporal dependencies.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        se_reduction: int = 8,
        bidirectional: bool = False
    ):
        """
        Initialize SE-GRN model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden dimension size
            num_layers: Number of GRU layers
            dropout: Dropout rate
            se_reduction: SE block reduction ratio
            bidirectional: Whether to use bidirectional GRU
        """
        super(SEGRN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Squeeze-and-Excitation block
        self.se_block = SqueezeExcitationBlock(hidden_dim, reduction=se_reduction)
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Temporal attention
        gru_output_dim = hidden_dim * self.num_directions
        self.attention = TemporalAttention(gru_output_dim)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SE-GRN.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Predictions of shape (batch, 1)
        """
        # Input projection: (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
        x = self.input_projection(x)
        
        # SE block: Channel-wise recalibration
        x = self.se_block(x)
        
        # GRU: (batch, seq_len, hidden_dim) -> (batch, seq_len, hidden_dim * num_directions)
        gru_out, _ = self.gru(x)
        
        # Temporal attention: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
        context, _ = self.attention(gru_out)
        
        # Output layer
        output = self.output_layer(context)
        
        return output


def train_segrn(
    model: SEGRN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 10,
    device: torch.device = None
) -> Tuple[SEGRN, List[float], List[float]]:
    """
    Train SE-GRN model with early stopping.
    
    Args:
        model: SE-GRN model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
        device: Device to train on
        
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training SE-GRN on {device}")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Training phase
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
        
        # Validation phase
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
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
        
        scheduler.step()
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_segrn(
    model: SEGRN,
    test_loader: DataLoader,
    preprocessor: DataPreprocessor,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate SE-GRN model on test data.
    
    Args:
        model: Trained SE-GRN model
        test_loader: Test data loader
        preprocessor: Data preprocessor for inverse transform
        device: Device for evaluation
        
    Returns:
        Tuple of (predictions, actuals, metrics)
    """
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
    
    # Convert to numpy arrays
    predictions_scaled = np.array(all_predictions)
    actuals_scaled = np.array(all_actuals)
    
    # Inverse transform
    predictions = preprocessor.inverse_transform_target(predictions_scaled)
    actuals = preprocessor.inverse_transform_target(actuals_scaled)
    
    # Calculate metrics
    metrics = calculate_metrics(actuals, predictions)
    
    return predictions, actuals, metrics


def run_segrn_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    seq_length: int = 100,
    hidden_dim: int = 128,
    num_layers: int = 2,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict:
    """
    Run complete SE-GRN experiment on Bitcoin price data.
    
    Args:
        data_path: Path to the dataset
        train_ratio: Ratio of data for training
        seq_length: Sequence length for input
        hidden_dim: Hidden dimension size
        num_layers: Number of GRU layers
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Dictionary containing predictions, metrics, and training history
    """
    print("="*60)
    print("SE-GRN Model - Bitcoin Price Prediction")
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
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    model = SEGRN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.2,
        se_reduction=8
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model, train_losses, val_losses = train_segrn(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=15,
        device=device
    )
    
    # Evaluate
    predictions, actuals, metrics = evaluate_segrn(model, test_loader, preprocessor, device)
    
    # Print results
    print_metrics(metrics, "SE-GRN")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model
    }


def visualize_results(results: Dict, save_path: Optional[str] = None) -> None:
    """Visualize SE-GRN prediction results."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Time series comparison
        ax1 = axes[0, 0]
        ax1.plot(results['actuals'][:500], label='Actual', alpha=0.8)
        ax1.plot(results['predictions'][:500], label='Predicted', alpha=0.8)
        ax1.set_title('SE-GRN: Actual vs Predicted (First 500 samples)')
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
        
        # Plot 3: Scatter plot
        ax3 = axes[1, 0]
        ax3.scatter(results['actuals'], results['predictions'], alpha=0.3, s=1)
        min_val = min(results['actuals'].min(), results['predictions'].min())
        max_val = max(results['actuals'].max(), results['predictions'].max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect')
        ax3.set_title('Prediction Scatter Plot')
        ax3.set_xlabel('Actual Price')
        ax3.set_ylabel('Predicted Price')
        ax3.legend()
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
    """
    Main execution block for SE-GRN model training and evaluation.
    """
    
    print("\n" + "="*60)
    print("SE-GRN MODEL - BITCOIN PRICE PREDICTION")
    print("Squeeze-and-Excitation Gated Recurrent Network")
    print("="*60 + "\n")
    
    # Run experiment
    results = run_segrn_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        seq_length=100,
        hidden_dim=128,
        num_layers=2,
        epochs=50,
        batch_size=32,
        learning_rate=0.001
    )
    
    # Display final metrics
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

