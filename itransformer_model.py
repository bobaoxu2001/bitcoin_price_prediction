"""
iTransformer Model for Bitcoin Price Prediction

Proposed model: A state-of-the-art time-series forecasting model that
restructures Transformers by embedding each time point as an independent
variable token. Designed to improve multivariate correlation modeling and
capture complex temporal dynamics for both short-term and long-term predictions.
Enhances sequence representation learning through a feed-forward network
while maintaining computational efficiency.

Reference:
    Liu et al., "iTransformer: Inverted Transformers Are Effective for
    Time Series Forecasting" (arXiv:2310.06625, 2023)

Report Results: MAE 1948.00, MSE 8,123,500.00

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
import math
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


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class FeedForward(nn.Module):
    """
    Feed-forward network with GELU activation.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = q.size(0)
        
        # Linear projections and reshape
        q = self.query(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(out)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer with self-attention and feed-forward.
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_out))
        
        return x


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for Time Series Forecasting.
    
    Instead of applying attention across time steps, iTransformer
    treats each feature/variable as a token and applies attention
    across features. This allows better capture of multivariate
    correlations in the data.
    
    Architecture:
    1. Temporal Embedding: Embed time series for each feature
    2. Feature-wise Attention: Attention across features (inverted)
    3. Prediction Head: Generate forecasts
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1
    ):
        """
        Initialize iTransformer.
        
        Args:
            input_dim: Number of input features (variables)
            seq_length: Length of input sequence
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(iTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Embed each feature's time series separately
        # Input: (batch, seq_len, input_dim) -> (batch, input_dim, d_model)
        self.feature_embedding = nn.Linear(seq_length, d_model)
        
        # Learnable positional encoding for features
        self.feature_pos_encoding = nn.Parameter(
            torch.zeros(1, input_dim, d_model)
        )
        nn.init.trunc_normal_(self.feature_pos_encoding, std=0.02)
        
        # Transformer encoder layers (attention across features)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(input_dim * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of iTransformer.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Predictions of shape (batch, 1)
        """
        batch_size = x.size(0)
        
        # Transpose to (batch, input_dim, seq_len) for feature-wise processing
        x = x.permute(0, 2, 1)
        
        # Embed each feature's time series: (batch, input_dim, d_model)
        x = self.feature_embedding(x)
        
        # Add feature positional encoding
        x = x + self.feature_pos_encoding
        
        # Apply transformer encoder layers (attention across features)
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # Flatten and predict
        x = x.reshape(batch_size, -1)  # (batch, input_dim * d_model)
        output = self.predictor(x)
        
        return output


def train_itransformer(
    model: iTransformer,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 10,
    device: torch.device = None
) -> Tuple[iTransformer, List[float], List[float]]:
    """
    Train iTransformer model with early stopping.
    
    Args:
        model: iTransformer model instance
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training iTransformer on {device}")
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


def evaluate_itransformer(
    model: iTransformer,
    test_loader: DataLoader,
    preprocessor: DataPreprocessor,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate iTransformer model on test data.
    
    Args:
        model: Trained iTransformer model
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


def run_itransformer_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    seq_length: int = 100,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict:
    """
    Run complete iTransformer experiment on Bitcoin price data.
    
    Args:
        data_path: Path to the dataset
        train_ratio: Ratio of data for training
        seq_length: Sequence length for input
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of encoder layers
        epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        
    Returns:
        Dictionary containing predictions, metrics, and training history
    """
    print("="*60)
    print("iTransformer Model - Bitcoin Price Prediction")
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
    print(f"Model dimension: {d_model}")
    print(f"Attention heads: {n_heads}")
    print(f"Encoder layers: {n_layers}")
    
    # Initialize model
    model = iTransformer(
        input_dim=input_dim,
        seq_length=seq_length,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        dropout=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    model, train_losses, val_losses = train_itransformer(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=15,
        device=device
    )
    
    # Evaluate
    predictions, actuals, metrics = evaluate_itransformer(model, test_loader, preprocessor, device)
    
    # Print results
    print_metrics(metrics, "iTransformer")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model
    }


def visualize_results(results: Dict, save_path: Optional[str] = None) -> None:
    """Visualize iTransformer prediction results."""
    try:
        # Try to use the comprehensive visualization module
        from visualization import generate_all_figures
        generate_all_figures(results, 'iTransformer', output_dir='figures')
    except ImportError:
        # Fallback to basic visualization
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Time series comparison
            ax1 = axes[0, 0]
            ax1.plot(results['actuals'][:500], label='Actual', alpha=0.8)
            ax1.plot(results['predictions'][:500], label='Predicted', alpha=0.8)
            ax1.set_title('iTransformer: Actual vs Predicted (First 500 samples)')
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
    Main execution block for iTransformer model training and evaluation.
    """
    
    print("\n" + "="*60)
    print("iTRANSFORMER MODEL - BITCOIN PRICE PREDICTION")
    print("Inverted Transformer for Time Series Forecasting")
    print("="*60 + "\n")
    
    # Run experiment
    results = run_itransformer_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        seq_length=100,
        d_model=64,
        n_heads=4,
        n_layers=3,
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

