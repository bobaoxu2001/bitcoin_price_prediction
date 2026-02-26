"""
Times-FM Inspired Model for Bitcoin Price Prediction

Proposed model: Architecture inspired by Google's Time-Series Foundation Model
(Times-FM), a state-of-the-art 200M-parameter model for zero-shot or one-shot
time-series forecasting. Implements an encoder-decoder architecture to model
complex temporal relationships with patch-based tokenization and causal attention.

This is a simplified implementation inspired by the Times-FM architecture.
The original model supports uni-variate and multivariate covariates.

Reference:
    Das et al., "A Decoder-Only Foundation Model for Time-Series Forecasting"
    (ICML 2024)

Report Results: MAE 2672.28, MSE 3,600,000.00

Authors: Sam Lai, Zexuan Yang, Yichao Yang, Ao Xu
Group 36 - NYU Capstone Project
Date: 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Dict, Optional, List
import math
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


class PatchEmbedding(nn.Module):
    """
    Patch-based embedding for time series.
    
    Divides the input time series into non-overlapping patches
    and projects them to the model dimension.
    """
    
    def __init__(
        self,
        input_dim: int,
        patch_size: int,
        d_model: int
    ):
        """
        Initialize patch embedding.
        
        Args:
            input_dim: Number of input features
            patch_size: Size of each patch
            d_model: Model dimension
        """
        super(PatchEmbedding, self).__init__()
        
        self.patch_size = patch_size
        self.projection = nn.Linear(input_dim * patch_size, d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert input to patches.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Patch embeddings (batch, num_patches, d_model)
        """
        batch, seq_len, input_dim = x.shape
        num_patches = seq_len // self.patch_size
        
        # Reshape into patches: (batch, num_patches, patch_size * input_dim)
        x = x[:, :num_patches * self.patch_size, :]
        x = x.reshape(batch, num_patches, self.patch_size * input_dim)
        
        # Project patches to model dimension
        x = self.projection(x)
        x = self.norm(x)
        
        return x


class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Positional Encoding (RoPE) for better positional information.
    
    RoPE encodes position information through rotation in the complex plane,
    allowing better generalization to different sequence lengths.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(RotaryPositionalEncoding, self).__init__()
        
        # Compute rotary frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position encodings
        positions = torch.arange(max_len).float()
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional encoding."""
        seq_len = x.shape[1]
        
        cos = self.cos_cached[:seq_len, :].unsqueeze(0)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0)
        
        # Apply rotation
        x_rotated = x * cos + self._rotate_half(x) * sin
        
        return x_rotated
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dims."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class TimesFMAttention(nn.Module):
    """
    Multi-head attention for Times-FM with causal masking.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1
    ):
        super(TimesFMAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask for autoregressive modeling
        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), 
                diagonal=1
            ).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(batch, seq_len, self.d_model)
        
        return self.out(out)


class TimesFMBlock(nn.Module):
    """
    Single decoder block for Times-FM.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super(TimesFMBlock, self).__init__()
        
        self.attention = TimesFMAttention(d_model, n_heads, dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attention(self.norm1(x))
        
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class TimesFMInspired(nn.Module):
    """
    Times-FM Inspired Foundation Model for Time Series.
    
    This model implements key architectural elements from Times-FM:
    1. Patch-based tokenization of multivariate time series
    2. Decoder-only transformer architecture
    3. Causal attention for autoregressive forecasting
    
    This is a smaller-scale implementation suitable for training
    on individual datasets rather than foundation model pre-training.
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        patch_size: int = 16,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize Times-FM inspired model.
        
        Args:
            input_dim: Number of input features
            seq_length: Input sequence length
            patch_size: Size of each patch for tokenization
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
        """
        super(TimesFMInspired, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_patches = seq_length // patch_size
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(input_dim, patch_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            TimesFMBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Times-FM inspired model.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Predictions (batch, 1)
        """
        # Patch embedding: (batch, num_patches, d_model)
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply decoder blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Use the last patch representation for prediction
        x = x[:, -1, :]
        
        # Output head
        output = self.head(x)
        
        return output


def train_timesfm(
    model: TimesFMInspired,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    learning_rate: float = 0.0005,
    weight_decay: float = 0.01,
    patience: int = 10,
    device: torch.device = None
) -> Tuple[TimesFMInspired, List[float], List[float]]:
    """
    Train Times-FM inspired model with early stopping.
    
    Args:
        model: Times-FM model instance
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
        weight_decay=weight_decay,
        betas=(0.9, 0.98)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"Training Times-FM on {device}")
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
            
            # Early stopping
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
        
        scheduler.step()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def evaluate_timesfm(
    model: TimesFMInspired,
    test_loader: DataLoader,
    preprocessor: DataPreprocessor,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate Times-FM model on test data."""
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


def run_timesfm_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    seq_length: int = 96,
    patch_size: int = 16,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 4,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.0005
) -> Dict:
    """Run complete Times-FM experiment."""
    print("="*60)
    print("Times-FM Inspired Model - Bitcoin Price Prediction")
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
    print(f"Patch size: {patch_size}")
    print(f"Number of patches: {seq_length // patch_size}")
    
    # Initialize model
    model = TimesFMInspired(
        input_dim=input_dim,
        seq_length=seq_length,
        patch_size=patch_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_model * 4,
        dropout=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model, train_losses, val_losses = train_timesfm(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=15,
        device=device
    )
    
    # Evaluate
    predictions, actuals, metrics = evaluate_timesfm(model, test_loader, preprocessor, device)
    
    print_metrics(metrics, "Times-FM Inspired")
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model
    }


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TIMES-FM INSPIRED MODEL - BITCOIN PRICE PREDICTION")
    print("Foundation Model Inspired Architecture")
    print("="*60 + "\n")
    
    results = run_timesfm_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        seq_length=96,
        patch_size=16,
        d_model=128,
        n_heads=8,
        n_layers=4,
        epochs=50,
        batch_size=32,
        learning_rate=0.0005
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"MSE:  {results['metrics']['MSE']:.4f}")
    print(f"MAE:  {results['metrics']['MAE']:.4f}")
    print(f"RMSE: {results['metrics']['RMSE']:.4f}")

