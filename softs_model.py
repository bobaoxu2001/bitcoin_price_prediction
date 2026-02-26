"""
SOFTS Model for Bitcoin Price Prediction

Proposed model: Designed for efficient and accurate multivariate time-series
forecasting. Employs a series-core fusion mechanism to model inter-series
relationships effectively. Optimized to capture both long-term trends and
short-term variations, balancing efficiency and accuracy.

SOFTS demonstrated superior predictive accuracy and stability in this study,
outperforming all baseline and advanced deep learning models.

Reference:
    Han et al., "SOFTS: Efficient Multivariate Time Series Forecasting
    with Series-Core Fusion" (NeurIPS 2024)

Report Results: MAE 304.23, MSE 183,679.47 (BEST among proposed models)

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


class TemporalConvBlock(nn.Module):
    """
    Temporal convolution block for capturing local temporal patterns.
    
    Uses dilated causal convolutions to capture patterns at different scales.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super(TemporalConvBlock, self).__init__()
        
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, seq_len)
            
        Returns:
            Output tensor (batch, channels, seq_len)
        """
        residual = self.residual(x)
        
        out = self.conv(x)
        # Remove future timesteps (causal)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        return out + residual


class SeriesCoreFusion(nn.Module):
    """
    Series-Core Fusion mechanism for capturing cross-series dependencies.
    
    This module creates a "core" representation that summarizes global
    information across all series, then fuses it back with individual series.
    """
    
    def __init__(
        self,
        n_series: int,
        d_model: int,
        n_cores: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize Series-Core Fusion.
        
        Args:
            n_series: Number of series (features)
            d_model: Model dimension
            n_cores: Number of core representations
            dropout: Dropout rate
        """
        super(SeriesCoreFusion, self).__init__()
        
        self.n_series = n_series
        self.d_model = d_model
        self.n_cores = n_cores
        
        # Core representations (learnable)
        self.cores = nn.Parameter(torch.randn(n_cores, d_model))
        nn.init.xavier_uniform_(self.cores)
        
        # Series-to-core projection
        self.series_to_core = nn.Linear(d_model, n_cores)
        
        # Core-to-series projection
        self.core_to_series = nn.Linear(n_cores * d_model, d_model)
        
        # Fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Series-Core Fusion.
        
        Args:
            x: Input tensor (batch, seq_len, n_series, d_model)
            
        Returns:
            Fused representation (batch, seq_len, n_series, d_model)
        """
        batch, seq_len, n_series, d_model = x.shape
        
        # Compute attention to cores: (batch, seq_len, n_series, n_cores)
        attn_scores = self.series_to_core(x)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Aggregate core information: (batch, seq_len, n_series, n_cores, d_model)
        expanded_cores = self.cores.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        expanded_cores = expanded_cores.expand(batch, seq_len, n_series, -1, -1)
        
        # Weighted sum of cores
        core_features = torch.einsum('bsnc,bsncd->bsnd', attn_weights, expanded_cores)
        
        # Flatten core features
        core_features = core_features.reshape(batch, seq_len, n_series, -1)
        core_features = self.core_to_series(core_features)
        
        # Fusion gate
        concat_features = torch.cat([x, core_features], dim=-1)
        gate = self.fusion_gate(concat_features)
        
        # Gated fusion
        fused = gate * x + (1 - gate) * core_features
        fused = self.norm(fused)
        fused = self.dropout(fused)
        
        return fused


class AdaptiveFeatureAggregation(nn.Module):
    """
    Adaptive feature aggregation across series.
    
    Learns to weight different series dynamically based on context.
    """
    
    def __init__(self, n_series: int, d_model: int):
        super(AdaptiveFeatureAggregation, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        self.projection = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aggregate features across series.
        
        Args:
            x: Input tensor (batch, seq_len, n_series, d_model)
            
        Returns:
            Aggregated tensor (batch, seq_len, d_model)
        """
        # Compute attention weights
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len, n_series)
        weights = F.softmax(scores, dim=-1)
        
        # Weighted aggregation
        aggregated = torch.einsum('bsn,bsnd->bsd', weights, x)
        
        return self.projection(aggregated)


class SOFTS(nn.Module):
    """
    SOFTS: Series-cOre Fused Time Series forecasting model.
    
    Architecture:
    1. Input Embedding: Project each series to model dimension
    2. Temporal Convolution: Capture local temporal patterns
    3. Series-Core Fusion: Capture cross-series dependencies
    4. Adaptive Aggregation: Dynamic feature weighting
    5. Prediction Head: Generate forecasts
    """
    
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        d_model: int = 64,
        n_cores: int = 4,
        n_conv_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize SOFTS model.
        
        Args:
            input_dim: Number of input features (series)
            seq_length: Input sequence length
            d_model: Model dimension
            n_cores: Number of core representations
            n_conv_layers: Number of temporal conv layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super(SOFTS, self).__init__()
        
        self.input_dim = input_dim
        self.seq_length = seq_length
        self.d_model = d_model
        
        # Series embedding (each feature -> d_model)
        self.series_embedding = nn.Linear(1, d_model)
        
        # Temporal convolution stack
        self.temp_convs = nn.ModuleList()
        for i in range(n_conv_layers):
            dilation = 2 ** i
            self.temp_convs.append(
                TemporalConvBlock(d_model, d_model, kernel_size, dilation, dropout)
            )
        
        # Series-Core Fusion
        self.series_core_fusion = SeriesCoreFusion(
            input_dim, d_model, n_cores, dropout
        )
        
        # Adaptive feature aggregation
        self.feature_aggregation = AdaptiveFeatureAggregation(input_dim, d_model)
        
        # Temporal aggregation
        self.temporal_attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SOFTS.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Predictions (batch, 1)
        """
        batch, seq_len, n_series = x.shape
        
        # Embed each series: (batch, seq_len, n_series, d_model)
        x = x.unsqueeze(-1)  # (batch, seq_len, n_series, 1)
        x = self.series_embedding(x)  # (batch, seq_len, n_series, d_model)
        
        # Apply temporal convolutions to each series
        # Reshape: (batch * n_series, d_model, seq_len)
        x = x.permute(0, 2, 3, 1).reshape(batch * n_series, self.d_model, seq_len)
        
        for conv in self.temp_convs:
            x = conv(x)
        
        # Reshape back: (batch, seq_len, n_series, d_model)
        x = x.reshape(batch, n_series, self.d_model, -1).permute(0, 3, 1, 2)
        
        # Series-Core Fusion
        x = self.series_core_fusion(x)
        
        # Aggregate across series: (batch, seq_len, d_model)
        x = self.feature_aggregation(x)
        
        # Temporal attention for aggregation
        temporal_scores = self.temporal_attention(x).squeeze(-1)  # (batch, seq_len)
        temporal_weights = F.softmax(temporal_scores, dim=-1)
        
        # Aggregate across time: (batch, d_model)
        x = torch.einsum('bs,bsd->bd', temporal_weights, x)
        
        # Prediction
        output = self.predictor(x)
        
        return output


def train_softs(
    model: SOFTS,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 0.01,
    patience: int = 10,
    device: torch.device = None
) -> Tuple[SOFTS, List[float], List[float]]:
    """Train SOFTS model with early stopping."""
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
    
    print(f"Training SOFTS on {device}")
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


def evaluate_softs(
    model: SOFTS,
    test_loader: DataLoader,
    preprocessor: DataPreprocessor,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluate SOFTS model on test data."""
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


def run_softs_experiment(
    data_path: str = 'filtered_df.csv',
    train_ratio: float = 0.8,
    seq_length: int = 100,
    d_model: int = 64,
    n_cores: int = 4,
    n_conv_layers: int = 3,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Dict:
    """Run complete SOFTS experiment."""
    print("="*60)
    print("SOFTS Model - Bitcoin Price Prediction")
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
    print(f"Number of cores: {n_cores}")
    
    # Initialize model
    model = SOFTS(
        input_dim=input_dim,
        seq_length=seq_length,
        d_model=d_model,
        n_cores=n_cores,
        n_conv_layers=n_conv_layers,
        dropout=0.1
    )
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model, train_losses, val_losses = train_softs(
        model, train_loader, val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        patience=15,
        device=device
    )
    
    # Evaluate
    predictions, actuals, metrics = evaluate_softs(model, test_loader, preprocessor, device)
    
    print_metrics(metrics, "SOFTS")
    
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
    print("SOFTS MODEL - BITCOIN PRICE PREDICTION")
    print("Series-cOre Fused Time Series Forecasting")
    print("="*60 + "\n")
    
    results = run_softs_experiment(
        data_path='filtered_df.csv',
        train_ratio=0.8,
        seq_length=100,
        d_model=64,
        n_cores=4,
        n_conv_layers=3,
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

