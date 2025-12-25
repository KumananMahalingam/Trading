import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import re
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
import glob

class ResidualLSTM(nn.Module):
    """LSTM with residual connections"""
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bidirectional = bidirectional
        output_size = hidden_size * 2 if bidirectional else hidden_size
        self.residual_fc = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        residual = self.residual_fc(x)
        # Ensure shapes match
        if residual.dim() == 2:
            residual = residual.unsqueeze(1)
        if residual.shape[1] != lstm_out.shape[1]:
            # Repeat or interpolate residual to match sequence length
            residual = residual.repeat(1, lstm_out.shape[1], 1)
        return lstm_out + residual

class MultiHeadAttention(nn.Module):
    """Multi-head attention for time series"""
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention = torch.softmax(attention, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.fc_out(out)

class TCNBlock(nn.Module):
    """Temporal Convolutional Network block"""
    def __init__(self, input_size, output_size, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # Use 'same' padding to maintain sequence length
        padding = ((kernel_size - 1) * dilation) // 2
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size,
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size,
                              padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        self.downsample = nn.Conv1d(input_size, output_size, 1) if input_size != output_size else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)

        # Apply first convolution
        out = self.relu(self.conv1(x))
        out = self.dropout(out)

        # Apply second convolution
        out = self.conv2(out)

        # Ensure shapes match before adding residual
        if out.shape != residual.shape:
            # Crop or pad to match shapes
            if out.shape[2] > residual.shape[2]:
                out = out[:, :, :residual.shape[2]]
            elif out.shape[2] < residual.shape[2]:
                residual = residual[:, :, :out.shape[2]]

        out = self.relu(out + residual)
        return out

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()
        self.class_weights = class_weights

    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)

        # Direction loss with class balancing
        pred_sign = torch.sign(predictions)
        true_sign = torch.sign(targets)

        if self.class_weights is not None:
            # Weight errors based on true class
            up_mask = (true_sign > 0).float()
            down_mask = (true_sign < 0).float()

            weights = up_mask * self.class_weights[0] + down_mask * self.class_weights[1]
            direction_loss = torch.mean((pred_sign != true_sign).float() * weights)
        else:
            direction_loss = torch.mean((pred_sign != true_sign).float())

        # Large movement emphasis
        large_moves = torch.abs(targets) > 0.02
        if large_moves.any():
            large_move_loss = torch.mean(torch.abs(predictions[large_moves] - targets[large_moves]))
        else:
            large_move_loss = torch.tensor(0.0).to(predictions.device)

        return self.alpha * mse_loss + self.beta * direction_loss + self.gamma * large_move_loss


class EnhancedStockDataset(Dataset):
    """
    Dataset with data augmentation and advanced features
    """
    def __init__(self, data, window_size=20, augment=True, dual_stream=True):
        super().__init__()
        self.window_size = window_size
        self.augment = augment
        self.dual_stream = dual_stream

        # Extract features
        alpha_cols = [col for col in data.columns if col.startswith('alpha_')]
        price_col = 'close'
        temporal_cols = ['day_of_week', 'day_of_month', 'month']

        # Extract price features (open, high, low, volume)
        price_features = ['open', 'high', 'low', 'volume']
        price_features = [col for col in price_features if col in data.columns]

        self.alphas = data[alpha_cols].values.astype(np.float32)
        self.prices = data[price_features].values.astype(np.float32)
        self.temporal = data[temporal_cols].values.astype(np.float32)
        self.targets = data['target'].values.astype(np.float32)
        self.dates = data['date'].values if 'date' in data.columns else None

        # Additional features for augmentation
        self.returns = data['target'].values.astype(np.float32) if 'target' in data.columns else None

        # Create sliding windows
        self.samples = []
        for i in range(len(data) - window_size):
            self.samples.append(i)

        print(f"  Created {len(self.samples)} samples with window_size={window_size}")

    def __len__(self):
        return len(self.samples)

    def augment_sequence(self, sequence, augmentation_type='all'):
        """Apply data augmentation to a sequence"""
        if not self.augment:
            return sequence

        if augmentation_type == 'all':
            augmentation_type = np.random.choice(['noise', 'scale', 'time_warp'])

        if augmentation_type == 'noise':
            # Add correlated noise (like Brownian motion)
            noise_level = np.random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, sequence.shape)
            noise = np.cumsum(noise, axis=0)
            return sequence + noise

        elif augmentation_type == 'scale':
            # Scale the entire sequence
            scale = np.random.uniform(0.95, 1.05)
            return sequence * scale

        elif augmentation_type == 'time_warp':
            # Time warping (stretching/squeezing)
            try:
                orig_time = np.arange(len(sequence))
                warp_points = np.random.normal(0, 0.1, 3).cumsum()
                warp_time = orig_time + warp_points * (len(sequence) / 10)
                cs = CubicSpline(warp_time, sequence, axis=0)
                return cs(orig_time)
            except:
                return sequence

        return sequence

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        end_idx = start_idx + self.window_size

        if self.dual_stream:
            # Alpha stream
            alphas_window = self.alphas[start_idx:end_idx].copy()

            # Price stream (close + temporal)
            price_window = self.prices[start_idx:end_idx, 0:1].copy()  # Only close price
            temporal_window = self.temporal[start_idx:end_idx].copy()

            # Apply augmentation during training
            if self.augment and idx % 3 == 0:  # Augment 1/3 of samples
                alphas_window = self.augment_sequence(alphas_window)
                price_window = self.augment_sequence(price_window)

            prices_temporal_window = np.concatenate([price_window, temporal_window], axis=1)

            target = self.targets[end_idx]

            return (
                torch.FloatTensor(alphas_window),
                torch.FloatTensor(prices_temporal_window),
                torch.FloatTensor([target])
            )
        else:
            # Single stream (all features concatenated)
            all_features = np.concatenate([
                self.alphas[start_idx:end_idx],
                self.prices[start_idx:end_idx],
                self.temporal[start_idx:end_idx]
            ], axis=1)

            # Apply augmentation
            if self.augment and idx % 3 == 0:
                all_features = self.augment_sequence(all_features)

            target = self.targets[end_idx]

            return (
                torch.FloatTensor(all_features),
                torch.FloatTensor([target])
            )

class EnhancedAlphaComputer:
    """Improved alpha computation with caching and validation"""
    def __init__(self, df):
        self.df = df.copy()
        self.available_columns = set(df.columns)
        self.alpha_cache = {}

    def parse_alpha_formula(self, formula_str):
        """Extract formula with better parsing"""
        # Remove alpha label and clean
        formula = re.sub(r'^[αA]lpha\s*\d+[\s:=→-]+', '', formula_str, flags=re.IGNORECASE)
        formula = formula.strip().strip('"').strip("'")

        # Replace special characters
        replacements = {
            '×': '*', '÷': '/', '−': '-', '–': '-',
            '^': '**', 'exp': 'np.exp', 'log': 'np.log',
            'sqrt': 'np.sqrt', 'abs': 'np.abs'
        }
        for old, new in replacements.items():
            formula = formula.replace(old, new)

        return formula

    def validate_formula(self, formula):
        """Validate formula syntax and columns"""
        try:
            # Check for invalid operations
            if re.search(r'/0([^.]|$)', formula) or '/ 0' in formula:
                print(f"    Warning: Potential division by zero: {formula[:50]}...")

            # Check column existence
            pattern = r'([A-Za-z][A-Za-z0-9_]*)(?=\s*[+\-*/()]|\s*$)'
            variables = re.findall(pattern, formula)

            missing_cols = []
            for var in variables:
                # Skip mathematical functions and constants
                if var in ['np', 'exp', 'log', 'sqrt', 'abs', 'e', 'pi']:
                    continue
                if var not in self.available_columns:
                    missing_cols.append(var)

            return len(missing_cols) == 0, missing_cols

        except Exception as e:
            return False, [f"Parse error: {str(e)}"]

    def compute_alpha(self, formula, alpha_name):
        """Compute alpha with caching and error handling"""
        # Check cache
        cache_key = f"{formula}_{len(self.df)}"
        if cache_key in self.alpha_cache:
            return self.alpha_cache[cache_key]

        try:
            is_valid, missing = self.validate_formula(formula)
            if not is_valid:
                print(f"    ✗ Invalid formula {alpha_name}: {missing}")
                return None

            # Show what columns are available
            print(f"    Computing {alpha_name}...")
            print(f"    Formula: {formula[:80]}...")

            # Extract variables used in formula
            pattern = r'\b([A-Za-z][A-Za-z0-9_]*)\b'
            variables_in_formula = set(re.findall(pattern, formula))
            variables_in_formula -= {'np', 'exp', 'log', 'sqrt', 'abs', 'e', 'pi'}

            missing_vars = [v for v in variables_in_formula if v not in self.available_columns]
            if missing_vars:
                print(f"    ⚠️  Missing columns: {missing_vars}")

            # Evaluate formula with access to DataFrame columns
            namespace = {'np': np}
            namespace.update(self.df.to_dict('series'))

            # Now eval can access both numpy functions and dataframe columns
            result = eval(formula, {"__builtins__": {}}, namespace)

            if result is None or (hasattr(result, 'size') and result.size == 0):
                return None

            # Convert to Series
            if isinstance(result, (int, float)):
                result = pd.Series([result] * len(self.df), index=self.df.index)
            elif isinstance(result, pd.DataFrame):
                if result.shape[1] > 0:
                    result = result.iloc[:, 0]
                else:
                    return None
            elif not isinstance(result, pd.Series):
                result = pd.Series(result, index=self.df.index)

            # Check for invalid values
            if result.isna().all() or result.isnull().all():
                return None

            # Remove extreme outliers (cap at 3 standard deviations)
            mean = result.mean()
            std = result.std()
            if std > 0:
                result = result.clip(mean - 3*std, mean + 3*std)

            # Cache result
            self.alpha_cache[cache_key] = result

            return result

        except Exception as e:
            print(f"    Error computing {alpha_name}: {str(e)[:100]}")
            import traceback
            traceback.print_exc()  # Print full error for debugging
            return None

    def add_alphas_from_text(self, alpha_text, max_alphas=5):
        """Parse and compute alphas from LLM text"""

        print(f"\n  DEBUG: Received alpha_text:")
        print(f"  Length: {len(alpha_text)} characters")
        print(f"  First 200 chars: {alpha_text[:200]}")
        print(f"  ---")

        lines = alpha_text.split('\n')
        print(f"  DEBUG: Split into {len(lines)} lines")

        alphas_added = 0
        attempted = 0

        print(f"  Available columns: {len(self.available_columns)} total")

        for line in lines:
            if alphas_added >= max_alphas:
                break

            # Show each line being processed
            if '=' in line and any(c.isalpha() for c in line):
                attempted += 1
                print(f"  DEBUG: Attempting line {attempted}: {line[:80]}")

                # Extract alpha name and formula
                match = re.match(r'([αA](?:lpha)?\s*\d+)\s*[:=\-→]+\s*(.+)', line.strip())
                if match:
                    alpha_name, formula_str = match.groups()
                    print(f"    Matched: {alpha_name} -> {formula_str[:60]}")
                    formula = self.parse_alpha_formula(formula_str)

                    if not formula or len(formula) < 3:
                        print(f"    ✗ Formula too short or empty: '{formula}'")
                        continue

                    # Compute alpha
                    alpha_values = self.compute_alpha(formula, alpha_name)

                    if alpha_values is not None:
                        alphas_added += 1
                        col_name = f'alpha_{alphas_added}'
                        self.df[col_name] = alpha_values

                        # Calculate correlation with target for debugging
                        if 'target' in self.df.columns:
                            corr = self.df[col_name].corr(self.df['target'])
                            print(f"  ✓ {col_name}: corr={corr:.3f}, {formula[:60]}...")
                        else:
                            print(f"  ✓ {col_name}: {formula[:60]}...")
                    else:
                        print(f"  Could not compute: {line[:80]}...")
                else:
                    print(f"    Regex didn't match: {line[:80]}")

        print(f"  Summary: {alphas_added}/{attempted} alphas computed successfully")

        if alphas_added == 0:
            print(f"  No alphas computed! Using simple technical indicators...")
            # Fallback to technical alphas
            self.df = self.add_fallback_alphas()
            alphas_added = 5

        return self.df, alphas_added

    def add_fallback_alphas(self):
        """Add fallback technical alphas when LLM fails"""
        print("  Adding fallback technical alphas...")

        # Simple momentum alphas
        if 'close' in self.df.columns:
            self.df['alpha_1'] = self.df['close'].pct_change(5)  # 5-day return
            self.df['alpha_2'] = self.df['close'].pct_change(10)  # 10-day return
            self.df['alpha_3'] = self.df['close'].rolling(20).mean() / self.df['close'] - 1  # SMA ratio

        if 'RSI' in self.df.columns:
            self.df['alpha_4'] = (self.df['RSI'] - 50) / 50  # Normalized RSI

        if 'MACD' in self.df.columns and 'MACD_Signal' in self.df.columns:
            self.df['alpha_5'] = self.df['MACD'] - self.df['MACD_Signal']  # MACD histogram

        return self.df

class ImprovedDualStreamLSTM(nn.Module):
    """
    Dual-stream LSTM with:
    1. Residual connections
    2. Multi-head attention
    3. TCN parallel processing
    4. Bayesian dropout for uncertainty
    """

    def __init__(self, num_alphas=5, hidden_size=128, num_layers=3, dropout=0.3, num_heads=4):
        super(ImprovedDualStreamLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout

        # Alpha stream with residual LSTM
        self.alpha_lstm = ResidualLSTM(
            input_size=num_alphas,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        # Price stream with residual LSTM
        self.price_lstm = ResidualLSTM(
            input_size=4,  # close + temporal features
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        # TCN for capturing local patterns (parallel to LSTM)
        self.alpha_tcn = nn.Sequential(
            TCNBlock(num_alphas, hidden_size // 2, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(hidden_size // 2, hidden_size, kernel_size=3, dilation=2, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        self.price_tcn = nn.Sequential(
            TCNBlock(4, hidden_size // 2, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(hidden_size // 2, hidden_size, kernel_size=3, dilation=2, dropout=dropout),
            nn.AdaptiveAvgPool1d(1)
        )

        # Multi-head attention mechanisms
        self.alpha_attention = MultiHeadAttention(hidden_size * 2, num_heads=num_heads)
        self.price_attention = MultiHeadAttention(hidden_size * 2, num_heads=num_heads)

        # Cross-attention between streams
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feature fusion layers - Use LayerNorm instead of BatchNorm
        total_features = hidden_size * 8  # (LSTM + TCN) * 2 streams * bidirectional

        self.fusion_layers = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.LayerNorm(512),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),  # Changed from BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final prediction with uncertainty estimation
        self.fc_mean = nn.Linear(64, 1)
        self.fc_std = nn.Linear(64, 1)
        self.softplus = nn.Softplus()

        # Dropout for Bayesian inference
        self.dropout_layer = nn.Dropout(dropout)

        # Added flag to control dropout behavior
        self.bayesian_dropout = True

    def attention_net(self, lstm_output, attention_layer):
        """Apply attention with residual connection"""
        attention_out = attention_layer(lstm_output)

        # Combine with original
        combined = lstm_output + attention_out

        # Global average pooling
        context = torch.mean(combined, dim=1)
        return context

    def forward(self, alphas, prices_temporal, n_samples=1, training=True):

        # Process alpha stream
        alpha_lstm_out = self.alpha_lstm(alphas)
        alpha_features = self.attention_net(alpha_lstm_out, self.alpha_attention)

        # Process price stream
        price_lstm_out = self.price_lstm(prices_temporal)
        price_features = self.attention_net(price_lstm_out, self.price_attention)

        # TCN processing (permute for Conv1d: [batch, features, seq_len])
        alpha_tcn_out = self.alpha_tcn(alphas.transpose(1, 2)).squeeze(-1)
        price_tcn_out = self.price_tcn(prices_temporal.transpose(1, 2)).squeeze(-1)

        # Cross-attention between streams
        cross_attn, _ = self.cross_attention(alpha_lstm_out, price_lstm_out, price_lstm_out)
        cross_features = torch.mean(cross_attn, dim=1)

        # Combine all features
        combined = torch.cat([
            alpha_features,
            price_features,
            alpha_tcn_out,
            price_tcn_out,
            cross_features
        ], dim=1)

        # Fusion layers
        features = self.fusion_layers(combined)

        # Monte-Carlo dropout
        if training:
            # During training: single forward pass with dropout
            features = self.dropout_layer(features)
            mean = self.fc_mean(features)
            return mean
        else:
            # During inference: multiple forward passes for uncertainty
            if n_samples <= 1:
                mean = self.fc_mean(features)
                std = torch.zeros_like(mean)
                return mean, std

            # Enable dropout during inference for Bayesian estimation
            self.dropout_layer.train()

            predictions = []
            for _ in range(n_samples):
                # Each sample gets different dropout mask
                sampled_features = self.dropout_layer(features)
                pred = self.fc_mean(sampled_features)
                predictions.append(pred)

            # Return to eval mode
            self.dropout_layer.eval()

            predictions = torch.stack(predictions, dim=0)
            mean_pred = predictions.mean(dim=0)

            std_pred = predictions.std(dim=0)
            # Add small epsilon to prevent zero uncertainty
            std_pred = std_pred + 1e-6

            return mean_pred, std_pred

def create_balanced_hybrid_loss(train_df):
    """
    Create loss function with proper class balancing
    """
    if train_df is None or 'target' not in train_df.columns:
        return HybridLoss(alpha=0.7, beta=0.2, gamma=0.1, class_weights=None)

    # Calculate class distribution
    up_count = (train_df['target'] > 0).sum()
    down_count = (train_df['target'] < 0).sum()
    neutral_count = (train_df['target'] == 0).sum()

    total = up_count + down_count  # Ignore neutral for weighting

    if up_count == 0 or down_count == 0 or total == 0:
        print("Extreme class imbalance detected!")
        return HybridLoss(alpha=0.7, beta=0.2, gamma=0.1, class_weights=None)

    # Inverse frequency weighting
    up_weight = total / (2.0 * up_count)
    down_weight = total / (2.0 * down_count)

    # Normalize weights so they sum to 2.0
    weight_sum = up_weight + down_weight
    up_weight = 2.0 * up_weight / weight_sum
    down_weight = 2.0 * down_weight / weight_sum

    print(f"\n  Class Distribution:")
    print(f"  UP samples: {up_count} ({up_count/total*100:.1f}%)")
    print(f"  DOWN samples: {down_count} ({down_count/total*100:.1f}%)")
    print(f"  Neutral: {neutral_count}")
    print(f"\n  Loss Weights:")
    print(f"  UP weight: {up_weight:.3f}")
    print(f"  DOWN weight: {down_weight:.3f}")

    return HybridLoss(alpha=0.7, beta=0.3, gamma=0.1, class_weights=[up_weight, down_weight])

def create_target_safely(df):
    """
    Create target WITHOUT data leakage
    """
    df = df.copy()

    # This ensures we don't use future information
    if 'close' in df.columns:
        # Next day's return
        df['target'] = df['close'].pct_change(1).shift(-1)

        # Drop the last row (has NaN target)
        df = df[:-1].copy()

    return df

def add_advanced_features(df):
    """Add advanced technical indicators and features"""
    df = df.copy()

    if df.empty:
        return df

    # Ensure numeric columns
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Price-based features
    if 'close' in df.columns:
        # Multi-timeframe returns
        for window in [1, 3, 5, 10, 20, 50]:
            df[f'return_{window}d'] = df['close'].pct_change(window)

        # Rolling volatility
        for window in [5, 10, 20, 50]:
            df[f'volatility_{window}d'] = df['close'].pct_change().rolling(window).std()

        # Price position within recent range
        for window in [20, 50, 200]:
            df[f'price_position_{window}d'] = (
                (df['close'] - df['low'].rolling(window).min()) /
                (df['high'].rolling(window).max() - df['low'].rolling(window).min())
            )

        # Trend strength (slope of linear regression)
        for window in [20, 50]:
            def calculate_slope(series):
                if len(series) < window:
                    return np.nan
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                return slope / np.std(series) if np.std(series) > 0 else 0

            df[f'trend_strength_{window}d'] = df['close'].rolling(window).apply(
                calculate_slope, raw=False
            )

    # Volume-based features
    if 'volume' in df.columns:
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

        # Volume-price correlation
        df['volume_price_corr_10'] = df['volume'].rolling(10).corr(df['close'])

    # Sentiment momentum features
    sentiment_cols = [col for col in df.columns if 'Sentiment' in col and not col.startswith('alpha_')]
    for col in sentiment_cols:
        df[f'{col}_MA5'] = df[col].rolling(5).mean()
        df[f'{col}_MA20'] = df[col].rolling(20).mean()
        df[f'{col}_Change'] = df[col].diff()
        df[f'{col}_Momentum'] = df[col] - df[col].rolling(5).mean()

    # Economic data features
    econ_cols = [col for col in df.columns if any(ind in col for ind in
                ['GDP', 'CPI', 'Unemployment', 'Fed_Funds', 'Treasury', 'VIX'])]
    for col in econ_cols:
        df[f'{col}_Change'] = df[col].pct_change()

    # Market regime detection
    if 'close' in df.columns and 'volatility_20d' in df.columns:
        df['market_regime'] = 0
        bull_condition = (
            (df['close'] > df['close'].rolling(50).mean()) &
            (df['volatility_20d'] < df['volatility_20d'].rolling(50).mean())
        )
        bear_condition = (
            (df['close'] < df['close'].rolling(50).mean()) &
            (df['volatility_20d'] > df['volatility_20d'].rolling(50).mean())
        )
        df.loc[bull_condition, 'market_regime'] = 1
        df.loc[bear_condition, 'market_regime'] = -1

    # Fill NaN values (forward fill for time series, 0 for others)
    for col in df.columns:
        if col not in ['date', 'target']:
            if df[col].dtype in ['float64', 'int64']:
                # Forward fill time series data
                df[col] = df[col].ffill().bfill().fillna(0)

    print(f"  Added {len(df.columns)} total features")
    return df

def select_top_features(df, target_col='target', top_k=30, method='correlation',
                       exclude_cols=None, min_correlation=0.02):
    """
    Select features with highest predictive power for the target
    """
    if exclude_cols is None:
        exclude_cols = ['date', 'ticker', 'symbol']

    # Get all feature columns
    feature_cols = [col for col in df.columns
                   if col != target_col and col not in exclude_cols]

    # Remove columns with too many NaNs
    valid_cols = []
    for col in feature_cols:
        if df[col].notna().sum() > len(df) * 0.5:
            valid_cols.append(col)

    print(f"\n{'='*80}")
    print(f"FEATURE SELECTION")
    print(f"{'='*80}")
    print(f"Total features available: {len(feature_cols)}")
    print(f"Features with sufficient data: {len(valid_cols)}")

    # Prepare data
    X = df[valid_cols].copy()
    y = df[target_col].copy()

    # Remove rows with NaN in target
    valid_idx = y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    # Fill remaining NaNs
    X = X.fillna(X.mean())

    feature_scores = pd.DataFrame({'feature': valid_cols})

    # Calculate correlations
    if method in ['correlation', 'both']:
        print(f"\nCalculating correlations...")
        correlations = []
        for col in valid_cols:
            try:
                corr = X[col].corr(y)
                correlations.append(abs(corr) if not np.isnan(corr) else 0)
            except:
                correlations.append(0)

        feature_scores['correlation'] = correlations
        feature_scores['abs_correlation'] = feature_scores['correlation'].abs()

    # Calculate mutual information
    if method in ['mutual_info', 'both']:
        print(f"Calculating mutual information...")
        try:
            mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=3)
            feature_scores['mutual_info'] = mi_scores
        except Exception as e:
            print(f"  Warning: Could not calculate mutual info: {e}")
            feature_scores['mutual_info'] = 0

    # Combined score
    if method == 'both':
        feature_scores['correlation_norm'] = (
            feature_scores['abs_correlation'] / feature_scores['abs_correlation'].max()
        )
        feature_scores['mi_norm'] = (
            feature_scores['mutual_info'] / feature_scores['mutual_info'].max()
            if feature_scores['mutual_info'].max() > 0 else 0
        )
        feature_scores['combined_score'] = (
            0.6 * feature_scores['correlation_norm'] +
            0.4 * feature_scores['mi_norm']
        )
        sort_by = 'combined_score'
    elif method == 'correlation':
        sort_by = 'abs_correlation'
    else:
        sort_by = 'mutual_info'

    # Sort and filter
    feature_scores = feature_scores.sort_values(by=sort_by, ascending=False)

    if method in ['correlation', 'both']:
        feature_scores = feature_scores[
            feature_scores['abs_correlation'] >= min_correlation
        ]

    # Select top K
    top_features = feature_scores.head(top_k)['feature'].tolist()

    # Print results
    print(f"\n{'='*80}")
    print(f"TOP {len(top_features)} FEATURES SELECTED")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Feature':<40} {'Corr':<10} {'Score':<10}")
    print(f"{'-'*80}")

    for i, row in feature_scores.head(top_k).iterrows():
        rank = feature_scores.index.get_loc(i) + 1
        feat = row['feature'][:38]
        corr_str = f"{row.get('abs_correlation', 0):.4f}"
        score_str = f"{row.get(sort_by, 0):.4f}"
        print(f"{rank:<6} {feat:<40} {corr_str:<10} {score_str:<10}")

    return top_features, feature_scores

def prepare_data_with_fixes(df, ticker, alpha_text, window_size=30,
                            use_feature_selection=True, top_k=30):
    """
    Enhanced data preparation with all fixes applied
    """
    print(f"\n{'='*80}")
    print(f"PREPARING DATA FOR {ticker} (WITH FIXES)")
    print(f"{'='*80}")

    # Add temporal features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_of_week'] = df['date'].dt.dayofweek / 6.0
        df['day_of_month'] = df['date'].dt.day / 31.0
        df['month'] = df['date'].dt.month / 12.0

    # Add advanced features
    df = add_advanced_features(df)

    # Create target SAFELY (no data leakage)
    df = create_target_safely(df)

    # Compute alphas
    alpha_computer = EnhancedAlphaComputer(df)
    df, num_alphas = alpha_computer.add_alphas_from_text(alpha_text, max_alphas=5)

    selected_features = None
    feature_scores = None

    if use_feature_selection:
        print(f"\n{'='*80}")
        print("APPLYING FEATURE SELECTION")
        print(f"{'='*80}")

        # Select top features
        selected_features, feature_scores = select_top_features(
            df=df,
            target_col='target',
            top_k=top_k,
            method='both',  # Use both correlation and mutual info
            exclude_cols=['date', 'ticker'],
            min_correlation=0.01  # Keep features with at least 1% correlation
        )

        # Always include alpha features
        alpha_features = [col for col in df.columns if col.startswith('alpha_')]
        for alpha_feat in alpha_features:
            if alpha_feat not in selected_features:
                selected_features.append(alpha_feat)

        # Always include temporal features
        temporal_features = ['day_of_week', 'day_of_month', 'month']
        temporal_features = [f for f in temporal_features if f in df.columns]
        for temp_feat in temporal_features:
            if temp_feat not in selected_features:
                selected_features.append(temp_feat)

        # Always include price features for dual stream
        price_features = ['open', 'high', 'low', 'close', 'volume']
        price_features = [f for f in price_features if f in df.columns]
        for price_feat in price_features:
            if price_feat not in selected_features:
                selected_features.append(price_feat)

        print(f"\n✓ Final feature set: {len(selected_features)} features")
        print(f"  - Alpha features: {len([f for f in selected_features if 'alpha_' in f])}")
        print(f"  - Temporal features: {len([f for f in selected_features if f in temporal_features])}")
        print(f"  - Price features: {len([f for f in selected_features if f in price_features])}")
        print(f"  - Selected features: {len([f for f in selected_features if f not in alpha_features + temporal_features + price_features])}")

        # Keep only selected features + target + date
        keep_cols = selected_features + ['target']
        if 'date' in df.columns:
            keep_cols.append('date')

        # Remove duplicates while preserving order
        keep_cols = list(dict.fromkeys(keep_cols))

        df = df[keep_cols].copy()

        print(f"\n✓ Reduced from original features to {len(selected_features)} features")

    # Check class balance BEFORE splitting
    if 'target' in df.columns:
        up_pct = (df['target'] > 0).sum() / len(df) * 100
        down_pct = (df['target'] < 0).sum() / len(df) * 100
        neutral_pct = (df['target'] == 0).sum() / len(df) * 100

        print(f"\n📊 Target Distribution (BEFORE split):")
        print(f"  UP: {up_pct:.1f}%")
        print(f"  DOWN: {down_pct:.1f}%")
        print(f"  NEUTRAL: {neutral_pct:.1f}%")

        # If extremely imbalanced
        if up_pct > 70 or down_pct > 70:
            print(f"⚠️  WARNING: Severe class imbalance detected!")
            print(f"  Consider using stratified sampling or collecting more data")

    # Handle missing values
    for col in df.columns:
        if col not in ['date', 'target']:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].ffill().bfill().fillna(0)

    # Drop rows with NaN target
    df = df.dropna(subset=['target'])

    if len(df) < window_size + 50:
        print(f"❌ ERROR: Insufficient data")
        return None, None, None, None, 0, None, None

    # Create binary labels for stratification
    df['target_class'] = (df['target'] > 0).astype(int)

    # This gives more stable validation metrics
    train_val_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=0.2,  # Keep 20% for test
        stratify=df['target_class'],
        random_state=42
    )

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.20,
        stratify=df.iloc[train_val_indices]['target_class'],
        random_state=42
    )

    # Sort indices to maintain temporal order within each set
    train_indices = np.sort(train_indices)
    val_indices = np.sort(val_indices)
    test_indices = np.sort(test_indices)

    train_df = df.iloc[train_indices].copy()
    val_df = df.iloc[val_indices].copy()
    test_df = df.iloc[test_indices].copy()

    # Remove temporary column
    for split_df in [train_df, val_df, test_df]:
        if 'target_class' in split_df.columns:
            split_df.drop('target_class', axis=1, inplace=True)

    print(f"\n Stratified split:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Check class distribution in each split
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        up_pct = (split_df['target'] > 0).sum() / len(split_df) * 100
        down_pct = (split_df['target'] < 0).sum() / len(split_df) * 100
        print(f"  {name} - UP: {up_pct:.1f}%, DOWN: {down_pct:.1f}%")

    # 8. Scale features (same as before)
    alpha_cols = [col for col in df.columns if col.startswith('alpha_')]
    price_cols = ['open', 'high', 'low', 'close', 'volume']

    scalers = {}

    for group_name, group_cols in [('alphas', alpha_cols), ('price', price_cols), ('target', ['target'])]:
        existing_cols = [col for col in group_cols if col in train_df.columns]
        if not existing_cols:
            continue

        scaler = StandardScaler()
        train_df[existing_cols] = scaler.fit_transform(train_df[existing_cols])
        val_df[existing_cols] = scaler.transform(val_df[existing_cols])
        test_df[existing_cols] = scaler.transform(test_df[existing_cols])

        scalers[group_name] = scaler

    # Create datasets
    train_dataset = EnhancedStockDataset(train_df, window_size=window_size, augment=True)
    val_dataset = EnhancedStockDataset(val_df, window_size=window_size, augment=False)
    test_dataset = EnhancedStockDataset(test_df, window_size=window_size, augment=False)

    # Adaptive batch size
    batch_size = min(64, max(16, len(train_dataset) // 10))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_dates = test_df['date'].values if 'date' in test_df.columns else None

    print(f"\n✓ Data preparation complete with fixes applied!")

    return train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df


def train_epoch_enhanced(model, train_loader, criterion, optimizer, device, scheduler=None):
    """Enhanced training epoch with gradient clipping and logging"""
    model.train()
    total_loss = 0
    total_direction_correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(train_loader):
        alphas, prices_temporal, targets = batch
        alphas = alphas.to(device)
        prices_temporal = prices_temporal.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(alphas, prices_temporal)
        loss = criterion(predictions, targets)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Calculate direction accuracy
        pred_sign = torch.sign(predictions).detach()
        true_sign = torch.sign(targets).detach()
        direction_correct = (pred_sign == true_sign).float().sum().item()

        total_loss += loss.item()
        total_direction_correct += direction_correct
        total_samples += len(targets)

        # Learning rate scheduler step
        if scheduler and hasattr(scheduler, 'batch_step'):
            scheduler.batch_step()

    avg_loss = total_loss / len(train_loader)
    direction_accuracy = (total_direction_correct / total_samples) * 100

    return avg_loss, direction_accuracy

def validate_enhanced(model, val_loader, criterion, device, n_samples=5):
    """Enhanced validation with uncertainty estimation"""
    model.eval()
    total_loss = 0
    total_direction_correct = 0
    total_samples = 0

    all_predictions = []
    all_targets = []
    all_uncertainties = []

    with torch.no_grad():
        for batch in val_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)
            targets = targets.to(device)

            # Skip batches with size 1 to avoid BatchNorm issues
            if alphas.shape[0] == 1:
                continue

            # Get predictions with uncertainty
            predictions, uncertainties = model(alphas, prices_temporal, n_samples=n_samples, training=False)

            loss = criterion(predictions, targets)

            # Calculate direction accuracy
            pred_sign = torch.sign(predictions)
            true_sign = torch.sign(targets)
            direction_correct = (pred_sign == true_sign).float().sum().item()

            total_loss += loss.item()
            total_direction_correct += direction_correct
            total_samples += len(targets)

            # Store for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())

    # Handle case where all batches were skipped
    if total_samples == 0:
        print("  All validation batches skipped (batch_size=1)")
        return {
            'loss': float('inf'),
            'direction_accuracy': 0,
            'mse': float('inf'),
            'mae': float('inf'),
            'uncertainty_corr': 0
        }, np.array([]), np.array([]), np.array([])

    avg_loss = total_loss / len(val_loader)
    direction_accuracy = (total_direction_correct / total_samples) * 100

    # Calculate additional metrics
    all_predictions = np.concatenate(all_predictions).flatten()
    all_targets = np.concatenate(all_targets).flatten()
    all_uncertainties = np.concatenate(all_uncertainties).flatten()

    mse = np.mean((all_predictions - all_targets) ** 2)
    mae = np.mean(np.abs(all_predictions - all_targets))

    # Calibration: uncertainty should correlate with error
    errors = np.abs(all_predictions - all_targets)
    uncertainty_corr = np.corrcoef(errors, all_uncertainties)[0, 1] if len(errors) > 1 else 0

    metrics = {
        'loss': avg_loss,
        'direction_accuracy': direction_accuracy,
        'mse': mse,
        'mae': mae,
        'uncertainty_corr': uncertainty_corr
    }

    return metrics, all_predictions, all_targets, all_uncertainties

def train_with_fixes(model, train_loader, val_loader, num_epochs=150,
                     learning_rate=3e-4, device='cpu', train_df=None):
    """
    Training with proper class balancing and regularization
    """
    print(f"\n{'='*80}")
    print("TRAINING WITH FIXES")
    print(f"{'='*80}")

    # Create balanced loss
    criterion = create_balanced_hybrid_loss(train_df)

    # Optimizer with stronger weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-3,
        betas=(0.9, 0.999)
    )

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0
    patience = 30

    # Track best balanced metric separately
    best_balanced_metric = 0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Track UP/DOWN precision separately
    train_up_precisions = []
    train_down_precisions = []
    val_up_precisions = []
    val_down_precisions = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        up_correct = 0
        up_total = 0
        down_correct = 0
        down_total = 0

        for batch in train_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(alphas, prices_temporal, training=True)
            loss = criterion(predictions, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Calculate metrics
            pred_sign = torch.sign(predictions).detach()
            true_sign = torch.sign(targets).detach()

            correct = (pred_sign == true_sign).float()
            total_correct += correct.sum().item()
            total_samples += len(targets)

            # Track UP/DOWN separately
            up_mask = true_sign > 0
            down_mask = true_sign < 0

            if up_mask.any():
                up_correct += (pred_sign[up_mask] > 0).float().sum().item()
                up_total += up_mask.sum().item()

            if down_mask.any():
                down_correct += (pred_sign[down_mask] < 0).float().sum().item()
                down_total += down_mask.sum().item()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        train_acc = (total_correct / total_samples) * 100
        train_up_prec = (up_correct / up_total * 100) if up_total > 0 else 0
        train_down_prec = (down_correct / down_total * 100) if down_total > 0 else 0

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_up_precisions.append(train_up_prec)
        train_down_precisions.append(train_down_prec)

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        val_up_correct = 0
        val_up_total = 0
        val_down_correct = 0
        val_down_total = 0

        with torch.no_grad():
            for batch in val_loader:
                alphas, prices_temporal, targets = batch
                alphas = alphas.to(device)
                prices_temporal = prices_temporal.to(device)
                targets = targets.to(device)

                predictions = model(alphas, prices_temporal, training=True)
                loss = criterion(predictions, targets)

                pred_sign = torch.sign(predictions)
                true_sign = torch.sign(targets)

                correct = (pred_sign == true_sign).float()
                val_correct += correct.sum().item()
                val_samples += len(targets)

                up_mask = true_sign > 0
                down_mask = true_sign < 0

                if up_mask.any():
                    val_up_correct += (pred_sign[up_mask] > 0).float().sum().item()
                    val_up_total += up_mask.sum().item()

                if down_mask.any():
                    val_down_correct += (pred_sign[down_mask] < 0).float().sum().item()
                    val_down_total += down_mask.sum().item()

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_acc = (val_correct / val_samples) * 100
        val_up_prec = (val_up_correct / val_up_total * 100) if val_up_total > 0 else 0
        val_down_prec = (val_down_correct / val_down_total * 100) if val_down_total > 0 else 0

        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_up_precisions.append(val_up_prec)
        val_down_precisions.append(val_down_prec)

        # Step scheduler
        scheduler.step()

        # Print progress with UP/DOWN precision
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]")
            print(f"  Train - Loss: {train_loss:.6f}, Acc: {train_acc:.1f}%, "
                  f"UP: {train_up_prec:.1f}%, DOWN: {train_down_prec:.1f}%")
            print(f"  Val   - Loss: {val_loss:.6f}, Acc: {val_acc:.1f}%, "
                  f"UP: {val_up_prec:.1f}%, DOWN: {val_down_prec:.1f}%")

        # Calculate balanced metric
        balanced_metric = min(val_up_prec, val_down_prec)

        # Save model if:
        # 1. Loss is improving AND balanced metric > 20%
        # OR
        # 2. Balanced metric is best so far (even if loss is worse)

        save_model = False

        if val_loss < best_val_loss and balanced_metric >= 20:
            # Both loss and balance are good
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            save_model = True
            print(f"  ✓ Saving: Better loss ({val_loss:.6f}) + balanced ({balanced_metric:.1f}%)")

        elif balanced_metric > best_balanced_metric and balanced_metric >= 25:
            # Best balance so far
            best_balanced_metric = balanced_metric
            best_val_acc = val_acc
            patience_counter = 0
            save_model = True
            print(f"  ✓ Saving: Best balance ({balanced_metric:.1f}%)")

        else:
            patience_counter += 1

        if save_model:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_up_prec': val_up_prec,
                'val_down_prec': val_down_prec,
                'balanced_metric': balanced_metric,
            }, 'best_model_checkpoint.pth')

            torch.save(model.state_dict(), 'best_model.pth')

        # Stop if no improvement for patience epochs
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            print(f"  Best validation accuracy: {best_val_acc:.1f}%")
            break

    # Load best model
    checkpoint = torch.load('best_model_checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\n✓ Training complete!")
    print(f"  Best val accuracy: {checkpoint['val_acc']:.1f}%")
    print(f"  Best UP precision: {checkpoint['val_up_prec']:.1f}%")
    print(f"  Best DOWN precision: {checkpoint['val_down_prec']:.1f}%")
    print(f"  Balanced metric: {checkpoint.get('balanced_metric', 0):.1f}%")

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_up_precisions': train_up_precisions,
        'train_down_precisions': train_down_precisions,
        'val_up_precisions': val_up_precisions,
        'val_down_precisions': val_down_precisions,
        'best_epoch': checkpoint['epoch']
    }


def evaluate_enhanced_model(model, test_loader, scalers, device, n_samples=10):
    """
    Enhanced evaluation with uncertainty and comprehensive metrics
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_uncertainties = []

    # Check if test_loader is empty
    if len(test_loader) == 0 or len(test_loader.dataset) == 0:
        print(" Test set is empty (likely too small for window_size)")
        # Return dummy metrics
        dummy_metrics = {
            'MSE': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'Directional Accuracy': 0.0,
            'Up Precision': 0.0,
            'Down Precision': 0.0,
            'Large Move Hit Rate': 0.0,
            'Uncertainty Correlation': 0.0,
            'Sharpe Ratio': 0.0,
            'Win Rate': 0.0,
            'Avg Uncertainty': 0.0,
            'Max Uncertainty': 0.0
        }
        return np.array([]), np.array([]), np.array([]), dummy_metrics

    with torch.no_grad():
        for batch in test_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)

            # Skip single-sample batches
            if alphas.shape[0] == 1:
                continue

            # Get predictions with uncertainty
            predictions, uncertainties = model(alphas, prices_temporal, n_samples=n_samples, training=False)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())

    # Check if we got any predictions
    if len(all_predictions) == 0:
        print(" No predictions made (all batches skipped or empty)")
        dummy_metrics = {
            'MSE': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'Directional Accuracy': 0.0,
            'Up Precision': 0.0,
            'Down Precision': 0.0,
            'Large Move Hit Rate': 0.0,
            'Uncertainty Correlation': 0.0,
            'Sharpe Ratio': 0.0,
            'Win Rate': 0.0,
            'Avg Uncertainty': 0.0,
            'Max Uncertainty': 0.0
        }
        return np.array([]), np.array([]), np.array([]), dummy_metrics

    # Concatenate all batches
    predictions = np.concatenate(all_predictions).flatten()
    actuals = np.concatenate(all_targets).flatten()
    uncertainties = np.concatenate(all_uncertainties).flatten()

    # Inverse transform if scaler is available
    if 'target' in scalers:
        predictions_unscaled = scalers['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_unscaled = scalers['target'].inverse_transform(actuals.reshape(-1, 1)).flatten()
    else:
        predictions_unscaled = predictions
        actuals_unscaled = actuals

    # Calculate comprehensive metrics
    mse = np.mean((predictions_unscaled - actuals_unscaled) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_unscaled - actuals_unscaled))

    # Directional accuracy
    actual_direction = np.sign(actuals_unscaled)
    pred_direction = np.sign(predictions_unscaled)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    # Precision for up/down predictions
    up_mask = actual_direction > 0
    down_mask = actual_direction < 0

    up_precision = np.mean(pred_direction[up_mask] > 0) * 100 if up_mask.any() else 0
    down_precision = np.mean(pred_direction[down_mask] < 0) * 100 if down_mask.any() else 0

    # Hit rate for large movements
    large_moves = np.abs(actuals_unscaled) > 0.02
    if large_moves.any():
        large_move_hit_rate = np.mean(np.sign(actuals_unscaled[large_moves]) == np.sign(predictions_unscaled[large_moves])) * 100
    else:
        large_move_hit_rate = 0

    # Uncertainty quality (calibration)
    errors = np.abs(predictions_unscaled - actuals_unscaled)
    uncertainty_corr = np.corrcoef(errors, uncertainties)[0, 1] if len(errors) > 1 else 0

    # Sharpe ratio of trading signal (if consistently predicting direction)
    signal_returns = np.where(pred_direction == actual_direction, np.abs(actuals_unscaled), -np.abs(actuals_unscaled))
    sharpe_ratio = np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252) if np.std(signal_returns) > 0 else 0

    # Win rate
    win_rate = np.mean(signal_returns > 0) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Directional Accuracy': directional_accuracy,
        'Up Precision': up_precision,
        'Down Precision': down_precision,
        'Large Move Hit Rate': large_move_hit_rate,
        'Uncertainty Correlation': uncertainty_corr,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate': win_rate,
        'Avg Uncertainty': np.mean(uncertainties),
        'Max Uncertainty': np.max(uncertainties)
    }

    return predictions_unscaled, actuals_unscaled, uncertainties, metrics

def evaluate_enhanced_model_with_viz(model, test_loader, test_df, scalers,
                                    ticker, device='cpu', n_samples=10):
    """
    Enhanced evaluation that also returns initial price for visualization

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        test_df: Original test DataFrame (before windowing)
        scalers: Dictionary of scalers
        ticker: Stock ticker
        device: 'cpu' or 'cuda'
        n_samples: Number of Monte Carlo samples

    Returns:
        predictions, actuals, uncertainties, metrics, dates, initial_price
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_uncertainties = []

    if len(test_loader) == 0 or len(test_loader.dataset) == 0:
        print(" Test set is empty")
        dummy_metrics = {
            'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0,
            'Directional Accuracy': 0.0, 'Up Precision': 0.0,
            'Down Precision': 0.0, 'Large Move Hit Rate': 0.0,
            'Uncertainty Correlation': 0.0, 'Sharpe Ratio': 0.0,
            'Win Rate': 0.0, 'Avg Uncertainty': 0.0, 'Max Uncertainty': 0.0
        }
        return np.array([]), np.array([]), np.array([]), dummy_metrics, None, None

    with torch.no_grad():
        for batch in test_loader:
            alphas, prices_temporal, targets = batch
            alphas = alphas.to(device)
            prices_temporal = prices_temporal.to(device)

            if alphas.shape[0] == 1:
                continue

            predictions, uncertainties = model(alphas, prices_temporal,
                                              n_samples=n_samples, training=False)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.numpy())
            all_uncertainties.append(uncertainties.cpu().numpy())

    if len(all_predictions) == 0:
        print(" No predictions made")
        dummy_metrics = {
            'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0,
            'Directional Accuracy': 0.0, 'Up Precision': 0.0,
            'Down Precision': 0.0, 'Large Move Hit Rate': 0.0,
            'Uncertainty Correlation': 0.0, 'Sharpe Ratio': 0.0,
            'Win Rate': 0.0, 'Avg Uncertainty': 0.0, 'Max Uncertainty': 0.0
        }
        return np.array([]), np.array([]), np.array([]), dummy_metrics, None, None

    # Concatenate all batches
    predictions = np.concatenate(all_predictions).flatten()
    actuals = np.concatenate(all_targets).flatten()
    uncertainties = np.concatenate(all_uncertainties).flatten()

    # Inverse transform
    if 'target' in scalers:
        predictions_unscaled = scalers['target'].inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_unscaled = scalers['target'].inverse_transform(actuals.reshape(-1, 1)).flatten()
    else:
        predictions_unscaled = predictions
        actuals_unscaled = actuals

    # Get dates and initial price from test_df
    dates = None
    initial_price = None

    if test_df is not None and 'date' in test_df.columns:
        # The test_df starts before the test_loader samples (due to windowing)
        # We need to get dates that correspond to the actual predictions
        window_size = test_loader.dataset.window_size

        # Dates for predictions (after window)
        dates = test_df['date'].values[window_size:window_size + len(predictions)]

        # Initial price is the close price just before first prediction
        if 'close' in test_df.columns:
            initial_price = test_df['close'].iloc[window_size - 1]

    # Calculate metrics (same as evaluate_enhanced_model)
    mse = np.mean((predictions_unscaled - actuals_unscaled) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_unscaled - actuals_unscaled))

    actual_direction = np.sign(actuals_unscaled)
    pred_direction = np.sign(predictions_unscaled)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    up_mask = actual_direction > 0
    down_mask = actual_direction < 0

    up_precision = np.mean(pred_direction[up_mask] > 0) * 100 if up_mask.any() else 0
    down_precision = np.mean(pred_direction[down_mask] < 0) * 100 if down_mask.any() else 0

    large_moves = np.abs(actuals_unscaled) > 0.02
    if large_moves.any():
        large_move_hit_rate = np.mean(np.sign(actuals_unscaled[large_moves]) == np.sign(predictions_unscaled[large_moves])) * 100
    else:
        large_move_hit_rate = 0

    errors = np.abs(predictions_unscaled - actuals_unscaled)
    uncertainty_corr = np.corrcoef(errors, uncertainties)[0, 1] if len(errors) > 1 else 0

    signal_returns = np.where(pred_direction == actual_direction, np.abs(actuals_unscaled), -np.abs(actuals_unscaled))
    sharpe_ratio = np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252) if np.std(signal_returns) > 0 else 0

    win_rate = np.mean(signal_returns > 0) * 100

    metrics = {
        'MSE': mse, 'RMSE': rmse, 'MAE': mae,
        'Directional Accuracy': directional_accuracy,
        'Up Precision': up_precision, 'Down Precision': down_precision,
        'Large Move Hit Rate': large_move_hit_rate,
        'Uncertainty Correlation': uncertainty_corr,
        'Sharpe Ratio': sharpe_ratio, 'Win Rate': win_rate,
        'Avg Uncertainty': np.mean(uncertainties),
        'Max Uncertainty': np.max(uncertainties)
    }

    return predictions_unscaled, actuals_unscaled, uncertainties, metrics, dates, initial_price

def plot_stock_price_predictions(actuals, predictions, uncertainties, ticker,
                                 test_dates, initial_price, save_path=None):
    """
    Create comprehensive stock price visualization with actual vs predicted prices
    """
    # Convert dates
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates

    # Convert % changes to actual prices
    actual_prices = [initial_price]
    predicted_prices = [initial_price]

    for i in range(len(actuals)):
        actual_next = actual_prices[-1] * (1 + actuals[i])
        predicted_next = predicted_prices[-1] * (1 + predictions[i])

        actual_prices.append(actual_next)
        predicted_prices.append(predicted_next)

    actual_prices = actual_prices[1:]
    predicted_prices = predicted_prices[1:]

    # Calculate prediction bands
    predicted_upper = []
    predicted_lower = []

    running_price = initial_price
    for i in range(len(predictions)):
        upper_return = predictions[i] + uncertainties[i]
        lower_return = predictions[i] - uncertainties[i]

        upper_price = running_price * (1 + upper_return)
        lower_price = running_price * (1 + lower_return)

        predicted_upper.append(upper_price)
        predicted_lower.append(lower_price)
        running_price = running_price * (1 + predictions[i])

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # Actual vs Predicted Stock Prices
    ax1 = axes[0, 0]

    ax1.plot(dates, actual_prices,
             label='Actual Stock Price',
             linewidth=3.5,
             alpha=0.9,
             color='#2E86DE',
             marker='o',
             markersize=8)

    ax1.plot(dates, predicted_prices,
             label='Predicted Stock Price',
             linewidth=3.5,
             alpha=0.9,
             color='#EE5A6F',
             marker='s',
             markersize=8,
             linestyle='--')

    if len(predicted_upper) > 0 and np.max(uncertainties) > 0:
        ax1.fill_between(dates,
                         predicted_lower,
                         predicted_upper,
                         alpha=0.2,
                         color='#EE5A6F',
                         label='95% Confidence')

    ax1.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Stock Price ($)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{ticker} - Actual vs Predicted Stock Prices',
                  fontsize=18, fontweight='bold', pad=20)

    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=12, loc='best', framealpha=0.9)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Annotations
    ax1.annotate(f'Start: ${actual_prices[0]:.2f}',
                xy=(dates[0], actual_prices[0]),
                xytext=(10, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                fontsize=10, fontweight='bold')

    ax1.annotate(f'End: ${actual_prices[-1]:.2f}',
                xy=(dates[-1], actual_prices[-1]),
                xytext=(-80, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7),
                fontsize=10, fontweight='bold')

    # 2. Daily Returns
    ax2 = axes[0, 1]
    x = np.arange(len(actuals))
    width = 0.35

    ax2.bar(x - width/2, actuals * 100, width,
            label='Actual Returns', alpha=0.8, color='#2E86DE')
    ax2.bar(x + width/2, predictions * 100, width,
            label='Predicted Returns', alpha=0.8, color='#EE5A6F')

    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Day', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Daily Returns Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Cumulative Returns
    ax3 = axes[1, 0]
    actual_cumulative = np.cumprod(1 + actuals) - 1
    predicted_cumulative = np.cumprod(1 + predictions) - 1

    ax3.plot(dates, actual_cumulative * 100,
             label='Actual', linewidth=2.5, color='#2E86DE', marker='o')
    ax3.plot(dates, predicted_cumulative * 100,
             label='Predicted', linewidth=2.5, color='#EE5A6F', marker='s', linestyle='--')

    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 4. Performance Metrics
    ax4 = axes[1, 1]
    ax4.axis('off')

    final_actual_return = (actual_prices[-1] / initial_price - 1) * 100
    final_predicted_return = (predicted_prices[-1] / initial_price - 1) * 100
    price_error = abs(actual_prices[-1] - predicted_prices[-1])
    price_error_pct = (price_error / actual_prices[-1]) * 100
    directional_accuracy = np.mean(np.sign(actuals) == np.sign(predictions)) * 100

    metrics_text = f"""
📊 PERFORMANCE METRICS
{'='*40}

Initial Price:        ${initial_price:.2f}
Final Actual Price:   ${actual_prices[-1]:.2f}
Final Predicted:      ${predicted_prices[-1]:.2f}

Price Error:          ${price_error:.2f} ({price_error_pct:.2f}%)

Actual Return:        {final_actual_return:+.2f}%
Predicted Return:     {final_predicted_return:+.2f}%

Directional Accuracy: {directional_accuracy:.1f}%
"""

    ax4.text(0.1, 0.5, metrics_text,
             fontsize=11,
             fontfamily='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round,pad=1',
                      facecolor='lightgray',
                      alpha=0.8,
                      edgecolor='black',
                      linewidth=2))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n Saved stock price visualization to {save_path}")

    plt.show()

    return fig


def plot_enhanced_predictions(actuals, predictions, uncertainties, ticker, test_dates, save_path=None):
    """
    Enhanced visualization with uncertainty bands
    """
    # Convert dates
    if isinstance(test_dates[0], str):
        dates = pd.to_datetime(test_dates)
    else:
        dates = test_dates

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Main prediction plot
    ax1 = axes[0, 0]
    ax1.plot(dates, actuals, label='Actual % Change', linewidth=2.5, alpha=0.8,
             color='#2E86DE', marker='o', markersize=6)
    ax1.plot(dates, predictions, label='Predicted % Change', linewidth=2.5, alpha=0.8,
             color='#EE5A6F', marker='s', markersize=6)

    # Add uncertainty bands
    ax1.fill_between(dates,
                    predictions - uncertainties,
                    predictions + uncertainties,
                    alpha=0.3, color='#EE5A6F', label='Uncertainty')

    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('Daily % Change')
    ax1.set_title(f'{ticker} - Predictions with Uncertainty')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Scatter plot with perfect prediction line
    ax2 = axes[0, 1]
    ax2.scatter(actuals, predictions, alpha=0.6, c=uncertainties, cmap='viridis')
    ax2.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()],
             'r--', alpha=0.5, label='Perfect Prediction')
    ax2.set_xlabel('Actual % Change')
    ax2.set_ylabel('Predicted % Change')
    ax2.set_title('Prediction vs Actual')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add colorbar for uncertainty
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=uncertainties.min(), vmax=uncertainties.max()))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label='Uncertainty')

    # Error distribution
    ax3 = axes[1, 0]
    errors = predictions - actuals
    ax3.hist(errors, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    ax3.grid(True, alpha=0.3)

    # Uncertainty vs Error
    ax4 = axes[1, 1]
    ax4.scatter(uncertainties, np.abs(errors), alpha=0.6)

    # Add trend line
    if len(errors) > 1:
        z = np.polyfit(uncertainties, np.abs(errors), 1)
        p = np.poly1d(z)
        ax4.plot(np.sort(uncertainties), p(np.sort(uncertainties)), "r--", alpha=0.8)

    ax4.set_xlabel('Uncertainty')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Uncertainty vs Error (should be correlated)')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved enhanced plot to {save_path}")

    plt.show()

def plot_training_progress(history, ticker, save_path=None):
    """Plot comprehensive training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(history['train_losses'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_losses'], label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{ticker} - Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(history['train_accuracies'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_accuracies'], label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Direction Accuracy (%)')
    ax2.set_title(f'{ticker} - Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(history['learning_rates'], linewidth=2, color='green')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title(f'{ticker} - Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Combined view
    ax4 = axes[1, 1]
    # Mark best epoch
    best_epoch = history.get('best_epoch', 0)
    ax4.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best Epoch: {best_epoch}')

    # Plot both losses on secondary y-axis
    ax4_twin = ax4.twinx()
    ln1 = ax4.plot(history['train_losses'], label='Train Loss', color='blue', linewidth=2)
    ln2 = ax4_twin.plot(history['train_accuracies'], label='Train Acc', color='orange', linewidth=2)

    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss', color='blue')
    ax4_twin.set_ylabel('Accuracy (%)', color='orange')
    ax4.set_title(f'{ticker} - Combined Training Progress')

    # Combine legends
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax4.legend(lns, labs, loc='upper right')

    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_stock_predictor(ticker, company_name, comprehensive_df, alpha_text, num_epochs=150, device='cpu'):
    """
    Main function to train enhanced stock predictor
    """
    print(f"\n{'='*80}")
    print(f"ENHANCED TRAINING FOR {company_name} ({ticker})")
    print(f"{'='*80}")

    # Prepare enhanced data
    train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df = \
        prepare_data_with_fixes(
            df=comprehensive_df,
            ticker=ticker,
            alpha_text=alpha_text,
            window_size=30,
            use_feature_selection=True,  # Enable feature selection
            top_k=30  # Select top 30 features
        )

    if train_loader is None:
        print(f" Data preparation failed for {ticker}")
        return None, None, None

    # Initialize enhanced model
    model = ImprovedDualStreamLSTM(
        num_alphas=num_alphas,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        num_heads=4
    ).to(device)

    print(f"\nModel Architecture:")
    print(f"  - Alpha stream: {num_alphas} features")
    print(f"  - Hidden size: {model.hidden_size}")
    print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Device: {device}")

    model, training_history = train_with_fixes(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=3e-4,
        device=device,
        train_df=train_df
    )

    # Evaluate on test set
    print(f"\n{'='*80}")
    print("FINAL EVALUATION ON TEST SET")
    print(f"{'='*80}")

    predictions, actuals, uncertainties, metrics = evaluate_enhanced_model(
        model=model,
        test_loader=test_loader,
        scalers=scalers,
        device=device,
        n_samples=10
    )

    # Print metrics
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print(f"  MAE: {metrics['MAE']:.6f}")
    print(f"  Directional Accuracy: {metrics['Directional Accuracy']:.2f}%")
    print(f"  Up Precision: {metrics['Up Precision']:.2f}%")
    print(f"  Down Precision: {metrics['Down Precision']:.2f}%")
    print(f"  Large Move Hit Rate: {metrics['Large Move Hit Rate']:.2f}%")
    print(f"  Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    print(f"  Win Rate: {metrics['Win Rate']:.2f}%")
    print(f"  Avg Uncertainty: {metrics['Avg Uncertainty']:.6f}")
    print(f"  Uncertainty Correlation: {metrics['Uncertainty Correlation']:.3f}")

    # Visualize results
    print(f"\nGenerating visualizations...")

    # Plot training progress
    plot_training_progress(training_history, ticker, save_path=f"{ticker}_training_progress.png")

    # Plot predictions with uncertainty
    plot_enhanced_predictions(
        actuals, predictions, uncertainties, ticker, test_dates,
        save_path=f"{ticker}_predictions_enhanced.png"
    )

    # Feature importance analysis (optional)
    print(f"\nFeature Analysis:")

    # Get alpha feature importance
    alpha_cols = [col for col in comprehensive_df.columns if col.startswith('alpha_')]
    if alpha_cols:
        print(f"  Alpha features: {', '.join(alpha_cols)}")

        # Simple correlation analysis
        for alpha_col in alpha_cols[:5]:  # Show top 5
            if alpha_col in comprehensive_df.columns:
                corr = comprehensive_df[alpha_col].corr(comprehensive_df['target'])
                print(f"    {alpha_col}: correlation with target = {corr:.3f}")

    return model, metrics, scalers


class ModelEnsemble(nn.Module):
    """Ensemble of multiple models for improved performance"""
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)

        # Learnable weights for ensemble
        if weights is None:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float32))

    def forward(self, alphas, prices_temporal, n_samples=5, training=True):
        predictions = []
        uncertainties = []

        for model in self.models:
            if training:
                pred = model(alphas, prices_temporal, training=training)
                predictions.append(pred)
            else:
                pred, unc = model(alphas, prices_temporal, n_samples=n_samples, training=False)
                predictions.append(pred)
                uncertainties.append(unc)

        predictions = torch.stack(predictions, dim=-1)

        # Weighted average
        weights = torch.softmax(self.weights, dim=0)
        weighted_pred = torch.sum(predictions * weights, dim=-1)

        if not training:
            uncertainties = torch.stack(uncertainties, dim=-1)
            weighted_unc = torch.sum(uncertainties * weights, dim=-1)
            return weighted_pred, weighted_unc

        return weighted_pred

def train_ensemble(ticker, company_name, comprehensive_df, alpha_text,
                  n_models=3, window_size=30, device='cpu'):
    """
    Train an ensemble of models for improved robustness
    """
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL ENSEMBLE ({n_models} models)")
    print(f"{'='*80}")

    models = []

    # Variables to store for visualization
    test_df = None
    val_df = None
    test_loader = None
    val_loader = None
    scalers = None

    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}...")

        result = prepare_data_with_fixes(
            comprehensive_df, ticker, alpha_text,
            window_size=window_size,
            use_feature_selection=True,
            top_k=30
        )

        if result[0] is None:
            continue

        train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df = result

        test_size = len(test_loader.dataset) + window_size if hasattr(test_loader, 'dataset') else 50
        val_size = len(val_loader.dataset) + window_size if hasattr(val_loader, 'dataset') else 50
        test_df = comprehensive_df.iloc[-(test_size):].copy()
        val_df = comprehensive_df.iloc[-(test_size + val_size):-test_size].copy()

        # Initialize model
        model = ImprovedDualStreamLSTM(
            num_alphas=num_alphas,
            hidden_size=128,
            num_layers=3,
            dropout=0.3 + i*0.05,  # Vary dropout
            num_heads=4
        ).to(device)

        # Train with different learning rate
        lr = 3e-4 * (0.8 ** i)  # Decaying learning rates
        model, _ = train_with_fixes(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,  # Fewer epochs per model
            learning_rate=lr,
            device=device,
            train_df=train_df
        )

        models.append(model)

    # Create ensemble
    if models:
        ensemble = ModelEnsemble(models).to(device)
        print(f"\n Ensemble created with {len(models)} models")

        # Determine which data to use for evaluation
        if test_loader is None or len(test_loader.dataset) == 0:
            print("\n Test set is empty, using validation set for evaluation")
            eval_loader = val_loader
            eval_df = val_df
        else:
            eval_loader = test_loader
            eval_df = test_df

        # Use new evaluation function with visualization support
        predictions, actuals, uncertainties, metrics, dates, initial_price = evaluate_enhanced_model_with_viz(
            model=ensemble,
            test_loader=eval_loader,
            test_df=eval_df,
            scalers=scalers,
            ticker=ticker,
            device=device,
            n_samples=10
        )

        # Only print and visualize if we have predictions
        if len(predictions) > 0:
            print(f"\nEnsemble Performance:")
            print(f"  Directional Accuracy: {metrics['Directional Accuracy']:.2f}%")
            print(f"  RMSE: {metrics['RMSE']:.6f}")

            if initial_price is not None and dates is not None:
                print(f"\n Creating stock price visualization...")

                plot_stock_price_predictions(
                    actuals=actuals,
                    predictions=predictions,
                    uncertainties=uncertainties,
                    ticker=ticker,
                    test_dates=dates,
                    initial_price=initial_price,
                    save_path=f'{ticker}_stock_price_prediction.png'
                )
        else:
            print(f"\n  No predictions made")
            metrics = None

        return ensemble, metrics, scalers

    return None, None, None


def load_single_model(ticker, num_alphas=5, device='cpu'):
    """Load a single trained model"""
    model_path = f'{ticker}_model.pth'

    if not os.path.exists(model_path):
        # Try alternative naming
        model_path = f'{ticker}_model_0.pth'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found for {ticker}")
            return None

    model = ImprovedDualStreamLSTM(
        num_alphas=num_alphas,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        num_heads=4
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f" Loaded single model for {ticker}")
    return model

def load_ensemble(ticker, num_alphas=5, device='cpu'):
    """
    Load a previously trained ensemble
    Returns: (ensemble_model, num_models_loaded)
    """

    # Look for ensemble file first
    ensemble_path = f'{ticker}_ensemble.pth'

    if os.path.exists(ensemble_path):
        # Load pre-saved ensemble
        try:
            # First create a dummy ensemble with 1 model
            dummy_model = ImprovedDualStreamLSTM(
                num_alphas=num_alphas,
                hidden_size=128,
                num_layers=3,
                dropout=0.3,
                num_heads=4
            ).to(device)

            ensemble = ModelEnsemble([dummy_model]).to(device)
            ensemble.load_state_dict(torch.load(ensemble_path, map_location=device))

            # Count actual models in ensemble
            num_models = len(ensemble.models)
            print(f" Loaded ensemble with {num_models} models for {ticker}")
            return ensemble, num_models

        except Exception as e:
            print(f" Could not load ensemble file, trying individual models: {e}")

    # Fallback: Load individual models and create ensemble
    print(f"Looking for individual model files for {ticker}...")

    # Pattern 1: ticker_model_X.pth
    model_files = sorted(glob.glob(f"{ticker}_model_[0-9]*.pth"))

    # Pattern 2: ticker_model_X.pth (if numbered differently)
    if not model_files:
        model_files = sorted(glob.glob(f"*{ticker}*model*.pth"))

    # Pattern 3: Any model file for this ticker
    if not model_files:
        all_model_files = sorted(glob.glob("*model*.pth"))
        model_files = [f for f in all_model_files if ticker in f]

    if not model_files:
        print(f" No model files found for {ticker}")
        return None, 0

    # Load each model
    models = []
    successful_loads = 0

    for model_file in model_files[:5]:  # Load max 5 models
        try:
            model = ImprovedDualStreamLSTM(
                num_alphas=num_alphas,
                hidden_size=128,
                num_layers=3,
                dropout=0.3,
                num_heads=4
            ).to(device)

            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()
            models.append(model)
            successful_loads += 1

            print(f"  Loaded: {os.path.basename(model_file)}")

        except Exception as e:
            print(f"  Failed to load {model_file}: {e}")
            continue

    if not models:
        print(f" Could not load any models for {ticker}")
        return None, 0

    # Create ensemble from loaded models
    ensemble = ModelEnsemble(models).to(device)

    # Save as ensemble file for future quick loading
    try:
        torch.save(ensemble.state_dict(), f'{ticker}_ensemble.pth')
        print(f"  Saved consolidated ensemble file: {ticker}_ensemble.pth")
    except Exception as e:
        print(f"  Could not save ensemble file: {e}")

    print(f" Created ensemble with {len(models)} models for {ticker}")
    return ensemble, len(models)

def predict_with_model(model, data_loader, scalers, device='cpu', is_ensemble=True):
    """
    Make predictions with a loaded model
    Returns: predictions, actuals, uncertainties, metrics
    """
    if is_ensemble and isinstance(model, ModelEnsemble):
        # Ensemble prediction
        return evaluate_enhanced_model(
            model=model,
            test_loader=data_loader,
            scalers=scalers,
            device=device,
            n_samples=10
        )
    else:
        # Single model prediction
        return evaluate_enhanced_model(
            model=model,
            test_loader=data_loader,
            scalers=scalers,
            device=device,
            n_samples=5
        )

def make_live_prediction(ticker, recent_data_df, alpha_text, model_type='ensemble', device='cpu'):
    """
    Make prediction on live/most recent data
    Args:
        ticker: Stock ticker
        recent_data_df: DataFrame with recent features (last N days)
        alpha_text: Alpha formulas for this ticker
        model_type: 'ensemble' or 'single'
        device: 'cpu' or 'cuda'
    """
    print(f"\n{'='*80}")
    print(f"LIVE PREDICTION FOR {ticker}")
    print(f"{'='*80}")

    # Prepare the data (same as training)
    _, _, test_loader, scalers, num_alphas, _ = prepare_data_with_fixes(
        recent_data_df, ticker, alpha_text, window_size=30
    )

    if test_loader is None:
        print("❌ Failed to prepare data")
        return None

    # Load the model
    if model_type == 'ensemble':
        model = load_ensemble(ticker, num_alphas, device)
        if model is None:
            print(" Falling back to single model...")
            model = load_single_model(ticker, num_alphas, device)
            model_type = 'single'
    else:
        model = load_single_model(ticker, num_alphas, device)

    if model is None:
        print(" Could not load any model")
        return None

    # Make prediction
    predictions, actuals, uncertainties, metrics = predict_with_model(
        model=model,
        data_loader=test_loader,
        scalers=scalers,
        device=device,
        is_ensemble=(model_type == 'ensemble')
    )

    # Format results
    latest_prediction = predictions[-1] if len(predictions) > 0 else 0
    latest_uncertainty = uncertainties[-1] if len(uncertainties) > 0 else 0

    result = {
        'ticker': ticker,
        'predicted_change': float(latest_prediction),
        'uncertainty': float(latest_uncertainty),
        'confidence_interval': [
            float(latest_prediction - 2*latest_uncertainty),
            float(latest_prediction + 2*latest_uncertainty)
        ],
        'direction': 'UP' if latest_prediction > 0 else 'DOWN',
        'strength': 'STRONG' if abs(latest_prediction) > 0.02 else 'WEAK',
        'model_type': model_type,
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Display results
    print(f"\n PREDICTION RESULTS:")
    print(f"  Ticker: {ticker}")
    print(f"  Predicted Change: {latest_prediction:.4%}")
    print(f"  Direction: {result['direction']}")
    print(f"  Strength: {result['strength']}")
    print(f"  Uncertainty: ±{latest_uncertainty:.4%}")
    print(f"  95% Confidence: [{result['confidence_interval'][0]:.4%}, {result['confidence_interval'][1]:.4%}]")
    print(f"  Model: {model_type}")

    if model_type == 'ensemble' and hasattr(model, 'models'):
        print(f"  Ensemble size: {len(model.models)} models")

    print(f"{'='*80}")

    return result


def batch_predict(tickers, data_dict, alpha_texts_dict, model_types=None, device='cpu'):
    """
    Make predictions for multiple tickers at once

    Args:
        tickers: List of ticker symbols
        data_dict: Dictionary {ticker: recent_data_df}
        alpha_texts_dict: Dictionary {ticker: alpha_text}
        model_types: Dictionary {ticker: 'ensemble' or 'single'} or 'ensemble' for all
        device: 'cpu' or 'cuda'

    Returns:
        Dictionary of prediction results
    """
    if model_types is None:
        model_types = {ticker: 'ensemble' for ticker in tickers}
    elif isinstance(model_types, str):
        model_types = {ticker: model_types for ticker in tickers}

    results = {}

    for ticker in tickers:
        if ticker not in data_dict or ticker not in alpha_texts_dict:
            print(f" Skipping {ticker}: missing data or alphas")
            continue

        model_type = model_types.get(ticker, 'ensemble')

        result = make_live_prediction(
            ticker=ticker,
            recent_data_df=data_dict[ticker],
            alpha_text=alpha_texts_dict[ticker],
            model_type=model_type,
            device=device
        )

        if result is not None:
            results[ticker] = result

    # Generate summary report
    if results:
        print(f"\n{'='*80}")
        print("BATCH PREDICTION SUMMARY")
        print(f"{'='*80}")

        up_count = sum(1 for r in results.values() if r['direction'] == 'UP')
        down_count = sum(1 for r in results.values() if r['direction'] == 'DOWN')
        strong_count = sum(1 for r in results.values() if r['strength'] == 'STRONG')

        print(f"  Total Predictions: {len(results)}")
        print(f"  BUY Signals (UP): {up_count}")
        print(f"  SELL Signals (DOWN): {down_count}")
        print(f"  Strong Signals: {strong_count}")

        # Sort by signal strength
        sorted_results = sorted(
            results.items(),
            key=lambda x: abs(x[1]['predicted_change']),
            reverse=True
        )

        print(f"\n  Top Signals:")
        for i, (ticker, result) in enumerate(sorted_results[:5]):
            print(f"    {i+1}. {ticker}: {result['predicted_change']:.4%} "
                  f"({result['direction']}, {result['strength']})")

    return results


def get_model_info(ticker):
    """Get information about available models for a ticker"""
    import glob

    print(f"\nModel information for {ticker}:")
    print("-" * 50)

    # Check for ensemble file
    ensemble_file = f'{ticker}_ensemble.pth'
    if os.path.exists(ensemble_file):
        file_size = os.path.getsize(ensemble_file) / (1024*1024)  # MB
        print(f" Ensemble file: {ensemble_file} ({file_size:.2f} MB)")

    # Check for individual model files
    model_files = glob.glob(f"*{ticker}*model*.pth")
    if model_files:
        print(f" Individual model files: {len(model_files)} found")
        for mf in model_files[:3]:  # Show first 3
            file_size = os.path.getsize(mf) / (1024*1024)
            print(f"  - {os.path.basename(mf)} ({file_size:.2f} MB)")
        if len(model_files) > 3:
            print(f"  ... and {len(model_files) - 3} more")
    else:
        print(" No model files found")

    # Check for training history
    history_files = glob.glob(f"*{ticker}*training*.png")
    if history_files:
        print(f" Training plots: {len(history_files)} found")

    print("-" * 50)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage
    print("Enhanced LSTM Stock Predictor")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test loading
    ticker = "AAPL"  # Example ticker

    # Get model info
    get_model_info(ticker)

    print("\nReady for predictions!")