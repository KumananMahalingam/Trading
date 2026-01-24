"""
Main dual-stream LSTM architecture
"""
import torch
import torch.nn as nn
from src.models.architectures.lstm_modules import ResidualLSTM, MultiHeadAttention, TCNBlock


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
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
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