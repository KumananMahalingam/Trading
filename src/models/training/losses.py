"""
Custom loss functions for stock prediction
"""
import torch
import torch.nn as nn
import numpy as np


class HybridLoss(nn.Module):
    """
    Hybrid loss combining MSE, directional accuracy, and large move emphasis
    """
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


def create_balanced_hybrid_loss(train_df):
    """
    Create loss function with proper class balancing

    Args:
        train_df: Training dataframe with 'target' column

    Returns:
        HybridLoss with balanced weights
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

    # **FIX 4: More aggressive inverse frequency weighting**
    # Use square root of inverse frequency for more aggressive balancing
    up_weight = np.sqrt(total / (2.0 * up_count))
    down_weight = np.sqrt(total / (2.0 * down_count))

    # Normalize weights so they sum to 2.0
    weight_sum = up_weight + down_weight
    up_weight = 2.0 * up_weight / weight_sum
    down_weight = 2.0 * down_weight / weight_sum

    # Apply additional boost to minority class (typically UP)
    if up_count < down_count:
        boost_factor = 1.5  # Boost minority class by 50%
        up_weight *= boost_factor
        # Renormalize
        weight_sum = up_weight + down_weight
        up_weight = 2.0 * up_weight / weight_sum
        down_weight = 2.0 * down_weight / weight_sum
    else:
        boost_factor = 1.5
        down_weight *= boost_factor
        weight_sum = up_weight + down_weight
        up_weight = 2.0 * up_weight / weight_sum
        down_weight = 2.0 * down_weight / weight_sum

    print(f"\n  Class Distribution:")
    print(f"  UP samples: {up_count} ({up_count/total*100:.1f}%)")
    print(f"  DOWN samples: {down_count} ({down_count/total*100:.1f}%)")
    print(f"  Neutral: {neutral_count}")
    print(f"\n  AGGRESSIVE Loss Weights (with {boost_factor}x minority boost):")
    print(f"  UP weight: {up_weight:.3f}")
    print(f"  DOWN weight: {down_weight:.3f}")

    # Increase beta (direction weight) to emphasize directional accuracy
    return HybridLoss(alpha=0.6, beta=0.35, gamma=0.05, class_weights=[up_weight, down_weight])