"""
Performance metrics calculation
"""
import numpy as np


def calculate_comprehensive_metrics(predictions, actuals, uncertainties):
    """
    Calculate comprehensive performance metrics

    Args:
        predictions: Predicted values
        actuals: Actual values
        uncertainties: Prediction uncertainties

    Returns:
        dict: Dictionary of metrics
    """
    # Basic regression metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Directional accuracy
    actual_direction = np.sign(actuals)
    pred_direction = np.sign(predictions)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100

    # Precision for up/down predictions
    up_mask = actual_direction > 0
    down_mask = actual_direction < 0

    up_precision = np.mean(pred_direction[up_mask] > 0) * 100 if up_mask.any() else 0
    down_precision = np.mean(pred_direction[down_mask] < 0) * 100 if down_mask.any() else 0

    # Hit rate for large movements
    large_moves = np.abs(actuals) > 0.02
    if large_moves.any():
        large_move_hit_rate = np.mean(
            np.sign(actuals[large_moves]) == np.sign(predictions[large_moves])
        ) * 100
    else:
        large_move_hit_rate = 0

    # Uncertainty quality (calibration)
    errors = np.abs(predictions - actuals)
    uncertainty_corr = np.corrcoef(errors, uncertainties)[0, 1] if len(errors) > 1 else 0

    # Sharpe ratio of trading signal
    signal_returns = np.where(
        pred_direction == actual_direction,
        np.abs(actuals),
        -np.abs(actuals)
    )
    sharpe_ratio = (
        np.mean(signal_returns) / np.std(signal_returns) * np.sqrt(252)
        if np.std(signal_returns) > 0 else 0
    )

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

    return metrics
