"""
Model evaluation functions
"""
import torch
import numpy as np
import pandas as pd
from src.models.evaluation.metrics import calculate_comprehensive_metrics


def evaluate_enhanced_model(model, test_loader, scalers, device, n_samples=10):
    """
    Enhanced evaluation with uncertainty and comprehensive metrics

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        scalers: Dictionary of scalers
        device: 'cpu' or 'cuda'
        n_samples: Number of Monte Carlo samples for uncertainty

    Returns:
        tuple: (predictions, actuals, uncertainties, metrics)
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_uncertainties = []

    # Check if test_loader is empty
    if len(test_loader) == 0 or len(test_loader.dataset) == 0:
        print(" Test set is empty (likely too small for window_size)")
        dummy_metrics = {
            'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0,
            'Directional Accuracy': 0.0, 'Up Precision': 0.0,
            'Down Precision': 0.0, 'Large Move Hit Rate': 0.0,
            'Uncertainty Correlation': 0.0, 'Sharpe Ratio': 0.0,
            'Win Rate': 0.0, 'Avg Uncertainty': 0.0, 'Max Uncertainty': 0.0
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
            'MSE': 0.0, 'RMSE': 0.0, 'MAE': 0.0,
            'Directional Accuracy': 0.0, 'Up Precision': 0.0,
            'Down Precision': 0.0, 'Large Move Hit Rate': 0.0,
            'Uncertainty Correlation': 0.0, 'Sharpe Ratio': 0.0,
            'Win Rate': 0.0, 'Avg Uncertainty': 0.0, 'Max Uncertainty': 0.0
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
    metrics = calculate_comprehensive_metrics(predictions_unscaled, actuals_unscaled, uncertainties)

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
        tuple: (predictions, actuals, uncertainties, metrics, dates, initial_price)
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
        window_size = test_loader.dataset.window_size
        dates = test_df['date'].values[window_size:window_size + len(predictions)]

        if 'close' in test_df.columns:
            initial_price = test_df['close'].iloc[window_size - 1]

    # Calculate metrics
    metrics = calculate_comprehensive_metrics(predictions_unscaled, actuals_unscaled, uncertainties)

    return predictions_unscaled, actuals_unscaled, uncertainties, metrics, dates, initial_price
