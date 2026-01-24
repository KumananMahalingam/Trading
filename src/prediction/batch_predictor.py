"""
Batch prediction functions
"""
from src.prediction.predictor import make_live_prediction


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
            print(f"Skipping {ticker}: missing data or alphas")
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
    import os

    print(f"\nModel information for {ticker}:")
    print("-" * 50)

    # Check for ensemble file
    ensemble_file = f'{ticker}_ensemble.pth'
    if os.path.exists(ensemble_file):
        file_size = os.path.getsize(ensemble_file) / (1024*1024)  # MB
        print(f"Ensemble file: {ensemble_file} ({file_size:.2f} MB)")

    # Check for individual model files
    model_files = glob.glob(f"*{ticker}*model*.pth")
    if model_files:
        print(f"Individual model files: {len(model_files)} found")
        for mf in model_files[:3]:
            file_size = os.path.getsize(mf) / (1024*1024)
            print(f"  - {os.path.basename(mf)} ({file_size:.2f} MB)")
        if len(model_files) > 3:
            print(f"  ... and {len(model_files) - 3} more")
    else:
        print("No model files found")

    # Check for training history
    history_files = glob.glob(f"*{ticker}*training*.png")
    if history_files:
        print(f"Training plots: {len(history_files)} found")

    print("-" * 50)
