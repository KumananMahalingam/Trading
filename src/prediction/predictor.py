"""
Live prediction functions
"""
import os
import glob
import torch
from datetime import datetime
from src.models.architectures.dual_stream_lstm import ImprovedDualStreamLSTM
from src.models.architectures.ensemble import ModelEnsemble
from src.models.training.trainer import prepare_data_with_fixes
from src.models.evaluation.evaluator import evaluate_enhanced_model


def load_single_model(ticker, num_alphas=5, device='cpu'):
    """Load a single trained model"""
    model_path = f'{ticker}_model.pth'

    if not os.path.exists(model_path):
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

    print(f"✓ Loaded single model for {ticker}")
    return model


def load_ensemble(ticker, num_alphas=5, device='cpu'):
    """
    Load a previously trained ensemble
    Returns: (ensemble_model, num_models_loaded)
    """
    ensemble_path = f'{ticker}_ensemble.pth'

    if os.path.exists(ensemble_path):
        try:
            dummy_model = ImprovedDualStreamLSTM(
                num_alphas=num_alphas,
                hidden_size=128,
                num_layers=3,
                dropout=0.3,
                num_heads=4
            ).to(device)

            ensemble = ModelEnsemble([dummy_model]).to(device)
            ensemble.load_state_dict(torch.load(ensemble_path, map_location=device))
            num_models = len(ensemble.models)
            print(f"✓ Loaded ensemble with {num_models} models for {ticker}")
            return ensemble, num_models
        except Exception as e:
            print(f"Could not load ensemble file, trying individual models: {e}")

    # Fallback: Load individual models
    print(f"Looking for individual model files for {ticker}...")
    model_files = sorted(glob.glob(f"{ticker}_model_[0-9]*.pth"))

    if not model_files:
        model_files = sorted(glob.glob(f"*{ticker}*model*.pth"))

    if not model_files:
        all_model_files = sorted(glob.glob("*model*.pth"))
        model_files = [f for f in all_model_files if ticker in f]

    if not model_files:
        print(f"No model files found for {ticker}")
        return None, 0

    models = []
    for model_file in model_files[:5]:
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
            print(f"  Loaded: {os.path.basename(model_file)}")
        except Exception as e:
            print(f"  Failed to load {model_file}: {e}")
            continue

    if not models:
        print(f"Could not load any models for {ticker}")
        return None, 0

    ensemble = ModelEnsemble(models).to(device)

    try:
        torch.save(ensemble.state_dict(), f'{ticker}_ensemble.pth')
        print(f"  Saved consolidated ensemble file: {ticker}_ensemble.pth")
    except Exception as e:
        print(f"  Could not save ensemble file: {e}")

    print(f"✓ Created ensemble with {len(models)} models for {ticker}")
    return ensemble, len(models)


def predict_with_model(model, data_loader, scalers, device='cpu', is_ensemble=True):
    """
    Make predictions with a loaded model
    Returns: predictions, actuals, uncertainties, metrics
    """
    if is_ensemble and isinstance(model, ModelEnsemble):
        return evaluate_enhanced_model(
            model=model,
            test_loader=data_loader,
            scalers=scalers,
            device=device,
            n_samples=10
        )
    else:
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

    Returns:
        dict: Prediction results
    """
    print(f"\n{'='*80}")
    print(f"LIVE PREDICTION FOR {ticker}")
    print(f"{'='*80}")

    # Prepare the data
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
            print("Falling back to single model...")
            model = load_single_model(ticker, num_alphas, device)
            model_type = 'single'
    else:
        model = load_single_model(ticker, num_alphas, device)

    if model is None:
        print("Could not load any model")
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
    print(f"\nPREDICTION RESULTS:")
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
