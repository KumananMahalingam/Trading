"""
Local testing script - simulates Lambda workflow without AWS
Tests: data fetching → feature preparation → model prediction
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add paths
sys.path.insert(0, 'utils')

print("="*80)
print("LOCAL PREDICTION TEST")
print("="*80)

# Test 1: Import your modules
print("\n1️⃣  Testing imports...")
try:
    from integrated_pipeline import (
        fetch_stock_data,
        calculate_technical_indicators,
        generate_simple_alphas,
        prepare_dataframe_for_alpha
    )
    from lstm_predictor import (
        ImprovedDualStreamLSTM,
        ModelEnsemble,
        prepare_data_with_fixes,
        load_ensemble
    )
    import torch
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Fetch real stock data
print("\n2️⃣  Fetching stock data...")
ticker = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

try:
    stock_df = fetch_stock_data(
        ticker,
        start_date.isoformat() + 'Z',
        end_date.isoformat() + 'Z'
    )
    print(f"   ✓ Fetched {len(stock_df)} days of data")
    print(f"   Latest close: ${stock_df.iloc[-1]['close']:.2f}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Calculate technical indicators
print("\n3️⃣  Calculating technical indicators...")
try:
    stock_df = calculate_technical_indicators(stock_df)
    print(f"   ✓ Added technical indicators")
    print(f"   Total columns: {len(stock_df.columns)}")

    # Show some indicators
    indicators = ['RSI', 'MACD', 'SMA_20', 'BB_Upper']
    available = [ind for ind in indicators if ind in stock_df.columns]
    print(f"   Available: {', '.join(available)}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Generate alphas
print("\n4️⃣  Generating alpha formulas...")
try:
    alpha_text = generate_simple_alphas(ticker)
    print(f"   ✓ Generated alphas")
    print(f"   Preview: {alpha_text[:100]}...")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    alpha_text = f"α1 = Return_5D\nα2 = RSI / 100"

# Test 5: Prepare comprehensive dataframe
print("\n5️⃣  Preparing features for model...")
try:
    comprehensive_df = prepare_dataframe_for_alpha(
        ticker,
        stock_df,
        {},  # No sentiments
        [],  # No related companies
        None,  # No alternative data
        None   # No economic data
    )
    print(f"   ✓ Prepared dataframe")
    print(f"   Shape: {comprehensive_df.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 6: Check for trained model
print("\n6️⃣  Checking for trained model...")
model_path = f"../{ticker}_ensemble.pth"

if not os.path.exists(model_path):
    # Try current directory
    model_path = f"{ticker}_ensemble.pth"

if not os.path.exists(model_path):
    # Try looking for individual models
    model_path = f"{ticker}_model_0.pth"

if os.path.exists(model_path):
    print(f"   ✓ Found model: {model_path}")
    model_size_mb = os.path.getsize(model_path) / (1024*1024)
    print(f"   Model size: {model_size_mb:.1f} MB")

    # Test 7: Load model
    print("\n7️⃣  Loading model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")

        # Prepare data
        train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df = \
            prepare_data_with_fixes(
                comprehensive_df,
                ticker,
                alpha_text,
                window_size=30,
                use_feature_selection=True,
                top_k=30
            )

        if test_loader is None or len(test_loader.dataset) == 0:
            print("   ⚠️  Not enough data for prediction window")
            print("   Need at least 60 days of data")
        else:
            # Create dummy model with correct architecture
            dummy_model = ImprovedDualStreamLSTM(
                num_alphas=num_alphas,
                hidden_size=128,
                num_layers=3,
                dropout=0.3,
                num_heads=4
            )

            ensemble = ModelEnsemble([dummy_model])
            ensemble.load_state_dict(torch.load(model_path, map_location=device))
            ensemble.eval()

            print(f"   ✓ Model loaded successfully")
            print(f"   Num alphas: {num_alphas}")

            # Test 8: Make prediction
            print("\n8️⃣  Making prediction...")

            with torch.no_grad():
                for batch in test_loader:
                    alphas, prices_temporal, targets = batch
                    break

                predictions, uncertainties = ensemble(
                    alphas,
                    prices_temporal,
                    n_samples=10,
                    training=False
                )

                predicted_change = float(predictions[-1][0])
                uncertainty = float(uncertainties[-1][0])

            # Inverse scale
            if 'target' in scalers:
                predicted_change = scalers['target'].inverse_transform([[predicted_change]])[0][0]

            current_price = float(stock_df.iloc[-1]['close'])
            target_price = current_price * (1 + predicted_change)
            direction = 'UP ⬆️' if predicted_change > 0 else 'DOWN ⬇️'
            confidence = (1 - min(uncertainty, 1.0)) * 100

            print(f"\n{'='*80}")
            print(f"🎯 PREDICTION RESULTS FOR {ticker}")
            print(f"{'='*80}")
            print(f"  Current Price:    ${current_price:.2f}")
            print(f"  Predicted Change: {predicted_change:+.2%}")
            print(f"  Target Price:     ${target_price:.2f}")
            print(f"  Direction:        {direction}")
            print(f"  Confidence:       {confidence:.1f}%")
            print(f"  Uncertainty:      ±{uncertainty:.4f}")
            print(f"{'='*80}")

            print("\n✅ LOCAL TEST SUCCESSFUL!")
            print("   Your pipeline is working correctly.")
            print("   Ready to deploy to AWS when you want.")

    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   ⚠️  No trained model found for {ticker}")
    print(f"   Expected location: {model_path}")
    print(f"\n   To train a model locally, run:")
    print(f"   cd ~/Trading && python integrated_pipeline.py")
    print(f"\n   Or continue with deployment and train on AWS.")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)