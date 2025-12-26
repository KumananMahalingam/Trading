import sys
import os

# Add paths so Python can find your modules
sys.path.insert(0, 'lambda-functions/predictions')
sys.path.insert(0, 'utils')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

# Mock AWS environment variables
os.environ['PREDICTIONS_TABLE'] = 'test-predictions'
os.environ['STOCK_DATA_TABLE'] = 'test-stock-data'
os.environ['MODELS_BUCKET'] = 'test-models'

print(" Testing Lambda Handler Imports...")
print("=" * 60)

# Test 1: Import handler functions
try:
    # Import directly from the handler module in lambda-functions/predictions
    import handler

    # Now access the functions
    prepare_features_for_inference = handler.prepare_features_for_inference
    calculate_rsi = handler.calculate_rsi
    calculate_macd = handler.calculate_macd

    print(" Successfully imported handler functions")
except ImportError as e:
    print(f" Failed to import handler: {e}")
    print(f"  Current sys.path: {sys.path[:3]}")
    print(f"  Looking for: lambda-functions/predictions/handler.py")
    sys.exit(1)

# Test 2: Import your model classes
try:
    from lstm_predictor import ImprovedDualStreamLSTM
    print(" Successfully imported LSTM model classes")
except ImportError as e:
    print(f" Failed to import LSTM model: {e}")
    sys.exit(1)

# Test 3: Test calculate_rsi function
try:
    import pandas as pd
    import numpy as np

    # Create sample data
    test_prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                            111, 110, 112, 114, 113, 115, 117])

    rsi = calculate_rsi(test_prices)

    print(f" RSI calculation works! Last RSI value: {rsi.iloc[-1]:.2f}")

except Exception as e:
    print(f" RSI calculation failed: {e}")

# Test 4: Test calculate_macd function
try:
    macd = calculate_macd(test_prices)
    print(f" MACD calculation works! Last MACD value: {macd.iloc[-1]:.4f}")
except Exception as e:
    print(f" MACD calculation failed: {e}")

# Test 5: Test feature preparation (with dummy data)
try:
    # Create dummy DataFrame
    dates = pd.date_range('2025-01-01', periods=60, freq='D')
    dummy_df = pd.DataFrame({
        'date': dates.astype(str),
        'open': np.random.uniform(100, 110, 60),
        'high': np.random.uniform(110, 120, 60),
        'low': np.random.uniform(90, 100, 60),
        'close': np.random.uniform(100, 110, 60),
        'volume': np.random.randint(1000000, 10000000, 60)
    })

    features = prepare_features_for_inference(dummy_df, 'AAPL')

    print(f" Feature preparation works!")
    print(f"  - Alpha features shape: {features['alphas'].shape}")
    print(f"  - Price/temporal features shape: {features['prices_temporal'].shape}")

except Exception as e:
    print(f" Feature preparation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test model initialization
try:
    model = ImprovedDualStreamLSTM(
        num_alphas=5,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        num_heads=4
    )
    print(f" Model initialization works!")
    print(f"  - Model has {sum(p.numel() for p in model.parameters()):,} parameters")

except Exception as e:
    print(f" Model initialization failed: {e}")

print("\n" + "=" * 60)
print(" All tests completed!")
print("=" * 60)