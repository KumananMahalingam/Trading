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

print("🔍 Testing Lambda Handler Imports...")
print("=" * 60)

# Test 1: Import handler
try:
    import handler
    print("✅ Successfully imported handler module")
    print(f"  - lambda_handler function exists: {hasattr(handler, 'lambda_handler')}")
except ImportError as e:
    print(f"❌ Failed to import handler: {e}")
    sys.exit(1)

# Test 2: Import your model classes and functions
try:
    from lstm_predictor import (
        ImprovedDualStreamLSTM,
        ModelEnsemble,
        prepare_data_with_fixes
    )
    print("✅ Successfully imported LSTM model classes and functions")
except ImportError as e:
    print(f"❌ Failed to import LSTM model: {e}")
    sys.exit(1)

# Test 3: Test prepare_data_with_fixes function with dummy data
try:
    import pandas as pd
    import numpy as np

    # Create dummy DataFrame with more realistic data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    dummy_df = pd.DataFrame({
        'date': dates.astype(str),
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(110, 120, 100),
        'low': np.random.uniform(90, 100, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.randint(1000000, 10000000, 100)
    })

    # Test data preparation
    train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df = \
        prepare_data_with_fixes(
            df=dummy_df,
            ticker='TEST',
            alpha_text='test alpha',
            window_size=30,
            use_feature_selection=True,
            top_k=30
        )

    print(f"✅ Data preparation works!")
    print(f"  - Number of alphas: {num_alphas}")
    print(f"  - Train batches: {len(train_loader) if train_loader else 0}")
    print(f"  - Val batches: {len(val_loader) if val_loader else 0}")
    print(f"  - Test batches: {len(test_loader) if test_loader else 0}")

except Exception as e:
    print(f"❌ Data preparation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test model initialization
try:
    model = ImprovedDualStreamLSTM(
        num_alphas=5,
        hidden_size=128,
        num_layers=3,
        dropout=0.3,
        num_heads=4
    )
    print(f"✅ Model initialization works!")
    print(f"  - Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test ensemble
    ensemble = ModelEnsemble([model])
    print(f"✅ Ensemble initialization works!")

except Exception as e:
    print(f"❌ Model initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Mock lambda_handler event
try:
    # Create a mock event (you'll need to mock boto3 for this to work fully)
    mock_event = {
        'ticker': 'AAPL',
        'stock_data': dummy_df.to_dict('records'),
        'alpha_text': 'test alpha',
        'forecast_days': 7
    }

    print(f"✅ Mock event created successfully")
    print(f"  - Ticker: {mock_event['ticker']}")
    print(f"  - Data points: {len(mock_event['stock_data'])}")

except Exception as e:
    print(f"❌ Mock event creation failed: {e}")

print("\n" + "=" * 60)
print("✅ All tests completed!")
print("=" * 60)