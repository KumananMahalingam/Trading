import json
import boto3
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

sys.path.insert(0, '/opt/python')
sys.path.insert(0, '/var/task/utils')

# Import YOUR actual model class
from lstm_predictor import ImprovedDualStreamLSTM, ModelEnsemble, prepare_data_with_fixes

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Cache for models
MODEL_CACHE = {}

def lambda_handler(event, context):
    """
    Makes predictions using YOUR trained model
    Called by orchestrator Lambda
    """
    try:
        ticker = event['ticker']
        stock_data_json = event['stock_data']
        alpha_text = event['alpha_text']
        forecast_days = event.get('forecast_days', 7)

        print(f"Making prediction for {ticker}")

        # ====================================================================
        # STEP 1: LOAD MODEL FROM S3 (or cache)
        # ====================================================================

        if ticker not in MODEL_CACHE:
            model_key = f'models/{ticker}_ensemble.pth'
            local_model_path = f'/tmp/{ticker}_ensemble.pth'

            try:
                s3.download_file(
                    os.environ['MODELS_BUCKET'],
                    model_key,
                    local_model_path
                )
            except Exception as e:
                print(f"Failed to download model: {e}")
                return {
                    'statusCode': 404,
                    'body': json.dumps({
                        'error': f'Model not found for {ticker}',
                        'suggestion': 'Trigger training first'
                    })
                }

            # Load YOUR actual model
            # Create dummy ensemble to load state
            dummy_model = ImprovedDualStreamLSTM(
                num_alphas=5,
                hidden_size=128,
                num_layers=3,
                dropout=0.3,
                num_heads=4
            )

            ensemble = ModelEnsemble([dummy_model])
            ensemble.load_state_dict(torch.load(local_model_path, map_location='cpu'))
            ensemble.eval()

            MODEL_CACHE[ticker] = ensemble
            print(f"Model loaded for {ticker}")
        else:
            ensemble = MODEL_CACHE[ticker]

        # ====================================================================
        # STEP 2: PREPARE DATA (using YOUR function)
        # ====================================================================

        # Convert JSON back to DataFrame
        stock_df = pd.DataFrame(stock_data_json)

        # Prepare data using YOUR actual function
        train_loader, val_loader, test_loader, scalers, num_alphas, test_dates, train_df = \
            prepare_data_with_fixes(
                df=stock_df,
                ticker=ticker,
                alpha_text=alpha_text,
                window_size=30,
                use_feature_selection=True,
                top_k=30
            )

        if test_loader is None or len(test_loader.dataset) == 0:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Insufficient data for prediction',
                    'ticker': ticker
                })
            }

        # ====================================================================
        # STEP 3: MAKE PREDICTION
        # ====================================================================

        ensemble.eval()

        with torch.no_grad():
            # Get the last batch from test_loader
            for batch in test_loader:
                alphas, prices_temporal, targets = batch

            # Make prediction with uncertainty
            predictions, uncertainties = ensemble(
                alphas,
                prices_temporal,
                n_samples=10,
                training=False
            )

            predicted_change = float(predictions[-1][0])
            uncertainty = float(uncertainties[-1][0])

        # Inverse scale if needed
        if 'target' in scalers:
            predicted_change_unscaled = scalers['target'].inverse_transform(
                [[predicted_change]]
            )[0][0]
        else:
            predicted_change_unscaled = predicted_change

        # ====================================================================
        # STEP 4: FORMAT RESULT
        # ====================================================================

        direction = 'UP' if predicted_change_unscaled > 0 else 'DOWN'
        confidence = 1 - min(uncertainty, 1.0)

        result = {
            'ticker': ticker,
            'predicted_change': float(predicted_change_unscaled),
            'direction': direction,
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'timestamp': datetime.now().isoformat()
        }

        # Cache in DynamoDB
        try:
            predictions_table = dynamodb.Table(os.environ['PREDICTIONS_TABLE'])
            predictions_table.put_item(Item={
                **result,
                'ttl': int(datetime.now().timestamp()) + 3600
            })
        except Exception as e:
            print(f"Warning: Could not cache prediction: {e}")

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'ticker': ticker if 'ticker' in locals() else 'unknown'
            })
        }