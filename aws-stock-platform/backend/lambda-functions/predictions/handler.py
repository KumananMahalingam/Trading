import json
import boto3
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import os
import sys

# Add utils to path for shared code
sys.path.insert(0, '/opt/python')
sys.path.insert(0, '/var/task/utils')

# Import your actual model class
from lstm_predictor import ImprovedDualStreamLSTM, EnhancedAlphaComputer

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

# Cache for models (persists between invocations)
MODEL_CACHE = {}

def lambda_handler(event, context):
    """
    GET /predictions/{ticker}
    Returns stock prediction with uncertainty
    """
    try:
        # Get ticker from API Gateway path parameters
        ticker = event['pathParameters']['ticker'].upper()

        print(f"Processing prediction request for {ticker}")

        # 1. Check DynamoDB cache (predictions expire after 1 hour)
        predictions_table = dynamodb.Table(os.environ['PREDICTIONS_TABLE'])

        try:
            response = predictions_table.get_item(Key={'ticker': ticker})

            if 'Item' in response:
                cached_time = datetime.fromisoformat(response['Item']['timestamp'])
                age_seconds = (datetime.now() - cached_time).total_seconds()

                if age_seconds < 3600:  # 1 hour cache
                    print(f"Returning cached prediction ({age_seconds:.0f}s old)")
                    return {
                        'statusCode': 200,
                        'headers': {
                            'Content-Type': 'application/json',
                            'Access-Control-Allow-Origin': '*'
                        },
                        'body': json.dumps(response['Item'])
                    }
        except Exception as e:
            print(f"Cache check failed: {e}")

        # 2. Load model from S3 (or use cached model)
        if ticker not in MODEL_CACHE:
            model_key = f'models/{ticker}_ensemble.pth'
            local_model_path = f'/tmp/{ticker}_ensemble.pth'

            print(f"Downloading model from s3://{os.environ['MODELS_BUCKET']}/{model_key}")

            try:
                s3.download_file(
                    os.environ['MODELS_BUCKET'],
                    model_key,
                    local_model_path
                )
            except Exception as e:
                print(f"Failed to download ensemble model: {e}")
                # Try single model as fallback
                model_key = f'models/{ticker}_model.pth'
                s3.download_file(
                    os.environ['MODELS_BUCKET'],
                    model_key,
                    local_model_path
                )

            # Load model (your actual model class)
            model = ImprovedDualStreamLSTM(
                num_alphas=5,  # Adjust based on your model
                hidden_size=128,
                num_layers=3,
                dropout=0.3,
                num_heads=4
            )

            model.load_state_dict(torch.load(local_model_path, map_location='cpu'))
            model.eval()

            MODEL_CACHE[ticker] = model
            print(f"Model loaded and cached for {ticker}")
        else:
            print(f"Using cached model for {ticker}")
            model = MODEL_CACHE[ticker]

        # 3. Get recent stock data from DynamoDB
        stock_data_table = dynamodb.Table(os.environ['STOCK_DATA_TABLE'])

        # Query last 60 days of data
        response = stock_data_table.query(
            KeyConditionExpression='ticker = :ticker',
            ExpressionAttributeValues={':ticker': ticker},
            ScanIndexForward=False,  # Most recent first
            Limit=60
        )

        if not response['Items']:
            return {
                'statusCode': 404,
                'body': json.dumps({'error': f'No data found for {ticker}'})
            }

        # 4. Prepare features (simplified version - you'll need to adapt)
        items = sorted(response['Items'], key=lambda x: x['date'])

        # Create dataframe
        df = pd.DataFrame(items)

        # YOUR ACTUAL FEATURE PREPARATION
        # This is simplified - adapt from your prepare_dataframe_for_alpha function
        features = prepare_features_for_inference(df, ticker)

        # 5. Make prediction
        with torch.no_grad():
            alphas = torch.FloatTensor(features['alphas']).unsqueeze(0)
            prices_temporal = torch.FloatTensor(features['prices_temporal']).unsqueeze(0)

            # Get prediction with uncertainty
            prediction, uncertainty = model(alphas, prices_temporal,
                                           n_samples=10, training=False)

            predicted_change = float(prediction[0][0])
            uncertainty_value = float(uncertainty[0][0])

        # 6. Format result
        result = {
            'ticker': ticker,
            'predicted_change': predicted_change,
            'direction': 'UP' if predicted_change > 0 else 'DOWN',
            'confidence': 1 - min(uncertainty_value, 1.0),  # Convert uncertainty to confidence
            'confidence_interval': [
                predicted_change - 2*uncertainty_value,
                predicted_change + 2*uncertainty_value
            ],
            'timestamp': datetime.now().isoformat(),
            'ttl': int(datetime.now().timestamp()) + 3600  # Expire in 1 hour
        }

        # 7. Cache in DynamoDB
        try:
            predictions_table.put_item(Item=result)
        except Exception as e:
            print(f"Failed to cache prediction: {e}")

        # 8. Return result
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }

    except Exception as e:
        print(f"Error in lambda_handler: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'ticker': event.get('pathParameters', {}).get('ticker', 'unknown')
            })
        }


def prepare_features_for_inference(df, ticker):
    """
    Simplified feature preparation for inference
    Adapt this based on your actual model's input requirements
    """
    # Calculate technical indicators (from your code)
    df['SMA_20'] = df['close'].rolling(20).mean()
    df['RSI'] = calculate_rsi(df['close'])
    df['MACD'] = calculate_macd(df['close'])

    # For now, return dummy features - YOU NEED TO ADAPT THIS
    # based on what your model actually expects

    window_size = 30
    last_window = df.tail(window_size)

    # Alpha features (simplified)
    alphas = last_window[['SMA_20', 'RSI', 'MACD', 'close', 'volume']].fillna(0).values

    # Price/temporal features
    prices_temporal = last_window[['close', 'volume']].fillna(0).values

    # Add temporal features (day of week, etc.)
    # Add more columns to match your model's expected input

    return {
        'alphas': alphas[:, :5],  # First 5 features as alphas
        'prices_temporal': prices_temporal
    }


def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices):
    """Calculate MACD"""
    ema_12 = prices.ewm(span=12, adjust=False).mean()
    ema_26 = prices.ewm(span=26, adjust=False).mean()
    return ema_12 - ema_26