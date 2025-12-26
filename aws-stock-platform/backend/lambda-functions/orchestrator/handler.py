import json
import boto3
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

# Add your utils to path
sys.path.insert(0, '/opt/python')
sys.path.insert(0, '/var/task/utils')

# Import YOUR actual functions from integrated_pipeline.py
from integrated_pipeline import (
    validate_ticker_quality,
    fetch_stock_data,
    calculate_technical_indicators
)

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')

VALID_EXCHANGES = ['NYSE', 'NASDAQ', 'NYQ', 'NMS']

def lambda_handler(event, context):
    """
    Main orchestrator - handles user requests
    POST /predict
    """
    try:
        body = json.loads(event.get('body', '{}'))
        ticker = body.get('ticker', '').upper()
        forecast_days = body.get('forecast_days', 7)

        print(f"🎯 Prediction request for {ticker}")

        # ====================================================================
        # STEP 1: VALIDATE TICKER (using YOUR function)
        # ====================================================================

        is_valid, reason = validate_ticker_quality(ticker)

        if not is_valid:
            return {
                'statusCode': 400,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'error': f'Invalid ticker: {reason}',
                    'ticker': ticker
                })
            }

        print(f"✓ {ticker} is valid: {reason}")

        # Get ticker info for response
        stock = yf.Ticker(ticker)
        info = stock.info
        exchange = info.get('exchange', 'Unknown')
        company_name = info.get('longName', ticker)

        # ====================================================================
        # STEP 2: CHECK IF MODEL EXISTS IN S3
        # ====================================================================

        model_key = f'models/{ticker}_ensemble.pth'

        try:
            s3.head_object(
                Bucket=os.environ['MODELS_BUCKET'],
                Key=model_key
            )
            model_exists = True
            print(f"✓ Model exists for {ticker}")
        except:
            model_exists = False
            print(f"⚠️  No model found for {ticker}")

        if not model_exists:
            # Trigger training Lambda (async)
            print(f"Initiating training for {ticker}...")

            training_response = lambda_client.invoke(
                FunctionName=os.environ['TRAINING_LAMBDA_ARN'],
                InvocationType='Event',  # Async - don't wait
                Payload=json.dumps({'ticker': ticker})
            )

            return {
                'statusCode': 202,  # Accepted
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'status': 'training',
                    'ticker': ticker,
                    'company_name': company_name,
                    'message': f'Model training initiated for {ticker}',
                    'estimated_time': '2-3 minutes'
                })
            }

        # ====================================================================
        # STEP 3: GET RECENT STOCK DATA (using YOUR function)
        # ====================================================================

        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        stock_data = fetch_stock_data(
            ticker,
            start_date.isoformat() + 'Z',
            end_date.isoformat() + 'Z'
        )

        if stock_data.empty or len(stock_data) < 60:
            return {
                'statusCode': 400,
                'headers': {'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'error': 'Insufficient historical data',
                    'ticker': ticker
                })
            }

        print(f"✓ Fetched {len(stock_data)} days of data")

        # ====================================================================
        # STEP 4: GET OR GENERATE ALPHA FORMULAS
        # ====================================================================

        alphas_table = dynamodb.Table(os.environ.get('ALPHAS_TABLE', 'stock-alphas'))

        try:
            response = alphas_table.get_item(Key={'ticker': ticker})

            if 'Item' in response:
                alpha_text = response['Item']['alphas']
                print(f"✓ Using cached alphas for {ticker}")
            else:
                # Generate new alphas (will be done by prediction Lambda)
                alpha_text = generate_simple_alphas(ticker)
                print(f"✓ Using fallback alphas for {ticker}")
        except:
            alpha_text = generate_simple_alphas(ticker)

        # ====================================================================
        # STEP 5: INVOKE PREDICTION LAMBDA
        # ====================================================================

        # Convert DataFrame to list of dicts for JSON
        stock_data_json = stock_data.tail(60).to_dict('records')

        prediction_payload = {
            'ticker': ticker,
            'stock_data': stock_data_json,
            'alpha_text': alpha_text,
            'forecast_days': forecast_days
        }

        prediction_response = lambda_client.invoke(
            FunctionName=os.environ['PREDICTION_LAMBDA_ARN'],
            InvocationType='RequestResponse',  # Wait for response
            Payload=json.dumps(prediction_payload)
        )

        prediction_result = json.loads(prediction_response['Payload'].read())

        if prediction_result.get('statusCode') != 200:
            raise Exception(f"Prediction failed: {prediction_result.get('body')}")

        prediction_data = json.loads(prediction_result['body'])

        # ====================================================================
        # STEP 6: FORMAT RESPONSE
        # ====================================================================

        current_price = float(stock_data.iloc[-1]['close'])
        predicted_change = prediction_data['predicted_change']
        target_price = current_price * (1 + predicted_change)

        # Generate chart data
        chart_data = {
            'historical': [],
            'forecast': []
        }

        # Last 30 days historical
        for _, row in stock_data.tail(30).iterrows():
            chart_data['historical'].append({
                'time': row['date'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': int(row['volume'])
            })

        # Forecast (7 days)
        last_date = datetime.strptime(stock_data.iloc[-1]['date'], '%Y-%m-%d')
        last_price = current_price

        for i in range(forecast_days):
            forecast_date = last_date + timedelta(days=i+1)
            forecasted_price = last_price * (1 + predicted_change)

            chart_data['forecast'].append({
                'time': forecast_date.strftime('%Y-%m-%d'),
                'value': forecasted_price
            })

            last_price = forecasted_price

        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'ticker': ticker,
                'exchange': exchange,
                'company_name': company_name,
                'current_price': current_price,
                'prediction': {
                    'direction': prediction_data['direction'],
                    'predicted_change': predicted_change,
                    'confidence': prediction_data['confidence'],
                    'target_price': target_price
                },
                'chart_data': chart_data,
                'timestamp': datetime.now().isoformat()
            })
        }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

        return {
            'statusCode': 500,
            'headers': {'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'error': str(e),
                'ticker': ticker if 'ticker' in locals() else 'unknown'
            })
        }


def generate_simple_alphas(ticker):
    """Fallback alphas from YOUR integrated_pipeline.py"""
    return f"""
α1 = Return_5D
α2 = (RSI - 50) / 50
α3 = MACD - MACD_Signal
α4 = (close - SMA_20) / SMA_20
α5 = Return_5D + 0.1 * {ticker}_Sentiment
"""