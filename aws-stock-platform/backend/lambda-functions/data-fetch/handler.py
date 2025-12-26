import json
import boto3
from datetime import datetime, timedelta
import yfinance as yf
import os

dynamodb = boto3.resource('dynamodb')

TICKERS = ['AAPL', 'JPM', 'PEP', 'TM', 'AMZN']

def lambda_handler(event, context):
    """
    Triggered by CloudWatch Events every hour
    Fetches latest stock data and stores in DynamoDB
    """
    print(f"Starting data fetch for {len(TICKERS)} tickers")

    stock_data_table = dynamodb.Table(os.environ['STOCK_DATA_TABLE'])

    success_count = 0
    error_count = 0

    for ticker in TICKERS:
        try:
            print(f"Fetching data for {ticker}")

            # Get last 2 days of data (in case market was closed yesterday)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=2)

            # Fetch from yfinance (same as your code)
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                print(f"No data returned for {ticker}")
                error_count += 1
                continue

            # Get the most recent day
            latest = df.iloc[-1]
            date_str = df.index[-1].strftime('%Y-%m-%d')

            # Store in DynamoDB
            item = {
                'ticker': ticker,
                'date': date_str,
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'close': float(latest['Close']),
                'volume': int(latest['Volume']),
                'timestamp': datetime.now().isoformat()
            }

            stock_data_table.put_item(Item=item)

            print(f"✓ Stored {ticker} data for {date_str}: close=${latest['Close']:.2f}")
            success_count += 1

        except Exception as e:
            print(f"✗ Error processing {ticker}: {e}")
            error_count += 1
            continue

    result = {
        'success_count': success_count,
        'error_count': error_count,
        'tickers_processed': TICKERS,
        'timestamp': datetime.now().isoformat()
    }

    print(f"\nData fetch complete: {success_count} succeeded, {error_count} failed")

    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }