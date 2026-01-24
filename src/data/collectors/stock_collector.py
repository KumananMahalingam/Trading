"""
Stock price data collection from Yahoo Finance
"""
import pandas as pd
import yfinance as yf


def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data from Yahoo Finance

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD or ISO format)
        end_date: End date (YYYY-MM-DD or ISO format)

    Returns:
        pd.DataFrame: Stock data with columns [date, open, high, low, close, volume]
    """
    try:
        print(f"  Fetching stock data for {ticker}...")

        # Extract date portion if ISO format
        start = start_date.split('T')[0] if 'T' in start_date else start_date
        end = end_date.split('T')[0] if 'T' in end_date else end_date

        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)

        if df.empty:
            print(f"  No data available for {ticker}")
            return pd.DataFrame()

        # Standardize column names
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'date': 'date'})
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # Select required columns
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        print(f"  Fetched {len(df)} days of stock data for {ticker}")
        return df

    except Exception as e:
        print(f"  Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()


def validate_ticker_quality(ticker, min_market_cap=5e9,
                            required_exchanges=['NYSE', 'NASDAQ', 'NYQ', 'NMS']):
    """
    Validate if a ticker meets quality requirements

    Args:
        ticker: Stock ticker symbol
        min_market_cap: Minimum market capitalization
        required_exchanges: List of acceptable exchanges

    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or len(info) < 5:
            return False, "Insufficient data"

        exchange = info.get('exchange', '').upper()
        if exchange not in required_exchanges:
            return False, f"Not on major exchange (found: {exchange})"

        market_cap = info.get('marketCap', 0)
        if market_cap < min_market_cap:
            return False, f"Market cap too low (${market_cap:,.0f})"

        volume = info.get('volume', 0)
        if volume < 100000:
            return False, "Low trading volume"

        return True, "Valid"

    except Exception as e:
        return False, f"Error: {str(e)}"