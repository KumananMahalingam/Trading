"""
Technical indicator calculations
"""
import numpy as np
import pandas as pd


def calculate_adx(df, period=14):
    """
    Calculate Average Directional Index (ADX)

    Args:
        df: DataFrame with high, low, close columns
        period: Lookback period

    Returns:
        pd.Series: ADX values
    """
    high_diff = df['high'].diff()
    low_diff = -df['low'].diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    atr = np.max(ranges, axis=1).rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx


def calculate_technical_indicators(df):
    """
    Calculate comprehensive technical indicators

    Args:
        df: DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    if df.empty:
        return df

    # Simple Moving Averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # Returns
    df['Return_1D'] = df['close'].pct_change(1)
    df['Return_5D'] = df['close'].pct_change(5)
    df['Return_20D'] = df['close'].pct_change(20)
    df['Return_60D'] = df['close'].pct_change(60)

    # Volatility
    df['Volatility_5D'] = df['Return_1D'].rolling(window=5).std()
    df['Volatility_20D'] = df['Return_1D'].rolling(window=20).std()
    df['Volatility_60D'] = df['Return_1D'].rolling(window=60).std()

    # Volume indicators
    df['Volume_SMA_20'] = df['volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['volume'] / df['Volume_SMA_20']

    # Momentum
    df['Momentum_5'] = df['close'] - df['close'].shift(5)
    df['Momentum_10'] = df['close'] - df['close'].shift(10)
    df['Momentum_20'] = df['close'] - df['close'].shift(20)

    # Rate of Change
    df['ROC_5'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
    df['ROC_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

    # Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

    # Money Flow Index (MFI)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi_ratio = positive_flow / negative_flow
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Stochastic Oscillator
    low_14 = df['low'].rolling(14).min()
    high_14 = df['high'].rolling(14).max()
    df['Stochastic_K'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
    df['Stochastic_D'] = df['Stochastic_K'].rolling(3).mean()

    # Williams %R
    df['Williams_R'] = -100 * (high_14 - df['close']) / (high_14 - low_14)

    # ADX (Trend strength)
    df['ADX'] = calculate_adx(df)

    return df