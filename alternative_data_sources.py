import pandas as pd
import re
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf

sia = SentimentIntensityAnalyzer()


# ============================================================================
# FRED ECONOMIC DATA
# ============================================================================

def fetch_fred_data(start_date, end_date, series_ids=None):

    if series_ids is None:
        # Comprehensive economic indicators for trading models
        series_ids = {
            # GDP & Economic Growth
            'GDP': 'GDP',                              # Gross Domestic Product
            'GDPC1': 'Real_GDP',                       # Real GDP (inflation-adjusted)
            'A191RL1Q225SBEA': 'GDP_Growth_Rate',      # GDP Growth Rate (QoQ)

            # Employment Data
            'PAYEMS': 'Nonfarm_Payrolls',              # Non-Farm Payrolls (NFP)
            'UNRATE': 'Unemployment_Rate',             # Unemployment Rate
            'CIVPART': 'Labor_Force_Participation',    # Labor Force Participation Rate
            'UNEMPLOY': 'Unemployed_Persons',          # Number of Unemployed
            'EMRATIO': 'Employment_Population_Ratio',  # Employment-Population Ratio
            'ICSA': 'Initial_Jobless_Claims',          # Initial Jobless Claims (weekly)

            # Inflation Data
            'CPIAUCSL': 'CPI',                         # Consumer Price Index (All Items)
            'CPILFESL': 'Core_CPI',                    # Core CPI (ex food & energy)
            'PPIACO': 'PPI',                           # Producer Price Index (All Commodities)
            'PPIFIS': 'PPI_Final_Demand',              # PPI for Final Demand
            'PCEPILFE': 'Core_PCE',                    # Core PCE Price Index (Fed's preferred)

            # Interest Rates & Monetary Policy
            'DFF': 'Fed_Funds_Rate',                   # Federal Funds Effective Rate
            'DGS10': 'Treasury_10Y',                   # 10-Year Treasury Rate
            'DGS2': 'Treasury_2Y',                     # 2-Year Treasury Rate
            'DGS5': 'Treasury_5Y',                     # 5-Year Treasury Rate
            'T10Y2Y': 'Yield_Curve_10Y2Y',             # 10Y-2Y Treasury Spread
            'T10Y3M': 'Yield_Curve_10Y3M',             # 10Y-3M Treasury Spread (recession indicator)
            'MORTGAGE30US': 'Mortgage_Rate_30Y',       # 30-Year Mortgage Rate

            # Retail Sales & Consumer Spending
            'RSXFS': 'Retail_Sales',                   # Advance Retail Sales
            'RETAILSMSA': 'Retail_Sales_Total',        # Retail and Food Services Sales
            'RRSFS': 'Retail_Sales_Ex_Auto',           # Retail Sales Ex Auto
            'PCE': 'Personal_Consumption',             # Personal Consumption Expenditures
            'PCEDG': 'PCE_Durable_Goods',              # PCE: Durable Goods

            # PMI & Business Conditions
            'MANEMP': 'Manufacturing_Employment',      # Manufacturing Employment
            'INDPRO': 'Industrial_Production',         # Industrial Production Index
            'IPMAN': 'Manufacturing_Production',       # Industrial Production: Manufacturing
            'TCU': 'Capacity_Utilization',             # Capacity Utilization

            # Consumer Confidence
            'UMCSENT': 'Consumer_Sentiment_Michigan',  # University of Michigan Consumer Sentiment
            'CSCICP03USM665S': 'Consumer_Confidence',  # OECD Consumer Confidence Index

            # Housing Market
            'HOUST': 'Housing_Starts',                 # Housing Starts (thousands of units)
            'PERMIT': 'Building_Permits',              # New Private Housing Units Authorized
            'ASPUS': 'Average_Sales_Price_Houses',     # Average Sales Price of Houses Sold
            'CSUSHPISA': 'Case_Shiller_Home_Price',    # S&P/Case-Shiller Home Price Index
            'MSPUS': 'Median_Sales_Price_Houses',      # Median Sales Price of Houses

            # Trade Balance
            'BOPGSTB': 'Trade_Balance_Goods_Services', # Trade Balance: Goods and Services
            'BOPGTB': 'Trade_Balance_Goods',           # Trade Balance: Goods
            'XTEXVA01USM667S': 'Exports',              # Exports of Goods and Services
            'XTIMVA01USM667S': 'Imports',              # Imports of Goods and Services

            # Market Indicators
            'VIXCLS': 'VIX',                           # CBOE Volatility Index
            'DEXUSEU': 'Dollar_Euro',                  # USD/EUR Exchange Rate
            'DEXJPUS': 'Dollar_Yen',                   # USD/JPY Exchange Rate
            'DEXCHUS': 'Dollar_Yuan',                  # USD/CNY Exchange Rate
            'DCOILWTICO': 'Oil_Price_WTI',             # WTI Crude Oil Price
            'DCOILBRENTEU': 'Oil_Price_Brent',         # Brent Crude Oil Price
            'GOLDAMGBD228NLBM': 'Gold_Price',          # Gold Fixing Price

            # Leading Economic Indicators
            'USSLIND': 'Leading_Index',                # Leading Index for the United States
            'M2SL': 'M2_Money_Supply',                 # M2 Money Stock
            'DPCREDIT': 'Total_Credit',                # Domestic Private Credit
        }

    print("  Fetching FRED economic data...")
    print(f"    Requesting {len(series_ids)} economic indicators...")

    try:
        from fredapi import Fred
    except ImportError:
        print("    Warning: fredapi not installed")
        return pd.DataFrame()

    try:
        import secret_key
        fred = Fred(api_key=secret_key.FRED_API_KEY)
    except:
        print("    Warning: FRED_API_KEY not found in secret_key.py")
        return pd.DataFrame()

    all_data = {}
    success_count = 0

    # Fetch in categories for better error handling
    categories = {
        'GDP & Growth': ['GDP', 'Real_GDP', 'GDP_Growth_Rate'],
        'Employment': ['Nonfarm_Payrolls', 'Unemployment_Rate', 'Labor_Force_Participation',
                      'Unemployed_Persons', 'Employment_Population_Ratio', 'Initial_Jobless_Claims'],
        'Inflation': ['CPI', 'Core_CPI', 'PPI', 'PPI_Final_Demand', 'Core_PCE'],
        'Interest Rates': ['Fed_Funds_Rate', 'Treasury_10Y', 'Treasury_2Y', 'Treasury_5Y',
                          'Yield_Curve_10Y2Y', 'Yield_Curve_10Y3M', 'Mortgage_Rate_30Y'],
        'Consumer': ['Retail_Sales', 'Retail_Sales_Total', 'Retail_Sales_Ex_Auto',
                    'Personal_Consumption', 'PCE_Durable_Goods', 'Consumer_Sentiment_Michigan', 'Consumer_Confidence'],
        'Manufacturing': ['Manufacturing_Employment', 'Industrial_Production', 'Manufacturing_Production',
                         'Capacity_Utilization'],
        'Housing': ['Housing_Starts', 'Building_Permits', 'Average_Sales_Price_Houses',
                   'Case_Shiller_Home_Price', 'Median_Sales_Price_Houses'],
        'Trade': ['Trade_Balance_Goods_Services', 'Trade_Balance_Goods', 'Exports', 'Imports'],
        'Markets': ['VIX', 'Dollar_Euro', 'Dollar_Yen', 'Dollar_Yuan', 'Oil_Price_WTI',
                   'Oil_Price_Brent', 'Gold_Price'],
        'Leading Indicators': ['Leading_Index', 'M2_Money_Supply', 'Total_Credit']
    }

    for category, indicators in categories.items():
        print(f"    Fetching {category}...")
        for series_id, column_name in series_ids.items():
            if column_name in indicators:
                try:
                    data = fred.get_series(series_id, start_date, end_date)
                    all_data[column_name] = data
                    success_count += 1
                except Exception as e:
                    # Silently skip unavailable series (some are quarterly/monthly only)
                    pass

    print(f"    ✓ Successfully fetched {success_count}/{len(series_ids)} indicators")

    if all_data:
        df = pd.DataFrame(all_data)
        df.index.name = 'date'
        df.reset_index(inplace=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # Forward fill missing values (economic data is often weekly/monthly/quarterly)
        df = df.fillna(method='ffill')

        # Backward fill any remaining NaN at the start
        df = df.fillna(method='bfill')

        print(f"    Total: {len(df)} days with {len(df.columns)-1} economic indicators")

        # Show which categories were successfully loaded
        loaded_categories = []
        for category, indicators in categories.items():
            if any(ind in df.columns for ind in indicators):
                loaded_categories.append(category)
        print(f"    Categories loaded: {', '.join(loaded_categories)}")

        return df

    return pd.DataFrame()


# ============================================================================
# SEC EDGAR FILINGS (10-K, 10-Q)
# ============================================================================

def fetch_sec_filings(ticker, start_date, end_date):
    """
    Fetch SEC filings (10-K annual reports, 10-Q quarterly reports)
    Analyzes sentiment of Management Discussion & Analysis section

    Returns: dict of {date: sentiment_score}
    """
    print(f"  Fetching SEC filings for {ticker}...")

    filing_sentiments = {}

    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        print("    Warning: sec-edgar-downloader not installed.")
        return filing_sentiments

    try:
        # Download filings
        dl = Downloader("MyCompany", "myemail@example.com")

        # Get 10-K and 10-Q filings
        for filing_type in ['10-K', '10-Q']:
            try:
                dl.get(filing_type, ticker, limit=20)

                # Parse downloaded files
                import os
                filing_dir = f"sec-edgar-filings/{ticker}/{filing_type}"

                if os.path.exists(filing_dir):
                    for root, dirs, files in os.walk(filing_dir):
                        for file in files:
                            if file.endswith('.txt'):
                                filepath = os.path.join(root, file)

                                # Extract filing date from directory structure
                                try:
                                    filing_date = root.split('/')[-1]

                                    # Read file
                                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                        content = f.read()

                                    # Extract MD&A section
                                    mda = extract_mda_section(content)

                                    if mda and len(mda) > 100:
                                        # Analyze sentiment (sample to avoid overwhelming VADER)
                                        sample = mda[:10000]  # First 10k characters
                                        sentiment = sia.polarity_scores(sample)['compound']
                                        filing_sentiments[filing_date] = sentiment
                                        print(f"    ✓ Analyzed {filing_type} filed on {filing_date}")

                                except Exception as e:
                                    continue

            except Exception as e:
                print(f"    Error fetching {filing_type}: {e}")

        print(f"    Total: {len(filing_sentiments)} SEC filings analyzed")

    except Exception as e:
        print(f"    Error with SEC data: {e}")

    return filing_sentiments


def extract_mda_section(filing_text):
    """
    Extract Management Discussion & Analysis section from SEC filing
    """
    # Common patterns for MD&A section
    patterns = [
        r'ITEM 7\.\s*MANAGEMENT\'?S DISCUSSION AND ANALYSIS(.*?)ITEM 8',
        r'ITEM 2\.\s*MANAGEMENT\'?S DISCUSSION AND ANALYSIS(.*?)ITEM 3',
        r'Management\'?s Discussion and Analysis(.*?)(?:Quantitative and Qualitative|Financial Statements)',
    ]

    for pattern in patterns:
        match = re.search(pattern, filing_text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)

    return ""


# ============================================================================
# YAHOO FINANCE EARNINGS TRANSCRIPTS
# ============================================================================

def fetch_earnings_transcripts(ticker, start_date, end_date):
    """
    Fetch earnings call transcripts and analyze sentiment

    Returns: dict of {date: sentiment_score}
    """
    print(f"  Fetching earnings transcripts for {ticker}...")

    transcript_sentiments = {}

    try:
        # Get earnings dates from yfinance
        stock = yf.Ticker(ticker)
        earnings_dates = stock.earnings_dates

        if earnings_dates is not None and not earnings_dates.empty:
            earnings_dates = earnings_dates.reset_index()
            earnings_dates['Earnings Date'] = pd.to_datetime(earnings_dates['Earnings Date'])

            # Filter by date range
            mask = (earnings_dates['Earnings Date'] >= start_date) & \
                   (earnings_dates['Earnings Date'] <= end_date)
            relevant_dates = earnings_dates[mask]

            print(f"    Found {len(relevant_dates)} earnings dates")

            # Note: Actual transcript scraping requires additional libraries
            # For now, we'll use a proxy: earnings surprise sentiment
            for _, row in relevant_dates.iterrows():
                date_str = row['Earnings Date'].strftime('%Y-%m-%d')

                # Earnings surprise as sentiment proxy
                surprise = row.get('Surprise(%)', 0)
                if pd.notna(surprise):
                    # Convert surprise % to sentiment (-1 to 1)
                    sentiment = max(-1, min(1, surprise / 100))
                    transcript_sentiments[date_str] = sentiment

            print(f"    Analyzed {len(transcript_sentiments)} earnings events")

    except Exception as e:
        print(f"    Error fetching earnings data: {e}")

    return transcript_sentiments


# ============================================================================
# INTEGRATION FUNCTION - COMBINES ALL SOURCES
# ============================================================================

def fetch_all_alternative_data(ticker, start_date, end_date):
    """
    Fetch all alternative data sources and combine them

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        tuple: (economic_data_df, filing_sentiments_dict, earnings_sentiments_dict)
    """
    print(f"\n{'='*80}")
    print(f"FETCHING ALTERNATIVE DATA FOR {ticker}")
    print(f"{'='*80}")

    # 1. FRED Economic Data (once per run, not per ticker - call separately)
    economic_data = fetch_fred_data(start_date, end_date)

    # 2. SEC Filings
    sec_sentiments = fetch_sec_filings(ticker, start_date, end_date)
    time.sleep(2)

    # 3. Earnings Transcripts
    earnings_sentiments = fetch_earnings_transcripts(ticker, start_date, end_date)

    print(f"\n{'='*80}")
    print(f"ALTERNATIVE DATA SUMMARY FOR {ticker}")
    print(f"{'='*80}")
    print(f"  SEC Filings: {len(sec_sentiments)}")
    print(f"  Earnings Events: {len(earnings_sentiments)}")
    print(f"{'='*80}\n")

    return sec_sentiments, earnings_sentiments


# ============================================================================
# HELPER: ADD ALTERNATIVE DATA TO DATAFRAME
# ============================================================================

def add_alternative_features_to_df(df, ticker, economic_data, sec_sentiment, earnings_sentiment):
    """
    Add all alternative data features to the main dataframe

    Args:
        df: Main stock dataframe with date column
        ticker: Stock ticker
        economic_data: DataFrame with economic indicators
        sec_sentiment: dict of {date: sentiment_score}
        earnings_sentiment: dict of {date: sentiment_score}

    Returns:
        Enhanced dataframe
    """
    df = df.copy()

    # 1. Merge economic data
    if not economic_data.empty:
        df = df.merge(economic_data, on='date', how='left')

        # Forward fill economic data (released weekly/monthly)
        econ_cols = [col for col in economic_data.columns if col != 'date']
        df[econ_cols] = df[econ_cols].fillna(method='ffill')

    # 2. Add SEC filing sentiment (forward fill between filings)
    df[f'{ticker}_SEC_Sentiment'] = df['date'].map(sec_sentiment)
    df[f'{ticker}_SEC_Sentiment'] = df[f'{ticker}_SEC_Sentiment'].fillna(method='ffill').fillna(0)

    # 3. Add earnings sentiment (binary flag + sentiment)
    df[f'{ticker}_Earnings_Event'] = df['date'].isin(earnings_sentiment.keys()).astype(int)
    df[f'{ticker}_Earnings_Sentiment'] = df['date'].map(earnings_sentiment).fillna(0)

    # Days since last earnings (useful feature)
    earnings_dates = sorted([d for d in earnings_sentiment.keys()])
    def days_since_earnings(date):
        date_obj = pd.to_datetime(date)
        previous_earnings = [pd.to_datetime(d) for d in earnings_dates if pd.to_datetime(d) <= date_obj]
        if previous_earnings:
            return (date_obj - max(previous_earnings)).days
        return 999  # No previous earnings

    df[f'{ticker}_Days_Since_Earnings'] = df['date'].apply(days_since_earnings)

    new_features = len([c for c in df.columns if c not in ['date', 'open', 'high', 'low', 'close', 'volume']])
    print(f"  ✓ Added alternative features (total features: {new_features})")

    return df