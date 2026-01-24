"""
Economic data collection from FRED (Federal Reserve Economic Data)
"""
import pandas as pd


def fetch_fred_data(start_date, end_date, series_ids=None):
    """
    Fetch economic indicators from FRED

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        series_ids: Dictionary of {series_id: column_name} or None for defaults

    Returns:
        pd.DataFrame: Economic data
    """
    if series_ids is None:
        # Comprehensive economic indicators for trading models
        series_ids = {
            # GDP & Economic Growth
            'GDP': 'GDP',
            'GDPC1': 'Real_GDP',
            'A191RL1Q225SBEA': 'GDP_Growth_Rate',

            # Employment Data
            'PAYEMS': 'Nonfarm_Payrolls',
            'UNRATE': 'Unemployment_Rate',
            'CIVPART': 'Labor_Force_Participation',
            'UNEMPLOY': 'Unemployed_Persons',
            'EMRATIO': 'Employment_Population_Ratio',
            'ICSA': 'Initial_Jobless_Claims',

            # Inflation Data
            'CPIAUCSL': 'CPI',
            'CPILFESL': 'Core_CPI',
            'PPIACO': 'PPI',
            'PPIFIS': 'PPI_Final_Demand',
            'PCEPILFE': 'Core_PCE',

            # Interest Rates & Monetary Policy
            'DFF': 'Fed_Funds_Rate',
            'DGS10': 'Treasury_10Y',
            'DGS2': 'Treasury_2Y',
            'DGS5': 'Treasury_5Y',
            'T10Y2Y': 'Yield_Curve_10Y2Y',
            'T10Y3M': 'Yield_Curve_10Y3M',
            'MORTGAGE30US': 'Mortgage_Rate_30Y',

            # Retail Sales & Consumer Spending
            'RSXFS': 'Retail_Sales',
            'RETAILSMSA': 'Retail_Sales_Total',
            'RRSFS': 'Retail_Sales_Ex_Auto',
            'PCE': 'Personal_Consumption',
            'PCEDG': 'PCE_Durable_Goods',

            # PMI & Business Conditions
            'MANEMP': 'Manufacturing_Employment',
            'INDPRO': 'Industrial_Production',
            'IPMAN': 'Manufacturing_Production',
            'TCU': 'Capacity_Utilization',

            # Consumer Confidence
            'UMCSENT': 'Consumer_Sentiment_Michigan',
            'CSCICP03USM665S': 'Consumer_Confidence',

            # Housing Market
            'HOUST': 'Housing_Starts',
            'PERMIT': 'Building_Permits',
            'ASPUS': 'Average_Sales_Price_Houses',
            'CSUSHPISA': 'Case_Shiller_Home_Price',
            'MSPUS': 'Median_Sales_Price_Houses',

            # Trade Balance
            'BOPGSTB': 'Trade_Balance_Goods_Services',
            'BOPGTB': 'Trade_Balance_Goods',
            'XTEXVA01USM667S': 'Exports',
            'XTIMVA01USM667S': 'Imports',

            # Market Indicators
            'VIXCLS': 'VIX',
            'DEXUSEU': 'Dollar_Euro',
            'DEXJPUS': 'Dollar_Yen',
            'DEXCHUS': 'Dollar_Yuan',
            'DCOILWTICO': 'Oil_Price_WTI',
            'DCOILBRENTEU': 'Oil_Price_Brent',
            'GOLDAMGBD228NLBM': 'Gold_Price',

            # Leading Economic Indicators
            'USSLIND': 'Leading_Index',
            'M2SL': 'M2_Money_Supply',
            'DPCREDIT': 'Total_Credit',
        }

    print("  Fetching FRED economic data...")
    print(f"    Requesting {len(series_ids)} economic indicators...")

    try:
        from fredapi import Fred
    except ImportError:
        print("    Warning: fredapi not installed")
        return pd.DataFrame()

    try:
        import config.secret_key as secret_key
        fred = Fred(api_key=secret_key.FRED_API_KEY)
    except:
        print("    Warning: FRED_API_KEY not found in config.secret_key")
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
                    'Personal_Consumption', 'PCE_Durable_Goods', 'Consumer_Sentiment_Michigan',
                    'Consumer_Confidence'],
        'Manufacturing': ['Manufacturing_Employment', 'Industrial_Production',
                         'Manufacturing_Production', 'Capacity_Utilization'],
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
                except Exception:
                    # Silently skip unavailable series
                    pass

    print(f"    Successfully fetched {success_count}/{len(series_ids)} indicators")

    if all_data:
        df = pd.DataFrame(all_data)
        df.index.name = 'date'
        df.reset_index(inplace=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # Forward and backward fill missing values
        df = df.ffill()
        df = df.bfill()

        print(f"    Total: {len(df)} days with {len(df.columns)-1} economic indicators")

        # Show loaded categories
        loaded_categories = []
        for category, indicators in categories.items():
            if any(ind in df.columns for ind in indicators):
                loaded_categories.append(category)
        print(f"    Categories loaded: {', '.join(loaded_categories)}")

        return df

    return pd.DataFrame()