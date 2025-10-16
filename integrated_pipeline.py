import time
import secret_key
from openpyxl import Workbook, load_workbook
from datetime import datetime, timezone
import spacy
from polygon import RESTClient
from collections import defaultdict, Counter
import json
import re
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from groq import Groq
import torch
import os

# Import PyTorch LSTM predictor
from lstm_predictor import train_stock_predictor

client = RESTClient(secret_key.API_KEY)
groq_client = Groq(api_key=secret_key.GROQ_API_KEY)
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()

companies = {
    "AAPL": "Apple",
    "JPM": "JPMorgan Chase & Co",
    "PEP": "Pepsi",
    "TM": "Toyota",
    "AMZN": "Amazon"
}

COMPANY_ALIASES = {
    "APPLE": ["AAPL"],
    "JPMORGAN": ["JPM", "JPM.N"],
    "PEPSI": ["PEP", "PEPSICO"],
    "TOYOTA": ["TM", "TYO"],
    "AMAZON": ["AMZN"]
}

# ============================================================================
# SENTIMENT CACHE FUNCTIONS
# ============================================================================

def load_existing_sentiment_data(excel_path="stock_analysis_complete.xlsx"):
    """
    Load existing sentiment data from Excel to avoid re-fetching
    Returns: dict of {ticker: {date: [scores]}}
    """
    if not os.path.exists(excel_path):
        print(f"No existing file found at {excel_path}, starting fresh...")
        return defaultdict(lambda: defaultdict(list))

    print(f"\nLoading existing sentiment data from {excel_path}...")
    daily_sentiments = defaultdict(lambda: defaultdict(list))

    try:
        wb = load_workbook(excel_path, read_only=True)

        # Check if Daily Sentiment sheet exists
        if "Daily Sentiment" not in wb.sheetnames:
            print("  No 'Daily Sentiment' sheet found, starting fresh...")
            wb.close()
            return daily_sentiments

        sheet = wb["Daily Sentiment"]

        # Skip header row
        rows = list(sheet.iter_rows(min_row=2, values_only=True))

        for row in rows:
            if len(row) >= 5:
                company_name, ticker, date, avg_sentiment, article_count = row[:5]

                if ticker and date and avg_sentiment is not None and article_count:
                    # Reconstruct the sentiment scores (approximate)
                    # Since we only have average, we'll create article_count copies
                    # This maintains the average while keeping track of article count
                    scores = [avg_sentiment] * int(article_count)
                    daily_sentiments[ticker][date] = scores

        wb.close()

        # Summary
        tickers_found = len(daily_sentiments)
        total_days = sum(len(dates) for dates in daily_sentiments.values())

        print(f"  ✓ Loaded sentiment data for {tickers_found} tickers")
        print(f"  ✓ Total days with sentiment: {total_days}")

        # Show what was loaded
        for ticker in sorted(daily_sentiments.keys()):
            days = len(daily_sentiments[ticker])
            print(f"    - {ticker}: {days} days")

        return daily_sentiments

    except Exception as e:
        print(f"  Error loading existing data: {e}")
        print("  Starting with fresh sentiment data...")
        return defaultdict(lambda: defaultdict(list))


def should_fetch_sentiment(ticker, daily_sentiments, min_days=30):
    """
    Check if we need to fetch sentiment for a ticker
    Returns: True if we should fetch, False if we already have enough data
    """
    if ticker not in daily_sentiments:
        return True

    days_with_data = len(daily_sentiments[ticker])

    if days_with_data < min_days:
        print(f"    Only {days_with_data} days of data for {ticker}, fetching more...")
        return True

    print(f"    Already have {days_with_data} days of data for {ticker}, skipping...")
    return False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_company_tickers_json(file_path="company_tickers.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} companies from {file_path}")
        ticker_to_info = {}
        name_to_ticker = {}
        name_variations = {}
        for key, company in data.items():
            ticker = company.get('ticker', '').upper()
            name = company.get('title', '')
            cik = company.get('cik_str', '')
            if ticker and name:
                ticker_to_info[ticker] = {'name': name, 'ticker': ticker, 'cik': cik}
                name_to_ticker[name.upper()] = ticker
                variations = generate_name_variations(name)
                for variation in variations:
                    if len(variation) >= 3:
                        name_variations[variation.upper()] = ticker
        print(f"Created {len(name_variations)} name variations for matching")
        return {
            'ticker_to_info': ticker_to_info,
            'name_to_ticker': name_to_ticker,
            'name_variations': name_variations
        }
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def generate_name_variations(company_name):
    variations = [company_name]
    suffixes_pattern = r'\s+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?|Holdings|Holding|LLC|L\.L\.C\.|plc|PLC)'
    base_name = re.sub(suffixes_pattern, '', company_name, flags=re.IGNORECASE).strip()
    variations.append(base_name)
    variations.extend([
        company_name.replace(',', '').strip(),
        company_name.replace('.', '').strip(),
        base_name.replace(',', '').strip(),
        base_name.replace('.', '').strip(),
        company_name.replace('&', 'and'),
        company_name.replace(' and ', ' & '),
        base_name.replace('&', 'and'),
        base_name.replace(' and ', ' & ')
    ])
    words = base_name.split()
    if len(words) > 1:
        acronym = ''.join([word[0] for word in words if word[0].isupper()])
        if len(acronym) >= 2:
            variations.append(acronym)
    return list(set([v.strip() for v in variations if v.strip()]))

def validate_company_exists(company_name, company_lookups):
    if not company_lookups:
        return False, None, None
    company_upper = company_name.upper().strip()
    if company_upper in company_lookups['name_to_ticker']:
        ticker = company_lookups['name_to_ticker'][company_upper]
        official_name = company_lookups['ticker_to_info'][ticker]['name']
        return True, ticker, official_name
    if company_upper in company_lookups['name_variations']:
        ticker = company_lookups['name_variations'][company_upper]
        official_name = company_lookups['ticker_to_info'][ticker]['name']
        return True, ticker, official_name
    if company_upper in company_lookups['ticker_to_info']:
        ticker = company_upper
        official_name = company_lookups['ticker_to_info'][ticker]['name']
        return True, ticker, official_name
    for variation, ticker in company_lookups['name_variations'].items():
        if len(company_upper) >= 4 and company_upper in variation:
            official_name = company_lookups['ticker_to_info'][ticker]['name']
            return True, ticker, official_name
        elif len(variation) >= 4 and variation in company_upper:
            official_name = company_lookups['ticker_to_info'][ticker]['name']
            return True, ticker, official_name
    return False, None, None

def is_same_company(ticker1, ticker2, name1, name2):
    if ticker1 == ticker2:
        return True
    for base_name, aliases in COMPANY_ALIASES.items():
        if ticker1 in aliases and ticker2 in aliases:
            return True
    name1_base = re.sub(r'\s+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?)', '', name1, flags=re.IGNORECASE).strip().upper()
    name2_base = re.sub(r'\s+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?)', '', name2, flags=re.IGNORECASE).strip().upper()
    if name1_base == name2_base:
        return True
    return False

def validate_ticker_quality(ticker, min_market_cap=5e9, required_exchanges=['NYSE', 'NASDAQ', 'NYQ', 'NMS']):
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

def fetch_stock_data(ticker, start_date, end_date):
    try:
        print(f"  Fetching stock data for {ticker}...")
        start = start_date.split('T')[0]
        end = end_date.split('T')[0]
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        if df.empty:
            print(f"  No data available for {ticker}")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.columns = df.columns.str.lower()
        df = df.rename(columns={'date': 'date'})
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        print(f"  Fetched {len(df)} days of stock data for {ticker}")
        return df
    except Exception as e:
        print(f"  Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    if df.empty:
        return df
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['Return_1D'] = df['close'].pct_change(1)
    df['Return_5D'] = df['close'].pct_change(5)
    df['Return_20D'] = df['close'].pct_change(20)
    df['Volatility_20D'] = df['Return_1D'].rolling(window=20).std()
    return df

def prepare_dataframe_for_alpha(ticker, stock_df, daily_sentiments, related_companies):
    if stock_df.empty:
        return None
    df = calculate_technical_indicators(stock_df.copy())
    sentiment_dict = {}
    for date, scores in daily_sentiments[ticker].items():
        sentiment_dict[date] = sum(scores) / len(scores) if scores else 0
    df[f'{ticker}_Sentiment'] = df['date'].map(sentiment_dict).fillna(0)
    for related_ticker in related_companies:
        if related_ticker in daily_sentiments:
            related_sentiment_dict = {}
            for date, scores in daily_sentiments[related_ticker].items():
                related_sentiment_dict[date] = sum(scores) / len(scores) if scores else 0
            df[f'{related_ticker}_Sentiment'] = df['date'].map(related_sentiment_dict).fillna(0)
    for related_ticker in related_companies:
        if f'{related_ticker}_Sentiment' in df.columns:
            df[f'Sentiment_Div_{ticker}_{related_ticker}'] = df[f'{ticker}_Sentiment'] - df[f'{related_ticker}_Sentiment']
    return df

def generate_predictive_alphas(ticker, company_name, dataframe_json, related_companies):
    try:
        related_list = ", ".join(related_companies) if related_companies else "None available"

        # Extract available sentiment columns from the dataframe
        sample_df = pd.read_json(dataframe_json)
        available_sentiment_cols = [col for col in sample_df.columns if 'Sentiment' in col]

        prompt = f"""Generating Predictive Alphas for {company_name} ({ticker}) Stock Prices

CRITICAL: You can ONLY use these exact column names:

Stock Features: close, open, high, low, volume
Technical Indicators: RSI, SMA_5, SMA_20, SMA_50, EMA_12, EMA_26, MACD, MACD_Signal, BB_Upper, BB_Middle, BB_Lower
Returns: Return_1D, Return_5D, Return_20D
Volatility: Volatility_20D
Sentiment Columns AVAILABLE: {', '.join(available_sentiment_cols)}

DO NOT use any other sentiment columns unless they appear in the list above.

Related Companies: {related_list}

DataFrame Sample:
{dataframe_json}

Generate 5 alpha formulas using ONLY the columns listed above. Focus on:
1. Momentum and mean reversion using technical indicators
2. Sentiment from available columns only
3. Combinations of price action and sentiment

Examples:
α1 = Return_5D + 0.2 * Return_20D
α2 = (RSI - 50) / 50
α3 = MACD - MACD_Signal
α4 = (close - SMA_20) / SMA_20
α5 = Return_5D + 0.1 * {ticker}_Sentiment

Provide ONLY 5 formulas."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a quantitative analyst. Generate alpha formulas using ONLY the exact column names provided. DO NOT invent column names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )

        alphas = response.choices[0].message.content
        return alphas

    except Exception as e:
        print(f"Error generating alphas for {ticker}: {e}")
        return None

def generate_simple_alphas(ticker):
    """Fallback alphas that don't require related company sentiment"""
    return f"""
α1 = Return_5D
α2 = (RSI - 50) / 50
α3 = MACD - MACD_Signal
α4 = (close - SMA_20) / SMA_20
α5 = Return_5D + 0.1 * {ticker}_Sentiment
"""

def fetch_news(ticker, start_date, end_date, batch_size=1000, sleep_time=12):
    results = []
    batch_count = 0
    try:
        response = client.list_ticker_news(
            ticker=ticker,
            published_utc_gte=start_date,
            published_utc_lte=end_date,
            limit=batch_size,
            order="asc"
        )
        current_batch = []
        for item in response:
            current_batch.append(item)
            if len(current_batch) >= batch_size:
                break
        while current_batch:
            results.extend(current_batch)
            batch_count += 1
            print(f"  Fetched batch {batch_count} ({len(current_batch)} articles) for {ticker}. Total: {len(results)}")
            if len(current_batch) < batch_size:
                print(f"  Reached end of articles for {ticker}")
                break
            last_article_date = current_batch[-1].published_utc
            if last_article_date >= end_date:
                print(f"  Reached end date for {ticker}")
                break
            print(f"  Waiting {sleep_time} seconds before next batch...")
            time.sleep(sleep_time)
            response = client.list_ticker_news(
                ticker=ticker,
                published_utc_gt=last_article_date,
                published_utc_lte=end_date,
                limit=batch_size,
                order='asc'
            )
            current_batch = []
            for item in response:
                current_batch.append(item)
                if len(current_batch) >= batch_size:
                    break
        print(f"  Total articles fetched for {ticker}: {len(results)}")
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
    return results

def fetch_sentiment_for_ticker(ticker, start_date, end_date, daily_sentiments):
    """
    Fetch news and compute sentiment for a specific ticker
    Updates daily_sentiments dict in place
    """
    print(f"  Fetching sentiment data for {ticker}...")

    # Check if already fetched
    if ticker in daily_sentiments and len(daily_sentiments[ticker]) > 0:
        print(f"    Already have sentiment data for {ticker}")
        return

    # Fetch news
    news_articles = fetch_news(ticker, start_date, end_date, batch_size=500, sleep_time=12)

    if not news_articles:
        print(f"    No news articles found for {ticker}")
        return

    # Process sentiment
    for item in news_articles:
        text = f"{item.title} {item.summary if hasattr(item, 'summary') else ''}"
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        article_date = item.published_utc.split('T')[0] if 'T' in item.published_utc else item.published_utc[:10]
        daily_sentiments[ticker][article_date].append(compound_score)

    print(f"    Processed {len(news_articles)} articles, {len(daily_sentiments[ticker])} days of sentiment")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*80)
print("STOCK PRICE PREDICTION PIPELINE")
print("Phase 0: Load Existing Sentiment Data")
print("Phase 1: News Fetching & Sentiment Analysis")
print("Phase 2: Related Company Sentiment Collection")
print("Phase 3: Alpha Generation & LSTM Training")
print("="*80)

print("\nLoading company tickers JSON...")
company_lookups = load_company_tickers_json("company_tickers.json")

if not company_lookups:
    print("Warning: Could not load company data.")

# ========================================================================
# PHASE 0: LOAD EXISTING SENTIMENT DATA
# ========================================================================

print("\n" + "="*80)
print("PHASE 0: CHECKING FOR EXISTING SENTIMENT DATA")
print("="*80)

daily_sentiments = load_existing_sentiment_data("stock_analysis_complete.xlsx")

# ========================================================================
# PREPARE EXCEL WORKBOOK
# ========================================================================

wb = Workbook()
sheet = wb.active
sheet.title = "News Data"
sheet.append(["Company", "Ticker", "Date", "Headline", "URL", "Summary", "Sentiment Score"])

daily_sheet = wb.create_sheet("Daily Sentiment")
daily_sheet.append(["Company", "Ticker", "Date", "Average Sentiment", "Article Count"])

alpha_sheet = wb.create_sheet("Predictive Alphas")
alpha_sheet.append(["Company", "Ticker", "Related Companies", "Generated Alphas"])

model_results_sheet = wb.create_sheet("Model Results")
model_results_sheet.append(["Company", "Ticker", "MSE", "RMSE", "MAE", "MAPE", "Directional Accuracy"])

start_date = datetime(2023, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
end_date   = datetime(2025, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

mentions_by_source = defaultdict(Counter)
validated_mentions = defaultdict(Counter)
stock_dataframes = {}
alpha_texts = {}
all_related_companies = {}

# ========================================================================
# PHASE 1: FETCH NEWS FOR PRIMARY COMPANIES (WITH CACHE CHECK)
# ========================================================================

print("\n" + "="*80)
print("PHASE 1: FETCHING NEWS FOR PRIMARY COMPANIES (CHECKING CACHE)")
print("="*80)

for ticker, name in companies.items():
    # CHECK IF WE ALREADY HAVE DATA
    if not should_fetch_sentiment(ticker, daily_sentiments, min_days=30):
        print(f"\n✓ Skipping {name} ({ticker}) - already have sentiment data")
        print(f"   No new data will be collected for this ticker")
        continue

    print(f"\nFetching news for {name} ({ticker})...")
    news_articles = fetch_news(ticker, start_date, end_date, batch_size=1000, sleep_time=12)

    for item in news_articles:
        text = f"{item.title} {item.summary if hasattr(item, 'summary') else ''}"
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        article_date = item.published_utc.split('T')[0] if 'T' in item.published_utc else item.published_utc[:10]
        daily_sentiments[ticker][article_date].append(compound_score)

        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                if company_lookups:
                    is_valid, mention_ticker, official_name = validate_company_exists(ent.text, company_lookups)
                    if is_valid:
                        validated_mentions[ticker][f"{official_name} ({mention_ticker})"] += 1

        row = [name, ticker, item.published_utc, item.title, item.article_url,
               item.summary if hasattr(item, "summary") else "", compound_score]
        sheet.append(row)

    print(f"Waiting 65 seconds before next company...")
    time.sleep(65)

# ========================================================================
# PHASE 1.5: IDENTIFY AND FETCH RELATED COMPANY SENTIMENT (WITH CACHE)
# ========================================================================

print("\n" + "="*80)
print("PHASE 1.5: IDENTIFYING AND FETCHING RELATED COMPANY SENTIMENT")
print("="*80)

all_related_tickers = set()

# First, identify all related companies
for ticker, name in companies.items():
    print(f"\nIdentifying related companies for {name} ({ticker})...")

    related_companies = []
    for company_str, count in validated_mentions[ticker].most_common(20):
        if len(related_companies) >= 5:
            break

        match = re.search(r'\(([A-Z]+)\)', company_str)
        if not match:
            continue

        related_ticker = match.group(1)

        if related_ticker == ticker:
            continue

        related_name = company_str.split('(')[0].strip()

        if is_same_company(ticker, related_ticker, name, related_name):
            continue

        is_valid, reason = validate_ticker_quality(related_ticker, min_market_cap=5e9)
        if not is_valid:
            print(f"    ✗ Skipped {related_ticker}: {reason}")
            continue

        related_companies.append(related_ticker)
        all_related_tickers.add(related_ticker)
        print(f"    ✓ Added {related_ticker} ({related_name}): {count} mentions")

    all_related_companies[ticker] = related_companies
    print(f"  Found {len(related_companies)} related companies")

# Now fetch sentiment for ALL unique related companies (WITH CACHE CHECK)
print(f"\n{'='*80}")
print(f"FETCHING SENTIMENT FOR {len(all_related_tickers)} UNIQUE RELATED COMPANIES (CHECKING CACHE)")
print(f"{'='*80}")

for related_ticker in all_related_tickers:
    # CHECK IF WE ALREADY HAVE DATA
    if not should_fetch_sentiment(related_ticker, daily_sentiments, min_days=30):
        continue

    print(f"\nFetching sentiment for {related_ticker}...")
    fetch_sentiment_for_ticker(related_ticker, start_date, end_date, daily_sentiments)

    print(f"Waiting 15 seconds before next company...")
    time.sleep(15)

# ========================================================================
# PHASE 2: POPULATE SENTIMENT SHEET
# ========================================================================

print("\n" + "="*80)
print("PHASE 2: CREATING DAILY SENTIMENT AGGREGATIONS")
print("="*80)

for ticker in sorted(daily_sentiments.keys()):
    company_name = companies.get(ticker, ticker)
    for date in sorted(daily_sentiments[ticker].keys()):
        scores = daily_sentiments[ticker][date]
        avg_sentiment = sum(scores) / len(scores)
        article_count = len(scores)
        daily_sheet.append([company_name, ticker, date, avg_sentiment, article_count])

# ========================================================================
# PHASE 3: STOCK DATA & ALPHA GENERATION
# ========================================================================

print("\n" + "="*80)
print("PHASE 3: FETCHING STOCK DATA AND GENERATING ALPHAS")
print("="*80)

for ticker, name in companies.items():
    print(f"\nProcessing {name} ({ticker})...")

    stock_df = fetch_stock_data(ticker, start_date, end_date)

    if not stock_df.empty:
        stock_dataframes[ticker] = stock_df

        related_companies = all_related_companies.get(ticker, [])

        print(f"  Using {len(related_companies)} related companies: {', '.join(related_companies)}")

        comprehensive_df = prepare_dataframe_for_alpha(
            ticker,
            stock_df,
            daily_sentiments,
            related_companies
        )

        if comprehensive_df is not None and not comprehensive_df.empty:
            sentiment_cols = [col for col in comprehensive_df.columns if 'Sentiment' in col]
            print(f"  Available sentiment columns: {', '.join(sentiment_cols)}")

            sample_df = comprehensive_df.tail(10).fillna(0)
            df_json = sample_df.to_json(orient='records', indent=2)

            print(f"  Generating predictive alphas for {ticker}...")
            alphas = generate_predictive_alphas(ticker, name, df_json, related_companies)

            if alphas:
                alpha_texts[ticker] = alphas
                alpha_sheet.append([name, ticker, ", ".join(related_companies) if related_companies else "None", alphas])

                print(f"\n{'='*80}")
                print(f"PREDICTIVE ALPHAS FOR {name} ({ticker})")
                print(f"Related Companies: {', '.join(related_companies) if related_companies else 'None'}")
                print(f"{'='*80}")
                print(alphas)
                print(f"{'='*80}\n")

        time.sleep(2)

# ========================================================================
# PHASE 4: PYTORCH LSTM MODEL TRAINING
# ========================================================================

print("\n" + "="*80)
print("PHASE 4: TRAINING PYTORCH LSTM MODELS")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

trained_models = {}

for ticker, name in companies.items():
    if ticker not in stock_dataframes or ticker not in alpha_texts:
        print(f"\nSkipping {ticker}: Missing data or alphas")
        continue

    print(f"\n{'='*80}")
    print(f"TRAINING PYTORCH LSTM MODEL FOR {name} ({ticker})")
    print(f"{'='*80}")

    stock_df = stock_dataframes[ticker]
    related_companies = all_related_companies.get(ticker, [])

    comprehensive_df = prepare_dataframe_for_alpha(
        ticker,
        stock_df,
        daily_sentiments,
        related_companies
    )

    if comprehensive_df is None or comprehensive_df.empty:
        print(f"Failed to prepare data for {ticker}")
        continue

    try:
        alpha_text = alpha_texts.get(ticker, None)

        if not alpha_text:
            print(f"  No LLM alphas, using simple alphas...")
            alpha_text = generate_simple_alphas(ticker)

        model, metrics, scalers = train_stock_predictor(
            ticker=ticker,
            company_name=name,
            comprehensive_df=comprehensive_df,
            alpha_text=alpha_text,
            window_size=5,
            num_epochs=50,
            device=device
        )

        # Fallback if training failed
        if model is None:
            print(f"  ⚠️  Training failed, retrying with simple alphas...")
            alpha_text = generate_simple_alphas(ticker)

            model, metrics, scalers = train_stock_predictor(
                ticker=ticker,
                company_name=name,
                comprehensive_df=comprehensive_df,
                alpha_text=alpha_text,
                window_size=5,
                num_epochs=50,
                device=device
            )

        if model is not None:
            trained_models[ticker] = {'model': model, 'metrics': metrics, 'scalers': scalers}
            model_results_sheet.append([
                name, ticker, metrics['MSE'], metrics['RMSE'],
                metrics['MAE'], metrics['MAPE'], metrics['Directional Accuracy']
            ])
            torch.save(model.state_dict(), f'{ticker}_model.pth')
            print(f"\n✓ Model saved as {ticker}_model.pth")

    except Exception as e:
        print(f"\n✗ Error training model for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        continue

    print(f"\nWaiting 5 seconds before next model...")
    time.sleep(5)

# ========================================================================
# SAVE RESULTS
# ========================================================================

wb.save("stock_analysis_complete.xlsx")
print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nResults saved to:")
print(f"  - stock_analysis_complete.xlsx")
print(f"  - <TICKER>_model.pth")
print(f"  - <TICKER>_predictions.png")
print(f"  - <TICKER>_training_history.png")

print(f"\nModels trained: {len(trained_models)}/{len(companies)}")
for ticker in trained_models:
    metrics = trained_models[ticker]['metrics']
    print(f"  {ticker}: RMSE={metrics['RMSE']:.2f}, Directional Accuracy={metrics['Directional Accuracy']:.2f}%")

print(f"\nSentiment data collected for:")
print(f"  - {len(companies)} primary companies")
print(f"  - {len(all_related_tickers)} related companies")
print(f"  - Total: {len(daily_sentiments)} unique tickers")