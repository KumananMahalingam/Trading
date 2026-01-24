"""
SEC filings and earnings data collection
"""
import re
import os
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()


def extract_mda_section(filing_text):
    """
    Extract Management Discussion & Analysis section from SEC filing

    Args:
        filing_text: Full SEC filing text

    Returns:
        str: MD&A section text
    """
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


def fetch_sec_filings(ticker, start_date, end_date):
    """
    Fetch SEC filings (10-K annual reports, 10-Q quarterly reports)
    Analyzes sentiment of Management Discussion & Analysis section

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        dict: {date: sentiment_score}
    """
    print(f"  Fetching SEC filings for {ticker}...")

    filing_sentiments = {}

    try:
        from sec_edgar_downloader import Downloader
    except ImportError:
        print("  sec-edgar-downloader not installed.")
        return filing_sentiments

    try:
        # Download filings
        dl = Downloader("MyCompany", "myemail@example.com")

        # Get 10-K and 10-Q filings
        for filing_type in ['10-K', '10-Q']:
            try:
                dl.get(filing_type, ticker, limit=20)

                # Parse downloaded files
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

                                except Exception:
                                    continue

            except Exception as e:
                print(f"    Error fetching {filing_type}: {e}")

        print(f"    Total: {len(filing_sentiments)} SEC filings analyzed")

    except Exception as e:
        print(f"    Error with SEC data: {e}")

    return filing_sentiments


def fetch_earnings_transcripts(ticker, start_date, end_date):
    """
    Fetch earnings call transcripts and analyze sentiment

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        dict: {date: sentiment_score}
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

            # Use earnings surprise as sentiment proxy
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


def fetch_all_alternative_data(ticker, start_date, end_date):
    """
    Fetch all alternative data sources and combine them

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        tuple: (sec_sentiments_dict, earnings_sentiments_dict)
    """
    print(f"\n{'='*80}")
    print(f"FETCHING ALTERNATIVE DATA FOR {ticker}")
    print(f"{'='*80}")

    # SEC Filings
    sec_sentiments = fetch_sec_filings(ticker, start_date, end_date)

    # Earnings Transcripts
    earnings_sentiments = fetch_earnings_transcripts(ticker, start_date, end_date)

    print(f"\n{'='*80}")
    print(f"ALTERNATIVE DATA SUMMARY FOR {ticker}")
    print(f"{'='*80}")
    print(f"  SEC Filings: {len(sec_sentiments)}")
    print(f"  Earnings Events: {len(earnings_sentiments)}")
    print(f"{'='*80}\n")

    return sec_sentiments, earnings_sentiments