import time
import secret_key
from openpyxl import Workbook
from datetime import datetime, timezone
import spacy
from polygon import RESTClient
from collections import defaultdict, Counter
import json
import re

client = RESTClient(secret_key.API_KEY)
nlp = spacy.load("en_core_web_sm")

companies = {
    "AAPL": "Apple",
    "JPM": "JPMorgan Chase & Co",
    "PEP": "Pepsi",
    "TM": "Toyota",
    "AMZN": "Amazon"
}

def load_company_tickers_json(file_path="company_tickers.json"):
    """
    Load the SEC company tickers JSON and create lookup dictionaries
    """
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
                ticker_to_info[ticker] = {
                    'name': name,
                    'ticker': ticker,
                    'cik': cik
                }

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
        print(f"Error: {file_path} not found. Please download it first.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None

def generate_name_variations(company_name):
    """
    Generate common variations of company names for better matching
    """
    variations = [company_name]

    suffixes_pattern = r'\s+(Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?|Holdings|Holding|LLC|L\.L\.C\.|plc|PLC)'
    base_name = re.sub(suffixes_pattern, '', company_name, flags=re.IGNORECASE).strip()
    variations.append(base_name)

    variations.extend([
        company_name.replace(',', '').strip(),
        company_name.replace('.', '').strip(),
        base_name.replace(',', '').strip(),
        base_name.replace('.', '').strip()
    ])

    variations.extend([
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
    """
    Check if a company name corresponds to a real publicly traded company
    Returns tuple: (is_valid, ticker, official_name)
    """
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

print("Loading company tickers JSON...")
company_lookups = load_company_tickers_json("company_tickers.json")

if not company_lookups:
    print("Warning: Could not load company data. Continuing with basic analysis...")

wb = Workbook()
sheet = wb.active
sheet.title = "News Data"
sheet.append(["Company", "Ticker", "Date", "Headline", "URL", "Summary"])

start_date = datetime(2023, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
end_date   = datetime(2025, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def fetch_news(ticker, start, end, batch_size=1000, sleep_time=12):
    """
    Fetch news for a ticker with pagination & backoff handling
    """
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

news_articles = []

for ticker, name in companies.items():
    print(f"Fetching news for {name} ({ticker})...")
    news_articles = fetch_news(ticker, start_date, end_date, batch_size=1000, sleep_time=12)

    for item in news_articles:
        row = [
            name,
            ticker,
            item.published_utc,
            item.title,
            item.article_url,
            item.summary if hasattr(item, "summary") else ""
        ]
        sheet.append(row)

    print(f"Waiting 65 seconds before next company...")
    time.sleep(65)

wb.save("news.xlsx")
print("News saved to news.xlsx")

mentions_by_source = defaultdict(Counter)
validated_mentions = defaultdict(Counter)
invalid_mentions = defaultdict(Counter)

print("\nAnalyzing company mentions...")

for row in sheet.iter_rows(min_row=2, values_only=True):
    source_company = row[0]
    source_ticker = row[1]
    headline = row[3]
    summary = row[5]

    text = f"{headline} {summary}"
    doc = nlp(text)

    for ent in doc.ents:
        if ent.label_ == "ORG":
            mentions_by_source[source_ticker][ent.text] += 1

            if company_lookups:
                is_valid, ticker, official_name = validate_company_exists(ent.text, company_lookups)

                if is_valid:
                    validated_mentions[source_ticker][f"{official_name} ({ticker})"] += 1
                else:
                    invalid_mentions[source_ticker][ent.text] += 1

for ticker, counter in mentions_by_source.items():
    print(f"\nNews for {ticker}:")
    for org, count in counter.most_common(30):
        print(f"  {org}: {count} mentions")

if company_lookups:
    print("\n" + "="*80)
    print("VALIDATED PUBLIC COMPANIES ONLY")
    print("="*80)

    for ticker, counter in validated_mentions.items():
        company_name = companies.get(ticker, ticker)
        print(f"\nValidated public companies mentioned in {company_name} ({ticker}) news:")
        for company, count in counter.most_common(15):
            print(f"  ✓ {company}: {count} mentions")

    print("\n" + "="*80)
    print("ENTITIES NOT FOUND IN PUBLIC MARKETS")
    print("="*80)

    for ticker, counter in invalid_mentions.items():
        company_name = companies.get(ticker, ticker)
        print(f"\nNon-public entities mentioned in {company_name} ({ticker}) news:")
        for entity, count in counter.most_common(15):
            print(f"  ✗ {entity}: {count} mentions")