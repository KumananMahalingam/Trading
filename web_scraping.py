import time
import secret_key
from openpyxl import Workbook
from datetime import datetime, timezone
import spacy
from polygon import RESTClient

from collections import defaultdict, Counter

client = RESTClient(secret_key.API_KEY)

nlp = spacy.load("en_core_web_sm")

companies = {
    "AAPL": "Apple",
    "HSBC": "HSBC",
    "PEP": "Pepsi",
    "TM": "Toyota",
    "TCEHY": "Tencent"
}

wb = Workbook()
sheet = wb.active
sheet.title = "News Data"
sheet.append(["Company", "Ticker", "Date", "Headline", "URL", "Summary"])

def fetch_news(ticker, start, end, limit=100, max_pages=5, sleep_time=65):
    """
        Fetch news for a ticker with pagination & backoff handling
    """
    results = []
    try:
        articles = client.list_ticker_news(
        ticker=ticker,
        published_utc_gte=datetime(2023, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        published_utc_lte=datetime(2025, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        limit=1000
        )

        page_count = 0
        for item in articles:
            results.append(item)
            if len(results) % limit == 0:
                page_count += 1
                if page_count >= max_pages:
                    print(f" Stopping early: reacher {max_pages} pages for {ticker}")
                    break
        
        time.sleep(sleep_time)

    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")

    return results

start = datetime(2023, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
end   = datetime(2025, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

news_articles = []

for ticker, name in companies.items():
    print(f"Fetching news for {name} ({ticker})...")
    news_articles = fetch_news(ticker, start, end)

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

wb.save("news.xlsx")
print("News saved to news.xlsx")


mentions_by_source = defaultdict(Counter)

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

for ticker, counter in mentions_by_source.items():
    print(f"\nNews for {ticker}:")
    for org, count in counter.most_common(30):  
        print(f"  {org}: {count} mentions")
