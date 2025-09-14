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

start_date = datetime(2023, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
end_date   = datetime(2025, 9, 1, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def fetch_news(ticker, start, end, batch_size=500, sleep_time=65):
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
    news_articles = fetch_news(ticker, start_date, end_date, batch_size=500, sleep_time=65)

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