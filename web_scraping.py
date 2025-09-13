import finnhub
import secret_key
from openpyxl import Workbook
from datetime import datetime, timezone
import spacy

from collections import Counter

finnhub_client = finnhub.Client(api_key=secret_key.API_KEY)

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

for ticker, name in companies.items():
    print(f"Fetching news for {name} ({ticker})...")
    news = finnhub_client.company_news(
        ticker, 
        _from="2025-01-01", 
        to="2025-09-01"
    )

    for item in news: 
        date_str = datetime.fromtimestamp(item["datetime"], tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        row = [
            name,
            ticker,
            date_str,
            item.get("headline", ""),
            item.get("url", ""),
            item.get("summary", "")
        ]
        sheet.append(row)

wb.save("news.xlsx")
print("News saved to news.xlsx")


company_mentions = Counter()

for row in sheet.iter_rows(min_row=2, values_only=True):  
    headline = row[3]  
    summary = row[5]  
    text = f"{headline} {summary}"
    
    doc = nlp(text)
    
    for ent in doc.ents:
        if ent.label_ == "ORG": 
            company_mentions[ent.text] += 1

print(company_mentions)


