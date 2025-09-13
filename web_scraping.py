import finnhub
import secret_key
from openpyxl import Workbook
from datetime import datetime, timezone

finnhub_client = finnhub.Client(api_key=secret_key.API_KEY)

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



