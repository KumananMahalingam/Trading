import finnhub
import secret_key

finnhub_client = finnhub.Client(api_key=secret_key.API_KEY)

print(finnhub_client.company_news('AAPL', _from="2025-01-01", to="2025-09-01"))




