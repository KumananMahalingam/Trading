"""
News data collection from Polygon API
"""
import time
from collections import defaultdict


def fetch_news(client, ticker, start_date, end_date, batch_size=1000, sleep_time=12):
    """
    Fetch news articles for a ticker from Polygon API

    Args:
        client: Polygon REST client
        ticker: Stock ticker symbol
        start_date: Start date (ISO format with Z)
        end_date: End date (ISO format with Z)
        batch_size: Number of articles per batch
        sleep_time: Seconds to wait between batches

    Returns:
        list: News articles
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


def fetch_sentiment_for_ticker(client, sia, ticker, start_date, end_date, daily_sentiments):
    """
    Fetch news and compute sentiment for a specific ticker

    Args:
        client: Polygon REST client
        sia: VADER SentimentIntensityAnalyzer
        ticker: Stock ticker
        start_date: Start date (ISO format)
        end_date: End date (ISO format)
        daily_sentiments: Dictionary to store results
    """
    print(f"  Fetching sentiment data for {ticker}...")

    news_articles = fetch_news(client, ticker, start_date, end_date,
                               batch_size=1000, sleep_time=12)

    if not news_articles:
        print(f"    No news articles found for {ticker}")
        return

    for item in news_articles:
        text = f"{item.title} {item.summary if hasattr(item, 'summary') else ''}"
        sentiment_scores = sia.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        article_date = item.published_utc.split('T')[0] if 'T' in item.published_utc else item.published_utc[:10]
        daily_sentiments[ticker][article_date].append(compound_score)

    print(f"    Processed {len(news_articles)} articles, {len(daily_sentiments[ticker])} days of sentiment")