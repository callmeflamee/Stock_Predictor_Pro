import os
import pandas as pd
from datetime import date, timedelta
from newsapi import NewsApiClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv
import time

load_dotenv()

def fetch_news_and_sentiment(stocks: list, start_date_str: str):
    """
    Fetches news headlines for a list of stocks from a given start date,
    analyzes their sentiment, and returns a structured DataFrame.
    This version uses intelligent batching to handle many stocks and respect API rate limits.
    """
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key or api_key == 'YOUR_NEW_NEWS_API_KEY':
        print("Warning: News API Key not found in .env file. Skipping news fetching.")
        return pd.DataFrame()

    try:
        newsapi = NewsApiClient(api_key=api_key)
        analyzer = SentimentIntensityAnalyzer()
    except Exception as e:
        print(f"Error initializing News API client: {e}")
        return pd.DataFrame()

    all_news_data = []
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    
    max_lookback_date = today - timedelta(days=29)
    effective_start_date = max(pd.to_datetime(start_date_str).date(), max_lookback_date)
    
    print(f"Fetching news headlines from {effective_start_date.strftime('%Y-%m-%d')} to {today_str}...")

    # Batching logic to handle a large number of stocks
    batch_size = 4 
    stock_batches = [stocks[i:i + batch_size] for i in range(0, len(stocks), batch_size)]

    for batch in stock_batches:
        query = " OR ".join(batch)
        print(f"\nFetching articles for batch: {query}")
        
        try:
            all_articles = newsapi.get_everything(
                q=query,
                from_param=effective_start_date.strftime('%Y-%m-%d'),
                to=today_str,
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            # --- NEW: Added informative print statement ---
            num_articles = len(all_articles['articles'])
            print(f"Found {num_articles} articles for this batch.")

            for article in all_articles['articles']:
                sentiment = analyzer.polarity_scores(article['title'])['compound']
                
                article_title = article['title'].lower()
                for stock in batch:
                    if stock.lower() in article_title:
                        all_news_data.append({
                            'date': pd.to_datetime(article['publishedAt']).date(),
                            'stock': stock,
                            'news_sentiment': sentiment
                        })
            
            time.sleep(1) 

        except Exception as e:
            print(f"An error occurred while fetching news for batch '{query}': {e}")
            continue
    
    if not all_news_data:
        print("\nNo relevant news articles were found across all stocks.")
        return pd.DataFrame()

    news_df = pd.DataFrame(all_news_data)
    daily_news_sentiment = news_df.groupby(['date', 'stock'])['news_sentiment'].mean().reset_index()
    
    return daily_news_sentiment

