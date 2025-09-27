import os
import pandas as pd
from datetime import date, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
import time
from transformers import pipeline, logging

# Suppress verbose logging from the transformers package
logging.set_verbosity_error()

load_dotenv()

# --- OPTIMIZATION: Initialize the FinBERT sentiment analysis pipeline once ---
# This is more efficient than reloading the model on every call.
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model=r"C:\stock_predictor\finbert_model")
except Exception as e:
    print(f"Error initializing FinBERT pipeline: {e}. News sentiment will not be available.")
    sentiment_pipeline = None

def fetch_news_and_sentiment(stocks: list, start_date_str: str):
    """
    Fetches news headlines, analyzes sentiment using FinBERT, saves titles,
    and returns a structured DataFrame.
    """
    if not sentiment_pipeline:
        return pd.DataFrame()

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key or api_key == 'YOUR_NEW_NEWS_API_KEY':
        print("Warning: News API Key not found in .env file. Skipping news fetching.")
        return pd.DataFrame()

    try:
        newsapi = NewsApiClient(api_key=api_key)
    except Exception as e:
        print(f"Error initializing News API client: {e}")
        return pd.DataFrame()

    all_news_data = []
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    
    max_lookback_date = today - timedelta(days=29)
    effective_start_date = max(pd.to_datetime(start_date_str).date(), max_lookback_date)
    
    print(f"Fetching news headlines from {effective_start_date.strftime('%Y-%m-%d')} to {today_str}...")

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
            
            num_articles = len(all_articles['articles'])
            print(f"Found {num_articles} articles for this batch.")

            for article in all_articles['articles']:
                if not article['title'] or article['title'] == '[Removed]':
                    continue

                # --- OPTIMIZATION: Use FinBERT for more accurate financial sentiment ---
                result = sentiment_pipeline(article['title'])[0]
                label = result['label']
                score = result['score']
                sentiment = score if label == 'positive' else -score if label == 'negative' else 0
                
                article_title_lower = article['title'].lower()
                for stock in batch:
                    if stock.lower() in article_title_lower:
                        all_news_data.append({
                            'date': pd.to_datetime(article['publishedAt']).date(),
                            'stock': stock,
                            'news_sentiment': sentiment,
                            'title': article['title'] 
                        })
            
            time.sleep(1) 

        except Exception as e:
            print(f"An error occurred while fetching news for batch '{query}': {e}")
            continue
    
    if not all_news_data:
        print("\nNo relevant news articles were found across all stocks.")
        return pd.DataFrame()

    news_df = pd.DataFrame(all_news_data)
    
    daily_news_sentiment = news_df.groupby(['date', 'stock']).agg({
        'news_sentiment': 'mean',
        'title': lambda x: ' || '.join(x)
    }).reset_index()
    
    return daily_news_sentiment

