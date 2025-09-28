import os
import pandas as pd
from datetime import date, timedelta
from newsapi import NewsApiClient
from dotenv import load_dotenv
import time
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
# --- NEW: Import torch to check for GPU availability ---
import torch

load_dotenv()

def fetch_news_and_sentiment(stocks: list, start_date_str: str):
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key or api_key == 'YOUR_NEW_NEWS_API_KEY':
        print("Warning: News API Key not found. Skipping news fetching.")
        return pd.DataFrame()

    try:
        newsapi = NewsApiClient(api_key=api_key)
        print("Initializing FinBERT for news sentiment analysis...")
        local_model_path = r"C:\stock_predictor\finbert_model"

        print(f"Checking local path: {local_model_path}")
        print(f"Path exists: {os.path.exists(local_model_path)}")
        if os.path.exists(local_model_path):
            print(f"Files in directory: {os.listdir(local_model_path)}")
        else:
            print("Local path does not exist—falling back to remote.")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True)
            print("Loaded FinBERT from local path successfully.")
        except Exception as local_err:
            print(f"Local load failed: {local_err}. Falling back to remote repo.")
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        
        # --- UPDATED: More informative GPU check ---
        if torch.cuda.is_available():
            device = 0 # Use the first available GPU
            print("GPU is available for PyTorch. Setting device to 'gpu'.")
        else:
            device = -1 # Use CPU
            print("GPU not available for PyTorch. Falling back to 'cpu'.")
            print("--> To enable GPU, please ensure you have a CUDA-enabled PyTorch version installed.")
            print("--> Visit https://pytorch.org/get-started/locally/ for installation commands.")
            
        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

    except Exception as e:
        print(f"Error initializing FinBERT pipeline during fallback: {e}. News sentiment will not be available.")
        return pd.DataFrame()

    all_news_data = []
    today = date.today()
    
    max_lookback_date = today - timedelta(days=29)
    effective_start_date = max(pd.to_datetime(start_date_str).date(), max_lookback_date)
    
    print(f"Fetching news headlines from {effective_start_date.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}...")

    batch_size = 4 
    stock_batches = [stocks[i:i + batch_size] for i in range(0, len(stocks), batch_size)]

    for batch in stock_batches:
        query = " OR ".join(batch)
        try:
            all_articles = newsapi.get_everything(
                q=query,
                from_param=effective_start_date.strftime('%Y-%m-%d'),
                to=today.strftime('%Y-%m-%d'),
                language='en',
                sort_by='publishedAt',
                page_size=100
            )
            
            print(f"Found {len(all_articles['articles'])} articles for batch: {query}")

            titles_to_process = [article['title'] for article in all_articles['articles'] if article['title']]
            if titles_to_process:
                results = sentiment_pipeline(titles_to_process, truncation=True, batch_size=8)
                
                for i, article in enumerate([a for a in all_articles['articles'] if a['title']]):
                    result = results[i]
                    sentiment = result['score'] if result['label'] == 'positive' else -result['score'] if result['label'] == 'negative' else 0
                    
                    article_title_lower = article['title'].lower()
                    for stock in batch:
                        if stock.lower() in article_title_lower:
                            all_news_data.append({
                                'date': pd.to_datetime(article['publishedAt']).date(),
                                'stock': stock,
                                'title': article['title'],
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
    # Aggregate sentiment and titles
    daily_news_sentiment = news_df.groupby(['date', 'stock']).agg({
        'news_sentiment': 'mean',
        'title': lambda x: list(x)
    }).reset_index()
    
    return daily_news_sentiment

