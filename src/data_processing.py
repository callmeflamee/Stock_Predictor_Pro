import pandas as pd
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch

def calculate_rsi(close_series, window=14):
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    
    # --- FIX: Prevent division by zero for stable RSI ---
    rs = avg_gain / avg_loss.replace(0, 1e-10) # Replace 0 with a very small number
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_data(config):
    data_config = config['Data'] if 'Data' in config else {}
    raw_stock_path = data_config.get('raw_stock_data_path', 'data/raw/stock_data.csv')
    raw_tweet_path = data_config.get('raw_tweet_data_path', 'data/raw/tweet_data.csv')
    raw_news_path = data_config.get('raw_news_data_path', 'data/raw/news_data.csv')
    processed_data_path = data_config.get('processed_data_path', 'data/processed/processed_data.csv')

    try:
        stock_df = pd.read_csv(raw_stock_path)
        stock_df['date'] = pd.to_datetime(stock_df['date'], format='mixed', errors='coerce').dt.date
    except FileNotFoundError:
        print("Error: Raw stock data not found. Please run data collection first.")
        return

    # --- FinBERT Sentiment Analysis for Tweets ---
    daily_tweet_sentiment = pd.DataFrame()
    try:
        tweet_df = pd.read_csv(raw_tweet_path)
        tweet_df['date'] = pd.to_datetime(tweet_df['date'], format='mixed', errors='coerce').dt.date
        
        print("Initializing FinBERT for tweet sentiment analysis...")
        local_model_path = r"C:\stock_predictor\finbert_model"

        if os.path.exists(local_model_path):
             print(f"Loading FinBERT from local path: {local_model_path}")
        else:
            print("Local FinBERT model not found. Falling back to remote.")

        try:
            tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True)
        except Exception:
            print(f"Local load failed. Falling back to remote repo.")
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        if torch.cuda.is_available():
            device = 0
            print("GPU is available for PyTorch. Setting device to 'gpu'.")
        else:
            device = -1
            print("GPU not available for PyTorch. Falling back to 'cpu'.")

        sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        
        print("Analyzing tweet sentiment with FinBERT...")
        tweet_df['text'] = tweet_df['text'].astype(str)
        results = sentiment_pipeline(tweet_df['text'].tolist(), batch_size=8, truncation=True)
        
        tweet_df['tweet_sentiment'] = [
            (r['score'] if r['label'] == 'positive' else -r['score'] if r['label'] == 'negative' else 0)
            for r in results
        ]
        daily_tweet_sentiment = tweet_df.groupby(['date', 'stock'])['tweet_sentiment'].mean().reset_index()

    except Exception as e:
        print(f"Error during FinBERT analysis: {e}. Tweet sentiment will use VADER.")
        try:
            if 'tweet_df' not in locals():
                 tweet_df = pd.read_csv(raw_tweet_path)
                 tweet_df['date'] = pd.to_datetime(tweet_df['date'], format='mixed', errors='coerce').dt.date
            analyzer = SentimentIntensityAnalyzer()
            tweet_df['tweet_sentiment'] = tweet_df['text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
            daily_tweet_sentiment = tweet_df.groupby(['date', 'stock'])['tweet_sentiment'].mean().reset_index()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print("Warning: Tweet data not found or empty.")

    # --- News Sentiment ---
    try:
        daily_news_sentiment = pd.read_csv(raw_news_path)
        daily_news_sentiment['date'] = pd.to_datetime(daily_news_sentiment['date'], format='mixed', errors='coerce').dt.date
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Warning: News data not found or empty.")
        daily_news_sentiment = pd.DataFrame()

    merged_df = stock_df
    if not daily_tweet_sentiment.empty:
        merged_df = pd.merge(merged_df, daily_tweet_sentiment, on=['date', 'stock'], how='left')
    if not daily_news_sentiment.empty:
        merged_df = pd.merge(merged_df, daily_news_sentiment, on=['date', 'stock'], how='left')
    
    merged_df['tweet_sentiment'] = merged_df.get('tweet_sentiment').fillna(method='ffill').fillna(0)
    merged_df['news_sentiment'] = merged_df.get('news_sentiment').fillna(method='ffill').fillna(0)

    weight_news, weight_tweet = 0.6, 0.4
    merged_df['sentiment'] = (merged_df['news_sentiment'] * weight_news + merged_df['tweet_sentiment'] * weight_tweet)
    
    merged_df['sentiment'] = merged_df.groupby('stock')['sentiment'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    merged_df.drop(columns=['tweet_sentiment', 'news_sentiment'], inplace=True, errors='ignore')

    merged_df['rsi'] = merged_df.groupby('stock')['close'].transform(lambda x: calculate_rsi(x))
    
    # --- NEW: CRITICAL DATA SANITIZATION STEP ---
    print("Sanitizing final dataset to remove corrupted data points...")
    # 1. Drop rows with invalid 'close' prices (0, negative, or NaN)
    initial_rows = len(merged_df)
    merged_df.dropna(subset=['close'], inplace=True)
    merged_df = merged_df[merged_df['close'] > 0]
    print(f"Removed {initial_rows - len(merged_df)} rows with invalid close prices.")

    # 2. Replace any NaN/Infinity in features with neutral values
    for col in ['sentiment', 'rsi']:
        # Replace infinite values with NaN
        merged_df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
        # Fill remaining NaNs with a neutral value
        fill_value = 0 if col == 'sentiment' else 50
        merged_df[col].fillna(fill_value, inplace=True)
        print(f"Sanitized '{col}' column.")

    merged_df.sort_values(by=['stock', 'date'], inplace=True)
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    merged_df.to_csv(processed_data_path, index=False)
    print(f"Sanitized and processed data saved to {processed_data_path}")

def run_data_processing(config):
    process_data(config)

