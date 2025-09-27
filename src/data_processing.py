import pandas as pd
import os
import numpy as np
from transformers import pipeline, logging

# Suppress verbose logging from the transformers package
logging.set_verbosity_error()

# Initialize the FinBERT sentiment analysis pipeline once
try:
    sentiment_pipeline = pipeline("sentiment-analysis", model=r"C:\stock_predictor\finbert_model")
except Exception as e:
    print(f"Error initializing FinBERT pipeline: {e}. Tweet sentiment will use fallback.")
    sentiment_pipeline = None

def calculate_rsi(close_series, window=14):
    """Calculates the Relative Strength Index (RSI) for a given price series."""
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Use exponential moving average for smoother RSI
    avg_gain = gain.ewm(com=window , min_periods=1).mean()
    avg_loss = loss.ewm(com=window , min_periods=1).mean()
    
    rs = avg_gain / avg_loss
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

    try:
        tweet_df = pd.read_csv(raw_tweet_path)
        tweet_df['date'] = pd.to_datetime(tweet_df['date'], format='mixed', errors='coerce').dt.date
        if sentiment_pipeline:
            print("Analyzing tweet sentiment with FinBERT...")
            def get_finbert_sentiment(text):
                try:
                    res = sentiment_pipeline(str(text))[0]
                    score = res['score']
                    return score if res['label'] == 'positive' else -score if res['label'] == 'negative' else 0
                except:
                    return 0
            tweet_df['tweet_sentiment'] = tweet_df['text'].apply(get_finbert_sentiment)
        else:
            print("Warning: FinBERT model not loaded. Tweet sentiment analysis skipped.")
            tweet_df['tweet_sentiment'] = 0.0
        
        daily_tweet_sentiment = tweet_df.groupby(['date', 'stock'])['tweet_sentiment'].mean().reset_index()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Warning: Tweet data not found or empty. Proceeding without tweet sentiment.")
        daily_tweet_sentiment = pd.DataFrame()

    try:
        daily_news_sentiment = pd.read_csv(raw_news_path)
        daily_news_sentiment['date'] = pd.to_datetime(daily_news_sentiment['date'], format='mixed', errors='coerce').dt.date
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Warning: News data not found or empty. Proceeding without news sentiment.")
        daily_news_sentiment = pd.DataFrame()

    merged_df = stock_df
    if not daily_tweet_sentiment.empty:
        merged_df = pd.merge(merged_df, daily_tweet_sentiment, on=['date', 'stock'], how='left')
    if not daily_news_sentiment.empty:
        merged_df = pd.merge(merged_df, daily_news_sentiment[['date', 'stock', 'news_sentiment']], on=['date', 'stock'], how='left')

    for col in ['tweet_sentiment', 'news_sentiment']:
        if col in merged_df.columns:
            merged_df[col] = merged_df.groupby('stock')[col].transform(lambda x: x.ffill())
            merged_df[col].fillna(0, inplace=True)
        else:
            merged_df[col] = 0

    weight_news = 0.6
    weight_tweet = 0.4

    merged_df['sentiment'] = np.where(
        (merged_df['news_sentiment'] != 0) & (merged_df['tweet_sentiment'] != 0),
        merged_df['news_sentiment'] * weight_news + merged_df['tweet_sentiment'] * weight_tweet,
        np.where(
            merged_df['news_sentiment'] != 0,
            merged_df['news_sentiment'],
            merged_df['tweet_sentiment']
        )
    )
    
    merged_df['sentiment'] = merged_df.groupby('stock')['sentiment'].transform(
        lambda x: x.ewm(span=5, adjust=False).mean()
    )
    
    # --- FIX: Add RSI as a new feature ---
    print("Calculating RSI for each stock...")
    merged_df['rsi'] = merged_df.groupby('stock')['close'].transform(lambda x: calculate_rsi(x))
    merged_df['rsi'].fillna(50, inplace=True)  # Fill initial NaNs with neutral 50

    merged_df.drop(columns=['tweet_sentiment', 'news_sentiment'], inplace=True, errors='ignore')
    merged_df.sort_values(by=['stock', 'date'], inplace=True)
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    merged_df.to_csv(processed_data_path, index=False)
    print(f"Processed data with sentiment and RSI saved to {processed_data_path}")

def run_data_processing(config):
    process_data(config)

