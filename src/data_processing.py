import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def process_data(config):
    # --- FIX: Use a robust method to access the config section ---
    if 'Data' in config:
        data_config = config['Data']
    else:
        print("Warning: [Data] section not found in config.ini. Using default paths.")
        data_config = {} # Use an empty dict to allow .get() with defaults below
        
    raw_stock_path = data_config.get('raw_stock_data_path', 'data/raw/stock_data.csv')
    raw_tweet_path = data_config.get('raw_tweet_data_path', 'data/raw/tweet_data.csv')
    raw_news_path = data_config.get('raw_news_data_path', 'data/raw/news_data.csv')
    processed_data_path = data_config.get('processed_data_path', 'data/processed/processed_data.csv')

    # Load raw stock data
    try:
        stock_df = pd.read_csv(raw_stock_path)
        stock_df['date'] = pd.to_datetime(stock_df['date'], format='mixed', errors='coerce').dt.date
    except FileNotFoundError:
        print("Error: Raw stock data not found. Please run data collection first.")
        return

    # --- 1. Process Tweet Sentiment ---
    try:
        tweet_df = pd.read_csv(raw_tweet_path)
        tweet_df['date'] = pd.to_datetime(tweet_df['date'], format='mixed', errors='coerce').dt.date
        analyzer = SentimentIntensityAnalyzer()
        tweet_df['tweet_sentiment'] = tweet_df['text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
        daily_tweet_sentiment = tweet_df.groupby(['date', 'stock'])['tweet_sentiment'].mean().reset_index()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Warning: Tweet data not found or empty. Proceeding without tweet sentiment.")
        daily_tweet_sentiment = pd.DataFrame(columns=['date', 'stock', 'tweet_sentiment'])

    # --- 2. Load Processed News Sentiment ---
    try:
        daily_news_sentiment = pd.read_csv(raw_news_path)
        daily_news_sentiment['date'] = pd.to_datetime(daily_news_sentiment['date'], format='mixed', errors='coerce').dt.date
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Warning: News data not found or empty. Proceeding without news sentiment.")
        daily_news_sentiment = pd.DataFrame(columns=['date', 'stock', 'news_sentiment'])

    # --- 3. Merge all data sources ---
    merged_df = stock_df

    if not daily_tweet_sentiment.empty:
        merged_df = pd.merge(merged_df, daily_tweet_sentiment, on=['date', 'stock'], how='left')
    else:
        merged_df['tweet_sentiment'] = 0
    
    if not daily_news_sentiment.empty:
        merged_df = pd.merge(merged_df, daily_news_sentiment, on=['date', 'stock'], how='left')
    else:
        merged_df['news_sentiment'] = 0

    # --- 4. Create a combined sentiment score ---
    merged_df['tweet_sentiment'].fillna(0, inplace=True)
    merged_df['news_sentiment'].fillna(0, inplace=True)

    merged_df['sentiment'] = (merged_df['tweet_sentiment'] + merged_df['news_sentiment']) / 2

    merged_df.drop(columns=['tweet_sentiment', 'news_sentiment'], inplace=True)
    
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    merged_df.to_csv(processed_data_path, index=False)
    print(f"Processed data with combined sentiment saved to {processed_data_path}")

def run_data_processing(config):
    process_data(config)

