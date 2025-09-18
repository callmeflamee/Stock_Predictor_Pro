import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def process_data(config):
    raw_stock_path = config['Data']['raw_stock_data_path']
    raw_tweet_path = config['Data']['raw_tweet_data_path']
    processed_data_path = config['Data']['processed_data_path']

    try:
        stock_df = pd.read_csv(raw_stock_path)
    except FileNotFoundError:
        print("Error: Raw stock data not found. Please run data collection first.")
        return

    stock_df.columns = stock_df.columns.str.lower()
    stock_df.rename(columns={'datetime': 'date'}, inplace=True, errors='ignore')
    stock_df['date'] = pd.to_datetime(stock_df['date'], format='mixed').dt.date

    try:
        tweet_df = pd.read_csv(raw_tweet_path)
        if not tweet_df.empty:
            tweet_df.columns = tweet_df.columns.str.lower()
            tweet_df.rename(columns={'datetime': 'date'}, inplace=True, errors='ignore')
            tweet_df['date'] = pd.to_datetime(tweet_df['date'], format='mixed').dt.date

            analyzer = SentimentIntensityAnalyzer()
            tweet_df['sentiment'] = tweet_df['text'].apply(lambda text: analyzer.polarity_scores(str(text))['compound'])
            
            daily_sentiment = tweet_df.groupby(['stock', 'date'])['sentiment'].mean().reset_index()
            
            merged_df = pd.merge(stock_df, daily_sentiment, on=['stock', 'date'], how='left')
            
            # --- FIX: Changed from inplace=True to direct assignment to resolve FutureWarning ---
            merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
        else:
            print("Tweet data file is empty. Proceeding without sentiment scores.")
            merged_df = stock_df
            merged_df['sentiment'] = 0
            
    except FileNotFoundError:
        print("Tweet data not found. Proceeding without sentiment scores.")
        merged_df = stock_df
        merged_df['sentiment'] = 0

    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    merged_df.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

def run_data_processing(config):
    process_data(config)

