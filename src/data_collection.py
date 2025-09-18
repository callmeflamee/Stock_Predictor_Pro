import asyncio
import pandas as pd
import yfinance as yf
from twscrape import API
from tqdm.asyncio import tqdm
import logging
import os
import sys
from datetime import date, timedelta, datetime
import requests
import time

logging.basicConfig(level=logging.WARNING)

async def scrape_tweets(api: API, stock: str, limit: int, since: str, until: str) -> list:
    # This function is working correctly and remains unchanged.
    tweets_list = []
    search_query = f"({stock} OR ${stock}) lang:en since:{since} until:{until} -is:retweet"
    try:
        async for tweet in tqdm(api.search(search_query, limit=limit), total=limit, desc=f"Scraping ${stock} tweets", file=sys.stdout):
            tweets_list.append([stock, tweet.date, tweet.rawContent])
    except Exception as e:
        print(f"An error occurred while scraping for ${stock}: {e}")
    return tweets_list

def fetch_all_stock_data(stocks: list, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock data by making a direct and robust API call to Yahoo Finance,
    bypassing the yf.download() function to resolve persistent local network/SSL issues.
    """
    print(f"Fetching stock data for {', '.join(stocks)} from {start_date} to {end_date}...")
    
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

    # Convert date strings to integer timestamps required by the API
    start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())

    all_data = []

    for stock in stocks:
        try:
            # Construct the direct API URL
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"
            
            response = session.get(url)
            response.raise_for_status()  # This will raise an exception for HTTP errors (like 404)
            
            data = response.json()
            
            # --- Parse the JSON response ---
            chart_data = data['chart']['result'][0]
            timestamps = chart_data['timestamp']
            ohlc = chart_data['indicators']['quote'][0]

            stock_df = pd.DataFrame({
                'date': pd.to_datetime(timestamps, unit='s').date,
                'open': ohlc['open'],
                'high': ohlc['high'],
                'low': ohlc['low'],
                'close': ohlc['close'],
                'volume': ohlc['volume'],
                'stock': stock
            })
            
            # Remove rows where all OHLC data is null (e.g., non-trading days)
            stock_df.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)
            all_data.append(stock_df)
            print(f"Successfully fetched data for {stock}.")
            time.sleep(0.5) # Be respectful to the API

        except Exception as e:
            print(f"A critical error occurred while fetching data for {stock}: {e}")
            print(f"Skipping {stock}. This could be due to an invalid ticker or a network issue.")
            continue
            
    if not all_data:
        print("Warning: Failed to download data for all tickers.")
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)


async def collect_data(config):
    # This logic remains the same, as it correctly handles incremental updates.
    stocks = config['Data']['stocks'].split(',')
    tweets_per_stock = int(config['Data']['tweets_per_stock'])
    raw_stock_path = config['Data']['raw_stock_data_path']
    raw_tweet_path = config['Data']['raw_tweet_data_path']
    end_date_str = date.today().strftime('%Y-%m-%d')

    start_date_str = config['Data']['start_date']
    if os.path.exists(raw_stock_path):
        existing_stock_df = pd.read_csv(raw_stock_path)
        if not existing_stock_df.empty:
            last_date = pd.to_datetime(existing_stock_df['date'], format='mixed').max()
            start_date_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Existing stock data found. Fetching new data from {start_date_str} onwards.")
    
    new_stock_df = fetch_all_stock_data(stocks, start_date_str, end_date_str)

    if not new_stock_df.empty:
        if os.path.exists(raw_stock_path):
            existing_stock_df = pd.read_csv(raw_stock_path)
            final_stock_df = pd.concat([existing_stock_df, new_stock_df])
        else:
            final_stock_df = new_stock_df
        
        final_stock_df.drop_duplicates(subset=['date', 'stock'], keep='last', inplace=True)
        final_stock_df.sort_values(by=['stock', 'date'], inplace=True)
        os.makedirs(os.path.dirname(raw_stock_path), exist_ok=True)
        final_stock_df.to_csv(raw_stock_path, index=False)
        print(f"Saved/updated raw stock data to {raw_stock_path}")
    else:
        print("No new stock data fetched.")

    since_date_str = (date.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    if os.path.exists(raw_tweet_path):
        existing_tweet_df = pd.read_csv(raw_tweet_path)
        if not existing_tweet_df.empty:
            last_tweet_date = pd.to_datetime(existing_tweet_df['date'], format='mixed').max()
            since_date_str = (last_tweet_date).strftime('%Y-%m-%d')
    
    print(f"Scraping new tweets from {since_date_str} onwards...")

    db_path = "twscrape.db"
    api = API(db_path)
    
    # await api.pool.add_account("user1", "pass1", "email1@example.com", "email_pass1")
    await api.pool.login_all()
    
    if not api.pool:
        print("Warning: No Twitter accounts configured. Skipping tweet scraping.")
        if not os.path.exists(raw_tweet_path):
             pd.DataFrame(columns=["stock", "date", "text"]).to_csv(raw_tweet_path, index=False)
        return

    tasks = [scrape_tweets(api, stock, tweets_per_stock, since_date_str, end_date_str) for stock in stocks]
    results = await asyncio.gather(*tasks)
    
    new_tweets = [tweet for stock_tweets in results for tweet in stock_tweets]
    if new_tweets:
        new_tweet_df = pd.DataFrame(new_tweets, columns=["stock", "date", "text"])
        if os.path.exists(raw_tweet_path):
            existing_tweet_df = pd.read_csv(raw_tweet_path)
            final_tweet_df = pd.concat([existing_tweet_df, new_tweet_df])
        else:
            final_tweet_df = new_tweet_df
        final_tweet_df.drop_duplicates(subset=['date', 'text'], keep='last', inplace=True)
        final_tweet_df.sort_values(by=['stock', 'date'], inplace=True)
        os.makedirs(os.path.dirname(raw_tweet_path), exist_ok=True)
        final_tweet_df.to_csv(raw_tweet_path, index=False)
        print(f"Saved/updated raw tweet data to {raw_tweet_path}")
    else:
        print("No new tweets were scraped.")

def run_data_collection(config):
    asyncio.run(collect_data(config))

