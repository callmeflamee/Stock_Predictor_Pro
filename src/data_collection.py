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
import random

logging.basicConfig(level=logging.WARNING)

async def scrape_tweets(api: API, stock: str, limit: int, since: str, until: str) -> list:
    """
    Scrapes tweets for a given stock, now with a random initial delay
    to prevent hitting rate limits when running multiple scrapers concurrently.
    """
    # --- NEW: Add a random "jitter" delay to stagger API requests ---
    initial_delay = random.uniform(1, 5)
    print(f"Scraper for ${stock} starting after a {initial_delay:.2f} second delay...")
    await asyncio.sleep(initial_delay)

    tweets_list = []
    search_query = f"({stock} OR ${stock}) lang:en since:{since} until:{until} -is:retweet"
    try:
        async for tweet in tqdm(api.search(search_query, limit=limit), total=limit, desc=f"Scraping ${stock} tweets", file=sys.stdout):
            tweets_list.append([stock, tweet.date, tweet.rawContent])
    except Exception as e:
        print(f"An error occurred while scraping for ${stock}: {e}")
    return tweets_list

def fetch_all_stock_data(stocks: list, start_date: str, end_date: str) -> pd.DataFrame:
    # This function is working correctly and remains unchanged.
    print(f"Fetching stock data for {', '.join(stocks)} from {start_date} to {end_date}...")
    session = requests.Session()
    session.headers['User-Agent'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    all_data = []
    for stock in stocks:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock}?period1={int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())}&period2={int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())}&interval=1d"
            response = session.get(url)
            response.raise_for_status()
            data = response.json()
            chart_data = data['chart']['result'][0]
            timestamps = chart_data['timestamp']
            ohlc = chart_data['indicators']['quote'][0]
            stock_df = pd.DataFrame({'date': pd.to_datetime(timestamps, unit='s').date, 'open': ohlc['open'], 'high': ohlc['high'], 'low': ohlc['low'], 'close': ohlc['close'], 'volume': ohlc['volume'], 'stock': stock})
            stock_df.dropna(subset=['open', 'high', 'low', 'close'], how='all', inplace=True)
            all_data.append(stock_df)
            print(f"Successfully fetched data for {stock}.")
            time.sleep(0.5)
        except Exception as e:
            print(f"A critical error occurred while fetching data for {stock}: {e}")
            continue
    if not all_data:
        print("Warning: Failed to download data for all tickers.")
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)


async def collect_data(config):
    # Stock fetching logic
    stocks = config['Data']['stocks'].split(',')
    tweets_per_stock = int(config['Data']['tweets_per_stock'])
    raw_stock_path = config['Data']['raw_stock_data_path']
    raw_tweet_path = config['Data']['raw_tweet_data_path']
    today_str = date.today().strftime('%Y-%m-%d')
    
    start_date_str = config['Data']['start_date']
    if os.path.exists(raw_stock_path):
        existing_stock_df = pd.read_csv(raw_stock_path)
        if not existing_stock_df.empty:
            last_date = pd.to_datetime(existing_stock_df['date'], format='mixed').max()
            start_date_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Existing stock data found. Will fetch new data from {start_date_str} onwards.")
    
    # --- NEW: Conditional check to see if fetching is necessary ---
    if start_date_str > today_str:
        print("Stock data is already up to date. No new data to fetch.")
    else:
        new_stock_df = fetch_all_stock_data(stocks, start_date_str, today_str)
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

    # Tweet scraping logic
    since_date_str = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d') 
    if os.path.exists(raw_tweet_path):
        final_tweet_df = pd.read_csv(raw_tweet_path)
        if not final_tweet_df.empty:
            last_tweet_date = pd.to_datetime(final_tweet_df['date'], format='mixed').max()
            since_date_str = (last_tweet_date).strftime('%Y-%m-%d')
    else:
        final_tweet_df = pd.DataFrame(columns=["stock", "date", "text"])

    print(f"Scraping new tweets from {since_date_str} onwards...")
    db_path = "twscrape.db"
    api = API(db_path)
    
    await api.pool.add_account("ShaikhBilquees", "Shaikhbilquees7", "shaikhbilquees2@gmail.com", "--manual")
    await api.pool.login_all()
    
    all_accounts = await api.pool.get_all()
    active_accounts = [acc for acc in all_accounts if acc.active]
    
    if not active_accounts:
        print("Warning: No active Twitter accounts available. Skipping tweet scraping.")
        os.makedirs(os.path.dirname(raw_tweet_path), exist_ok=True)
        final_tweet_df.to_csv(raw_tweet_path, index=False)
        return

    tasks = [scrape_tweets(api, stock, tweets_per_stock, since_date_str, today_str) for stock in stocks]
    results = await asyncio.gather(*tasks)
    
    new_tweets = [tweet for stock_tweets in results for tweet in stock_tweets]
    
    if new_tweets:
        print(f"Found {len(new_tweets)} new tweets. Appending to dataset.")
        new_tweet_df = pd.DataFrame(new_tweets, columns=["stock", "date", "text"])
        final_tweet_df = pd.concat([final_tweet_df, new_tweet_df])
        
        final_tweet_df.drop_duplicates(subset=['date', 'text'], keep='last', inplace=True)
        final_tweet_df.sort_values(by=['stock', 'date'], inplace=True)
    else:
        print("No new tweets were scraped.")

    os.makedirs(os.path.dirname(raw_tweet_path), exist_ok=True)
    final_tweet_df.to_csv(raw_tweet_path, index=False)
    print(f"Raw tweet data file is up to date at {raw_tweet_path}")


def run_data_collection(config):
    asyncio.run(collect_data(config))

