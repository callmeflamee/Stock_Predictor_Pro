import asyncio
import pandas as pd
from datetime import date, timedelta, datetime
import os
import sys
import requests
import time
import random

# Import the news fetching function from its dedicated module
from news_fetcher import fetch_news_and_sentiment
from twscrape import API
from tqdm.asyncio import tqdm

logging.basicConfig(level=logging.WARNING)

# --- TWITTER SCRAPING LOGIC (No changes needed) ---
async def scrape_tweets(api: API, stock: str, limit: int, since: str, until: str) -> list:
    # This function is correct
    initial_delay = random.uniform(1, 5)
    print(f"Scraper for ${stock} starting after a {initial_delay:.2f} second delay...")
    await asyncio.sleep(initial_delay)
    tweets_list = []
    search_query = f"({stock} OR ${stock}) lang:en since:{since} until:{until} -is:retweet"
    try:
        async for tweet in api.search(search_query, limit=limit):
            tweets_list.append([stock, tweet.date, tweet.rawContent])
    except Exception as e:
        print(f"An error occurred while scraping for ${stock}: {e}")
    return tweets_list

# --- STOCK DATA FETCHING LOGIC (No changes needed) ---
def fetch_all_stock_data(stocks: list, start_date: str, end_date: str) -> pd.DataFrame:
    # This function is correct
    if not stocks:
        return pd.DataFrame()
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
        return pd.DataFrame()
    return pd.concat(all_data, ignore_index=True)

# --- MAIN DATA COLLECTION ORCHESTRATOR ---
async def collect_data(config):
    data_config = config['Data'] if 'Data' in config else {}
    stocks_in_config = data_config.get('stocks', 'AAPL,NVDA').split(',')
    raw_stock_path = data_config.get('raw_stock_data_path', 'data/raw/stock_data.csv')
    raw_tweet_path = data_config.get('raw_tweet_data_path', 'data/raw/tweet_data.csv')
    raw_news_path = data_config.get('raw_news_data_path', 'data/raw/news_data.csv')
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')
    default_start_date = data_config.get('start_date', '2023-01-01')

    # --- 1. STOCK DATA (Intelligent, Per-Stock Incremental Updates) ---
    new_stocks_to_fetch = []
    incremental_stocks_to_fetch = []
    final_stock_df = pd.DataFrame()

    if os.path.exists(raw_stock_path):
        existing_stock_df = pd.read_csv(raw_stock_path)
        final_stock_df = existing_stock_df.copy()
        existing_tickers = existing_stock_df['stock'].unique().tolist()
        new_stocks_to_fetch = [s for s in stocks_in_config if s not in existing_tickers]
        
        last_date = pd.to_datetime(existing_stock_df['date'], format='mixed').max().date()
        if last_date < today:
            incremental_stocks_to_fetch = existing_tickers
        else:
            print("Stock data is already up to date.")
    else:
        new_stocks_to_fetch = stocks_in_config

    if new_stocks_to_fetch:
        print(f"\nNew stocks detected: {', '.join(new_stocks_to_fetch)}. Fetching full history...")
        new_stock_history_df = fetch_all_stock_data(new_stocks_to_fetch, default_start_date, today_str)
        final_stock_df = pd.concat([final_stock_df, new_stock_history_df], ignore_index=True)

    if incremental_stocks_to_fetch:
        start_date_incremental = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"\nFetching incremental updates for existing stocks from {start_date_incremental}...")
        incremental_df = fetch_all_stock_data(incremental_stocks_to_fetch, start_date_incremental, today_str)
        final_stock_df = pd.concat([final_stock_df, incremental_df], ignore_index=True)

    if new_stocks_to_fetch or incremental_stocks_to_fetch:
        if not final_stock_df.empty:
            final_stock_df.drop_duplicates(subset=['date', 'stock'], keep='last', inplace=True)
            final_stock_df.sort_values(by=['stock', 'date'], inplace=True)
            os.makedirs(os.path.dirname(raw_stock_path), exist_ok=True)
            final_stock_df.to_csv(raw_stock_path, index=False)
            print(f"\nSaved/updated raw stock data to {raw_stock_path}")
    else:
        print("\nNo new stock data to fetch or save.")

    # --- 2. NEWS DATA (NEW Intelligent, Independent Logic) ---
    should_fetch_news = True
    news_start_date = default_start_date
    if os.path.exists(raw_news_path):
        existing_news_df = pd.read_csv(raw_news_path)
        if not existing_news_df.empty:
            last_news_date = pd.to_datetime(existing_news_df['date']).max().date()
            if last_news_date >= today:
                print("\nNews data is already up to date.")
                should_fetch_news = False
            else:
                news_start_date = (last_news_date + timedelta(days=1)).strftime('%Y-%m-%d')

    if should_fetch_news:
        news_sentiment_df = fetch_news_and_sentiment(stocks_in_config, news_start_date)
        if not news_sentiment_df.empty:
            if os.path.exists(raw_news_path):
                final_news_df = pd.concat([pd.read_csv(raw_news_path), news_sentiment_df])
            else:
                final_news_df = news_sentiment_df
            final_news_df.drop_duplicates(subset=['date', 'stock'], keep='last', inplace=True)
            final_news_df.sort_values(by=['date', 'stock'], inplace=True)
            os.makedirs(os.path.dirname(raw_news_path), exist_ok=True)
            final_news_df.to_csv(raw_news_path, index=False)
            print(f"Saved/updated raw news sentiment data to {raw_news_path}")

    ## --- 3. TWEET DATA (NEW Intelligent, Independent Logic) ---
    ## commenting out tweet scraping for now, handed in tweetscraper.py
    ##should_fetch_tweets = True
    ##tweet_since_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    ##if os.path.exists(raw_tweet_path):
    ##    final_tweet_df = pd.read_csv(raw_tweet_path)
    ##    if not final_tweet_df.empty:
    ##        last_tweet_date = pd.to_datetime(final_tweet_df['date'], format='mixed').max().date()
    ##        if last_tweet_date >= today:
    ##            print("\nTweet data is already up to date.")
    ##            should_fetch_tweets = False
    ##        else:
    ##            tweet_since_date = (last_tweet_date).strftime('%Y-%m-%d')
    ##else:
    ##    final_tweet_df = pd.DataFrame(columns=["stock", "date", "text"])
##
    ##if should_fetch_tweets:
    ##    print(f"\nScraping new tweets from {tweet_since_date} onwards...")
    ##    db_path = "twscrape.db"
    ##    api = API(db_path)
    ##    await api.pool.login_all()
    ##    all_accounts = await api.pool.get_all()
    ##    active_accounts = [acc for acc in all_accounts if acc.active]
    ##    if not active_accounts:
    ##        print("Warning: No active Twitter accounts available. Skipping tweet scraping.")
    ##    else:
    ##        tweets_per_stock = int(data_config.get('tweets_per_stock', 2000))
    ##        tasks = [scrape_tweets(api, stock, tweets_per_stock, tweet_since_date, today_str) for stock in stocks_in_config]
    ##        results = await asyncio.gather(*tasks)
    ##        new_tweets = [tweet for stock_tweets in results for tweet in stock_tweets]
    ##        if new_tweets:
    ##            new_tweet_df = pd.DataFrame(new_tweets, columns=["stock", "date", "text"])
    ##            final_tweet_df = pd.concat([final_tweet_df, new_tweet_df])
    ##            final_tweet_df.drop_duplicates(subset=['date', 'text'], keep='last', inplace=True)
    ##            final_tweet_df.sort_values(by=['stock', 'date'], inplace=True)
    ##    
    ##    os.makedirs(os.path.dirname(raw_tweet_path), exist_ok=True)
    ##    final_tweet_df.to_csv(raw_tweet_path, index=False)
    ##    print(f"Raw tweet data file is up to date at {raw_tweet_path}")

def run_data_collection(config):
    asyncio.run(collect_data(config))

