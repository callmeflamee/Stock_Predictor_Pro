import asyncio
import pandas as pd
from twscrape import API
import logging
import os
from datetime import date, timedelta
import random
import configparser
import sys
import time

# Use a basic logger for cleaner output
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

async def scrape_tweets(api: API, stock: str, limit: int, since: str, until: str) -> list:
    """
    Scrapes tweets for a given stock, with a random initial delay.
    """
    initial_delay = random.uniform(1, 5)
    logging.info(f"Scraper for ${stock} starting after a {initial_delay:.2f} second delay...")
    await asyncio.sleep(initial_delay)

    tweets_list = []
    search_query = f"({stock} OR ${stock}) lang:en since:{since} until:{until} -is:retweet"
    try:
        # Use api.search which is an async generator
        async for tweet in api.search(search_query, limit=limit):
            tweets_list.append([stock, tweet.date, tweet.rawContent])
        logging.info(f"Finished scraping for {stock}, found {len(tweets_list)} tweets.")
    except Exception as e:
        logging.error(f"An error occurred while scraping for ${stock}: {e}")
    return tweets_list

async def main_scrape():
    """
    Main function to run the standalone tweet scraping process with robust initialization.
    """
    print("\n--- Standalone Tweet Scraper ---")
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    data_config = config['Data'] if 'Data' in config else {}
    stocks_in_config = data_config.get('stocks', '').split(',')
    raw_tweet_path = data_config.get('raw_tweet_data_path', 'data/raw/tweet_data.csv')
    tweets_per_stock = int(data_config.get('tweets_per_stock', 2000))
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')

    # --- NEW: More robust logic for determining the start date ---
    # Default to a 7-day lookback if no data exists
    tweet_since_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    
    if os.path.exists(raw_tweet_path):
        try:
            final_tweet_df = pd.read_csv(raw_tweet_path)
            if not final_tweet_df.empty:
                last_tweet_date = pd.to_datetime(final_tweet_df['date'], format='mixed').max().date()
                if last_tweet_date >= today:
                    print("Tweet data is already up to date. Exiting.")
                    return
                # --- CRITICAL FIX: Create a 3-day overlapping window ---
                # This makes the scrape resilient to days with low tweet volume.
                tweet_since_date = (last_tweet_date - timedelta(days=2)).strftime('%Y-%m-%d')
        except (pd.errors.EmptyDataError, KeyError):
            final_tweet_df = pd.DataFrame(columns=["stock", "date", "text"])
    else:
        final_tweet_df = pd.DataFrame(columns=["stock", "date", "text"])

    # --- SEQUENTIAL INITIALIZATION: Setup DB and Accounts FIRST ---
    print(f"\nInitializing scraper... Scraping new tweets from {tweet_since_date} onwards...")
    db_path = "twscrape.db"
    api = API(db_path)
    
    # --- IMPORTANT: Re-add your accounts ONE TIME if the DB is fresh or corrupted ---
    # await api.pool.add_account("user1", "pass1", "email1", "email_pass1")
    
    await api.pool.login_all()
    
    active_accounts = [acc for acc in await api.pool.get_all() if acc.active]
    
    if not active_accounts:
        logging.error("No active Twitter accounts available. Please add accounts and try again. Exiting.")
        return

    logging.info(f"Successfully logged in with {len(active_accounts)} active account(s).")

    # --- CREATE AND RUN SCRAPING TASKS ---
    print("\nStarting sequential scraping for all stocks...")
    
    all_new_tweets = []
    for stock in stocks_in_config:
        stock_tweets = await scrape_tweets(api, stock, tweets_per_stock, tweet_since_date, today_str)
        all_new_tweets.extend(stock_tweets)
        
        inter_stock_delay = random.uniform(15, 45)
        logging.info(f"Pausing for {inter_stock_delay:.2f} seconds before next stock...")
        time.sleep(inter_stock_delay)

    if all_new_tweets:
        print(f"\nFound {len(all_new_tweets)} new tweets in total. Appending to dataset.")
        new_tweet_df = pd.DataFrame(all_new_tweets, columns=["stock", "date", "text"])
        
        if not final_tweet_df.empty:
            final_tweet_df = pd.concat([final_tweet_df, new_tweet_df], ignore_index=True)
        else:
            final_tweet_df = new_tweet_df
            
        final_tweet_df.drop_duplicates(subset=['date', 'text'], keep='last', inplace=True)
        final_tweet_df.sort_values(by=['stock', 'date'], inplace=True)
    else:
        print("\nNo new tweets were scraped in this run.")
    
    os.makedirs(os.path.dirname(raw_tweet_path), exist_ok=True)
    final_tweet_df.to_csv(raw_tweet_path, index=False)
    print(f"Raw tweet data file is up to date at {raw_tweet_path}")

if __name__ == "__main__":
    asyncio.run(main_scrape())

