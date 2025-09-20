import asyncio
import pandas as pd
from twscrape import API
from tqdm.asyncio import tqdm
import logging
import os
import sys
from datetime import date, timedelta
import random
import configparser

logging.basicConfig(level=logging.WARNING)

async def scrape_tweets(api: API, stock: str, limit: int, since: str, until: str) -> list:
    """
    Scrapes tweets for a given stock, with a random initial delay.
    """
    initial_delay = random.uniform(1, 5)
    print(f"Scraper for ${stock} starting after a {initial_delay:.2f} second delay...")
    await asyncio.sleep(initial_delay)

    tweets_list = []
    search_query = f"({stock} OR ${stock}) lang:en since:{since} until:{until} -is:retweet"
    try:
        # The tqdm progress bar is disabled here for a cleaner background process log
        async for tweet in api.search(search_query, limit=limit):
            tweets_list.append([stock, tweet.date, tweet.rawContent])
        print(f"Finished scraping for {stock}.")
    except Exception as e:
        print(f"An error occurred while scraping for ${stock}: {e}")
    return tweets_list

async def main_scrape():
    """
    Main function to run the standalone tweet scraping process.
    """
    print("--- Standalone Tweet Scraper ---")
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    data_config = config['Data'] if 'Data' in config else {}
    stocks_in_config = data_config.get('stocks', 'AAPL,NVDA').split(',')
    raw_tweet_path = data_config.get('raw_tweet_data_path', 'data/raw/tweet_data.csv')
    tweets_per_stock = int(data_config.get('tweets_per_stock', 2000))
    today = date.today()
    today_str = today.strftime('%Y-%m-%d')

    should_fetch_tweets = True
    tweet_since_date = (today - timedelta(days=7)).strftime('%Y-%m-%d')
    
    if os.path.exists(raw_tweet_path):
        final_tweet_df = pd.read_csv(raw_tweet_path)
        if not final_tweet_df.empty:
            last_tweet_date = pd.to_datetime(final_tweet_df['date'], format='mixed').max().date()
            if last_tweet_date >= today:
                print("Tweet data is already up to date. Exiting.")
                should_fetch_tweets = False
            else:
                tweet_since_date = (last_tweet_date).strftime('%Y-%m-%d')
    else:
        final_tweet_df = pd.DataFrame(columns=["stock", "date", "text"])

    if should_fetch_tweets:
        print(f"Scraping new tweets from {tweet_since_date} onwards...")
        db_path = "twscrape.db"
        api = API(db_path)
        
        # --- IMPORTANT ---
        # Uncomment and add your accounts ONCE to populate the database.
        # Then, comment them out again for all future runs.
        # await api.pool.add_account("user1", "pass1", "email1", "email_pass1")
        # await api.pool.add_account("user2", "pass2", "email2", "email_pass2")
        
        await api.pool.login_all()
        
        all_accounts = await api.pool.get_all()
        active_accounts = [acc for acc in all_accounts if acc.active]
        
        if not active_accounts:
            print("Warning: No active Twitter accounts available. Exiting.")
            return

        print(f"Using {len(active_accounts)} active account(s) for scraping.")
        tasks = [scrape_tweets(api, stock, tweets_per_stock, tweet_since_date, today_str) for stock in stocks_in_config]
        results = await asyncio.gather(*tasks)
        new_tweets = [tweet for stock_tweets in results for tweet in stock_tweets]
        
        if new_tweets:
            print(f"Found {len(new_tweets)} new tweets. Appending to dataset.")
            new_tweet_df = pd.DataFrame(new_tweets, columns=["stock", "date", "text"])
            final_tweet_df = pd.concat([final_tweet_df, new_tweet_df])
            final_tweet_df.drop_duplicates(subset=['date', 'text'], keep='last', inplace=True)
            final_tweet_df.sort_values(by=['stock', 'date'], inplace=True)
        else:
            print("No new tweets were scraped in this run.")
        
        os.makedirs(os.path.dirname(raw_tweet_path), exist_ok=True)
        final_tweet_df.to_csv(raw_tweet_path, index=False)
        print(f"Raw tweet data file is up to date at {raw_tweet_path}")

if __name__ == "__main__":
    asyncio.run(main_scrape())
