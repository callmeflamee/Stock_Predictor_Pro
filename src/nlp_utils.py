import pandas as pd
import google.generativeai as genai
import os

def generate_dynamic_summary(stock: str, raw_tweet_df: pd.DataFrame, raw_news_df: pd.DataFrame):
    """
    Analyzes recent tweets AND news headlines using the Gemini API to generate a
    high-quality, comprehensive summary.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        return "Google API Key not found in .env file. AI summary is disabled."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return "Failed to initialize the AI model. Please check your API key."

    # --- 1. Process Tweet Data ---
    tweets_context = "No recent tweets found."
    if not raw_tweet_df.empty:
        raw_tweet_df['date'] = pd.to_datetime(raw_tweet_df['date'], errors='coerce').dt.tz_localize(None)
        seven_days_ago = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=7)
        stock_tweets = raw_tweet_df[(raw_tweet_df['stock'] == stock) & (raw_tweet_df['date'] >= seven_days_ago)]
        if not stock_tweets.empty:
            tweets_text = "\n".join("- " + str(text) for text in stock_tweets.nlargest(20, 'date')['text'])
            tweets_context = f"Recent Tweets:\n---\n{tweets_text}\n---"

    # --- 2. Process News Data ---
    news_context = "No recent news headlines found."
    if not raw_news_df.empty and 'title' in raw_news_df.columns:
        raw_news_df['date'] = pd.to_datetime(raw_news_df['date'], errors='coerce').dt.tz_localize(None)
        seven_days_ago = pd.Timestamp.now().tz_localize(None) - pd.Timedelta(days=7)
        stock_news = raw_news_df[(raw_news_df['stock'] == stock) & (raw_news_df['date'] >= seven_days_ago) & (raw_news_df['title'].notna())]
        
        if not stock_news.empty:
            # --- OPTIMIZATION: Correctly parse aggregated titles ---
            # Combine all title strings from the last 7 days.
            all_titles_str = ' || '.join(stock_news['title'])
            # Split the combined string and get unique, non-empty titles.
            unique_titles = list(pd.unique([title.strip() for title in all_titles_str.split('||') if title.strip()]))
            
            # Use the most recent unique titles for the summary.
            news_text = "\n".join("- " + title for title in unique_titles[:15])
            news_context = f"Recent News Headlines:\n---\n{news_text}\n---"

    # --- 3. Construct the Combined Prompt ---
    prompt = f"""
    As a neutral financial analyst, analyze the following recent social media posts and news headlines about ${stock}.
    Provide a concise, well-rounded summary (strictly 2-3 sentences, maximum 4 lines) of the key themes, catalysts, and overall market sentiment.
    Synthesize information from both sources if available. Do not use hashtags or promotional language.

    {tweets_context}

    {news_context}

    Comprehensive Summary:
    """

    safety_settings = {'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE'}
    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)
        if not response.parts:
            block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
            print(f"Warning: Summary generation for {stock} was blocked for reason: {block_reason}")
            return f"AI summary for {stock} could not be generated due to content safety filters."
            
        summary = response.text.strip().replace('*', '')
    except Exception as e:
        print(f"An error occurred during the Gemini API call: {e}")
        summary = f"AI summary generation for {stock} failed."

    return summary
