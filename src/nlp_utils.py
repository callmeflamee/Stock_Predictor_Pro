import pandas as pd
import google.generativeai as genai

def generate_dynamic_summary(stock: str, raw_tweet_df: pd.DataFrame, api_key: str):
    """
    Analyzes recent tweets using the Gemini API to generate a high-quality summary.
    This version includes safety checks and more robust error handling.
    """
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        return "Google API Key not configured in config.ini. AI summary is disabled."
        
    if raw_tweet_df.empty:
        return "No recent tweet data available to generate a summary."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return "Failed to initialize the AI model. Please check your API key."

    # --- Date Handling (Timezone-naive) ---
    raw_tweet_df['date'] = pd.to_datetime(raw_tweet_df['date'], errors='coerce')
    if pd.api.types.is_datetime64_any_dtype(raw_tweet_df['date']) and raw_tweet_df['date'].dt.tz is not None:
        raw_tweet_df['date'] = raw_tweet_df['date'].dt.tz_localize(None)

    now_naive = pd.Timestamp.now().tz_localize(None)
    seven_days_ago = now_naive - pd.Timedelta(days=7)
    
    recent_tweets_df = raw_tweet_df[raw_tweet_df['date'] >= seven_days_ago]
    stock_tweets = recent_tweets_df[recent_tweets_df['stock'] == stock]

    # --- NEW: Minimum Tweet Threshold ---
    # Ensure there's enough data to create a meaningful summary.
    MIN_TWEET_COUNT = 5
    if len(stock_tweets) < MIN_TWEET_COUNT:
        return f"Not enough recent tweets (found {len(stock_tweets)}) for {stock} to generate a reliable summary."

    tweets_text = "\n".join(
        "- " + str(text) for text in stock_tweets.nlargest(30, 'date')['text']
    )

    prompt = f"""
    As a neutral financial analyst, please analyze the following recent tweets about the stock ${stock}.
    Provide a concise summary (2-3 sentences) of the key themes, topics of discussion, and overall market sentiment.
    Identify any potential catalysts or concerns mentioned. Do not use hashtags or overly promotional language.

    Tweets:
    ---
    {tweets_text}
    ---
    Summary:
    """

    # --- NEW: Configure Safety Settings ---
    # Financial tweets can sometimes contain harsh language that might trigger default filters.
    # We relax the thresholds for categories that are less likely to be problematic in this context.
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_MEDIUM_AND_ABOVE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_ONLY_HIGH',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_ONLY_HIGH',
    }

    try:
        response = model.generate_content(prompt, safety_settings=safety_settings)

        # --- NEW: Check for safety blocks in the response ---
        if not response.parts:
            # This indicates the response was blocked.
            block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback else "Unknown"
            print(f"Warning: Summary generation for {stock} was blocked for reason: {block_reason}")
            return f"AI summary for {stock} could not be generated due to content safety filters."
            
        summary = response.text.strip().replace('*', '')

    except Exception as e:
        print(f"An error occurred during the Gemini API call: {e}")
        summary = f"AI summary generation for {stock} failed. Please check API key and network connection."

    return summary

