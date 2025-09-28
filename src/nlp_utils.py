import pandas as pd
import google.generativeai as genai
import os

def generate_dynamic_summary(stock: str, raw_tweet_df: pd.DataFrame, raw_news_df: pd.DataFrame):
    """
    Analyzes recent news headlines using the Gemini API to generate a
    high-quality, comprehensive summary.
    """
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        return "Google API Key not found in .env file. AI summary is disabled."
    
    try:
        genai.configure(api_key=api_key)
        # --- FIX: Updated model name to the latest stable version ---
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        return "Failed to initialize the AI model. Please check your API key."

    # --- Process News Data ---
    news_context = "No recent news headlines found."
    if not raw_news_df.empty and 'title' in raw_news_df.columns:
        # The news_fetcher now aggregates titles into a list for each day.
        # We need to flatten this list for the prompt.
        
        # Ensure 'date' is in datetime format for proper sorting
        raw_news_df['date'] = pd.to_datetime(raw_news_df['date'], errors='coerce')
        
        # Filter for the specific stock and the last 7 days
        seven_days_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
        stock_news = raw_news_df[
            (raw_news_df['stock'] == stock) & 
            (raw_news_df['date'] >= seven_days_ago)
        ].copy()

        if not stock_news.empty:
            # Flatten the list of lists of titles
            all_titles = [title for sublist in stock_news['title'] for title in sublist]
            
            # Get unique titles while preserving order (to some extent)
            unique_titles = list(pd.unique([title.strip() for title in all_titles if title.strip()]))

            # Take the most recent 20 unique headlines for the prompt
            news_text = "\n".join("- " + title for title in unique_titles[:20])
            news_context = f"Recent News Headlines:\n---\n{news_text}\n---"
    else:
         news_context = "News headlines not available in the current data format."


    # --- Construct the Prompt ---
    prompt = f"""
    As a neutral financial analyst, analyze the following recent news headlines about ${stock}.
    Provide a concise, well-rounded summary (strictly 2-3 sentences, maximum 4 lines) of the key themes, catalysts, and overall market sentiment.
    Do not use hashtags or promotional language.

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

