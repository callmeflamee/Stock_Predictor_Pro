import pandas as pd
import numpy as np
from keras.models import load_model
import os
import json
from datetime import date, timedelta, datetime
import google.api_core.exceptions
from nlp_utils import generate_dynamic_summary

from dotenv import load_dotenv
load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def scale_data(data, params):
    min_ = np.array(params['min_'])
    scale_ = np.array(params['scale_'])
    return data * scale_ + min_

def inverse_scale_data(data, params):
    min_ = np.array(params['min_'])
    scale_ = np.array(params['scale_'])
    return (data - min_[0]) / scale_[0]

def calculate_accuracy(stock: str, historical_predictions_path: str, processed_df: pd.DataFrame):
    if not os.path.exists(historical_predictions_path): return None, None
    
    past_predictions_df = pd.read_csv(historical_predictions_path)
    past_predictions_df['date'] = pd.to_datetime(past_predictions_df['date']).dt.date
    today = pd.to_datetime(date.today()).date()
    
    eval_df = past_predictions_df[past_predictions_df['date'] <= today].copy()
    if eval_df.empty: return None, None
    
    actuals_df = processed_df[processed_df['stock'] == stock].copy()
    actuals_df['date'] = pd.to_datetime(actuals_df['date']).dt.date
    actuals_df.rename(columns={'close': 'actual_close'}, inplace=True)
    actuals_df['actual_prev_close'] = actuals_df['actual_close'].shift(1)
    
    merged_df = pd.merge(eval_df, actuals_df[['date', 'actual_close', 'actual_prev_close']], on='date', how='inner')
    merged_df.dropna(subset=['actual_close', 'actual_prev_close'], inplace=True)
    
    if merged_df.empty: return None, None

    mape = np.abs((merged_df['actual_close'] - merged_df['predicted_close']) / merged_df['actual_close']).mean() * 100
    
    merged_df['predicted_direction'] = (merged_df['predicted_close'] > merged_df['actual_prev_close']).astype(int)
    merged_df['actual_direction'] = (merged_df['actual_close'] > merged_df['actual_prev_close']).astype(int)
    correct_directions = (merged_df['predicted_direction'] == merged_df['actual_direction']).sum()
    directional_accuracy = (correct_directions / len(merged_df)) * 100 if len(merged_df) > 0 else 0
    
    return mape, directional_accuracy

def generate_predictions(config):
    # --- FIX: Correctly access config sections ---
    data_config = config['Data'] if 'Data' in config else {}
    model_config = config['Model'] if 'Model' in config else {}
    prediction_config = config['Prediction'] if 'Prediction' in config else {}
    
    processed_data_path = data_config.get('processed_data_path', 'data/processed/processed_data.csv')
    raw_tweet_path = data_config.get('raw_tweet_data_path', 'data/raw/tweet_data.csv')
    raw_news_path = data_config.get('raw_news_data_path', 'data/raw/news_data.csv')
    model_dir = model_config.get('model_dir', 'models/')
    timesteps = int(model_config.get('timesteps', 60))
    prediction_days = int(prediction_config.get('prediction_days', 30))
    output_dir = prediction_config.get('output_dir', 'public/predictions/')
    
    try:
        df = pd.read_csv(processed_data_path)
        raw_tweet_df = pd.read_csv(raw_tweet_path) if os.path.exists(raw_tweet_path) else pd.DataFrame()
        raw_news_df = pd.read_csv(raw_news_path) if os.path.exists(raw_news_path) else pd.DataFrame()
    except FileNotFoundError as e:
        print(f"Error: A required data file was not found: {e}. Please run the full data pipeline.")
        return

    os.makedirs(output_dir, exist_ok=True)
    stocks_to_predict = [s for s in data_config.get('stocks', '').split(',') if s in df['stock'].unique()]
    successful_predictions = []

    for stock in stocks_to_predict:
        print(f"--- Generating predictions for {stock} ---")
        model_path = os.path.join(model_dir, f'{stock}_model.keras')
        params_path = os.path.join(model_dir, f'{stock}_scaling_params.json')
        prediction_json_path = os.path.join(output_dir, f'{stock}_prediction.json')

        if not all(os.path.exists(p) for p in [model_path, params_path]):
            print(f"Model or scaling parameters for {stock} not found. Ensure it has been trained.")
            continue

        model = load_model(model_path)
        with open(params_path, 'r') as f:
            scaling_params = json.load(f)
            
        stock_df = df[df['stock'] == stock].copy()
        features = ['close', 'sentiment', 'rsi']
        last_sequence = stock_df[features].values[-timesteps:]
        
        scaled_sequence = scale_data(last_sequence, scaling_params)
        
        predictions_scaled = []
        current_batch = scaled_sequence.reshape(1, timesteps, len(features))
        
        scale_ = np.array(scaling_params['scale_'])
        min_ = np.array(scaling_params['min_'])
        scaled_neutral_sentiment = 0 * scale_[1] + min_[1]
        
        rsi_window = 14
        last_closes = stock_df['close'].values[-rsi_window:]

        for _ in range(prediction_days):
            next_pred_scaled = model.predict(current_batch, verbose=0)
            predictions_scaled.append(next_pred_scaled[0, 0])
            
            next_pred_unscaled = inverse_scale_data(next_pred_scaled[0, 0], scaling_params)
            
            last_closes = np.append(last_closes[1:], next_pred_unscaled)
            delta = np.diff(last_closes)
            gain = delta[delta > 0].mean() if (delta > 0).any() else 0
            loss = -delta[delta < 0].mean() if (delta < 0).any() else 1e-10
            rs = gain / loss
            new_rsi = 100 - (100 / (1 + rs))
            scaled_new_rsi = new_rsi * scale_[2] + min_[2]
            
            new_row = np.array([[next_pred_scaled[0, 0], scaled_neutral_sentiment, scaled_new_rsi]])
            
            current_batch = np.append(current_batch[:, 1:, :], new_row.reshape(1, 1, len(features)), axis=1)

        actual_predictions = inverse_scale_data(np.array(predictions_scaled), scaling_params)
        next_day_prediction = float(actual_predictions[0]) if len(actual_predictions) > 0 else np.nan

        last_date = pd.to_datetime(stock_df['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
        
        prediction_df = pd.DataFrame({'date': future_dates, 'predicted_close': actual_predictions.flatten()})
        
        historical_predictions_path = os.path.join(output_dir, f'{stock}_historical_predictions.csv')
        if future_dates:
            next_day_df = pd.DataFrame({'date': [future_dates[0].date()], 'predicted_close': [next_day_prediction]})
            if os.path.exists(historical_predictions_path):
                historical_df = pd.read_csv(historical_predictions_path)
                historical_df['date'] = pd.to_datetime(historical_df['date']).dt.date
                combined_historical = pd.concat([historical_df, next_day_df])
            else:
                combined_historical = next_day_df
            combined_historical.drop_duplicates(subset=['date'], keep='last', inplace=True)
            combined_historical.sort_values(by=['date'], inplace=True)
            combined_historical.to_csv(historical_predictions_path, index=False)
        
        mape, directional_accuracy = calculate_accuracy(stock, historical_predictions_path, df)
        
        summary = "Summary generation failed or was skipped."
        try:
            summary = generate_dynamic_summary(stock, raw_tweet_df, raw_news_df)
        except google.api_core.exceptions.ResourceExhausted:
            summary = "AI summary generation failed due to API quota limits."
        except Exception as e:
            summary = f"An error occurred during summary generation: {e}"

        output_data = {
            'stock': stock,
            'historical': stock_df[['date', 'open', 'high', 'low', 'close']].to_dict(orient='records'),
            'predictions': prediction_df.assign(date=lambda df: df.date.dt.strftime('%Y-%m-%d')).to_dict(orient='records'),
            'next_day_prediction': next_day_prediction,
            'summary': summary,
            'accuracy': {'mape': mape, 'directional_accuracy': directional_accuracy}
        }
        
        with open(prediction_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Prediction data for {stock} saved to {prediction_json_path}")
        
        successful_predictions.append(stock)
    
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(successful_predictions, f)
    print(f"Manifest file updated at {manifest_path} with {len(successful_predictions)} stocks.")

def run_prediction(config):
    generate_predictions(config)

