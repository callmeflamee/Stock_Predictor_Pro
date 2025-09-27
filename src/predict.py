import pandas as pd
import numpy as np
from keras.models import load_model
import os
import json
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from nlp_utils import generate_dynamic_summary
import time
import requests
from dotenv import load_dotenv
import configparser
import google.api_core.exceptions

load_dotenv()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Corrected scaling functions
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
    if merged_df.empty: return None, None
    merged_df.dropna(subset=['actual_close', 'actual_prev_close'], inplace=True)
    if merged_df.empty: return None, None
    mape = np.abs((merged_df['actual_close'] - merged_df['predicted_close']) / merged_df['actual_close']).mean() * 100
    merged_df['predicted_direction'] = (merged_df['predicted_close'] > merged_df['actual_prev_close']).astype(int)
    merged_df['actual_direction'] = (merged_df['actual_close'] > merged_df['actual_prev_close']).astype(int)
    correct_directions = (merged_df['predicted_direction'] == merged_df['actual_direction']).sum()
    directional_accuracy = (correct_directions / len(merged_df)) * 100
    return mape, directional_accuracy

def generate_predictions(config):
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
        raw_tweet_df = pd.read_csv(raw_tweet_path)
        raw_news_df = pd.read_csv(raw_news_path)
    except FileNotFoundError as e:
        print(f"Error: A required data file was not found: {e}. Please run the full data pipeline.")
        return

    os.makedirs(output_dir, exist_ok=True)
    stocks_to_predict = [s for s in data_config.get('stocks', '').split(',') if s in df['stock'].unique()]

    for stock in stocks_to_predict:
        print(f"--- Generating predictions for {stock} with dynamic RSI ---")
        model_path = os.path.join(model_dir, f'{stock}_model.keras')
        params_path = os.path.join(model_dir, f'{stock}_scaling_params.json')
        prediction_json_path = os.path.join(output_dir, f'{stock}_prediction.json')

        if not os.path.exists(model_path) or not os.path.exists(params_path):
            print(f"Model or scaling parameters for {stock} not found. Ensure it has been trained.")
            continue

        model = load_model(model_path)
        with open(params_path, 'r') as f:
            scaling_params = json.load(f)
        
        min_ = np.array(scaling_params['min_'])
        scale_ = np.array(scaling_params['scale_'])
            
        stock_df = df[df['stock'] == stock].copy()
        
        # --- FIX: Update features to include RSI ---
        features = ['close', 'sentiment', 'rsi']
        last_sequence = stock_df[features].values[-timesteps:]
        scaled_sequence = scale_data(last_sequence, scaling_params)
        
        predictions_scaled = []
        current_batch = scaled_sequence.reshape(1, timesteps, len(features))
        
        # --- FIX: Pre-calculate scaled neutral sentiment ---
        scaled_neutral_sentiment = 0 * scale_[1] + min_[1]
        
        # --- FIX: Setup for dynamic RSI calculation ---
        rsi_window = 14
        last_closes = stock_df['close'].values[-rsi_window:]

        for _ in range(prediction_days):
            next_pred_scaled = model.predict(current_batch, verbose=0)
            predictions_scaled.append(next_pred_scaled[0, 0])
            
            # --- FIX: Dynamic RSI calculation within the loop ---
            # 1. Inverse scale the new prediction to get the actual price
            next_pred_price = inverse_scale_data(next_pred_scaled[0, 0], scaling_params)
            
            # 2. Update the recent close prices and calculate new RSI
            last_closes = np.append(last_closes[1:], next_pred_price)
            delta = np.diff(last_closes)
            gain = delta[delta > 0].mean() if (delta > 0).any() else 0
            loss = -delta[delta < 0].mean() if (delta < 0).any() else 1e-10 # Avoid zero division
            rs = gain / loss
            new_rsi = 100 - (100 / (1 + rs))

            # 3. Scale the new RSI for model input
            scaled_new_rsi = new_rsi * scale_[2] + min_[2]
            
            # 4. Create the new input row with predicted price, neutral sentiment, and dynamic RSI
            new_row = np.array([[next_pred_scaled[0, 0], scaled_neutral_sentiment, scaled_new_rsi]])
            
            current_batch = np.append(current_batch[:, 1:, :], new_row.reshape(1, 1, len(features)), axis=1)

        actual_predictions = inverse_scale_data(np.array(predictions_scaled), scaling_params)

        next_day_prediction = float(actual_predictions[0]) if len(actual_predictions) > 0 else np.nan
        print(f"LSTM Next-Day Prediction for {stock}: {next_day_prediction:.2f}")

        last_date = pd.to_datetime(stock_df['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
        prediction_df_chart = pd.DataFrame({'date': future_dates, 'predicted_close': actual_predictions.flatten()})
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=pd.to_datetime(stock_df['date']), open=stock_df['open'], high=stock_df['high'], low=stock_df['low'], close=stock_df['close'], name='Historical Price', increasing_line_color='#22c55e', decreasing_line_color='#d20f39'))
        last_actual_point = stock_df[['date', 'close']].iloc[-1:].rename(columns={'close': 'predicted_close'})
        full_prediction_line = pd.concat([last_actual_point, prediction_df_chart], ignore_index=True)
        fig.add_trace(go.Scatter(x=pd.to_datetime(full_prediction_line['date']), y=full_prediction_line['predicted_close'], mode='lines+markers', name='Predicted Trend', line=dict(color='#ea76cb', width=2), marker=dict(size=4), hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Predicted Price</b>: $%{y:.2f}<extra></extra>'))
        fig.update_layout(title=f'{stock} Price Prediction', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#161b22', xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        chart_path = os.path.join(output_dir, f'{stock}_chart.html')
        fig.write_html(chart_path, include_plotlyjs='cdn')
        
        historical_predictions_path = os.path.join(output_dir, f'{stock}_historical_predictions.csv')
        # ... (rest of the file is unchanged) ...
        if future_dates:
            next_day_df = pd.DataFrame({'date': [future_dates[0]], 'predicted_close': [next_day_prediction]})
            if os.path.exists(historical_predictions_path):
                historical_df = pd.read_csv(historical_predictions_path)
                historical_df['date'] = pd.to_datetime(historical_df['date'])
                combined_historical = pd.concat([historical_df, next_day_df])
            else:
                combined_historical = next_day_df
            combined_historical.drop_duplicates(subset=['date'], keep='last', inplace=True)
            combined_historical.sort_values(by=['date'], inplace=True)
            combined_historical.to_csv(historical_predictions_path, index=False)
        
        mape, directional_accuracy = None, None
        try:
            mape, directional_accuracy = calculate_accuracy(stock, historical_predictions_path, df)
            if mape is not None:
                print(f"Accuracy for {stock}: MAPE={mape:.2f}%, Directional={directional_accuracy:.2f}%")
        except Exception as e:
            print(f"Warning: Could not calculate accuracy for {stock}: {e}")
        
        summary = "Summary will be generated soon."
        # ... (summary generation logic is unchanged) ...
        needs_new_summary = True
        if os.path.exists(prediction_json_path):
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(prediction_json_path))
            if (datetime.now() - file_mod_time).days < 1: # Cache summary for 1 day
                try:
                    with open(prediction_json_path, 'r') as f:
                        summary = json.load(f).get('summary', summary)
                    needs_new_summary = False
                except (json.JSONDecodeError, KeyError): pass
        
        if needs_new_summary:
            print("Generating new AI summary...")
            try:
                summary = generate_dynamic_summary(stock, raw_tweet_df, raw_news_df)
            except Exception as e:
                summary = "An error occurred while generating the AI summary."

        output_data = {
            'stock': stock,
            'historical': stock_df[['date', 'open', 'high', 'low', 'close']].to_dict(orient='records'),
            'predictions': prediction_df_chart.assign(date=lambda df: df.date.dt.strftime('%Y-%m-%d')).to_dict(orient='records'),
            'next_day_prediction': next_day_prediction,
            'summary': summary,
            'accuracy': {'mape': mape, 'directional_accuracy': directional_accuracy}
        }
        
        with open(prediction_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Prediction data for {stock} saved successfully.")
        
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(stocks_to_predict, f)
    print(f"Manifest file updated with {len(stocks_to_predict)} stocks.")

def run_prediction(config):
    generate_predictions(config)

