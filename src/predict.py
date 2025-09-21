import pandas as pd
import numpy as np
from keras.models import load_model
import joblib
import os
import json
from datetime import date, timedelta, datetime
import plotly.graph_objects as go
from .nlp_utils import generate_dynamic_summary
import time
import requests
from dotenv import load_dotenv

# Load variables from .env file into the environment
load_dotenv()

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def calculate_accuracy(stock: str, historical_predictions_path: str, processed_df: pd.DataFrame):
    # This function is working correctly and remains unchanged.
    if not os.path.exists(historical_predictions_path):
        return None, None
    past_predictions_df = pd.read_csv(historical_predictions_path)
    past_predictions_df['date'] = pd.to_datetime(past_predictions_df['date']).dt.date
    today = pd.to_datetime(date.today()).date()
    eval_df = past_predictions_df[past_predictions_df['date'] <= today].copy()
    if eval_df.empty:
        return None, None
    actuals_df = processed_df[processed_df['stock'] == stock].copy()
    actuals_df['date'] = pd.to_datetime(actuals_df['date']).dt.date
    actuals_df.rename(columns={'close': 'actual_close'}, inplace=True)
    actuals_df['actual_prev_close'] = actuals_df['actual_close'].shift(1)
    merged_df = pd.merge(eval_df, actuals_df[['date', 'actual_close', 'actual_prev_close']], on='date', how='inner')
    merged_df.dropna(subset=['actual_close', 'actual_prev_close'], inplace=True)
    if merged_df.empty:
        return None, None
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
    model_dir = model_config.get('model_dir', 'models/')
    timesteps = int(model_config.get('timesteps', 60))
    prediction_days = int(prediction_config.get('prediction_days', 30))
    output_dir = prediction_config.get('output_dir', 'public/predictions/')

    try:
        df = pd.read_csv(processed_data_path)
        raw_tweet_df = pd.read_csv(raw_tweet_path)
        raw_tweet_df.columns = raw_tweet_df.columns.str.lower()
    except FileNotFoundError:
        print("Error: Required data files not found. Please run previous steps first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    stocks = df['stock'].unique()
    successful_predictions = []

    for stock in stocks:
        print(f"--- Generating predictions for {stock} ---")
        model_path = os.path.join(model_dir, f'{stock}_model.keras')
        scaler_path = os.path.join(model_dir, f'{stock}_scaler.gz')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Model or scaler for {stock} not found. Skipping prediction.")
            continue

        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        stock_df = df[df['stock'] == stock].copy()
        
        # Prediction logic
        features = ['close', 'sentiment']
        last_sequence = stock_df[features].values[-timesteps:]
        scaled_sequence = scaler.transform(last_sequence)
        predictions = []
        current_batch = scaled_sequence.reshape(1, timesteps, len(features))
        for _ in range(prediction_days):
            next_pred_scaled = model.predict(current_batch, verbose=0)
            predictions.append(next_pred_scaled[0, 0])
            new_row = np.array([[next_pred_scaled[0, 0], current_batch[0, -1, 1]]])
            current_batch = np.append(current_batch[:, 1:, :], new_row.reshape(1, 1, len(features)), axis=1)
        dummy_array = np.zeros((len(predictions), len(features)))
        dummy_array[:, 0] = predictions
        actual_predictions = scaler.inverse_transform(dummy_array)[:, 0]
        last_date = pd.to_datetime(stock_df['date'].iloc[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(1, prediction_days + 1)]
        prediction_df = pd.DataFrame({'date': future_dates, 'predicted_close': actual_predictions})
        
        # Chart generation
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=pd.to_datetime(stock_df['date']), open=stock_df['open'], high=stock_df['high'], low=stock_df['low'], close=stock_df['close'], name='Historical Price', increasing_line_color='#22c55e', decreasing_line_color='#d20f39'))
        last_actual_point = stock_df[['date', 'close']].iloc[-1:].rename(columns={'close': 'predicted_close'})
        full_prediction_line = pd.concat([last_actual_point, prediction_df], ignore_index=True)
        fig.add_trace(go.Scatter(x=pd.to_datetime(full_prediction_line['date']), y=full_prediction_line['predicted_close'], mode='lines+markers', name='Predicted Price', line=dict(color='#ea76cb', width=2), marker=dict(size=4), hovertemplate='<b>Date</b>: %{x|%Y-%m-%d}<br><b>Predicted Price</b>: $%{y:.2f}<extra></extra>'))
        fig.update_layout(title=f'{stock} Price Prediction', xaxis_title='Date', yaxis_title='Price (USD)', template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#161b22', xaxis_rangeslider_visible=False, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        chart_path = os.path.join(output_dir, f'{stock}_chart.html')
        fig.write_html(chart_path, include_plotlyjs='cdn')
        
        historical_predictions_path = os.path.join(output_dir, f'{stock}_historical_predictions.csv')
        
        mape, directional_accuracy = None, None
        try:
            mape, directional_accuracy = calculate_accuracy(stock, historical_predictions_path, df)
            print(f"Accuracy for {stock}: MAPE={mape}, Directional Accuracy={directional_accuracy}")
        except Exception as e:
            print(f"Warning: Could not calculate accuracy for {stock} due to an error: {e}")
        
        # Historical prediction saving
        if os.path.exists(historical_predictions_path):
            historical_df = pd.read_csv(historical_predictions_path)
            historical_df['date'] = pd.to_datetime(historical_df['date'])
            combined_historical = pd.concat([historical_df, prediction_df])
        else:
            combined_historical = prediction_df
        combined_historical.drop_duplicates(subset=['date'], keep='last', inplace=True)
        combined_historical.sort_values(by='date', inplace=True)
        combined_historical.to_csv(historical_predictions_path, index=False)
        print(f"Historical predictions for {stock} updated.")
        
        # --- UPDATED: No longer passes the API key as it's loaded within the function ---
        summary = generate_dynamic_summary(stock, raw_tweet_df)
        
        output_data = {
            'stock': stock,
            'historical': stock_df[['date', 'open', 'high', 'low', 'close']].to_dict(orient='records'),
            'predictions': prediction_df.assign(date=lambda df: df.date.dt.strftime('%Y-%m-%d')).to_dict(orient='records'),
            'summary': summary,
            'accuracy': {
                'mape': mape,
                'directional_accuracy': directional_accuracy
            }
        }
        
        with open(os.path.join(output_dir, f'{stock}_prediction.json'), 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Prediction data for {stock} saved successfully.")
        
        successful_predictions.append(stock)
    
    manifest_path = os.path.join(output_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(successful_predictions, f)
    print(f"Manifest file updated at {manifest_path}")

def run_prediction(config):
    generate_predictions(config)

