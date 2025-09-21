import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import os
import time
import json 

# Import the ARIMA training function
from arima_model import train_and_save_arima

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_and_train_model(config):
    data_config = config['Data'] if 'Data' in config else {}
    model_config = config['Model'] if 'Model' in config else {}

    processed_data_path = data_config.get('processed_data_path', 'data/processed/processed_data.csv')
    model_dir = model_config.get('model_dir', 'models/')
    timesteps = int(model_config.get('timesteps', 60))

    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print("Error: Processed data not found.")
        return

    os.makedirs(model_dir, exist_ok=True)
    
    stocks = df['stock'].unique()
    for stock in stocks:
        print(f"--- Processing models for {stock} ---")
        
        stock_df = df[df['stock'] == stock].copy()
        model_path = os.path.join(model_dir, f'{stock}_model.keras')
        
        needs_update = True
        if os.path.exists(model_path):
            data_mod_time = os.path.getmtime(processed_data_path)
            model_mod_time = os.path.getmtime(model_path)
            if model_mod_time >= data_mod_time:
                needs_update = False

        if not needs_update:
            print(f"Models for {stock} are already up to date. Skipping training.")
            continue
        
        print(f"--- Training LSTM model for {stock} ---")
        if len(stock_df) < timesteps + 1:
            print(f"Not enough data for {stock} to train. Skipping.")
            continue
        
        features = ['close', 'sentiment']
        dataset = stock_df[features].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        scaling_params = {
            'min_': scaler.min_.tolist(),
            'scale_': scaler.scale_.tolist()
        }
        params_path = os.path.join(model_dir, f'{stock}_scaling_params.json')
        with open(params_path, 'w') as f:
            json.dump(scaling_params, f)
        print(f"Scaling parameters for {stock} saved successfully.")

        X_train, y_train = [], []
        for i in range(timesteps, len(scaled_data)):
            X_train.append(scaled_data[i-timesteps:i, :])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=50, return_sequences=True),
            LSTM(units=50, return_sequences=False),
            Dense(units=25),
            # --- CRITICAL FIX: Add a Softplus activation to the final layer ---
            # This constrains the model's output to be non-negative, preventing negative price predictions.
            Dense(units=1, activation='softplus')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
        
        model.save(model_path)
        print(f"LSTM model for {stock} saved successfully.")

        train_and_save_arima(stock_df, stock, model_dir)


def run_model_training(config):
    build_and_train_model(config)
