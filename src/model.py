import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import joblib
import os

# Suppress TensorFlow informational messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_and_train_model(config):
    # Safely get config values with defaults
    data_config = config['Data'] if 'Data' in config else {}
    model_config = config['Model'] if 'Model' in config else {}

    processed_data_path = data_config.get('processed_data_path', 'data/processed/processed_data.csv')
    model_dir = model_config.get('model_dir', 'models/')
    timesteps = int(model_config.get('timesteps', 60))

    try:
        df = pd.read_csv(processed_data_path)
    except FileNotFoundError:
        print("Error: Processed data not found. Please run data collection and processing first.")
        return

    os.makedirs(model_dir, exist_ok=True)
    
    stocks = df['stock'].unique()
    for stock in stocks:
        print(f"--- Processing model for {stock} ---")
        
        model_path = os.path.join(model_dir, f'{stock}_model.keras')
        scaler_path = os.path.join(model_dir, f'{stock}_scaler.gz')

        # --- NEW: Check if model needs updating ---
        # If a model exists, compare its modification time to the data's modification time.
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model_mtime = os.path.getmtime(model_path)
                data_mtime = os.path.getmtime(processed_data_path)
                
                if data_mtime <= model_mtime:
                    print(f"Model for {stock} is up-to-date. Skipping training.")
                    continue
            except FileNotFoundError:
                # If data file is not found (edge case), proceed to train
                pass

        print(f"--- Training new or updated model for {stock} ---")
        stock_df = df[df['stock'] == stock].copy()
        
        if len(stock_df) < timesteps + 1:
            print(f"Not enough data for {stock} to train a model (requires at least {timesteps + 1} days). Skipping.")
            continue

        features = ['close', 'sentiment']
        dataset = stock_df[features].values
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        X_train, y_train = [], []
        for i in range(timesteps, len(scaled_data)):
            X_train.append(scaled_data[i-timesteps:i, :])
            y_train.append(scaled_data[i, 0]) # Predicting the 'close' price

        X_train, y_train = np.array(X_train), np.array(y_train)

        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=50, return_sequences=True),
            LSTM(units=50, return_sequences=False),
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=20, verbose=0)
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"Model and scaler for {stock} saved successfully.")

def run_model_training(config):
    build_and_train_model(config)

