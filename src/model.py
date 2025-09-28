import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam
import os
import json
# --- NEW: Import TensorFlow to check for GPUs ---
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def build_and_train_model(config):
    # --- NEW: Check for available GPUs and configure them ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to prevent TensorFlow from allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"TensorFlow is using {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    else:
        print("TensorFlow did not find any GPUs. Model training will run on CPU.")

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
        params_path = os.path.join(model_dir, f'{stock}_scaling_params.json')

        if os.path.exists(model_path):
            model_mod_time = os.path.getmtime(model_path)
            data_mod_time = os.path.getmtime(processed_data_path)
            
            if data_mod_time <= model_mod_time:
                print(f"Model for {stock} is already up-to-date. Skipping training.")
                continue
            else:
                print(f"Data has been updated. Retraining model for {stock}...")
        
        stock_df = df[df['stock'] == stock].copy()
        
        if len(stock_df) < timesteps + 50:
            print(f"Not enough data for {stock} to train a model. Skipping.")
            continue
        
        features = ['close', 'sentiment', 'rsi']
        dataset = stock_df[features].values
        
        train_data, val_data = train_test_split(dataset, test_size=0.15, shuffle=False)
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_val_data = scaler.transform(val_data)

        scaling_params = {'min_': scaler.min_.tolist(), 'scale_': scaler.scale_.tolist()}
        with open(params_path, 'w') as f:
            json.dump(scaling_params, f)

        def create_sequences(data, timesteps):
            X, y = [], []
            for i in range(timesteps, len(data)):
                X.append(data[i-timesteps:i, :])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(scaled_train_data, timesteps)
        X_val, y_val = create_sequences(scaled_val_data, timesteps)

        if len(X_train) == 0 or len(X_val) == 0:
            print(f"Not enough data for {stock} to create train/validation sequences. Skipping.")
            continue

        model = Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(units=64, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            LSTM(units=32, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.001)),
            Dropout(0.3),
            Dense(units=25, activation='relu'),
            BatchNormalization(),
            Dense(units=1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            batch_size=32, 
            epochs=100,
            callbacks=[early_stopping],
            verbose=1
        )
        
        model.save(model_path)
        print(f"Model and scaler for {stock} saved successfully.")

def run_model_training(config):
    build_and_train_model(config)

