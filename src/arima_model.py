import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
import warnings

# Suppress statistical warnings from the ARIMA model
warnings.filterwarnings("ignore", category=UserWarning)

def train_and_save_arima(stock_df: pd.DataFrame, stock: str, model_dir: str):
    """
    Trains a simple ARIMA model on the closing price and saves it.
    """
    print(f"--- Training ARIMA model for {stock} ---")
    
    # ARIMA works on a single time series of historical prices
    history = stock_df['close'].values
    
    # A common starting point for ARIMA parameters (p,d,q) - (AutoRegressive, Integrated, Moving Average)
    order = (5, 1, 0)
    
    try:
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        
        # Save the trained model using joblib for efficiency
        model_path = os.path.join(model_dir, f'{stock}_arima.pkl')
        joblib.dump(model_fit, model_path)
        print(f"ARIMA model for {stock} saved successfully to {model_path}")
        
    except Exception as e:
        print(f"Error training ARIMA for {stock}: {e}")

def get_arima_prediction(stock: str, model_dir: str, history: list):
    """
    Loads a saved ARIMA model and generates a single-step forecast.
    """
    model_path = os.path.join(model_dir, f'{stock}_arima.pkl')
    if not os.path.exists(model_path):
        print(f"Warning: ARIMA model for {stock} not found.")
        return None # No model available

    try:
        # To make the best forecast, the model is refit with the most up-to-date history
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()

        # Forecast one step ahead
        prediction = model_fit.forecast(steps=1)
        return prediction[0]
    except Exception as e:
        print(f"Error making ARIMA prediction for {stock}: {e}")
        return None
