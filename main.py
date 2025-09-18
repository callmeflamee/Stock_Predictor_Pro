import configparser
import os
from src.data_collection import run_data_collection
from src.data_processing import run_data_processing
from src.model import run_model_training
from src.predict import run_prediction

def main():
    """
    Runs the complete data pipeline:
    1. Collects stock and tweet data.
    2. Processes and merges the data, adding sentiment scores.
    3. Trains the LSTM model on the processed data.
    4. Generates and saves future predictions.
    """
    config_path = 'config.ini'
    config = configparser.ConfigParser()
    
    # --- FIX: Added a check to ensure the config file is read correctly ---
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please make sure the file exists in the project's root directory.")
        return

    config.read(config_path)

    # Validate that the config file has the necessary sections
    if 'Data' not in config or 'Model' not in config:
        print(f"Error: The '{config_path}' file is missing the required [Data] or [Model] sections.")
        return

    print("--- Step 1: Starting Data Collection ---")
    #run_data_collection(config)
    print("\n--- Data Collection Finished ---\n")

    print("--- Step 2: Starting Data Processing & Sentiment Analysis ---")
    #run_data_processing(config)
    print("\n--- Data Processing Finished ---\n")

    print("--- Step 3: Starting Model Training ---")
    #run_model_training(config)
    print("\n--- Model Training Finished ---\n")

    print("--- Step 4: Starting Prediction Generation ---")
    run_prediction(config)
    print("\n--- Prediction Generation Finished ---\n")

    print("✅✅✅ Pipeline Complete! You can now open public/index.html to see the results. ✅✅✅")

if __name__ == "__main__":
    main()

