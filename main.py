import configparser
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_collection import run_data_collection
from data_processing import run_data_processing
from model import run_model_training
from predict import run_prediction

def main():
    """
    Main function to run the entire stock prediction pipeline.
    """
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    
    if not os.path.exists(config_path):
        print(f"Error: Configuration file '{config_path}' not found.")
        return
        
    config.read(config_path)

    print("\n--- Stock Predictor Pro Pipeline ---")
    
    # Step 1: Data Collection
    print("\n--- Step 1: Starting Data Collection ---")
    run_data_collection(config)
    print("\n--- Data Collection Finished ---")

    # Step 2: Data Processing
    print("\n--- Step 2: Starting Data Processing & Sentiment Analysis ---")
    run_data_processing(config)
    print("\n--- Data Processing Finished ---")

    # Step 3: Model Training
    print("\n--- Step 3: Starting Model Training ---")
    run_model_training(config)
    print("\n--- Model Training Finished ---")
    
    # Step 4: Prediction Generation
    print("\n--- Step 4: Starting Prediction Generation ---")
    run_prediction(config)
    print("\n--- Prediction Generation Finished ---")
    
    print("\n--- Pipeline execution complete! ---")
    print("You can now open the 'public/index.html' file in your browser to see the results.")


if __name__ == "__main__":
    main()
