Stock Predictor Pro 📈
Stock Predictor Pro is a comprehensive Python-based application that leverages machine learning and natural language processing to forecast stock prices. By analyzing historical price data, technical indicators, and real-time market sentiment from news headlines, it trains unique LSTM models for each stock to provide a 30-day price prediction. The results are displayed in a clean, interactive web dashboard.

This project was developed as an academic exploration into the application of modern AI techniques in financial forecasting.

✨ Features
Automated Data Pipeline: Intelligently collects and updates stock prices (from Yahoo Finance) and news data (from NewsAPI) on a per-stock basis.

Advanced Sentiment Analysis: Utilizes the FinBERT model, a language model specifically fine-tuned for financial text, to accurately gauge market sentiment from news headlines.

Robust LSTM Models: Trains a unique Long Short-Term Memory (LSTM) neural network for each stock, learning from historical prices, sentiment scores, and technical indicators like the Relative Strength Index (RSI).

Intelligent Training: Models are automatically retrained only when new data is available, saving significant time on subsequent runs.

AI-Powered Summaries: Integrates with the Google Gemini API to generate concise, human-readable summaries of the overall market sentiment for a selected stock.

Interactive Web Dashboard: A modern, responsive frontend built with HTML, CSS, and JavaScript that visualizes historical data and future predictions using Plotly.js charts.

GPU Acceleration: Automatically utilizes available NVIDIA GPUs (via CUDA) to significantly speed up sentiment analysis and model training.

🛠️ Tech Stack
Backend: Python

Machine Learning: TensorFlow (Keras), Scikit-learn

NLP: Hugging Face Transformers (for FinBERT), Google Generative AI (for Gemini)

Data Handling: Pandas, NumPy

Data Collection: NewsAPI Client, Requests

Frontend: HTML5, Tailwind CSS, JavaScript

Charting: Plotly.js

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.9+

An environment manager like pip or conda.

NVIDIA GPU with CUDA Toolkit and cuDNN installed (for GPU acceleration).

1. Clone the Repository
git clone [https://github.com/your-username/stock-predictor-pro.git](https://github.com/your-username/stock-predictor-pro.git)
cd stock-predictor-pro

2. Set Up a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt

(Note: You may need to create a requirements.txt file by running pip freeze > requirements.txt in your terminal.)

4. Set Up API Keys
You will need API keys for NewsAPI and Google Gemini.

Create a file named .env in the root directory of the project.

Add your API keys to this file as follows:

NEWS_API_KEY="YOUR_NEWS_API_KEY_HERE"
GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

5. Download the FinBERT Model (Offline Use)
To ensure the application runs smoothly without relying on an internet connection for the model, download the FinBERT files manually:

Go to the ProsusAI/finbert model page on Hugging Face.

Download the following files: config.json, pytorch_model.bin, special_tokens_map.json, tokenizer_config.json, and vocab.txt.

Create a folder in your local system (e.g., C:\stock_predictor\finbert_model).

Place all the downloaded files into this new folder.

Ensure the file path in data_processing.py and news_fetcher.py matches this location.

🏃‍♀️ Usage
The entire pipeline is orchestrated by main.py.

Configure Your Stocks: Open config.ini and list the stock tickers you want to analyze (e.g., stocks = AAPL,GOOGL,TSLA,NVDA).

Run the Pipeline: Execute the main script from the root directory.

python main.py

The script will perform all four steps automatically:

Step 1: Collect new stock and news data.

Step 2: Process data, calculate sentiment and RSI.

Step 3: Train models (only if new data was found).

Step 4: Generate 30-day predictions.

View the Results: Once the pipeline is complete, open the public/index.html file in your web browser to see the interactive dashboard.

📂 Project Structure
.
├── public/
│   ├── predictions/      # Generated JSON data and manifest
│   ├── index.html        # The main dashboard page
│   ├── script.js         # Frontend logic for interactivity and charting
│   └── style.css         # Styling for the dashboard
├── src/
│   ├── data_collection.py  # Fetches stock prices and news
│   ├── data_processing.py  # Cleans data, calculates sentiment and RSI
│   ├── model.py            # Builds and trains the LSTM models
│   ├── news_fetcher.py     # Handles news fetching and sentiment analysis
│   ├── nlp_utils.py        # Generates AI summaries with Gemini
│   └── predict.py          # Generates future predictions from trained models
├── .env                  # Stores API keys (must be created)
├── config.ini            # Main configuration for stocks and paths
├── main.py               # Orchestrator script to run the full pipeline
└── README.md             # This file

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
This project utilizes the NewsAPI for headline data.

Sentiment analysis is powered by the FinBERT model.

AI summaries are generated using Google's Gemini API.

Charting is made possible by the excellent Plotly.js library.