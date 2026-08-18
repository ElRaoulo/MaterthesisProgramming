"""
Master's Thesis Data Collection: LLM Stock Price Prediction
This script collects stock price predictions and stores them in a JSONL file.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from newsapi import NewsApiClient
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
import time
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
import json
warnings.filterwarnings('ignore')


# ========================
# CONFIGURATION
# ========================

# Number of stocks to predict
NUMBER_OF_STOCKS = 30
# Seed for reproducibility
RANDOM_SEED = 2025
# Output file
LOG_FILE = 'stock_predictions_log.jsonl'
# Today's date
today = date.today()


# ========================
# INITIALIZATION
# ========================

def initialize():
    """Initialize the environment and load API keys."""
    load_dotenv()
    print("[OK] All core libraries imported successfully!")
    print("Current timestamp:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Setup: {NUMBER_OF_STOCKS} stocks.\n")


# ========================
# STOCK SELECTION
# ========================

def load_and_select_tickers(n_stocks, seed):
    """Loads S&P 500 tickers and selects a fixed random sample."""
    try:
        sp500 = pd.read_csv("sp500_constitutes.csv")
        sp500_tickers = sp500['Symbol'].tolist()
    except FileNotFoundError:
        print(f"[WARNING] Error: 'sp500_constitutes.csv' not found. Using a fallback list.")
        sp500_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG', 'HD', 'MA', 
            'UNH', 'NVDA', 'TSLA', 'XOM', 'WMT', 'KO', 'PEP', 'DIS', 'NFLX', 'SBUX', 
            'ADBE', 'CRM', 'CSCO', 'CMCSA', 'DHR', 'INTC', 'TMO', 'TXN', 'QCOM', 
            'MCD', 'NKE', 'ORCL', 'ABBV', 'MDT', 'NEE'
        ]
        
    if sp500_tickers:
        random.seed(seed)
        selected_tickers = random.sample(sp500_tickers, min(n_stocks, len(sp500_tickers)))
        print(f"[INFO] Total S&P 500 tickers available: {len(sp500_tickers)}")
        print(f"[INFO] Randomly selected {len(selected_tickers)} fixed tickers for analysis.")
        print(f"Tickers: {selected_tickers}\n")
        return selected_tickers
    else:
        print("[ERROR] Failed to load S&P 500 tickers")
        return []


# ========================
# UTILITY FUNCTIONS
# ========================

def get_current_stock_price(ticker):
    """Fetch current market data for a given ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try to get the latest available price
        current_price = (info.get('regularMarketPrice') or 
                        info.get('currentPrice') or 
                        info.get('previousClose'))
            
        if current_price is not None:
            return round(current_price, 2)
        else:
            # Fallback to 1-day history
            hist = stock.history(period="1d")
            if not hist.empty:
                return round(hist['Close'].iloc[-1], 2)
            return None
            
    except Exception as e:
        print(f"[ERROR] Error fetching price for {ticker}: {e}")
        return None


def log_full_response(filepath, log_entry):
    """Appends a log entry (dictionary) as a JSON line to the specified file."""
    try:
        with open(filepath, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"[WARNING] Error logging data to file {filepath}: {e}")


# ========================
# LLM PREDICTION
# ========================

def initialize_openai_client():
    """Initialize OpenAI client."""
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        print("[OK] OpenAI client initialized\n")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}. Check OPENAI_API_KEY.")
        return None


def get_llm_prediction(client, ticker, current_price):
    """
    Get stock price prediction from GPT-5-nano for multiple horizons.
    Returns the raw response object with all metadata.
    """
    if not client:
        return None
        
    try:
        system_prompt = """You are a world-class financial analyst specializing in short-term market movements. Based on the provided current stock price, recent news, and macroeconomic factors, analyze the potential price of the stock at the close of today's trading session, in one week, and in one month. Your analysis must be grounded in real-time information obtained through web search. Provide a concise prediction following the exact format specified."""
        
        user_prompt = f"""Stock Ticker: {ticker}
Current Price: ${current_price}

Do an online research to predict the trading price for {ticker} at the following three future points:
1. Closing of Today's Trading Session.
2. Closing in one Week ({today + timedelta(days=7)}) from now.
3. Closing in one Month ({today + relativedelta(months=1)}) from now.

Your response must be ONLY the following format, with no additional text, explanations, or prefixes:

STOCK: {ticker}
PREDICTED_PRICE_DAY: [your predicted numerical value in USD at the end of today's trading day, rounded to two decimal places]
PREDICTED_PRICE_WEEK: [your predicted numerical value in USD in one week {today + timedelta(days=7)}, rounded to two decimal places]
PREDICTED_PRICE_MONTH: [your predicted numerical value in USD in one month{today + relativedelta(months=1)}, rounded to two decimal places]
CONFIDENCE: [High/Medium/Low]
REASONING: [A single, concise sentence explaining the primary factor for your prediction across all three horizons.]"""

        response = client.responses.create(
            model="gpt-5-nano-2025-08-07",
            instructions=system_prompt,
            input=user_prompt,
            tools=[{"type": "web_search"}],
            include=["web_search_call.results", "web_search_call.action.sources", "reasoning.encrypted_content"],
        )
        
        print(f"  -> LLM Output: {response.output_text.replace(chr(10), ' | ')}")
        return response
        
    except Exception as e:
        print(f"[ERROR] Error getting LLM prediction for {ticker}: {e}")
        return None


# ========================
# DATA COLLECTION
# ========================

def run_data_collection(client, tickers, log_file):
    """Simulates one day of data collection for all selected tickers."""
    print(f"\n------------------------------------------")
    print(f"Starting Data Collection for {len(tickers)} stocks.")
    print(f"------------------------------------------")
    
    log_count = 0
    
    for i, ticker in enumerate(tickers, 1):
        print(f"  Processing {i}/{len(tickers)}: {ticker}")
        
        # Fetch Price
        current_price = get_current_stock_price(ticker)
        
        if current_price is None:
            print(f"    [SKIP] Skipping {ticker}: Could not fetch price.")
            continue
        
        print(f"    [PRICE] Current Price: ${current_price:.2f}")

        # Get LLM Prediction
        llm_response = get_llm_prediction(client, ticker, current_price)
        
        if llm_response:
            # Store the full response object and key metadata
            log_entry = {
                'collection_timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'start_price': current_price,
                'llm_response_dump': llm_response.model_dump()
            }
            log_full_response(log_file, log_entry)
            log_count += 1
            # Delay to avoid API rate limiting
            time.sleep(1)
        else:
            print(f"    [ERROR] Failed to get LLM response for {ticker}.")
            
    print(f"Data Complete. Logged {log_count} entries.")
    return log_count


# ========================
# MAIN EXECUTION
# ========================

def main():
    """Main execution function."""
    # Initialize
    initialize()
    
    # Load and select tickers
    selected_tickers = load_and_select_tickers(NUMBER_OF_STOCKS, RANDOM_SEED)
    
    if not selected_tickers:
        print("[ERROR] No tickers selected. Exiting.")
        return
    
    # Initialize OpenAI client
    client = initialize_openai_client()
    
    if not client:
        print("[ERROR] Cannot proceed without OpenAI client. Exiting.")
        return
    
    # Run data collection
    total_logged_entries = run_data_collection(client, selected_tickers, LOG_FILE)
    
    # Summary
    print("------------------------------------------")
    print(f"Completed Data Collection. Date: {today}.")
    print(f"[OK] Data saved to: {LOG_FILE}")
    print(f"Total entries logged: {total_logged_entries}")
    print("------------------------------------------")


if __name__ == "__main__":
    main()
