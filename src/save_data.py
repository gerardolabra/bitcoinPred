import os
import sys
import pandas as pd
from binance.client import Client
import time
import streamlit as st

api_key = st.secrets["BINANCE_API_KEY"]
api_secret = st.secrets["BINANCE_API_SECRET"]

if not api_key or not api_secret:
    raise ValueError("API keys are missing. Please check your Streamlit secrets.")

client = Client(api_key, api_secret)

def save_crypto_data(tokens):
    """
    Save historical data for a list of tokens to CSV files.
    """
    for symbol, filename in tokens:
        try:
            df = get_historical_data(symbol)
            # Save to the `data/raw/` directory
            file_path = os.path.join("data", "raw", filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
            df.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
            time.sleep(1)  # Pause to avoid hitting API rate limits
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")

def get_historical_data(symbol):
    """
    Fetch historical klines (candlestick data) for a given symbol from Binance.
    """
    try:
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, "1 Jan 2017")
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

def get_24h_volume(symbol):
    """
    Get the 24-hour trading volume for a given symbol from Binance.
    """
    try:
        ticker = client.get_ticker(symbol=symbol)
        volume = ticker['quoteVolume']
        return volume
    except Exception as e:
        print(f"Error fetching 24-hour volume for {symbol}: {e}")
        return None

if __name__ == "__main__":
    # List of tokens and their corresponding filenames
    tokens = [
        ("BTCUSDT", "btc_data.csv"),
        # Add more tokens here as needed
    ]
    
    # Save historical data
    save_crypto_data(tokens)
    
    # Get and print the 24-hour trading volume for BTC
    btc_volume = get_24h_volume("BTCUSDT")
    if btc_volume:
        print(f"24-hour trading volume for BTC: ${btc_volume}")