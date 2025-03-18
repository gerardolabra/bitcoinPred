import os
import sys
import pandas as pd
from binance.client import Client

# Add the parent directory to the system path to import get_historical_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.binance_data import get_historical_data  # Use absolute import

# Initialize Binance client
api_key = 'your_api_key'
api_secret = 'your_api_secret'
client = Client(api_key, api_secret)

def save_crypto_data(tokens):
    for symbol, filename in tokens:
        df = get_historical_data(symbol)
        file_path = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)  # Ensure directory exists
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

def get_historical_data(symbol):
    """Fetch historical klines (candlestick data) for a given symbol from Binance."""
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

def get_24h_volume(symbol):
    """Get the 24-hour trading volume for a given symbol from Binance."""
    ticker = client.get_ticker(symbol=symbol)
    volume = ticker['quoteVolume']
    return volume

if __name__ == "__main__":
    # List of tokens and their corresponding filenames
    tokens = [
        ("BTCUSDT", "btc_data.csv"),
        ("FTTUSDT", "ftt_data.csv"),
        ("USTCUSDT", "usct_data.csv"),
        ("RUNEUSDT", "rune_data.csv"),
        ("BNBUSDT", "bnb_data.csv"),
        ("ETHUSDT", "eth_data.csv"),
        ("XRPUSDT", "xrp_data.csv"),
        # Add more tokens here as needed
    ]
    
    # Save historical data
    save_crypto_data(tokens)
    
    # Get and print the 24-hour trading volume for BTC
    btc_volume = get_24h_volume("BTCUSDT")
    print(f"24-hour trading volume for BTC: ${btc_volume}")