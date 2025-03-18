from binance.client import Client
import pandas as pd
from datetime import datetime
from .config import load_env, get_env_variable

def get_binance_client():
    load_env()
    api_key = get_env_variable('BINANCE_API_KEY')
    api_secret = get_env_variable('BINANCE_API_SECRET')
    return Client(api_key, api_secret)

def get_historical_data(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1DAY, start="1 Jan, 2022"):
    client = get_binance_client()
    klines = client.get_historical_klines(symbol, interval, start)
    
    # Convert to DataFrame
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    
    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)

    return df

# Fetch FTT data
df_ftt = get_historical_data(symbol="FTTUSDT", start="1 Jan, 2022")

# Inspect the FTT data
def inspect_data(df):
    """Print basic info and statistics."""
    print("First 5 rows:\n", df.head(), "\n")
    print("Data Summary:\n", df.info(), "\n")
    print("Descriptive Statistics:\n", df.describe(), "\n")
    print("Missing Values:\n", df.isnull().sum(), "\n")

if __name__ == "__main__":
    inspect_data(df_ftt)