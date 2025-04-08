import os
import pandas as pd
from binance.client import Client
import time
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = "eu-central-1"  # Frankfurt region
S3_BUCKET_NAME = "bitcoin-data-gl"  # Replace with your bucket name

if not API_KEY or not API_SECRET:
    raise ValueError("Binance API keys are missing. Please check your .env file.")

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS keys are missing. Please check your .env file.")

# Initialize Binance client
client = Client(API_KEY, API_SECRET)

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
)

def save_crypto_data(tokens):
    """
    Save historical data for a list of tokens to CSV files and upload to S3.
    """
    for symbol, filename in tokens:
        try:
            df = get_historical_data(symbol)
            # Save to the `data/raw/` directory
            local_file_path = os.path.join("data", "raw", filename)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)  # Ensure directory exists
            df.to_csv(local_file_path, index=False)
            print(f"Data saved locally to {local_file_path}")

            # Upload to S3
            s3_file_path = f"raw/{filename}"  # Path in the S3 bucket
            s3.upload_file(local_file_path, S3_BUCKET_NAME, s3_file_path)
            print(f"File uploaded to S3: s3://{S3_BUCKET_NAME}/{s3_file_path}")

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