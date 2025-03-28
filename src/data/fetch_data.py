import os
import requests
import pandas as pd
import streamlit as st

# Access API keys from Streamlit Secrets
api_key = st.secrets["BINANCE_API_KEY"]
api_secret = st.secrets["BINANCE_API_SECRET"]

print(f"API Key: {api_key}")
print(f"API Secret: {api_secret}")

# Function to fetch historical BTC data from Binance API
def fetch_btc_data(symbol='BTCUSDT', interval='1d', start_str='1 Jan 2017'):
    base_url = 'https://api.binance.com'
    endpoint = '/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': int(pd.Timestamp(start_str).timestamp() * 1000),
        'limit': 1000
    }
    
    headers = {
        'X-MBX-APIKEY': api_key  # Corrected variable name
    }
    
    all_data = []
    while True:
        response = requests.get(base_url + endpoint, headers=headers, params=params)
        data = response.json()
        
        if not data:
            break
        
        all_data.extend(data)
        
        # Update startTime to the last timestamp + 1 to avoid overlapping
        params['startTime'] = data[-1][0] + 1
    
    # Convert data to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

# Fetch data and save to CSV
if __name__ == '__main__':
    df = fetch_btc_data()
    
    # Ensure the directory exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, '../../data/raw')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the data
    output_file = os.path.join(output_dir, 'btc_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Data fetched and saved to {output_file}")