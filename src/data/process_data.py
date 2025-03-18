import os
import pandas as pd

# In process_data.py, update process_btc_data()
def process_btc_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '../../data/raw/btc_data.csv')
    output_dir = os.path.join(script_dir, '../../data/processed')
    output_file = os.path.join(output_dir, 'btc_data_processed.csv')
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    
    # Add returns
    df['returns'] = df['close'].pct_change()
    
    # Features (volatility on returns now)
    df['moving_avg_7'] = df['close'].rolling(window=7).mean()
    df['moving_avg_30'] = df['close'].rolling(window=30).mean()
    df['volatility_7'] = df['returns'].rolling(window=7).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()
    
    df.to_csv(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

if __name__ == '__main__':
    process_btc_data()