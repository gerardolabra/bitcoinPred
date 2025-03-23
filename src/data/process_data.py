import os
import pandas as pd

# In process_data.py, update process_btc_data()
def process_btc_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '../../data/raw/btc_data.csv')
    output_dir = os.path.join(script_dir, '../../data/processed')
    output_file = os.path.join(output_dir, 'btc_data_processed.csv')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the raw data
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    
    # Add returns
    df['returns'] = df['close'].pct_change()
    
    # Add features (moving averages and volatility)
    df['moving_avg_7'] = df['close'].rolling(window=7).mean()
    df['moving_avg_30'] = df['close'].rolling(window=30).mean()
    df['volatility_7'] = df['returns'].rolling(window=7).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()
    
    # Drop unnecessary columns
    if 'ignore' in df.columns:
        df = df.drop(columns=['ignore'])
    
    # Drop rows with NaN values (first 31 rows due to rolling calculations)
    df = df.dropna().reset_index(drop=True)

    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Data processed and saved to {output_file}")

if __name__ == '__main__':
    process_btc_data()