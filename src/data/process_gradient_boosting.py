import os
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
import numpy as np


def process_gradient_boosting_data():
    """
    Processes raw Bitcoin data to generate features for Gradient Boosting models.
    Includes technical indicators, lagged features, and rolling statistics.
    Saves the processed data to a CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '../../data/raw/btc_data.csv')
    output_dir = os.path.join(script_dir, '../../data/processed')
    output_file = os.path.join(output_dir, 'btc_gradient_boosting.csv')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the raw data
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')
    
    # Add returns
    df['returns'] = df['close'].pct_change()
    
    # Add technical indicators
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['upper_band'] = bb.bollinger_hband()
    df['lower_band'] = bb.bollinger_lband()
    df['middle_band'] = bb.bollinger_mavg()
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()
    
    # Add lagged features for 'close' and 'returns'
    for lag in range(1, 6):  # Add lagged features for the last 5 days
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
    
    # Add rolling statistics
    df['rolling_mean_7'] = df['close'].rolling(window=7).mean()
    df['rolling_std_7'] = df['close'].rolling(window=7).std()
    df['rolling_mean_30'] = df['close'].rolling(window=30).mean()
    df['rolling_std_30'] = df['close'].rolling(window=30).std()
    
    # Add volatility (e.g., high-low range)
    df['volatility'] = df['high'] - df['low']
    
    # Define the target variable (next day's close price)
    df['target'] = df['close'].shift(-1)
    
    # Drop unnecessary columns
    columns_to_keep = [
        'close', 'open', 'high', 'low', 'RSI', 'upper_band', 'lower_band',
        'MACD', 'MACD_signal', 'MACD_hist', 'close_lag_1', 'close_lag_2',
        'close_lag_3', 'close_lag_4', 'close_lag_5', 'returns_lag_1',
        'returns_lag_2', 'returns_lag_3', 'returns_lag_4', 'returns_lag_5',
        'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30',
        'volatility', 'target'
    ]
    df = df[columns_to_keep]
    
    # Drop rows with NaN values (caused by rolling calculations and lagging)
    df = df.dropna().reset_index(drop=True)

    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Gradient Boosting-ready data processed and saved to {output_file}")


if __name__ == '__main__':
    process_gradient_boosting_data()