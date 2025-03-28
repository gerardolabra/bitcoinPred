import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
import joblib

def process_lstm_live(window_size=1000):
    """
    Processes raw Bitcoin data to generate features for live next-day prediction.
    Prepares the last `window_size` rows as a sequence for LSTM prediction.
    Saves the processed sequence to a CSV file for live predictions.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '../../data/raw/btc_data.csv')
    numerical_scaler_file = os.path.join(script_dir, '../../data/processed/numerical_scaler.pkl')
    output_dir = os.path.join(script_dir, '../../data/processed')
    output_file = os.path.join(output_dir, 'btc_lstm_live.csv')
    os.makedirs(output_dir, exist_ok=True)

    # Load the raw data
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp')

    # Extract numerical features from the timestamp
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day

    # Add technical indicators
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()  # RSI with a 14-day window
    bb = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['upper_band'] = bb.bollinger_hband()
    df['lower_band'] = bb.bollinger_lband()
    df['middle_band'] = bb.bollinger_mavg()
    macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    # Add lagged features for 'close' and 'volume'
    for lag in [1, 3, 5, 15, 30]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'volume_lag_{lag}'] = df['volume'].shift(lag)

    # Add rolling volatility (standard deviation) for 'close'
    for window in [5, 15, 30]:
        df[f'volatility_{window}'] = df['close'].rolling(window=window).std()

    # Add percentage returns for 'close'
    for window in [5, 15, 30]:
        df[f'returns_{window}'] = df['close'].pct_change(periods=window)

    # Drop rows with NaN values (caused by rolling calculations and lagging)
    df = df.dropna().reset_index(drop=True)

     # Drop the 'timestamp' column
    df = df.drop(columns=['timestamp'])


    # Load the saved numerical scaler
    numerical_scaler = joblib.load(numerical_scaler_file)

    # Define numerical features to normalize (exclude 'year', 'month', 'day')
    numerical_features = [
        'close', 'volume', 'RSI', 'upper_band', 'lower_band',
        'middle_band', 'MACD', 'MACD_signal', 'MACD_hist',
        'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_15', 'close_lag_30',
        'volume_lag_1', 'volume_lag_3', 'volume_lag_5', 'volume_lag_15', 'volume_lag_30',
        'volatility_5', 'volatility_15', 'volatility_30',
        'returns_5', 'returns_15', 'returns_30'
    ]

    # Normalize numerical features
    df[numerical_features] = numerical_scaler.transform(df[numerical_features])

    
    # Save the processed sequence to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Live LSTM sequence saved to {output_file}")


if __name__ == '__main__':
    process_lstm_live()