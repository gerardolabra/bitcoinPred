import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import MACD
import joblib

def process_linear():
    """
    Processes raw Bitcoin data to generate features for linear regression, including
    technical indicators, lagged features, rolling statistics, and normalized data.
    Saves the processed data to a CSV file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '../../data/raw/btc_data.csv')
    output_dir = os.path.join(script_dir, '../../data/processed')
    output_file = os.path.join(output_dir, 'btc_linear.csv')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the raw data
    df = pd.read_csv(input_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp to datetime
    df = df.sort_values(by='timestamp')  # Ensure data is sorted by time
    
    # Extract numerical features from the timestamp
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day

    # Add returns
    df['returns'] = df['close'].pct_change()
    
    # Add moving averages and volatility
    df['moving_avg_7'] = df['close'].rolling(window=7).mean()
    df['moving_avg_30'] = df['close'].rolling(window=30).mean()
    df['volatility_7'] = df['returns'].rolling(window=7).std()
    df['volatility_30'] = df['returns'].rolling(window=30).std()
    
    # Add rolling mean and standard deviation for additional windows
    df['rolling_mean_14'] = df['close'].rolling(window=14).mean()
    df['rolling_std_14'] = df['close'].rolling(window=14).std()
    df['rolling_mean_30'] = df['close'].rolling(window=30).mean()
    df['rolling_std_30'] = df['close'].rolling(window=30).std()
    
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
    for lag in range(1, 4):  # Add lagged features for the last 3 days
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
    
    # Drop the original 'timestamp' column after extracting features
    df = df.drop(columns=["timestamp"])
    
    # Drop rows with NaN values (caused by rolling calculations and lagging)
    df = df.dropna().reset_index(drop=True)

    # Save a separate scaler for the 'close' column BEFORE normalization
    close_scaler = MinMaxScaler()
    close_scaler.fit(df[['close']])  # Fit the scaler to the original 'close' column
    close_scaler_path = os.path.join(output_dir, 'close_scaler.pkl')
    joblib.dump(close_scaler, close_scaler_path)
    print(f"Close scaler saved to {close_scaler_path}")

    # Normalize numerical features using Min-Max scaling
    scaler = MinMaxScaler()
    numerical_features = [
        'close', 'returns', 'moving_avg_7', 'moving_avg_30', 'volatility_7', 'volatility_30',
        'rolling_mean_14', 'rolling_std_14', 'rolling_mean_30', 'rolling_std_30',
        'RSI', 'upper_band', 'lower_band', 'middle_band', 'MACD', 'MACD_signal', 'MACD_hist',
        'close_lag_1', 'close_lag_2', 'close_lag_3', 'returns_lag_1', 'returns_lag_2', 'returns_lag_3'
    ]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Save the scaler for all numerical features
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"Data processed and saved to {output_file}")


if __name__ == '__main__':
    process_linear()