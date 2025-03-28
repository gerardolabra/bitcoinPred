import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_lstm_model(window_size=1500, epochs=100, batch_size=8):
    """
    Trains an LSTM model for next-day Bitcoin price prediction and evaluates it on an unseen test set.

    Args:
        window_size (int): The number of timesteps in each input sequence.
        epochs (int): The number of training epochs.
        batch_size (int): The batch size for training.

    Saves:
        The trained LSTM model as 'lstm_daily_model.keras'.
        The predictions and actual values as 'lstm_results.csv'.
    """
    # Paths to the processed data and scalers
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, '../../data/processed/btc_daily.csv')
    numerical_scaler_file = os.path.join(script_dir, '../../data/processed/numerical_scaler.pkl')
    target_scaler_file = os.path.join(script_dir, '../../data/processed/target_scaler.pkl')
    model_output_path = os.path.join(script_dir, 'lstm_daily_model.keras')
    results_output_path = os.path.join(script_dir, '../../data/processed/lstm_results.csv')

    # Load the processed daily data
    df = pd.read_csv(input_file)

    # Load the scalers
    numerical_scaler = joblib.load(numerical_scaler_file)
    target_scaler = joblib.load(target_scaler_file)

    # Features and target
    features = [
        'close', 'volume', 'RSI', 'upper_band', 'lower_band',
        'middle_band', 'MACD', 'MACD_signal', 'MACD_hist',
        'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_15', 'close_lag_30',
        'volume_lag_1', 'volume_lag_3', 'volume_lag_5', 'volume_lag_15', 'volume_lag_30',
        'volatility_5', 'volatility_15', 'volatility_30',
        'returns_5', 'returns_15', 'returns_30'
    ]
    target = 'next_day_close'

    # Prepare sequences for LSTM
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df[features].iloc[i:i+window_size].values)  # Sequence of features
        y.append(df[target].iloc[i+window_size])  # Target value after the sequence

    X = np.array(X)
    y = np.array(y)

    # Split the data into training and unseen test sets
    test_size = int(len(X) * 0.05)  # Use the last 5% of data as the unseen test set
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    # Further split the training data into training and validation sets
    split_index = int(0.8 * len(X_train))  # 80% training, 20% validation
    X_train, X_val = X_train[:split_index], X_train[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]

    # Build the LSTM model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(1)  # Output layer for predicting the next day's price
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the trained model
    model.save(model_output_path)
    print(f"Trained LSTM model saved to {model_output_path}")

    # Evaluate the model on the unseen test set
    print("\nEvaluating the model on the unseen test set...")
    y_test_pred = model.predict(X_test)

    # Denormalize predictions and actual values
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_original = target_scaler.inverse_transform(y_test_pred).flatten()

    # Debugging: Print the denormalized values
    print("\nDebugging: Denormalized Values")
    print("Actual Prices:", y_actual[:5])  # Print the first 5 actual prices
    print("Predicted Prices:", y_pred_original[:5])  # Print the first 5 predicted prices

    # Add year, month, and day columns to the results
    results_df = df.iloc[-test_size:][["year", "month", "day"]].copy()
    results_df["actual"] = y_actual  # Use denormalized actual prices
    results_df["predicted"] = y_pred_original  # Use denormalized predicted prices

    # Debugging: Print the DataFrame before saving
    print("\nDebugging: Results DataFrame")
    print(results_df.head())  # Print the first 5 rows of the DataFrame

    # Save results to a CSV file
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}")

    # Calculate metrics on the original scale
    mae_original = mean_absolute_error(y_actual, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_actual, y_pred_original))
    
    print("\nUnseen Test Set Results (Original Scale):")
    print(f"MAE: {mae_original:.2f}")
    print(f"RMSE: {rmse_original:.2f}")
    
    return history


if __name__ == '__main__':
    train_lstm_model()