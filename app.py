import streamlit as st
import pandas as pd
import subprocess
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Import your models and stochastic simulation function
from src.models.stochastic_sim import simulate_btc_prices
from tensorflow.keras.models import load_model
import pickle

import subprocess
import sys

def check_installed_packages():
    required_packages = ["requests", "pandas"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Package {package} is missing. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", package])

check_installed_packages()



# Function to execute a script
def execute_script(script_path):
    import sys
    print(f"Executing script with Python: {sys.executable}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],  # Use sys.executable to ensure the correct Python interpreter is used
            check=True,
            capture_output=True,
            text=True
        )
        st.success(f"Executed {script_path} successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Error executing {script_path}: {e.stderr}")

# Function to load data
def load_data(file_path):
    """
    Loads the processed data from a CSV file.
    """
    return pd.read_csv(file_path)

# Function to plot predictions
def plot_predictions(actual, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual Prices", color="blue")
    plt.plot(predicted, label="Predicted Prices", color="orange")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(plt)

# Streamlit app
st.title("Bitcoin Price Prediction App")
st.sidebar.title("Options")

# Sidebar options
st.sidebar.header("Actions")
fetch_data_button = st.sidebar.button("Fetch and Process Data")
model_choice = st.sidebar.selectbox("Select a Model", ["LSTM", "Stochastic Simulation", "Linear Regression"])

# Fetch and process data if button is clicked
if fetch_data_button:
    # Execute all required scripts
    execute_script("src/save_data.py")
    execute_script("src/data/process_data.py")
    execute_script("src/data/process_linear_next.py")
    execute_script("src/data/process_linear_next_live.py")
    execute_script("src/data/process_lstm.py")
    execute_script("src/data/process_lstm_live.py")
    
    

# Main section
if model_choice == "Linear Regression":
    st.header("Linear Regression Predictions")
    try:
        # Load the processed features for live prediction
        live_data = load_data("data/processed/btc_linear_next_live.csv")

        # Load the training dataset to get the column structure
        training_data = load_data("data/processed/btc_linear_next.csv")

        # Load the trained model
        linear_model = joblib.load("src/models/linear_regression_next.pkl")

        # Load the scaler used for the 'close' column
        close_scaler = joblib.load("data/processed/close_scaler_next.pkl")

        # --- Live Prediction ---
        # Ensure the live data has the same feature order as the training data (excluding next_day_close)
        latest_features = live_data.iloc[-1][training_data.columns.drop("next_day_close")].values.reshape(1, -1)

        # Predict the next day's close price
        predicted_next_day_close = linear_model.predict(latest_features)[0]

        # Denormalize the predicted value
        denormalized_price = close_scaler.inverse_transform([[predicted_next_day_close]])[0][0]
        print(f"Denormalized Predicted Price: {denormalized_price}")

        # Extract the latest date from the live dataset
        latest_row = live_data.iloc[-1]
        latest_year = int(latest_row["year"])
        latest_month = int(latest_row["month"])
        latest_day = int(latest_row["day"])

        # Display the live prediction with the date
        st.subheader("Predicted Next Day Close Price")
        st.write(
            f"The predicted close price for tomorrow, using Binance information up to "
            f"**{latest_year}-{latest_month:02d}-{latest_day:02d}**, is: **{predicted_next_day_close:.2f}**"
        )

        # --- Historical Evaluation ---
        # Use the last 7 rows for evaluation
        N = 7  # Number of rows to compare
        if len(live_data) >= N:
            # Load the scaler for the 'close' column
            close_scaler_path = "data/processed/close_scaler_next.pkl"  # Use the correct scaler
            close_scaler = joblib.load(close_scaler_path)

            # Load the scaler for all numerical features (excluding 'close')
            scaler_path = "data/processed/scaler_next.pkl"  # Use the correct scaler
            scaler = joblib.load(scaler_path)

            # Extract the last 7 rows of the 'close' column as y_actual_normalized
            y_actual_normalized = live_data["close"].iloc[-N:]  # Normalized actual close prices

            # Denormalize the actual close prices
            y_actual = close_scaler.inverse_transform(y_actual_normalized.values.reshape(-1, 1)).flatten()

            # Ensure the 'close' column is included in the feature set
            X_live = live_data[training_data.columns.drop("next_day_close")].iloc[-N:]

            # Predict the close prices
            y_pred = linear_model.predict(X_live)

            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)

            # Display metrics
            st.subheader("Model Performance on Historical Data (Last 7 Rows)")
            st.write(f"Mean Absolute Error (MAE): **{mae:.2f}**")
            st.write(f"Root Mean Squared Error (RMSE): **{rmse:.2f}**")
            st.write(f"R² Score: **{r2:.2f}**")

            # Plot predictions vs actual values
            st.subheader("Prediction vs Actual (Last 7 Days)")
            plot_predictions(y_actual, y_pred, "Linear Regression: Actual vs Predicted")

    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

elif model_choice == "LSTM":
    st.header("LSTM Predictions")
    try:
        # Load the processed live data and the trained LSTM model
        live_data = load_data("data/processed/btc_lstm_live.csv")
        lstm_model = load_model("src/models/lstm_daily_model.keras")

        # Define the features used during training
        features = [
            'close', 'volume', 'RSI', 'upper_band', 'lower_band',
            'middle_band', 'MACD', 'MACD_signal', 'MACD_hist',
            'close_lag_1', 'close_lag_3', 'close_lag_5', 'close_lag_15', 'close_lag_30',
            'volume_lag_1', 'volume_lag_3', 'volume_lag_5', 'volume_lag_15', 'volume_lag_30',
            'volatility_5', 'volatility_15', 'volatility_30',
            'returns_5', 'returns_15', 'returns_30'
        ]

        # Create a DataFrame for extracting the latest date (includes year, month, day)
        full_data = live_data.copy()

        # Create a DataFrame for feeding the model (only includes features)
        live_data = live_data[features]

        # Check if there are enough rows for the window size
        window_size = 30  # Ensure this matches the window size used during training
        num_features = len(features)  # Number of features in the live data
        if len(live_data) < window_size:
            st.error(
                f"Not enough data for prediction. The live data contains only {len(live_data)} rows, "
                f"but at least {window_size} rows are required. Please wait for more data to accumulate."
            )
        else:
            # Extract the last `window_size` rows for prediction
            live_sequence = live_data.iloc[-window_size:].values.reshape(1, window_size, num_features)

            # Make the prediction
            predicted_next_day_close = lstm_model.predict(live_sequence)[0][0]

            # Load the target scaler to denormalize the prediction
            target_scaler = joblib.load("data/processed/target_scaler.pkl")
            denormalized_price = target_scaler.inverse_transform([[predicted_next_day_close]])[0][0]

            # Extract the latest date from the full dataset
            latest_row = full_data.iloc[-1]
            latest_year = int(latest_row["year"])
            latest_month = int(latest_row["month"])
            latest_day = int(latest_row["day"])

            # Display the live prediction with the date
            st.subheader("Predicted Next Day Close Price")
            st.write(
                f"The predicted close price for tomorrow, using Binance information up to "
                f"**{latest_year}-{latest_month:02d}-{latest_day:02d}**, is: **${denormalized_price:.2f}**"
            )

            # --- Historical Evaluation ---
            # Use the last 180 rows for evaluation
            N = 180  # Number of rows to compare

            # Load the btc_daily.csv file for historical evaluation
            btc_daily = load_data("data/processed/btc_daily.csv")

            if len(btc_daily) >= N + window_size:  # Ensure enough rows for the sliding window
                # Extract the last N + window_size rows for evaluation
                evaluation_data = btc_daily.iloc[-(N + window_size):]

                # Prepare features and target for evaluation
                X_live = []
                y_actual = []
                for i in range(len(evaluation_data) - window_size):
                    # Use only the features for the input sequence
                    X_live.append(evaluation_data[features].iloc[i:i + window_size].values)
                    # Use the 'next_day_close' column for the target
                    y_actual.append(evaluation_data.iloc[i + window_size]["next_day_close"])

                # Convert to numpy arrays
                X_live = np.array(X_live)
                y_actual = np.array(y_actual)

                # Make predictions
                y_pred_normalized = lstm_model.predict(X_live).flatten()

                # Denormalize the predictions and actual values
                target_scaler = joblib.load("data/processed/target_scaler.pkl")
                y_pred = target_scaler.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()
                y_actual = target_scaler.inverse_transform(y_actual.reshape(-1, 1)).flatten()

                # Calculate metrics
                mae = mean_absolute_error(y_actual, y_pred)
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                r2 = r2_score(y_actual, y_pred)

                # Display metrics
                st.subheader("Model Performance on Historical Data (Last 180 Rows)")
                st.write(f"Mean Absolute Error (MAE): **{mae:.2f}**")
                st.write(f"Root Mean Squared Error (RMSE): **{rmse:.2f}**")
                st.write(f"R² Score: **{r2:.2f}**")

                # Plot predictions vs actual values
                st.subheader("Prediction vs Actual (Last 180 Days)")
                plot_predictions(y_actual, y_pred, "LSTM: Actual vs Predicted")
            else:
                st.error(
                    f"Not enough data for historical evaluation. The btc_daily dataset contains only {len(btc_daily)} rows, "
                    f"but at least {N + window_size} rows are required."
                )

    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

elif model_choice == "Stochastic Simulation":
    st.header("Stochastic Simulation")
    try:
        # Load the dataset
        data = load_data("data/processed/btc_data_processed.csv")
        
        # Extract the latest date directly from the 'timestamp' column
        latest_date = data.iloc[-1]["timestamp"]  # The latest date is already in 'YYYY-MM-DD' format

        # Display the description with the latest date
        st.write(
            f"Stochastic Simulator using Geometric Brownian Motion, fed with Binance information dated "
            f"**{latest_date}**."
        )

        # Run the stochastic simulation
        st.write("Running Stochastic Simulation for the next 30 days...")
        prices, drop_prob, increase_prob = simulate_btc_prices(data)

        # Display the probabilities
        st.subheader("Simulation Results")
        st.write(f"Probability of >20% drop in 30 days: **{drop_prob:.2f}%**")
        st.write(f"Probability of >20% increase in 30 days: **{increase_prob:.2f}%**")

    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")