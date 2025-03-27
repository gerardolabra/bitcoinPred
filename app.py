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

# Function to execute a script
def execute_script(script_path):
    """
    Executes a Python script.
    """
    try:
        subprocess.run(["python", script_path], check=True)
        st.success(f"Executed {script_path} successfully!")
    except Exception as e:
        st.error(f"Error executing {script_path}: {e}")

# Function to load data
@st.cache
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
model_choice = st.sidebar.selectbox("Select a Model", ["Linear Regression", "RNN", "LSTM", "XGBoost", "LightGBM", "Stochastic Simulation"])

# Fetch and process data if button is clicked
if fetch_data_button:
    # Execute all required scripts
    execute_script("src/data/process_linear_next.py")
    execute_script("src/data/process_linear_next_live.py")
    execute_script("src/data/process_lstm.py")
    execute_script("src/data/process_gradient_boosting.py")
    execute_script("src/data/fetch_data.py")
    execute_script("src/data/process_data.py")
    

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

elif model_choice == "RNN":
    st.header("RNN Predictions")
    try:
        # Load the processed data and model
        rnn_data = load_data("data/processed/btc_lstm.csv")
        rnn_model = load_model("src/models/rnn_model.h5")

        # Make predictions dynamically
        X = rnn_data.drop(columns=["close"]).values  # Use all features except the target column
        y_actual = rnn_data["close"]  # The actual target values
        X = X.reshape((X.shape[0], 30, -1))  # Adjust shape for RNN
        y_pred = rnn_model.predict(X)

        # Add 'actual' and 'predicted' columns to the DataFrame
        rnn_data["actual"] = y_actual
        rnn_data["predicted"] = y_pred.flatten()

        # Plot predictions
        plot_predictions(rnn_data["actual"], rnn_data["predicted"], "RNN: Actual vs Predicted")
        mae = mean_absolute_error(rnn_data["actual"], rnn_data["predicted"])
        rmse = np.sqrt(mean_squared_error(rnn_data["actual"], rnn_data["predicted"]))
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

elif model_choice == "LSTM":
    st.header("LSTM Predictions")
    try:
        # Load the processed data and model
        lstm_data = load_data("data/processed/btc_lstm.csv")
        lstm_model = load_model("src/models/lstm_model.h5")

        # Make predictions dynamically
        X = lstm_data.drop(columns=["close"]).values  # Use all features except the target column
        y_actual = lstm_data["close"]  # The actual target values
        X = X.reshape((X.shape[0], 30, -1))  # Adjust shape for LSTM
        y_pred = lstm_model.predict(X)

        # Add 'actual' and 'predicted' columns to the DataFrame
        lstm_data["actual"] = y_actual
        lstm_data["predicted"] = y_pred.flatten()

        # Plot predictions
        plot_predictions(lstm_data["actual"], lstm_data["predicted"], "LSTM: Actual vs Predicted")
        mae = mean_absolute_error(lstm_data["actual"], lstm_data["predicted"])
        rmse = np.sqrt(mean_squared_error(lstm_data["actual"], lstm_data["predicted"]))
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

elif model_choice == "XGBoost":
    st.header("XGBoost Predictions")
    try:
        # Load the processed data and model
        xgboost_data = load_data("data/processed/btc_gradient_boosting.csv")
        with open("src/models/xgboost_model.pkl", "rb") as f:
            xgboost_model = pickle.load(f)

        # Make predictions dynamically
        X = xgboost_data.drop(columns=["close"])  # Use all features except the target column
        y_actual = xgboost_data["close"]  # The actual target values
        y_pred = xgboost_model.predict(X)

        # Add 'actual' and 'predicted' columns to the DataFrame
        xgboost_data["actual"] = y_actual
        xgboost_data["predicted"] = y_pred

        # Plot predictions
        plot_predictions(xgboost_data["actual"], xgboost_data["predicted"], "XGBoost: Actual vs Predicted")
        mae = mean_absolute_error(xgboost_data["actual"], xgboost_data["predicted"])
        rmse = np.sqrt(mean_squared_error(xgboost_data["actual"], xgboost_data["predicted"]))
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

elif model_choice == "LightGBM":
    st.header("LightGBM Predictions")
    try:
        # Load the processed data and model
        lightgbm_data = load_data("data/processed/btc_gradient_boosting.csv")
        with open("src/models/lightgbm_model.pkl", "rb") as f:
            lightgbm_model = pickle.load(f)

        # Make predictions dynamically
        X = lightgbm_data.drop(columns=["close"])  # Use all features except the target column
        y_actual = lightgbm_data["close"]  # The actual target values
        y_pred = lightgbm_model.predict(X)

        # Add 'actual' and 'predicted' columns to the DataFrame
        lightgbm_data["actual"] = y_actual
        lightgbm_data["predicted"] = y_pred

        # Plot predictions
        plot_predictions(lightgbm_data["actual"], lightgbm_data["predicted"], "LightGBM: Actual vs Predicted")
        mae = mean_absolute_error(lightgbm_data["actual"], lightgbm_data["predicted"])
        rmse = np.sqrt(mean_squared_error(lightgbm_data["actual"], lightgbm_data["predicted"]))
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

elif model_choice == "Stochastic Simulation":
    st.header("Stochastic Simulation")
    try:
        # Load the dataset
        data = load_data("data/processed/btc_data_processed.csv")
        
        # Run the stochastic simulation
        st.write("Running stochastic simulation for the next 30 days...")
        simulate_btc_prices(data)
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")