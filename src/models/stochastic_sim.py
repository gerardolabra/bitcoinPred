import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st 

def load_data(file_path="data/processed/btc_data_processed.csv"):
    """Load the dataset into a pandas DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found. Run fetch_data.py first.")
    
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])  # Convert timestamps
    return df

import streamlit as st  # Import Streamlit for rendering plots

def simulate_btc_prices(df):
    """
    Simulates Bitcoin prices for the next 30 days using a stochastic model.

    Args:
        df (pd.DataFrame): DataFrame containing historical BTC prices.

    Returns:
        np.ndarray: Simulated price paths.
        float: Probability of >20% drop in 30 days.
        float: Probability of >20% increase in 30 days.
    """
    # Last 30 days for params
    recent = df.tail(30)
    S0 = recent['close'].iloc[-1]  # Latest price
    mu = recent['close'].pct_change().mean()  # Drift
    sigma = recent['close'].pct_change().std()  # Volatility

    # Simulation parameters
    T = 30  # 30 days forward
    n_paths = 100
    dt = 1

    # Simulate
    np.random.seed(42)
    prices = np.zeros((T, n_paths))
    prices[0] = S0

    for t in range(1, T):
        rand = np.random.normal(0, 1, n_paths)
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)

    # Stats
    final_prices = prices[-1]
    drop_prob = np.mean(final_prices < S0 * 0.8) * 100  # Probability of >20% drop
    increase_prob = np.mean(final_prices > S0 * 1.2) * 100  # Probability of >20% increase

    # Plot
    plt.figure(figsize=(14, 7))
    for i in range(n_paths):
        plt.plot(prices[:, i], lw=1, alpha=0.5)
    plt.axhline(S0 * 0.8, color='r', linestyle='--', label='20% Drop')
    plt.axhline(S0 * 1.2, color='g', linestyle='--', label='20% Increase')
    plt.title(f"BTC Price Simulation (S0={S0:.0f}, mu={mu:.4f}, sigma={sigma:.4f})")
    plt.xlabel("Days")
    plt.ylabel("Price (USD)")
    plt.legend()

    # Render the plot in Streamlit
    st.pyplot(plt)

    # Return probabilities
    return prices, drop_prob, increase_prob