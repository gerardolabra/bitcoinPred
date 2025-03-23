import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_data(file_path):
    """
    Prepares the data for machine learning by adding a target column,
    splitting into training and testing sets, and scaling the features.

    Args:
        file_path (str): Path to the processed CSV file.

    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test: Scaled features and target splits.
    """
    # Load the processed data
    df = pd.read_csv(file_path)

    # Add the target column for the next day's closing price
    df['target'] = df['close'].shift(-1)

    # Drop the last row with NaN target
    df = df.dropna()

    # Drop non-numeric columns (e.g., 'timestamp') from features
    X = df.drop(columns=['target', 'timestamp'])  # Exclude 'timestamp' and 'target'
    y = df['target']

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Main execution block
if __name__ == "__main__":
    file_path = "data/processed/btc_data_processed.csv"  # Path to your processed data
    try:
        X_train, X_test, y_train, y_test = prepare_data(file_path)
        print("Data preparation completed successfully!")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

