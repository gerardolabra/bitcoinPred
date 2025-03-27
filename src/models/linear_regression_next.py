import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_linear_regression_next(data_path, model_output_path, results_output_path, k_folds=5):
    """
    Trains a Linear Regression model for next-day prediction using K-Fold Cross-Validation.
    Evaluates the model on an unseen test set and saves the trained model and predictions to a CSV file.
    """
    # Load the processed dataset (already normalized)
    df = pd.read_csv(data_path)

    # Debug: Check if the 'close' column is already normalized
    print("First 5 Rows of 'close' Column in Training Data:")
    print(df["close"].head())

    # Check if the 'close' column is already normalized
    if df["close"].max() > 1 or df["close"].min() < 0:
        print("The 'close' column is not normalized. Ensure preprocessing is correct.")
    else:
        print("The 'close' column is already normalized.")

    # Load the saved scalers for denormalization
    close_scaler_path = "data/processed/close_scaler_next.pkl"  # Path to the saved close scaler
    close_scaler = joblib.load(close_scaler_path)
    print(f"Close scaler loaded from {close_scaler_path}")

    # Debug: Verify the close scaler
    print("Close Scaler Type:", type(close_scaler))
    print("Close Scaler Attributes:")
    print(f"Min: {close_scaler.data_min_}, Max: {close_scaler.data_max_}")

    # Separate the last 5% of rows as the unseen test set
    test_size = int(len(df) * 0.05)
    train_df = df[:-test_size]
    test_df = df[-test_size:]

    # Separate features (X) and target (y) for training and testing
    X_train = train_df.drop(columns=["next_day_close"])
    y_train = train_df["next_day_close"]
    X_test = test_df.drop(columns=["next_day_close"])
    y_test = test_df["next_day_close"]

    # Initialize K-Fold Cross-Validation
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    # Initialize metrics
    mae_scores = []
    rmse_scores = []
    r2_scores = []

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Perform K-Fold Cross-Validation
    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        print(f"Training fold {fold + 1}/{k_folds}...")

        # Split the data into training and validation sets for this fold
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train the model
        model.fit(X_fold_train, y_fold_train)

        # Make predictions
        y_pred = model.predict(X_fold_val)

        # Calculate metrics
        mae = mean_absolute_error(y_fold_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
        r2 = r2_score(y_fold_val, y_pred)

        # Store metrics
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        print(f"Fold {fold + 1} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

    # Print average metrics
    print("\nK-Fold Cross-Validation Results:")
    print(f"Average MAE: {np.mean(mae_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Average R²: {np.mean(r2_scores):.4f}")

    # Train the model on the entire training dataset
    print("\nTraining the final model on the entire training dataset...")
    model.fit(X_train, y_train)

    # Save the trained model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f:
        joblib.dump(model, f)
    print(f"Trained model saved to {model_output_path}")

    # Evaluate the model on the unseen test set
    print("\nEvaluating the model on the unseen test set...")
    y_test_pred = model.predict(X_test)

    # Debug: Print raw predictions
    print("Raw Predictions (First 5):", y_test_pred[:5])

    # Normalize predictions to the range [0, 1] using the scaler
    y_test_pred_normalized = (y_test_pred - close_scaler.data_min_) / (close_scaler.data_max_ - close_scaler.data_min_)

    # Debug: Print normalized predictions
    print("Normalized Predictions (First 5):", y_test_pred_normalized[:5])

    # Denormalize predictions using the scaler
    y_pred_original = close_scaler.inverse_transform(y_test_pred_normalized.reshape(-1, 1)).flatten()

    # Debug: Print denormalized predictions
    print("Denormalized Predictions (First 5):", y_pred_original[:5])

    # Use actual values directly (no denormalization needed)
    y_actual = y_test.values

    # Debug: Print actual values
    print("Actual Values (First 5):", y_actual[:5])

    # Add year, month, and day columns to the results
    results_df = test_df[["year", "month", "day"]].copy()
    results_df["actual"] = y_actual
    results_df["predicted"] = y_pred_original

    # Debugging: Print the DataFrame before saving
    print("\nDebugging: Results DataFrame")
    print(results_df.head())  # Print the first 5 rows of the DataFrame

    # Save results to a CSV file
    results_df.to_csv(results_output_path, index=False)
    print(f"Results saved to {results_output_path}")

    # Calculate metrics on the original scale
    mae_original = mean_absolute_error(y_actual, y_pred_original)
    rmse_original = np.sqrt(mean_squared_error(y_actual, y_pred_original))
    r2_original = r2_score(y_actual, y_pred_original)

    print("\nUnseen Test Set Results (Original Scale):")
    print(f"MAE: {mae_original:.2f}")
    print(f"RMSE: {rmse_original:.2f}")
    print(f"R²: {r2_original:.4f}")


if __name__ == "__main__":
    # Define paths
    data_path = "data/processed/btc_linear_next.csv"  # Path to the processed dataset
    model_output_path = "src/models/linear_regression_next.pkl"  # Path to save the trained model
    results_output_path = "data/processed/linear_results_next.csv"  # Path to save the predictions and actual values

    # Train the Linear Regression model for next-day prediction
    train_linear_regression_next(data_path, model_output_path, results_output_path)