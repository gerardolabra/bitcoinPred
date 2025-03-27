import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_linear_regression(data_path, model_output_path, results_output_path, k_folds=5):
    """
    Trains a Linear Regression model using K-Fold Cross-Validation and evaluates it on an unseen test set.
    Saves the trained model and predictions to a CSV file.
    
    Parameters:
        data_path (str): Path to the processed dataset (e.g., btc_linear.csv).
        model_output_path (str): Path to save the trained model (e.g., linear_regression.pkl).
        results_output_path (str): Path to save the predictions and actual values (e.g., linear_results.csv).
        k_folds (int): Number of folds for K-Fold Cross-Validation.
    """
    # Load the processed dataset
    df = pd.read_csv(data_path)

    # Load the saved scaler
    close_scaler_path = "data/processed/close_scaler.pkl"  # Path to the saved close scaler
    close_scaler = joblib.load(close_scaler_path)
    print(f"Close scaler loaded from {close_scaler_path}")
    
    # Separate the last 5% of rows as the unseen test set
    test_size = int(len(df) * 0.05)
    train_df = df[:-test_size]
    test_df = df[-test_size:]
    
    # Separate features (X) and target (y) for training and testing
    X_train = train_df.drop(columns=["close"])
    y_train = train_df["close"]
    X_test = test_df.drop(columns=["close"])
    y_test = test_df["close"]
    
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

    # Inverse-transform predictions and actual values using the close scaler
    y_actual = close_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
    y_pred_original = close_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()

    # Debugging: Print the denormalized values
    print("\nDebugging: Denormalized Values")
    print("Actual Prices:", y_actual[:5])  # Print the first 5 actual prices
    print("Predicted Prices:", y_pred_original[:5])  # Print the first 5 predicted prices

    # Add year, month, and day columns to the results
    results_df = test_df[["year", "month", "day"]].copy()
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
    r2_original = r2_score(y_actual, y_pred_original)

    print("\nUnseen Test Set Results (Original Scale):")
    print(f"MAE: {mae_original:.2f}")
    print(f"RMSE: {rmse_original:.2f}")
    print(f"R²: {r2_original:.4f}")


if __name__ == "__main__":
    # Define paths
    data_path = "data/processed/btc_linear.csv"  # Path to the processed dataset
    model_output_path = "src/models/linear_regression.pkl"  # Path to save the trained model
    results_output_path = "data/processed/linear_results.csv"  # Path to save the predictions and actual values
    
    # Train the Linear Regression model
    train_linear_regression(data_path, model_output_path, results_output_path)