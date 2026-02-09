import numpy as np
import pandas as pd
from model import ESPAnomalyDetector
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os

def evaluate_dataset(detector, df, failure_start_idx=None, dataset_name="Dataset"):
    print(f"\n--- Evaluating {dataset_name} ---")
    
    # 1. Predict Anomaly Scores
    scores = detector.predict_anomaly_score(df)
    
    # 2. Define Ground Truth
    y_true = np.zeros(len(df))
    if failure_start_idx is not None:
        y_true[failure_start_idx:] = 1
        
    # 3. Define Threshold (using contamination logic or statistical)
    # Here we use a fixed threshold based on training data statistics if available, 
    # or a percentile of the current scores if we assume some contamination.
    # For this test, let's assume a threshold of 0.0 (since Isolation Forest scores are centered at 0, 
    # and we inverted them: positive = anomaly, negative = normal)
    # Wait, in model.py: scores = -self.model.decision_function(X)
    # Decision function: positive for inliers, negative for outliers.
    # So -decision_function: negative for inliers, positive for outliers.
    # Threshold should be 0.
    
    threshold = 0.0
    y_pred = (scores > threshold).astype(int)
    
    # 4. Calculate Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Threshold: {threshold}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "scores": scores
    }

if __name__ == "__main__":
    # Load Model
    detector = ESPAnomalyDetector()
    if os.path.exists('models/esp_iso_forest.joblib'):
        detector.load_model()
        print("Model loaded successfully.")
    else:
        print("Model not found. Please train it first.")
        exit()

    # 1. Evaluate on My Synthetic Gas Lock Data
    # In data_generator.py: n_samples=2000, start_idx = 0.7 * 2000 = 1400
    if os.path.exists('data/esp_gas_lock.csv'):
        df_my_gas = pd.read_csv('data/esp_gas_lock.csv')
        evaluate_dataset(detector, df_my_gas, failure_start_idx=1400, dataset_name="My Synthetic Gas Lock Data")

    # 2. Evaluate on User Snippet Data (if exists)
    # In user snippet: n_samples=1000, start_idx=800
    if os.path.exists('data/user_gas_lock.csv'):
        df_user_gas = pd.read_csv('data/user_gas_lock.csv')
        evaluate_dataset(detector, df_user_gas, failure_start_idx=800, dataset_name="User Snippet Gas Lock Data")
