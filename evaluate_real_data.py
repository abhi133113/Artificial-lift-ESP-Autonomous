"""
Evaluate ESP Model on Real Pump Sensor Data

This script evaluates the trained Isolation Forest model on the
real pump sensor data to compare performance with synthetic data.
"""

import numpy as np
import pandas as pd
from model import ESPAnomalyDetector
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import os

def evaluate_real_data():
    # Load Model
    detector = ESPAnomalyDetector()
    if os.path.exists('models/esp_iso_forest.joblib'):
        detector.load_model()
        print("Model loaded successfully.\n")
    else:
        print("Model not found. Please train it first.")
        return

    # Load Real Data
    if not os.path.exists('data/real_pump_esp_format.csv'):
        print("Real data not found. Run download_real_data.py first.")
        return
        
    df = pd.read_csv('data/real_pump_esp_format.csv')
    
    # Use a smaller sample to avoid memory issues
    df = df.sample(n=min(2000, len(df)), random_state=42).reset_index(drop=True)
    
    print(f"Evaluating on {len(df)} samples from real pump sensor data")
    print(f"Status distribution:\n{df['machine_status'].value_counts()}\n")

    # Create ground truth labels (1 = anomaly for BROKEN/RECOVERING)
    y_true = (df['machine_status'] != 'NORMAL').astype(int).values
    
    # Predict anomaly scores
    scores = detector.predict_anomaly_score(df)
    
    # Use threshold of 0 for Isolation Forest
    threshold = 0.0
    y_pred = (scores > threshold).astype(int)
    
    # Calculate metrics
    print("=" * 50)
    print("EVALUATION RESULTS ON REAL PUMP SENSOR DATA")
    print("=" * 50)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"              Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
    
    # Compare with synthetic data results
    print("\n" + "=" * 50)
    print("COMPARISON: SYNTHETIC vs REAL DATA")
    print("=" * 50)
    print("\n| Dataset | Precision | Recall | F1 Score | Accuracy |")
    print("|---------|-----------|--------|----------|----------|")
    print(f"| User Synthetic | 82.06% | 91.50% | 86.52% | 94.30% |")
    print(f"| Real Pump Data | {precision*100:.2f}% | {recall*100:.2f}% | {f1*100:.2f}% | {accuracy*100:.2f}% |")

if __name__ == "__main__":
    evaluate_real_data()
