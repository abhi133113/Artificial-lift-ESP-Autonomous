"""
Train and Evaluate Model on Real Pump Sensor Data

This script trains a new Isolation Forest model on the "NORMAL" portion
of the real pump sensor data, then evaluates it to get fair metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import joblib
import os

def main():
    # Load Real Data
    if not os.path.exists('data/real_pump_esp_format.csv'):
        print("Real data not found. Run download_real_data.py first.")
        return
        
    df = pd.read_csv('data/real_pump_esp_format.csv')
    print(f"Loaded {len(df)} samples from real pump sensor data")
    print(f"Status distribution:\n{df['machine_status'].value_counts()}\n")
    
    # Split into train (NORMAL only) and test (all)
    features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']
    
    df_train = df[df['machine_status'] == 'NORMAL'].sample(n=2000, random_state=42)
    df_test = df.sample(n=2000, random_state=123)  # Mixed sample for testing
    
    print(f"Training on {len(df_train)} NORMAL samples")
    print(f"Testing on {len(df_test)} samples (mixed status)\n")
    
    # Prepare training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    
    # Train Isolation Forest
    print("Training Isolation Forest on REAL normal data...")
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X_train)
    print("Training complete!\n")
    
    # Evaluate on test set
    X_test = scaler.transform(df_test[features])
    
    # Get predictions (-1 for anomaly, 1 for normal in sklearn)
    y_pred_raw = model.predict(X_test)
    y_pred = (y_pred_raw == -1).astype(int)  # Convert to 1 for anomaly
    
    # Ground truth
    y_true = (df_test['machine_status'] != 'NORMAL').astype(int).values
    
    # Calculate metrics
    print("=" * 50)
    print("EVALUATION RESULTS (Model Trained on Real Data)")
    print("=" * 50)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
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
    
    # Save the retrained model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/esp_iso_forest_real.joblib')
    joblib.dump(scaler, 'models/scaler_real.joblib')
    print("\nRetrained model saved to models/esp_iso_forest_real.joblib")
    
    # Final comparison
    print("\n" + "=" * 50)
    print("FINAL COMPARISON: SYNTHETIC vs REAL DATA")
    print("=" * 50)
    print("\n| Dataset | Precision | Recall | F1 Score | Accuracy |")
    print("|---------|-----------|--------|----------|----------|")
    print(f"| Synthetic (User) | 82.06% | 91.50% | 86.52% | 94.30% |")
    print(f"| Real Pump Data   | {precision*100:.2f}% | {recall*100:.2f}% | {f1*100:.2f}% | {accuracy*100:.2f}% |")
    
    # Save results to file
    with open('data/evaluation_results.txt', 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("EVALUATION RESULTS (Model Trained on Real Data)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Precision: {precision:.4f} ({precision*100:.2f}%)\n")
        f.write(f"Recall:    {recall:.4f} ({recall*100:.2f}%)\n")
        f.write(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)\n")
        f.write(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"              Predicted\n")
        f.write(f"              Normal  Anomaly\n")
        f.write(f"Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}\n")
        f.write(f"Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}\n\n")
        f.write("=" * 50 + "\n")
        f.write("COMPARISON: SYNTHETIC vs REAL DATA\n")
        f.write("=" * 50 + "\n\n")
        f.write("| Dataset | Precision | Recall | F1 Score | Accuracy |\n")
        f.write("|---------|-----------|--------|----------|----------|\n")
        f.write(f"| Synthetic (User) | 82.06% | 91.50% | 86.52% | 94.30% |\n")
        f.write(f"| Real Pump Data   | {precision*100:.2f}% | {recall*100:.2f}% | {f1*100:.2f}% | {accuracy*100:.2f}% |\n")
    print("\nResults saved to data/evaluation_results.txt")

if __name__ == "__main__":
    main()
