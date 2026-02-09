"""
Evaluate LSTM Autoencoder vs Isolation Forest

Compare performance of both models on the same test data.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import os
import sys

# Add src to path
sys.path.insert(0, 'src')

from lstm_model import ESPLSTMDetector
from model import ESPAnomalyDetector

def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print metrics."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def main():
    print("="*60)
    print("LSTM AUTOENCODER vs ISOLATION FOREST COMPARISON")
    print("="*60)
    
    # Load test data (Gas Lock with ground truth)
    df_test = pd.read_csv('data/esp_gas_lock.csv')
    
    # Ground truth: failure starts at index 1400 (70% of 2000)
    y_true = np.zeros(len(df_test))
    y_true[1400:] = 1
    
    results = {}
    
    # 1. Evaluate LSTM Autoencoder
    if os.path.exists('models/lstm_autoencoder.pth'):
        lstm_detector = ESPLSTMDetector()
        lstm_detector.load_model()
        
        scores = lstm_detector.predict_anomaly_score(df_test)
        # Align predictions with original data (account for time_steps offset)
        y_pred_lstm = np.zeros(len(df_test))
        y_pred_lstm[lstm_detector.time_steps:] = (scores > lstm_detector.threshold).astype(int)
        
        results['LSTM'] = evaluate_model(y_true, y_pred_lstm, "LSTM AUTOENCODER")
    else:
        print("LSTM model not found. Run lstm_model.py first.")
    
    # 2. Evaluate Isolation Forest
    if os.path.exists('models/esp_iso_forest.joblib'):
        iso_detector = ESPAnomalyDetector()
        iso_detector.load_model()
        
        scores = iso_detector.predict_anomaly_score(df_test)
        y_pred_iso = (scores > 0).astype(int)
        
        results['IsoForest'] = evaluate_model(y_true, y_pred_iso, "ISOLATION FOREST")
    else:
        print("Isolation Forest model not found. Run model.py first.")
    
    # Final Comparison Table
    if results:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print("\n| Model | Precision | Recall | F1 Score | Accuracy |")
        print("|-------|-----------|--------|----------|----------|")
        for name, metrics in results.items():
            print(f"| {name:15} | {metrics['precision']*100:6.2f}% | {metrics['recall']*100:5.2f}% | {metrics['f1']*100:7.2f}% | {metrics['accuracy']*100:7.2f}% |")
        
        # Save to file
        with open('data/model_comparison.txt', 'w') as f:
            f.write("MODEL COMPARISON: LSTM Autoencoder vs Isolation Forest\n")
            f.write("="*60 + "\n\n")
            f.write("| Model | Precision | Recall | F1 Score | Accuracy |\n")
            f.write("|-------|-----------|--------|----------|----------|\n")
            for name, metrics in results.items():
                f.write(f"| {name:15} | {metrics['precision']*100:6.2f}% | {metrics['recall']*100:5.2f}% | {metrics['f1']*100:7.2f}% | {metrics['accuracy']*100:7.2f}% |\n")
        print("\nResults saved to data/model_comparison.txt")

if __name__ == "__main__":
    main()
