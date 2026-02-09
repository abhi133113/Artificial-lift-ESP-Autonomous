"""
State-of-the-Art Anomaly Detection Models using PyOD

This script implements and compares the best anomaly detection models:
1. COPOD - Copula-Based Outlier Detection (Fast, Parameter-free)
2. ECOD - Empirical Cumulative Distribution Based (Fast, Interpretable)
3. IForest - Isolation Forest (Tree-based, Robust)
4. LOF - Local Outlier Factor (Density-based)
5. SUOD - Scalable Unsupervised Outlier Detection (Ensemble)

These are considered state-of-the-art according to research.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# PyOD models
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.suod import SUOD
from pyod.models.knn import KNN

def evaluate_model(y_true, y_pred, model_name):
    """Calculate metrics for a model."""
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Accuracy': accuracy
    }

def main():
    print("="*70)
    print("STATE-OF-THE-ART ANOMALY DETECTION MODELS COMPARISON")
    print("="*70)
    
    # Load cleaned data
    df = pd.read_csv('data/real_pump_cleaned.csv')
    print(f"\nLoaded {len(df)} samples")
    
    # Features
    features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration',
                'PIP_rolling_mean', 'PIP_rolling_std',
                'Amps_rolling_mean', 'Amps_rolling_std',
                'Vibration_rolling_mean', 'Vibration_rolling_std']
    
    # Split data
    df_normal = df[df['machine_status'] == 'NORMAL']
    df_anomaly = df[df['machine_status'] != 'NORMAL']
    
    # Training set: Only normal data
    df_train = df_normal.sample(n=2000, random_state=42)
    
    # Test set: Mixed data
    df_test = df.sample(n=1500, random_state=99)
    
    print(f"Train: {len(df_train)} normal samples")
    print(f"Test: {len(df_test)} mixed samples")
    
    # Prepare data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    X_test = scaler.transform(df_test[features])
    y_test = (df_test['machine_status'] != 'NORMAL').astype(int).values
    
    print(f"\nTest anomaly ratio: {y_test.mean()*100:.1f}%\n")
    
    # Define models
    models = {
        'COPOD (Copula-Based)': COPOD(contamination=0.1),
        'ECOD (Empirical CDF)': ECOD(contamination=0.1),
        'IForest (Isolation Forest)': IForest(contamination=0.1, random_state=42),
        'LOF (Local Outlier Factor)': LOF(n_neighbors=20, contamination=0.1),
        'KNN (K-Nearest Neighbors)': KNN(n_neighbors=10, contamination=0.1),
    }
    
    # Train and evaluate each model
    results = []
    
    print("Training and evaluating models...")
    print("-"*70)
    
    for name, model in models.items():
        print(f"  Training {name}...", end=' ')
        
        # Train on normal data only
        model.fit(X_train)
        
        # Predict on test data
        y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, name)
        results.append(metrics)
        
        print(f"F1: {metrics['F1']*100:.2f}%")
    
    # Create ensemble using SUOD
    print(f"\n  Training SUOD (Ensemble)...", end=' ')
    
    base_estimators = [
        COPOD(contamination=0.1),
        ECOD(contamination=0.1),
        IForest(contamination=0.1, random_state=42),
        LOF(n_neighbors=20, contamination=0.1),
    ]
    
    suod = SUOD(base_estimators=base_estimators, contamination=0.1, 
                combination='average', verbose=False)
    suod.fit(X_train)
    y_pred_suod = suod.predict(X_test)
    
    metrics_suod = evaluate_model(y_test, y_pred_suod, 'SUOD (Ensemble)')
    results.append(metrics_suod)
    print(f"F1: {metrics_suod['F1']*100:.2f}%")
    
    # Results table
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('F1', ascending=False)
    
    print("\n| Model | Precision | Recall | F1 Score | Accuracy |")
    print("|" + "-"*35 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*10 + "|")
    
    for _, row in df_results.iterrows():
        print(f"| {row['Model']:33} | {row['Precision']*100:8.2f}% | {row['Recall']*100:5.2f}% | {row['F1']*100:7.2f}% | {row['Accuracy']*100:7.2f}% |")
    
    # Find best model
    best_model = df_results.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   F1 Score: {best_model['F1']*100:.2f}%")
    print(f"   Precision: {best_model['Precision']*100:.2f}%")
    print(f"   Recall: {best_model['Recall']*100:.2f}%")
    
    # Save results
    with open('data/sota_models_results.txt', 'w') as f:
        f.write("STATE-OF-THE-ART ANOMALY DETECTION MODELS\n")
        f.write("="*60 + "\n\n")
        f.write("| Model | Precision | Recall | F1 Score | Accuracy |\n")
        f.write("|" + "-"*35 + "|" + "-"*11 + "|" + "-"*8 + "|" + "-"*10 + "|" + "-"*10 + "|\n")
        for _, row in df_results.iterrows():
            f.write(f"| {row['Model']:33} | {row['Precision']*100:8.2f}% | {row['Recall']*100:5.2f}% | {row['F1']*100:7.2f}% | {row['Accuracy']*100:7.2f}% |\n")
        f.write(f"\nBest Model: {best_model['Model']}\n")
        f.write(f"F1 Score: {best_model['F1']*100:.2f}%\n")
    
    print("\nResults saved to data/sota_models_results.txt")

if __name__ == "__main__":
    main()
