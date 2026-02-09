"""
Train and Save COPOD Model

Trains the COPOD (Copula-Based Outlier Detection) model on the
cleaned real pump sensor data and saves it for the dashboard.
"""

import pandas as pd
import numpy as np
from pyod.models.copod import COPOD
from sklearn.preprocessing import StandardScaler
import joblib
import os

def main():
    print("Training COPOD Model...")
    
    # Load cleaned data
    if not os.path.exists('data/real_pump_cleaned.csv'):
        print("Cleaned data not found. Run clean_data.py first.")
        return
        
    df = pd.read_csv('data/real_pump_cleaned.csv')
    
    # Filter for NORMAL data only
    df_normal = df[df['machine_status'] == 'NORMAL']
    print(f"Training on {len(df_normal)} normal samples")
    
    # Features (same as used in evaluation)
    features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration',
                'PIP_rolling_mean', 'PIP_rolling_std',
                'Amps_rolling_mean', 'Amps_rolling_std',
                'Vibration_rolling_mean', 'Vibration_rolling_std']
    
    # Prepare data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_normal[features])
    
    # Train COPOD
    # contamination=0.05 assumes ~5% anomalies in production data
    model = COPOD(contamination=0.05)
    model.fit(X_train)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/copod_model.joblib')
    joblib.dump(scaler, 'models/copod_scaler.joblib')
    
    print("Model saved to models/copod_model.joblib")
    print("Scaler saved to models/copod_scaler.joblib")

if __name__ == "__main__":
    main()
