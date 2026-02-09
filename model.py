import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ESPAnomalyDetector:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']

    def prepare_data(self, df, training=True):
        data = df[self.features].values
        
        if training:
            self.scaler.fit(data)
            
        scaled_data = self.scaler.transform(data)
        return scaled_data

    def train(self, df):
        X = self.prepare_data(df, training=True)
        self.model.fit(X)
        
        # Save model and scaler
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/esp_iso_forest.joblib')
        joblib.dump(self.scaler, 'models/scaler.joblib')
        print("Model and scaler saved to models/")

    def load_model(self):
        self.model = joblib.load('models/esp_iso_forest.joblib')
        self.scaler = joblib.load('models/scaler.joblib')

    def predict_anomaly_score(self, df):
        X = self.prepare_data(df, training=False)
        # Isolation Forest decision_function returns the anomaly score (lower is more anomalous)
        # We invert it so higher is more anomalous
        scores = -self.model.decision_function(X)
        # Normalize scores to be roughly between 0 and 1 for easier thresholding
        # (Note: Isolation Forest scores are centered around 0)
        return scores

if __name__ == "__main__":
    # Load normal data for training
    if os.path.exists('data/esp_normal.csv'):
        df_normal = pd.read_csv('data/esp_normal.csv')
        detector = ESPAnomalyDetector()
        print("Training Isolation Forest...")
        detector.train(df_normal)
        
        # Test on Gas Lock data
        df_gas_lock = pd.read_csv('data/esp_gas_lock.csv')
        scores = detector.predict_anomaly_score(df_gas_lock)
        print(f"Max Anomaly Score (Gas Lock): {np.max(scores):.4f}")
        print(f"Mean Anomaly Score (Gas Lock): {np.mean(scores):.4f}")
    else:
        print("Normal data not found. Run data_generator.py first.")
