"""
Train and Evaluate LSTM Autoencoder on Real Pump Sensor Data

This script trains the LSTM Autoencoder on the NORMAL portion of real
pump sensor data and evaluates it on the full dataset with ground truth labels.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import joblib
import os

# Import LSTM model
import sys
sys.path.insert(0, 'src')
from lstm_model import LSTMAutoencoder

class RealDataLSTMDetector:
    def __init__(self, time_steps=10, n_features=5, hidden_dim=64, latent_dim=32):
        self.time_steps = time_steps
        self.n_features = n_features
        self.features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']
        self.scaler = StandardScaler()
        self.model = LSTMAutoencoder(n_features, hidden_dim, latent_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = None
        
    def prepare_sequences(self, df, training=True):
        data = df[self.features].values
        
        if training:
            self.scaler.fit(data)
        
        scaled_data = self.scaler.transform(data)
        
        sequences = []
        for i in range(len(scaled_data) - self.time_steps):
            sequences.append(scaled_data[i:i + self.time_steps])
        
        return np.array(sequences)
    
    def train(self, df, epochs=50, batch_size=32, lr=0.001):
        print(f"Training LSTM Autoencoder on {len(df)} NORMAL samples...")
        print(f"Device: {self.device}")
        
        X = self.prepare_sequences(df, training=True)
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
        
        # Calculate threshold
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
            self.threshold = np.percentile(mse, 95)
            print(f"Threshold set at: {self.threshold:.6f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
        }, 'models/lstm_real_data.pth')
        joblib.dump(self.scaler, 'models/lstm_real_scaler.joblib')
        print("Model saved!")
        
    def predict_anomaly_score(self, df):
        X = self.prepare_sequences(df, training=False)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
        
        return mse

def main():
    print("="*60)
    print("LSTM AUTOENCODER ON REAL PUMP SENSOR DATA")
    print("="*60)
    
    # Load real data
    if not os.path.exists('data/real_pump_esp_format.csv'):
        print("Real data not found. Run download_real_data.py first.")
        return
        
    df = pd.read_csv('data/real_pump_esp_format.csv')
    print(f"\nLoaded {len(df)} samples from real pump sensor data")
    print(f"Status distribution:\n{df['machine_status'].value_counts()}\n")
    
    # Split: Train on NORMAL only, Test on mixed sample
    df_train = df[df['machine_status'] == 'NORMAL'].sample(n=2000, random_state=42)
    df_test = df.sample(n=2000, random_state=123)
    
    print(f"Training on {len(df_train)} NORMAL samples")
    print(f"Testing on {len(df_test)} samples\n")
    
    # Train LSTM
    detector = RealDataLSTMDetector()
    detector.train(df_train, epochs=30, batch_size=64)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    scores = detector.predict_anomaly_score(df_test)
    
    # Ground truth (adjusted for time_steps offset)
    y_true_full = (df_test['machine_status'] != 'NORMAL').astype(int).values
    y_true = y_true_full[detector.time_steps:]
    
    # Predictions
    y_pred = (scores > detector.threshold).astype(int)
    
    # Metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nPrecision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}")
    
    # Save results
    results = f"""
LSTM AUTOENCODER - REAL PUMP SENSOR DATA RESULTS
{'='*50}

Precision: {precision*100:.2f}%
Recall:    {recall*100:.2f}%
F1 Score:  {f1*100:.2f}%
Accuracy:  {accuracy*100:.2f}%

Confusion Matrix:
              Predicted
              Normal  Anomaly
Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}
Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}
"""
    
    with open('data/lstm_real_results.txt', 'w') as f:
        f.write(results)
    print("\nResults saved to data/lstm_real_results.txt")

if __name__ == "__main__":
    main()
