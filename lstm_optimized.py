"""
Optimized LSTM Autoencoder for Real Pump Sensor Data

Improvements:
- More training epochs (50)
- Better threshold selection using validation data
- Optimized for real pump sensor characteristics
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
import sys

sys.path.insert(0, 'src')
from lstm_model import LSTMAutoencoder

class OptimizedLSTMDetector:
    def __init__(self, time_steps=10, n_features=5):
        self.time_steps = time_steps
        self.n_features = n_features
        self.features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']
        self.scaler = StandardScaler()
        self.model = LSTMAutoencoder(n_features, hidden_dim=64, latent_dim=32)
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
    
    def train(self, df_train, df_val_normal, df_val_anomaly, epochs=50, batch_size=32):
        print(f"Training LSTM on {len(df_train)} NORMAL samples...")
        print(f"Validation: {len(df_val_normal)} normal, {len(df_val_anomaly)} anomaly")
        print(f"Device: {self.device}\n")
        
        # Prepare training data
        X_train = self.prepare_sequences(df_train, training=True)
        X_tensor = torch.FloatTensor(X_train)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training
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
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
        
        # Find optimal threshold using validation data
        print("\nFinding optimal threshold...")
        self.model.eval()
        
        with torch.no_grad():
            # Get scores for normal validation data
            X_val_normal = self.prepare_sequences(df_val_normal, training=False)
            X_val_normal_tensor = torch.FloatTensor(X_val_normal).to(self.device)
            recon_normal = self.model(X_val_normal_tensor)
            scores_normal = torch.mean((X_val_normal_tensor - recon_normal) ** 2, dim=(1, 2)).cpu().numpy()
            
            # Get scores for anomaly validation data
            X_val_anom = self.prepare_sequences(df_val_anomaly, training=False)
            X_val_anom_tensor = torch.FloatTensor(X_val_anom).to(self.device)
            recon_anom = self.model(X_val_anom_tensor)
            scores_anom = torch.mean((X_val_anom_tensor - recon_anom) ** 2, dim=(1, 2)).cpu().numpy()
        
        # Find threshold that maximizes F1-score
        best_f1 = 0
        best_threshold = 0
        
        for percentile in range(80, 100):
            threshold = np.percentile(scores_normal, percentile)
            y_true = np.concatenate([np.zeros(len(scores_normal)), np.ones(len(scores_anom))])
            y_pred = np.concatenate([
                (scores_normal > threshold).astype(int),
                (scores_anom > threshold).astype(int)
            ])
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.threshold = best_threshold
        print(f"Optimal threshold: {self.threshold:.6f} (F1: {best_f1:.4f})")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
        }, 'models/lstm_optimized.pth')
        joblib.dump(self.scaler, 'models/lstm_optimized_scaler.joblib')
        print("Model saved!")
        
    def predict(self, df):
        X = self.prepare_sequences(df, training=False)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            recon = self.model(X_tensor)
            scores = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).cpu().numpy()
        
        return scores, (scores > self.threshold).astype(int)

def main():
    print("="*60)
    print("OPTIMIZED LSTM AUTOENCODER - REAL PUMP DATA")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data/real_pump_esp_format.csv')
    print(f"\nTotal samples: {len(df)}")
    print(f"Status distribution:\n{df['machine_status'].value_counts()}\n")
    
    # Split data
    df_normal = df[df['machine_status'] == 'NORMAL']
    df_anomaly = df[df['machine_status'] != 'NORMAL']
    
    # Training: 70% of normal data
    df_train = df_normal.sample(n=int(len(df_normal)*0.7), random_state=42)
    
    # Validation: remaining 30% normal + all anomaly
    df_val_normal = df_normal.drop(df_train.index).sample(n=500, random_state=42)
    df_val_anomaly = df_anomaly.sample(n=min(200, len(df_anomaly)), random_state=42)
    
    # Test: Fresh sample
    df_test = df.sample(n=1500, random_state=99)
    
    print(f"Train: {len(df_train)} normal samples")
    print(f"Val: {len(df_val_normal)} normal, {len(df_val_anomaly)} anomaly")
    print(f"Test: {len(df_test)} mixed samples\n")
    
    # Train
    detector = OptimizedLSTMDetector()
    detector.train(df_train, df_val_normal, df_val_anomaly, epochs=50)
    
    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    
    scores, y_pred = detector.predict(df_test)
    
    # Ground truth
    y_true = (df_test['machine_status'] != 'NORMAL').values[detector.time_steps:]
    
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
    with open('data/lstm_final_results.txt', 'w') as f:
        f.write("LSTM AUTOENCODER - REAL PUMP DATA (OPTIMIZED)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall:    {recall*100:.2f}%\n")
        f.write(f"F1 Score:  {f1*100:.2f}%\n")
        f.write(f"Accuracy:  {accuracy*100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"              Predicted\n")
        f.write(f"              Normal  Anomaly\n")
        f.write(f"Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}\n")
        f.write(f"Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}\n")
    
    print("\nResults saved to data/lstm_final_results.txt")

if __name__ == "__main__":
    main()
