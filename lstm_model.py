"""
LSTM Autoencoder for ESP Anomaly Detection

This module implements an LSTM-based Autoencoder for detecting anomalies
in ESP sensor data. The model learns to reconstruct "normal" patterns,
and high reconstruction error indicates anomalies.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import os

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder Architecture:
    
    Encoder: LSTM(64) -> LSTM(32) -> Latent Space
    Decoder: LSTM(32) -> LSTM(64) -> Dense(n_features)
    """
    def __init__(self, n_features, hidden_dim=64, latent_dim=32):
        super(LSTMAutoencoder, self).__init__()
        
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_lstm1 = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.encoder_lstm2 = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        
        # Decoder
        self.decoder_lstm1 = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, n_features)
        
    def forward(self, x):
        # Encode
        x, _ = self.encoder_lstm1(x)
        x, (hidden, cell) = self.encoder_lstm2(x)
        
        # Decode (use last hidden state repeated for sequence)
        x, _ = self.decoder_lstm1(x)
        x, _ = self.decoder_lstm2(x)
        x = self.output_layer(x)
        
        return x


class ESPLSTMDetector:
    """
    ESP Anomaly Detector using LSTM Autoencoder.
    
    Workflow:
    1. Train on "normal" data to learn reconstruction
    2. Predict: High reconstruction error = Anomaly
    """
    def __init__(self, time_steps=10, n_features=5, hidden_dim=64, latent_dim=32):
        self.time_steps = time_steps
        self.n_features = n_features
        self.features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration']
        self.scaler = StandardScaler()
        self.model = LSTMAutoencoder(n_features, hidden_dim, latent_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = None
        
    def prepare_sequences(self, df, training=True):
        """Convert dataframe to sequences for LSTM."""
        data = df[self.features].values
        
        if training:
            self.scaler.fit(data)
        
        scaled_data = self.scaler.transform(data)
        
        sequences = []
        for i in range(len(scaled_data) - self.time_steps):
            sequences.append(scaled_data[i:i + self.time_steps])
        
        return np.array(sequences)
    
    def train(self, df, epochs=50, batch_size=32, lr=0.001):
        """Train the LSTM Autoencoder on normal data."""
        print(f"Training LSTM Autoencoder on {len(df)} samples...")
        print(f"Device: {self.device}")
        
        # Prepare data
        X = self.prepare_sequences(df, training=True)
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Model setup
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
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
        
        # Calculate threshold based on training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            X_tensor = X_tensor.to(self.device)
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
            # Set threshold at 95th percentile of training errors
            self.threshold = np.percentile(mse, 95)
            print(f"Threshold set at: {self.threshold:.6f}")
        
        # Save model
        self.save_model()
        print("Training complete!")
        
    def save_model(self):
        """Save model weights and scaler."""
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'time_steps': self.time_steps,
            'n_features': self.n_features
        }, 'models/lstm_autoencoder.pth')
        joblib.dump(self.scaler, 'models/lstm_scaler.joblib')
        print("Model saved to models/lstm_autoencoder.pth")
        
    def load_model(self):
        """Load model weights and scaler."""
        checkpoint = torch.load('models/lstm_autoencoder.pth', map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.threshold = checkpoint['threshold']
        self.scaler = joblib.load('models/lstm_scaler.joblib')
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
        
    def predict_anomaly_score(self, df):
        """Calculate reconstruction error as anomaly score."""
        X = self.prepare_sequences(df, training=False)
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=(1, 2)).cpu().numpy()
        
        return mse
    
    def predict(self, df):
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        scores = self.predict_anomaly_score(df)
        return (scores > self.threshold).astype(int)


if __name__ == "__main__":
    # Test the LSTM Autoencoder
    if os.path.exists('data/esp_normal.csv'):
        print("=" * 50)
        print("LSTM AUTOENCODER FOR ESP ANOMALY DETECTION")
        print("=" * 50)
        
        # Load normal data for training
        df_normal = pd.read_csv('data/esp_normal.csv')
        
        # Initialize and train
        detector = ESPLSTMDetector()
        detector.train(df_normal, epochs=30, batch_size=64)
        
        # Test on Gas Lock data
        print("\nTesting on Gas Lock data...")
        df_gas_lock = pd.read_csv('data/esp_gas_lock.csv')
        scores = detector.predict_anomaly_score(df_gas_lock)
        
        print(f"Normal threshold: {detector.threshold:.6f}")
        print(f"Max Anomaly Score (Gas Lock): {np.max(scores):.6f}")
        print(f"Mean Anomaly Score (Gas Lock): {np.mean(scores):.6f}")
        print(f"Anomalies detected: {np.sum(scores > detector.threshold)} / {len(scores)}")
    else:
        print("Normal data not found. Run data_generator.py first.")
