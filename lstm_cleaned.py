"""
Train LSTM Autoencoder on Cleaned Real Pump Data

Uses the cleaned data with rolling features for improved anomaly detection.
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

def main():
    print("="*60)
    print("LSTM ON CLEANED REAL PUMP DATA")
    print("="*60)
    
    # Load cleaned data
    df = pd.read_csv('data/real_pump_cleaned.csv')
    print(f"\nLoaded {len(df)} cleaned samples")
    print(f"Status distribution:\n{df['machine_status'].value_counts()}\n")
    
    # Use extended features including rolling stats
    features = ['PIP', 'Discharge_Press', 'Amps', 'Temp', 'Vibration',
                'PIP_rolling_mean', 'PIP_rolling_std',
                'Amps_rolling_mean', 'Amps_rolling_std',
                'Vibration_rolling_mean', 'Vibration_rolling_std',
                'PIP_rate', 'Amps_rate']
    
    n_features = len(features)
    time_steps = 10
    
    # Split data
    df_normal = df[df['machine_status'] == 'NORMAL']
    df_anomaly = df[df['machine_status'] != 'NORMAL']
    
    df_train = df_normal.sample(n=int(len(df_normal)*0.7), random_state=42)
    df_val_normal = df_normal.drop(df_train.index).sample(n=500, random_state=42)
    df_val_anomaly = df_anomaly.sample(n=min(200, len(df_anomaly)), random_state=42)
    df_test = df.sample(n=1500, random_state=99)
    
    print(f"Train: {len(df_train)} normal")
    print(f"Val: {len(df_val_normal)} normal, {len(df_val_anomaly)} anomaly")
    print(f"Test: {len(df_test)} mixed\n")
    
    # Prepare data
    scaler = StandardScaler()
    
    def prepare_sequences(data, training=True):
        if training:
            scaled = scaler.fit_transform(data[features])
        else:
            scaled = scaler.transform(data[features])
        seqs = []
        for i in range(len(scaled) - time_steps):
            seqs.append(scaled[i:i + time_steps])
        return np.array(seqs)
    
    X_train = prepare_sequences(df_train, training=True)
    print(f"Training sequences shape: {X_train.shape}")
    
    # Build model with more features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMAutoencoder(n_features, hidden_dim=64, latent_dim=32).to(device)
    
    # Train
    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training...")
    model.train()
    for epoch in range(50):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {total_loss/len(dataloader):.6f}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    model.eval()
    
    with torch.no_grad():
        X_val_n = torch.FloatTensor(prepare_sequences(df_val_normal, False)).to(device)
        X_val_a = torch.FloatTensor(prepare_sequences(df_val_anomaly, False)).to(device)
        
        scores_n = torch.mean((X_val_n - model(X_val_n))**2, dim=(1,2)).cpu().numpy()
        scores_a = torch.mean((X_val_a - model(X_val_a))**2, dim=(1,2)).cpu().numpy()
    
    best_f1, best_thresh = 0, 0
    for p in range(80, 100):
        thresh = np.percentile(scores_n, p)
        y_true = np.concatenate([np.zeros(len(scores_n)), np.ones(len(scores_a))])
        y_pred = np.concatenate([(scores_n > thresh).astype(int), (scores_a > thresh).astype(int)])
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh
    
    print(f"Optimal threshold: {best_thresh:.6f} (Val F1: {best_f1:.4f})")
    
    # Evaluate on test
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    X_test = prepare_sequences(df_test, False)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    
    with torch.no_grad():
        scores = torch.mean((X_test_tensor - model(X_test_tensor))**2, dim=(1,2)).cpu().numpy()
    
    y_true = (df_test['machine_status'] != 'NORMAL').values[time_steps:]
    y_pred = (scores > best_thresh).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\nPrecision: {precision*100:.2f}%")
    print(f"Recall:    {recall*100:.2f}%")
    print(f"F1 Score:  {f1*100:.2f}%")
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}")
    print(f"Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}")
    
    # Save results
    with open('data/lstm_cleaned_results.txt', 'w') as f:
        f.write("LSTM ON CLEANED REAL PUMP DATA\n")
        f.write("="*50 + "\n\n")
        f.write(f"Features used: {len(features)} (including rolling stats)\n\n")
        f.write(f"Precision: {precision*100:.2f}%\n")
        f.write(f"Recall:    {recall*100:.2f}%\n")
        f.write(f"F1 Score:  {f1*100:.2f}%\n")
        f.write(f"Accuracy:  {accuracy*100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"              Normal  Anomaly\n")
        f.write(f"Actual Normal   {cm[0,0]:>5}    {cm[0,1]:>5}\n")
        f.write(f"Actual Anomaly  {cm[1,0]:>5}    {cm[1,1]:>5}\n")
    
    # Save model
    torch.save({'model': model.state_dict(), 'threshold': best_thresh}, 'models/lstm_cleaned.pth')
    joblib.dump(scaler, 'models/lstm_cleaned_scaler.joblib')
    
    print("\nResults saved to data/lstm_cleaned_results.txt")
    print("Model saved to models/lstm_cleaned.pth")

if __name__ == "__main__":
    main()
