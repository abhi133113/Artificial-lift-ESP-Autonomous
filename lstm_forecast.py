import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Seq2SeqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Decoder
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, target_len):
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Decode
        outputs = []
        decoder_input = x[:, -1, :].unsqueeze(1) # Start with last input
        
        for _ in range(target_len):
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            out = self.fc(out)
            outputs.append(out)
            decoder_input = out # Next input is current output
            
        return torch.cat(outputs, dim=1)

class ESPForecaster:
    def __init__(self, input_len=60, output_len=30, n_features=2):
        self.input_len = input_len
        self.output_len = output_len
        self.n_features = n_features # PIP, Amps
        self.features = ['PIP', 'Amps']
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = Seq2SeqLSTM(n_features, 64, n_features).to(self.device)
        
    def prepare_data(self, df):
        data = df[self.features].values
        self.scaler.fit(data)
        scaled = self.scaler.transform(data)
        
        X, y = [], []
        for i in range(len(scaled) - self.input_len - self.output_len):
            X.append(scaled[i : i + self.input_len])
            y.append(scaled[i + self.input_len : i + self.input_len + self.output_len])
            
        return np.array(X), np.array(y)
        
    def train(self, df, epochs=50, batch_size=32, lr=0.001):
        print(f"Training Forecaster on {len(df)} samples...")
        X, y = self.prepare_data(df)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.model(batch_x, self.output_len)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
                
        self.save_model()
        
    def save_model(self):
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), 'models/lstm_forecast.pth')
        joblib.dump(self.scaler, 'models/forecast_scaler.joblib')
        print("Forecast model saved!")
        
    def load_model(self):
        if not os.path.exists('models/lstm_forecast.pth'):
            print("Model not found!")
            return False
            
        self.model.load_state_dict(torch.load('models/lstm_forecast.pth', map_location=self.device, weights_only=True))
        self.scaler = joblib.load('models/forecast_scaler.joblib')
        self.model.eval()
        print("Forecast model loaded!")
        return True
        
    def predict(self, history):
        # history: list of dicts or array of shape (input_len, n_features)
        if isinstance(history, list):
            df = pd.DataFrame(history)
            data = df[self.features].values
        else:
            data = history
            
        scaled = self.scaler.transform(data)
        X_tensor = torch.FloatTensor(scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(X_tensor, self.output_len)
            
        prediction = self.scaler.inverse_transform(output.cpu().numpy()[0])
        return prediction

if __name__ == "__main__":
    # Train if run directly
    if os.path.exists('data/esp_normal.csv'):
        df = pd.read_csv('data/esp_normal.csv')
        forecaster = ESPForecaster()
        forecaster.train(df, epochs=30)
    else:
        print("Data not found. Run data_generator.py first.")
