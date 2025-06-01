import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from gpu_model import modelForPredictions as ourPredictionModel

class training_process:
    def __init__(self, gpu_name, gpu_year, data_path='../data/data.csv', sequence_length=30):
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load and preprocess data
        gpu_data = np.loadtxt(data_path, delimiter=',', dtype=str)

        # Extract relevant columns
        raw_priceData = gpu_data[:, 2].astype(float).reshape(-1, 1)
        raw_usedPriceData = gpu_data[:, 3].astype(float).reshape(-1, 1)
        raw_gpuData = gpu_data[:, 1]
        raw_dateData = gpu_data[:, 0]
        year_ofPurchase = np.array([int("20" + d.split('-')[2]) for d in raw_dateData]).reshape(-1, 1)

        # Encode and scale data
        gpu_encoder = LabelEncoder()
        gpu_ids = gpu_encoder.fit_transform(raw_gpuData).reshape(-1, 1)
        year_scaler = MinMaxScaler()
        retail_scaler = MinMaxScaler()
        used_scaler = MinMaxScaler()
        years_scaled = year_scaler.fit_transform(year_ofPurchase)
        prices_scaled = retail_scaler.fit_transform(raw_priceData)
        usedPrices_scaled = used_scaler.fit_transform(raw_usedPriceData)

        # Combine features
        X = np.hstack((years_scaled, prices_scaled, usedPrices_scaled, gpu_ids))
        y = prices_scaled  # We'll predict future retail price

        # Create sequences for LSTM
        X_seq, y_seq = self.create_sequences(X, y)

        # Convert to tensors
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)

        # Initialize model
        self.model = ourPredictionModel(input_size=4, hidden_size=16).to(self.device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        # Training loop
        for i in range(300):  # Adjust epochs as needed
            self.model.train()
            output = self.model(X_tensor)
            loss = loss_fn(output, Y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 50 == 0:
                print(f"Epoch {i+1}, Loss: {loss.item():.4f}")

        # Predict target year
        self.predict(gpu_name, gpu_year, gpu_encoder, year_scaler, retail_scaler, used_scaler)

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, gpu_name, year, gpu_encoder, year_scaler, retail_scaler, used_scaler):
        self.model.eval()
        with torch.no_grad():
            placeholder_price = 1.0
            scaled_year = year_scaler.transform(np.array([[year]]))[0][0]
            scaled_price = retail_scaler.transform([[placeholder_price]])[0][0]
            scaled_used = used_scaler.transform([[placeholder_price]])[0][0]
            gpu_encoded = gpu_encoder.transform([gpu_name])[0]

            test_seq = np.array([[scaled_year, scaled_price, scaled_used, gpu_encoded]] * self.sequence_length)
            test_tensor = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0).to(self.device)

            pred_scaled = self.model(test_tensor).item()
            self.retailPrice = retail_scaler.inverse_transform([[pred_scaled]])[0][0]
            self.usedPrice = used_scaler.inverse_transform([[pred_scaled]])[0][0]

            print(f"Predicted retail price for the {year} {gpu_name}: ${self.retailPrice:.2f}")
            print(f"Predicted used price for the {year} {gpu_name}: ${self.usedPrice:.2f}")

    def get_prediction(self):
        return self.retailPrice, self.usedPrice
