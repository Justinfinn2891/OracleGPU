import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from models.gpu_model import modelForPredictions as ourPredictionModel

class training_process:
    def __init__(self, gpu_name, gpu_year, data_path='../data/gtx_7800_prices.csv', sequence_length=30):
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_history = []

        gpu_data = np.loadtxt(data_path, delimiter=',', dtype=str)
        raw_priceData = gpu_data[:, 2].astype(float).reshape(-1, 1)
        raw_usedPriceData = gpu_data[:, 3].astype(float).reshape(-1, 1)
        raw_gpuData = gpu_data[:, 1]
        raw_dateData = gpu_data[:, 0]
        year_ofPurchase = np.array([int("20" + d.split('-')[2]) for d in raw_dateData]).reshape(-1, 1)

        gpu_encoder = LabelEncoder()
        gpu_ids = gpu_encoder.fit_transform(raw_gpuData).reshape(-1, 1)
        year_scaler = MinMaxScaler()
        retail_scaler = MinMaxScaler()
        used_scaler = MinMaxScaler()

        years_scaled = year_scaler.fit_transform(year_ofPurchase)
        prices_scaled = retail_scaler.fit_transform(raw_priceData)
        usedPrices_scaled = used_scaler.fit_transform(raw_usedPriceData)

        X = np.hstack((years_scaled, prices_scaled, usedPrices_scaled, gpu_ids))
        y = np.hstack((prices_scaled, usedPrices_scaled))

        X_seq, y_seq = self.create_sequences(X, y)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(self.device)

        self.model = ourPredictionModel(input_size=4, hidden_size=64, output_size=2).to(self.device)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for i in range(50):
            self.model.train()
            output = self.model(X_tensor)
            loss = loss_fn(output, Y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

        self.predict(gpu_name, gpu_year, gpu_encoder, year_scaler, retail_scaler, used_scaler, X_seq)

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def predict(self, gpu_name, year, gpu_encoder, year_scaler, retail_scaler, used_scaler, X_seq):
        self.model.eval()
        with torch.no_grad():
            year = int(year)
            gpu_encoded = gpu_encoder.transform([gpu_name])[0]
            scaled_year = year_scaler.transform(np.array([[year]]))[0][0]

            last_seq = X_seq[-1].copy()
            placeholder_retail = last_seq[-1, 1]
            placeholder_used = last_seq[-1, 2]
            new_input = np.array([scaled_year, placeholder_retail, placeholder_used, gpu_encoded])
            new_seq = np.vstack((last_seq[1:], new_input))

            test_tensor = torch.tensor(new_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            predicted = self.model(test_tensor).squeeze().cpu().numpy()

            # Save for UI
            self.retailPrice = float(retail_scaler.inverse_transform([[predicted[0], 0]])[0][0])
            self.usedPrice = float(used_scaler.inverse_transform([[0, predicted[1]]])[0][1])
     
