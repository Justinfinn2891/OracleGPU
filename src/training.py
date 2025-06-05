import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from models.gpu_model import modelForPredictions as ourPredictionModel

class training_process:
    def __init__(self, gpu_name, gpu_year, train=False, data_path='../data/data.csv', sequence_length=30):
        self.sequence_length = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "gpu_model.pth"
        self.loss_history = []

        # Load data
        gpu_data = np.loadtxt(data_path, delimiter=',', dtype=str)
        raw_priceData = gpu_data[:, 2].astype(float).reshape(-1, 1)
        raw_usedPriceData = gpu_data[:, 3].astype(float).reshape(-1, 1)
        raw_gpuData = gpu_data[:, 1]
        raw_dateData = gpu_data[:, 0]
        year_ofPurchase = np.array([int("20" + d.split('-')[2]) for d in raw_dateData]).reshape(-1, 1)

        # Encode and scale
        self.gpu_encoder = LabelEncoder()
        gpu_ids = self.gpu_encoder.fit_transform(raw_gpuData).reshape(-1, 1)
        self.year_scaler = MinMaxScaler()
        self.retail_scaler = MinMaxScaler()
        self.used_scaler = MinMaxScaler()

        years_scaled = self.year_scaler.fit_transform(year_ofPurchase)
        prices_scaled = self.retail_scaler.fit_transform(raw_priceData)
        usedPrices_scaled = self.used_scaler.fit_transform(raw_usedPriceData)

        X = np.hstack((years_scaled, prices_scaled, usedPrices_scaled, gpu_ids))
        y = np.hstack((prices_scaled, usedPrices_scaled))

        self.X_seq, self.y_seq = self.create_sequences(X, y)

        # Model
        self.model = ourPredictionModel(input_size=4, hidden_size=64, output_size=2).to(self.device)

        if train or not os.path.exists(self.model_path):
            self.train_model()
            torch.save(self.model.state_dict(), self.model_path)
        else:
            self.load_model()

        self.predict(gpu_name, gpu_year)

    def create_sequences(self, X, y):
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        return np.array(X_seq), np.array(y_seq)

    def train_model(self):
        X_tensor = torch.tensor(self.X_seq, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(self.y_seq, dtype=torch.float32).to(self.device)

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(100):
            self.model.train()
            output = self.model(X_tensor)
            loss = loss_fn(output, Y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    def predict(self, gpu_name, year):
        self.model.eval()
        with torch.no_grad():
            year = int(year)
            gpu_encoded = self.gpu_encoder.transform([gpu_name])[0]
            scaled_year = self.year_scaler.transform(np.array([[year]]))[0][0]

            last_seq = self.X_seq[-1].copy()
            placeholder_retail = last_seq[-1, 1]
            placeholder_used = last_seq[-1, 2]
            new_input = np.array([scaled_year, placeholder_retail, placeholder_used, gpu_encoded])
            new_seq = np.vstack((last_seq[1:], new_input))

            test_tensor = torch.tensor(new_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            predicted = self.model(test_tensor).squeeze().cpu().numpy()

            self.retailPrice = float(self.retail_scaler.inverse_transform([[predicted[0], 0]])[0][0])
            self.usedPrice = float(self.used_scaler.inverse_transform([[0, predicted[1]]])[0][1])

    def get_prediction(self):
        return self.retailPrice, self.usedPrice