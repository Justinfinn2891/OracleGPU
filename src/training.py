import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
from gpu_model import modelForPredictions as ourPredictionModel

class training_process():
    def __init__(self, gpu_name, gpu_year):
        
        gpu_data = np.loadtxt('../data/data.csv', delimiter=',', dtype=str)

        #Breaking data apart out of the csv
        raw_priceData = gpu_data[:, 2].astype(float).reshape(-1,1)
        raw_usedPriceData = gpu_data[:, 3].astype(float).reshape(-1,1)
        raw_gpuData = gpu_data[:, 1]
        raw_dateData = gpu_data[:, 0]
        year_ofPurchase = np.array([int("20" + d.split('-')[2]) for d in raw_dateData]).reshape(-1,1)
        
        
        gpu_encoder = LabelEncoder()
        gpu_ids = gpu_encoder.fit_transform(raw_gpuData).reshape(-1, 1)
        year_scaler = MinMaxScaler()
        retail_scaler = MinMaxScaler()
        used_scaler = MinMaxScaler()
        years_scaled = year_scaler.fit_transform(year_ofPurchase)
        prices_scaled = retail_scaler.fit_transform(raw_priceData)
        usedPrices_scaled = used_scaler.fit_transform(raw_usedPriceData)
        
        X = np.hstack((years_scaled, prices_scaled, usedPrices_scaled, gpu_ids))
        y = np.roll(prices_scaled, -1, axis = 0)
        
        X = X[:-1]
        y = y[:-1]
        
        X_tensor = torch.tensor(X, dtype = torch.float32).unsqueeze(1)
        Y_tensor = torch.tensor(y, dtype=torch.float32)
        
        
        model = ourPredictionModel(input_size=4, hidden_size = 16)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.01)
        
        epoch = 1000
        for i in range(epoch):
            model.train()
            output = model(X_tensor)
            loss = loss_fn(output, Y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i +1) % 50 == 0:
                print(f"Epoch {i+1}, Loss: {loss.item():.4f}")
        
        predicted_year = gpu_year
        graphics_card = gpu_name
        #prediction stage
        placeholder_price = 1.0
        test_year = year_scaler.transform(np.array([[predicted_year]]))[0][0]
        prices_scaled = retail_scaler.transform(np.array([[placeholder_price]]))[0][0]
        used_scaled = used_scaler.transform(np.array([[placeholder_price]]))[0][0]
        test_gpu = gpu_encoder.transform([graphics_card])[0]
        
        test_retailInput = np.array([[test_year, prices_scaled, prices_scaled, test_gpu]], dtype=np.float32)
        test_usedInput = np.array([[test_year, used_scaled, used_scaled, test_gpu]], dtype=np.float32)
        test_retailTensor = torch.tensor(test_retailInput, dtype=torch.float32).unsqueeze(0)
        test_usedTensor = torch.tensor(test_usedInput, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(test_retailTensor).item()
            pred_usedScaled = model(test_usedTensor).item()
            pred_retailPrice = retail_scaler.inverse_transform([[pred_scaled]])[0][0]
            pred_usedPrice = retail_scaler.inverse_transform([[pred_usedScaled]])[0][0]
            self.retailPrice = pred_retailPrice
            self.usedPrice = pred_usedPrice
            print(f"Predicted retail price for the {predicted_year} {graphics_card}: ${pred_retailPrice:.2f}")
            print(f"Predicted used price for the {predicted_year} {graphics_card}: ${pred_usedPrice:.2f}")
        

if __name__ == "__main__":
    training_process() 


# We need a lot more entries
# We need to add monthly 
# We need to connect this to app.py so we can have user input