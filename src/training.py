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
    def __init__(self):
        
        gpu_data = np.loadtxt('../data/data.csv', delimiter=',', dtype=str)

        #Breaking data apart out of the csv
        raw_priceData = gpu_data[:, 2].astype(float).reshape(-1,1)
        raw_usedPriceData = gpu_data[:, 3].astype(float).reshape(-1,1)
        raw_gpuData = gpu_data[:, 1]
        raw_dateData = gpu_data[:, 0]
        year_ofPurchase = np.array([int("20" + d.split('-')[2]) for d in raw_dateData]).reshape(-1,1)
        
        
        gpu_encoder = LabelEncoder()
        gpu_ids = gpu_encoder.fit_transform(raw_gpuData).reshape(-1, 1)
        scaler = MinMaxScaler()
        years_scaled = scaler.fit_transform(year_ofPurchase)
        prices_scaled = scaler.fit_transform(raw_priceData)
        usedPrices_scaled = scaler.fit_transform(raw_usedPriceData)
        
        X = np.hstack((years_scaled, prices_scaled, usedPrices_scaled, gpu_ids))
        y = np.roll(prices_scaled, -1, axis = 0)
        
        X = X[:-1]
        y = y[:-1]
        
        X_tensor = torch.tensor(X, dtype = torch.float32).unsqueeze(1)
        Y_tensor = torch.tensor(y, dtype=torch.float32)
        
        
        model = ourPredictionModel(input_size=4, hidden_size = 16)
        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = 0.01)
        
        for epoch in range(200):
            model.train()
            output = model(X_tensor)
            loss = loss_fn(output, Y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch +1) % 50 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        predicted_year = 2023
        graphics_card = "GeForce RTX 3070"
        #prediction stage
        test_year = scaler.transform(np.array([[predicted_year]]))
        place_holder = scaler.transform(np.array([[1]]))
        test_gpu = gpu_encoder.transform([graphics_card]).reshape(-1, 1)
        test_price = scaler.transform([[499]]).astype(float).reshape(-1,1) #last known
        test_usedPrice = scaler.transform([[300]]).astype(float).reshape(-1,1)
        test_retailInput = np.hstack((test_year, test_gpu, test_price, place_holder))
        test_usedInput = np.hstack((test_year, test_gpu, test_usedPrice, place_holder))
        test_retailTensor = torch.tensor(test_retailInput, dtype=torch.float32).unsqueeze(0)
        test_usedTensor = torch.tensor(test_usedInput, dtype=torch.float32).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            pred_scaled = model(test_retailTensor).item()
            pred_usedScaled = model(test_usedTensor).item()
            pred_retailPrice = scaler.inverse_transform([[pred_scaled]])[0][0]
            pred_usedPrice = scaler.inverse_transform([[pred_usedScaled]])[0][0]
            print(f"Predicted retail price for the {predicted_year} RTX 4070: ${pred_retailPrice:.2f}")
            print(f"Predicted used price for the {predicted_year} RTX 4070: ${pred_usedPrice:.2f}")
        

training_process()
    
    