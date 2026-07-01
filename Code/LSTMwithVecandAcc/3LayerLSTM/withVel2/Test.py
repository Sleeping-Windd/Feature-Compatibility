import numpy as np
import torch
from torch.utils.data import DataLoader
import torch
import joblib
import os
import sys

current_dir = os.getcwd()
github_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

if github_dir not in sys.path:
    sys.path.insert(0, github_dir)

from DataModel import LSTMDataset, LSTMModel, train_lstm_with_preprocessed

scaler_raw = joblib.load("./datasets/guiderail/scaler_coord.pkl")
X_input_net= np.load("./datasets/guiderail/test_input.npy")  
Y_target   = np.load("./datasets/guiderail/test_target.npy")      
X_input_net = X_input_net[:,:,:10]
print(X_input_net.shape)
print(Y_target.shape)


dataset = LSTMDataset(X_input_net, Y_target)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=10, num_layers=3, output_size=8).to(device)
model.load_state_dict(torch.load("./best_lstm_model.pt", map_location=device))
model.eval()

X_data = torch.tensor(X_input_net,dtype=torch.float32)
batch_size = 256      
pred_list = []

model.eval()

with torch.no_grad():
    for i in range(0, len(X_data), batch_size):
        xb = X_data[i:i+batch_size].to(device)
        yb = model(xb)
        pred_list.append(yb.cpu())
        del xb, yb

Y_pred = torch.cat(pred_list, dim=0).numpy()

torch.cuda.empty_cache()

Y_pred_final = scaler_raw.inverse_transform(Y_pred)
Y_target_final = scaler_raw.inverse_transform(Y_target)

feature_names = [r"C_{l}X_{l}",r"C_{l}Y_{l}",r"C_{l}X_{r}",r"C_{l}Y_{r}",r"C_{r}X_{l}",r"C_{r}Y_{l}",r"C_{r}X_{r}",r"C_{r}Y_{r}"]

mae_list = []
rmse_list = []

print("===== Feature Errors =====")

for i in range(0, 8, 2):
    diff = Y_target_final[:, i] - Y_pred_final[:, i]
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.var(diff))  
    mae_list.append(mae)
    rmse_list.append(rmse)
    print(f"{feature_names[i]:<12s} | MAE = {mae:.6f} | RMSE = {rmse:.6f}")

torch.cuda.empty_cache()