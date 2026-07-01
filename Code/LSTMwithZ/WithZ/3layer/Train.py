import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys

current_dir = os.getcwd()
github_dir = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))

if github_dir not in sys.path:
    sys.path.insert(0, github_dir)

from DataModel import LSTMDataset, LSTMModel, train_lstm_with_preprocessed

X_input_net= np.load("./datasets/guiderail/train_input.npy")  # (N, 50, 32)
Y_target   = np.load("./datasets/guiderail/train_target.npy")         # (N, 8)

if isinstance(X_input_net, np.ndarray):
    X_input_net = torch.from_numpy(X_input_net).float()
X_input_net = torch.cat([X_input_net[:, :, :8], X_input_net[:, :, 32:34]],dim=2)
print(X_input_net.shape)
print(Y_target.shape)
X_input_net = X_input_net.detach().cpu().numpy()



dataset = LSTMDataset(X_input_net, Y_target)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)

input_size = X_input_net.shape[2]
output_size = Y_target.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Model = LSTMModel(input_size=input_size,num_layers=3,output_size=output_size).to(device)

LSTM = train_lstm_with_preprocessed(X_input_net=X_input_net,Y_target=Y_target,batch_size=1024,epochs=60,
                                    Model = Model, 
                                    device = device, 
                                    lr=5e-4,val_ratio=0.2,step_size=8,
                                    gamma=0.95,
                                    seed=42,
                                    save_path="./best_lstm_model.pt")
