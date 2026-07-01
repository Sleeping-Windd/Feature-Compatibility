import numpy as np
import torch
from torch.utils.data import  DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib
from DataModel import LSTMModel, LSTMDataset

##load data   
scaler_coord = joblib.load("./datasets/guiderail/scaler_coord.pkl")
X_input_net= np.load("./datasets/guiderail/test_input.npy")  
Y_target   = np.load("./datasets/guiderail/test_target.npy")      

X_input_net = X_input_net[:,:,:16]        #Notice the input configuration！！！ 
print(X_input_net.shape)

X_all = X_input_net.reshape(-1, X_input_net.shape[-1]) 
X_coord = X_all.reshape(X_input_net.shape[0], -1, X_all.shape[-1])
print("Flattened shape:", X_all.shape)

##input spectropy
pca = PCA()
pca.fit(X_all)

spectrum = pca.explained_variance_ratio_
cum_ratio = np.cumsum(spectrum)
k90 = np.where(cum_ratio >= 0.9)[0][0] + 1
print("Dimension for 90% variance:", k90)

eigvals = pca.explained_variance_
PR = (eigvals.sum()**2) / np.sum(eigvals**2)
print("Participation Ratio:", PR)

p = spectrum / spectrum.sum()
spectral_entropy = -np.sum(p * np.log(p + 1e-12))
print("Spectral Entropy:", spectral_entropy)

print("PCA spectrum:")
print(spectrum)

plt.figure(figsize=(6,4))
plt.plot(spectrum, 'o-')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Input PCA Spectrum")
plt.grid(True)
plt.show()

##representation spectropy
dataset = LSTMDataset(X_coord, Y_target)
loader = DataLoader(dataset, batch_size=1024, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=16, num_layers = 2, output_size=8).to(device)              #Notice the model configuration！！！ 
model.load_state_dict(torch.load("./best_lstm_model.pt", map_location=device))
model.eval()

hidden_states = []
with torch.no_grad():
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        output, (h_n, c_n) = model.lstm(X_batch)

        H = output.reshape(-1, output.shape[-1])
        #H = output[:,-1:,:].reshape(-1, output.shape[-1])
        
        hidden_states.append(H.cpu())
H_all = torch.cat(hidden_states, dim=0).numpy()
print("Hidden state matrix shape:", H_all.shape)
print(X_batch.shape)

pca = PCA()
pca.fit(H_all)

eigvals = pca.explained_variance_
ratio = pca.explained_variance_ratio_
cum_ratio = np.cumsum(ratio)

print("First 10 explained variance ratio:")
print(ratio[:10])

pr = (eigvals.sum()**2) / (np.sum(eigvals**2))
print("Hidden Participation Ratio:", pr)
k90 = np.where(cum_ratio >= 0.9)[0][0] + 1
print("Dimension for 90% variance:", k90)

p = ratio + 1e-12  # 防止 log(0)
entropy = -np.sum(p * np.log(p))
print("Spectral entropy:", entropy)

plt.figure(figsize=(6,4))
plt.plot(ratio[:10], 'o-')
plt.xlabel("PC index")
plt.ylabel("explained variance ratio")
plt.title("Hidden State Eigen Spectrum")
plt.grid(True)
plt.show()
plt.figure(figsize=(6,4))
plt.semilogy(ratio[:10], 'o-')
plt.xlabel("PC index")
plt.ylabel("explained variance ratio (log)")
plt.title("Hidden State Eigen Spectrum (log)")
plt.grid(True)
plt.show()