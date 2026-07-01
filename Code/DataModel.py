import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch.nn as nn
from sklearn.model_selection import train_test_split
import random

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class RawToSmoothDataset(Dataset):
    def __init__(self, list_of_windows):
        self.X = []
        self.Y = []
        for raw50, smooth50 in list_of_windows:
            self.X.append(raw50)
            self.Y.append(smooth50)
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def train_lstm_with_preprocessed(
        X_input_net,
        Y_target,
        batch_size,
        epochs,
        Model,
        device,
        lr=5e-4,
        val_ratio=0.2,
        step_size=10,
        gamma=0.98,
        seed=42,
        save_path="./best_lstm_model.pt"):

    set_random_seed(seed)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_input_net,
        Y_target,
        test_size=val_ratio,
        shuffle=True,
        random_state=seed
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val, dtype=torch.float32)
    )

    # DataLoader 随机种子
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    output_size = Y_target.shape[1]

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):

        #####################
        # Train
        #####################
        Model.train()
        total_train_loss = 0.0

        for X_batch, Y_batch in train_loader:

            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()

            Y_pred = Model(X_batch)

            loss = criterion(Y_pred, Y_batch)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * X_batch.size(0)

        total_train_loss /= len(train_dataset)

        #####################
        # Validation
        #####################
        Model.eval()

        total_val_loss = 0.0
        total_val_mae = 0.0

        with torch.no_grad():

            for X_batch, Y_batch in val_loader:

                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

                Y_pred = Model(X_batch)

                loss = criterion(Y_pred, Y_batch)

                total_val_loss += loss.item() * X_batch.size(0)
                total_val_mae += torch.abs(Y_pred - Y_batch).sum().item()

        total_val_loss /= len(val_dataset)
        total_val_mae /= (len(val_dataset) * output_size)

        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save(Model.state_dict(), save_path)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {total_train_loss:.6f} "
            f"Val Loss: {total_val_loss:.6f} "
            f"Val MAE: {total_val_mae:.6f}"
        )

        scheduler.step()

    print(f"Best model saved to: {save_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return Model, save_path

class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=8):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)  # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :]   
        out = self.fc(out)     # [batch, output_size]
        return out
    
