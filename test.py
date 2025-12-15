import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# 1. Load df safely
# -------------------------
df = pd.read_hdf("speed.h5", key="df")
data = df.values.astype(np.float32)

# If sensors are rows, transpose:
if data.shape[0] == 11160:
    data = data.T  # now shape is [timesteps, sensors]

timesteps, num_sensors = data.shape
print("Data shape:", data.shape)

input_len = 12     # 1 hour of 5-min intervals
output_len = 3     # next 15 minutes
total_len = input_len + output_len

# -------------------------
# 2. Memory-safe Dataset
# -------------------------
class SlidingWindowDataset(Dataset):
    def __init__(self, data, input_len, output_len):
        self.data = data
        self.input_len = input_len
        self.output_len = output_len
        self.total_len = input_len + output_len
        self.max_index = len(data) - self.total_len

    def __len__(self):
        return self.max_index

    def __getitem__(self, idx):
        X = self.data[idx : idx + self.input_len]
        Y = self.data[idx + self.input_len : idx + self.total_len]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# -------------------------
# 3. Train / Val split (by index)
# -------------------------
dataset = SlidingWindowDataset(data, input_len, output_len)

train_end = int(0.7 * len(dataset))
val_end = int(0.8 * len(dataset))

train_dataset = torch.utils.data.Subset(dataset, range(0, train_end))
val_dataset   = torch.utils.data.Subset(dataset, range(train_end, val_end))
test_dataset  = torch.utils.data.Subset(dataset, range(val_end, len(dataset)))

batch_size = 16  # smaller batch due to many sensors

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# 4. LSTM Model
# -------------------------
class TrafficLSTM(nn.Module):
    def __init__(self, num_sensors, hidden=64, output_len=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_sensors, hidden_size=hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, num_sensors * output_len)
        self.num_sensors = num_sensors
        self.output_len = output_len

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last hidden state
        out = self.fc(out)
        return out.view(-1, self.output_len, self.num_sensors)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficLSTM(num_sensors).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------
# 5. Training Loop
# -------------------------
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, Y_batch in train_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            out = model(X_batch.to(device))
            val_loss += criterion(out, Y_batch.to(device)).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
