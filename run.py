import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

from train_test import load_dataset, load_labels
from randperm import get_invperm

# Load datasets
X_train = load_dataset("pems+sf/PEMS_train")
y_train = load_labels("pems+sf/PEMS_trainlabels")

X_test = load_dataset("pems+sf/PEMS_test")
y_test = load_labels("pems+sf/PEMS_testlabels")

invperm = get_invperm()

# Combine datasets
X_all = np.concatenate([X_train, X_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

# Reorder dataset based on inverse permutation
X_calendar = X_all[invperm]
y_calendar = y_all[invperm]

# Perform 70-20-10 split
train_percentage = 70
validation_percentage = 20
test_percentage = 10

assert train_percentage + validation_percentage + test_percentage == 100

N = len(X_calendar)
train_end = int((train_percentage) / 100 * N)
val_end = int((train_percentage + validation_percentage) / 100 * N)

X_train_final = X_calendar[:train_end]
y_train_final = y_calendar[:train_end]

X_val_final = X_calendar[train_end:val_end]
y_val_final = y_calendar[train_end:val_end]

X_test_final = X_calendar[val_end:]
y_test_final = y_calendar[val_end:]

# Prepare for MLP, CNN, SVM
# CNN
X_train_cnn = X_train_final[..., np.newaxis]
X_val_cnn = X_val_final[..., np.newaxis]
X_test_cnn = X_test_final[..., np.newaxis]

# MLP and SVM
X_train_flat = X_train_final.reshape(len(X_train_final), -1)
X_val_flat = X_val_final.reshape(len(X_val_final), -1)
X_test_flat = X_test_final.reshape(len(X_test_final), -1)

# Reshape data
num_samples_train = X_train_final.shape[0]
num_samples_val = X_val_final.shape[0]
num_samples_test = X_test_final.shape[0]

num_sensors = 963
num_time = 144

# Reshape back to (samples, sensors, time)
X_train_final = X_train_final.reshape(num_samples_train, num_sensors, num_time)
X_val_final = X_val_final.reshape(num_samples_val, num_sensors, num_time)
X_test_final = X_test_final.reshape(num_samples_test, num_sensors, num_time)

# Add channel dimension for Conv2D
X_train_cnn = X_train_final[..., np.newaxis]
X_val_cnn = X_val_final[..., np.newaxis]
X_test_cnn = X_test_final[..., np.newaxis]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train_cnn, dtype=torch.float32).permute(0,3,1,2).to(device) # (N,1,963,144)
y_train_tensor = torch.tensor(y_train_final, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val_cnn, dtype=torch.float32).permute(0,3,1,2).to(device)
y_val_tensor = torch.tensor(y_val_final, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32).permute(0,3,1,2).to(device)
y_test_tensor = torch.tensor(y_test_final, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

num_classes = len(np.unique(y_train_final))

class_counts = Counter(y_train_final)

print("Class counts:")
for k in sorted(class_counts.keys()):
    print(f"Number of {k} targets: {class_counts[k]}")

class PEMS_CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3,3)),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3,3)),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3,3)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*35*5, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

cnn_model = PEMS_CNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3)

# -----------------------
# 4. CNN Training Loop
# -----------------------
num_epochs = 20
for epoch in range(num_epochs):
    cnn_model.train()
    running_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = cnn_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)
    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    cnn_model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = cnn_model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == y_batch).sum().item()
            val_total += y_batch.size(0)
    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"[CNN] Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# CNN Test evaluation
cnn_model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = cnn_model(X_batch)
        preds = outputs.argmax(dim=1)
        test_correct += (preds == y_batch).sum().item()
        test_total += y_batch.size(0)
cnn_acc = test_correct / test_total
print(f"[CNN] Test Accuracy: {cnn_acc:.4f}")

# -----------------------
# 5. MLP (Sklearn)
# -----------------------
mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
mlp.fit(X_train_flat, y_train_final)
y_val_pred_mlp = mlp.predict(X_val_flat)
y_test_pred_mlp = mlp.predict(X_test_flat)
print(f"[MLP] Val Accuracy: {accuracy_score(y_val_final, y_val_pred_mlp):.4f}")
print(f"[MLP] Test Accuracy: {accuracy_score(y_test_final, y_test_pred_mlp):.4f}")

# -----------------------
# 6. SVM (Sklearn)
# -----------------------
svm = SVC(kernel='rbf', C=10)
svm.fit(X_train_flat, y_train_final)
y_val_pred_svm = svm.predict(X_val_flat)
y_test_pred_svm = svm.predict(X_test_flat)
print(f"[SVM] Val Accuracy: {accuracy_score(y_val_final, y_val_pred_svm):.4f}")
print(f"[SVM] Test Accuracy: {accuracy_score(y_test_final, y_test_pred_svm):.4f}")