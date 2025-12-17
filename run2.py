import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, log_loss, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

from train_test import load_dataset, load_labels

import os

os.makedirs("graphs", exist_ok=True)

# -------------------------------
# 1. Load original train/test
# -------------------------------
X_train_orig = load_dataset("pems+sf/PEMS_train")
y_train_orig = load_labels("pems+sf/PEMS_trainlabels")

X_test_final = load_dataset("pems+sf/PEMS_test")
y_test_final = load_labels("pems+sf/PEMS_testlabels")

# -------------------------------
# 2. Create validation set from train
# -------------------------------
val_fraction = 0.2  # 20% for validation
num_train_samples = len(X_train_orig)
val_size = int(num_train_samples * val_fraction)

# Shuffle indices for train/validation split
indices = np.arange(num_train_samples)
np.random.seed(42)
np.random.shuffle(indices)

val_indices = indices[:val_size]
train_indices = indices[val_size:]

X_val_final = X_train_orig[val_indices]
y_val_final = y_train_orig[val_indices]

X_train_final = X_train_orig[train_indices]
y_train_final = y_train_orig[train_indices]

# -------------------------------
# 3. Reshape / add channel dimension for CNN
# -------------------------------
num_samples_train = X_train_final.shape[0]
num_samples_val = X_val_final.shape[0]
num_samples_test = X_test_final.shape[0]

num_sensors = 963
num_time = 144

# Reshape to (samples, sensors, time)
X_train_final = X_train_final.reshape(num_samples_train, num_sensors, num_time)
X_val_final = X_val_final.reshape(num_samples_val, num_sensors, num_time)
X_test_final = X_test_final.reshape(num_samples_test, num_sensors, num_time)

print("Train samples:", len(X_train_final))
print("Validation samples:", len(X_val_final))
print("Test samples:", len(X_test_final))

# Add channel dimension for Conv2D
X_train_cnn = X_train_final[..., np.newaxis]
X_val_cnn = X_val_final[..., np.newaxis]
X_test_cnn = X_test_final[..., np.newaxis]

# -------------------------------
# 4. Flatten for MLP/SVM
# -------------------------------
X_train_flat = X_train_final.reshape(num_samples_train, -1)
X_val_flat = X_val_final.reshape(num_samples_val, -1)
X_test_flat = X_test_final.reshape(num_samples_test, -1)

scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_val_flat = scaler.transform(X_val_flat)
X_test_flat = scaler.transform(X_test_flat)

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
def performCNN():
    num_epochs = 100
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
def performMLP(
        hidden_layer_sizes: tuple = (128, 64),
        activation: str = 'relu',
        solver: str = 'adam',
        alpha: float = 0.0005,
        learning_rate: float = 0.001,
        identifier: int = 1,
        epochs: int = 50
    ):
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size=32,
        learning_rate_init=learning_rate,
        validation_fraction=0.2,
        max_iter=500,
        early_stopping=True,
        random_state=42,
        verbose=True
    )
    mlp.fit(X_train_flat, y_train_final)
    # Loss
    train_loss = np.array([mlp.loss_])
    val_loss = np.array([log_loss(y_val_final, mlp.predict_proba(X_val_flat))])
    test_loss = np.array([log_loss(y_test_final, mlp.predict_proba(X_test_flat))])

    # Predictions
    y_val_pred_mlp  = mlp.predict(X_val_flat)
    y_test_pred_mlp = mlp.predict(X_test_flat)

    # Probabilities (needed for AUC)
    y_val_prob_mlp  = mlp.predict_proba(X_val_flat)
    y_test_prob_mlp = mlp.predict_proba(X_test_flat)

    # Print statements
    print(f"[MLP {identifier}] Training loss: {mlp.loss_}")
    print(f"[MLP {identifier}] Val loss: {log_loss(y_val_final, mlp.predict_proba(X_val_flat))}")
    print(f"[MLP {identifier}] Val Accuracy:  {accuracy_score(y_val_final, y_val_pred_mlp):.4f}")
    print(f"[MLP {identifier}] Val Precision: {precision_score(y_val_final, y_val_pred_mlp, average='macro'):.4f}")
    print(f"[MLP {identifier}] Val AUC:       {roc_auc_score(y_val_final, y_val_prob_mlp, multi_class='ovr', average='macro'):.4f}")
    print(f"[MLP {identifier}] Test loss: {log_loss(y_test_final, mlp.predict_proba(X_test_flat))}")
    print(f"[MLP {identifier}] Test Accuracy:  {accuracy_score(y_test_final, y_test_pred_mlp):.4f}")
    print(f"[MLP {identifier}] Test Precision: {precision_score(y_test_final, y_test_pred_mlp, average='macro'):.4f}")
    print(f"[MLP {identifier}] Test AUC:       {roc_auc_score(y_test_final, y_test_prob_mlp, multi_class='ovr', average='macro'):.4f}")
    print()

    # Plot figures
    plt.figure()
    plt.plot(mlp.loss_curve_, label='Train Loss', marker='o')
    # plt.plot(mlp.validation_scores_, label='Validation Accuracy', marker='s')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("MLP Confusion Matrix")
    plt.legend()
    plt.savefig(os.path.join("graphs", f"mlp_loss_curve_{identifier}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------
# 6. SVM (Sklearn)
# -----------------------
def performSVM(C: float = 10, gamma: float = 0, identifier: int = 1):
    svm = SVC(kernel='rbf', C=C, gamma=gamma or 'scale')
    svm.fit(X_train_flat, y_train_final)
    y_val_pred_svm = svm.predict(X_val_flat)
    y_test_pred_svm = svm.predict(X_test_flat)
    print(f"[SVM {identifier}] Val Accuracy: {accuracy_score(y_val_final, y_val_pred_svm):.4f}")
    print(f"[SVM {identifier}] Test Accuracy: {accuracy_score(y_test_final, y_test_pred_svm):.4f}")

    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test_final, y_test_pred_svm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"SVM Confusion Matrix (C={C}, gamma={gamma or 'scale'})")
    plt.savefig(os.path.join("graphs", f"svm_confusion_matrix_{identifier}.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    # performCNN()

    performMLP()
    # performMLP(alpha=0.001, identifier=2)
    # performMLP(alpha=0.0001, identifier=3)
    # performMLP(learning_rate=0.005, identifier=4)
    # performMLP(learning_rate=0.0005, identifier=5)
    # performMLP(hidden_layer_sizes=(256, 128), identifier=6)
    # performMLP(hidden_layer_sizes=(64, 32), identifier=7)

    # performSVM()
    # performSVM(C=0.1, identifier=2)
    # performSVM(C=1, identifier=3)
    # performSVM(C=100, identifier=4)
    # performSVM(gamma=0.01, identifier=5)
    # performSVM(gamma=0.001, identifier=6)
