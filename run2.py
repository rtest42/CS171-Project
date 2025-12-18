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

import random

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
seed = 42
val_fraction = 0.2  # 20% for validation
num_train_samples = len(X_train_orig)
val_size = int(num_train_samples * val_fraction)

# Shuffle indices for train/validation split
indices = np.arange(num_train_samples)
np.random.seed(seed)
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
X_train_cnn = X_train_final# [..., np.newaxis]
X_val_cnn = X_val_final# [..., np.newaxis]
X_test_cnn = X_test_final# [..., np.newaxis]

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

X_train_tensor = torch.tensor(X_train_cnn, dtype=torch.float32).to(device) # (N,1,963,144)
y_train_tensor = torch.tensor(y_train_final, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val_cnn, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_final, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test_cnn, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_final, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(np.unique(y_train_final))

class PEMS_CNN(nn.Module):
    def __init__(self, num_classes, dropout: float = 0.3, c: int = 32, k: int = 3, p: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=num_sensors, out_channels=c, kernel_size=k, padding=(k - 1) // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=p, stride=p),
            nn.BatchNorm1d(c),
            nn.Conv1d(in_channels=c, out_channels=c * 2, kernel_size=k, padding=(k - 1) // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=p, stride=p),
            nn.BatchNorm1d(c * 2),
            nn.Conv1d(in_channels=c * 2, out_channels=c * 4, kernel_size=k, padding=(k - 1) // 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=p, stride=p),
            nn.BatchNorm1d(c * 4),
        )
        self._to_linear = None
        with torch.no_grad():
            x = torch.zeros(1, num_sensors, num_time)
            x = self.conv(x)
            self._to_linear = x.numel()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, c * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c * 4, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

# -----------------------
# 4. CNN Training Loop
# -----------------------
def performCNN(dropout: float = 0.3, epochs: int = 100, learning_rate: float = 1e-3, identifier: int = 1, patience: int = 10, alpha: float = 1e-4, c: int = 32, k: int = 3, p: int = 3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cnn_model = PEMS_CNN(num_classes=num_classes, dropout=dropout, c=c, k=k, p=p).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=alpha)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = cnn_model.state_dict()
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []
    num_epochs = epochs
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

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)

        if epoch + 1 >= num_epochs - 10:
            print(f"[CNN {identifier}] Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = cnn_model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[CNN {identifier}] Early stopping triggered after {epoch + 1} epochs.")
                print(f"[CNN {identifier}] Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                cnn_model.load_state_dict(best_model_wts)
                break

    # CNN Test evaluation
    cnn_model.eval()
    test_correct, test_total = 0, 0
    all_logits = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = cnn_model(X_batch)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == y_batch).sum().item()
            test_total += y_batch.size(0)

            all_logits.append(outputs.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())
    cnn_acc = test_correct / test_total
    logits = torch.cat(all_logits)
    y_test_pred_cnn = torch.cat(all_preds).numpy()
    y_test_true = torch.cat(all_targets).numpy()
    y_test_prob_cnn = torch.softmax(logits, dim=1).numpy()
    print(f"[CNN {identifier}] Test Accuracy: {cnn_acc:.4f}")

    
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test_final, y_test_pred_cnn)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    disp.plot(cmap='Greens', ax=ax)
    plt.title(f"CNN Confusion Matrix #{identifier}")
    fig.savefig(os.path.join("graphs", f"cnn_confusion_matrix_{identifier}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    epochs_range = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs_range, train_losses, label="Training Loss")
    plt.plot(epochs_range, val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"CNN Loss #{identifier}")
    plt.legend()
    plt.savefig(os.path.join("graphs", f"cnn_loss_curve_{identifier}.png"), dpi=300, bbox_inches="tight")
    plt.close()

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
        # verbose=True
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

    print(f"[MLP {identifier}] Val loss:      {log_loss(y_val_final, mlp.predict_proba(X_val_flat))}")
    print(f"[MLP {identifier}] Val Accuracy:  {accuracy_score(y_val_final, y_val_pred_mlp):.4f}")
    print(f"[MLP {identifier}] Val Precision: {precision_score(y_val_final, y_val_pred_mlp, average='macro'):.4f}")
    print(f"[MLP {identifier}] Val AUC:       {roc_auc_score(y_val_final, y_val_prob_mlp, multi_class='ovr', average='macro'):.4f}")

    print(f"[MLP {identifier}] Test loss:      {log_loss(y_test_final, mlp.predict_proba(X_test_flat))}")
    print(f"[MLP {identifier}] Test Accuracy:  {accuracy_score(y_test_final, y_test_pred_mlp):.4f}")
    print(f"[MLP {identifier}] Test Precision: {precision_score(y_test_final, y_test_pred_mlp, average='macro'):.4f}")
    print(f"[MLP {identifier}] Test AUC:       {roc_auc_score(y_test_final, y_test_prob_mlp, multi_class='ovr', average='macro'):.4f}")
    print()

    # Plot figures
    # plt.figure()
    # plt.plot(mlp.loss_curve_, label='Train Loss', marker='o')
    # plt.plot(mlp.validation_scores_, label='Validation Accuracy', marker='s')
    # plt.xlabel("iterations")
    # plt.ylabel("loss")
    # plt.title("MLP Confusion Matrix")
    # plt.legend()
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test_final, y_test_pred_mlp)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    disp.plot(cmap='Reds', ax=ax)
    plt.title(f"MLP Confusion Matrix #{identifier}")
    plt.savefig(os.path.join("graphs", f"mlp_confusion_matrix_{identifier}.png"), dpi=300, bbox_inches="tight")
    plt.close()

# -----------------------
# 6. SVM (Sklearn)
# -----------------------
def performSVM(C: float = 10, gamma: float = 0, identifier: int = 1):
    svm = SVC(kernel='rbf', C=C, gamma=gamma or 'scale')
    svm.fit(X_train_flat, y_train_final)
    y_val_pred_svm = svm.predict(X_val_flat)
    y_test_pred_svm = svm.predict(X_test_flat)

    # y_val_prob_svm = svm.predict_proba(X_val_flat)
    # y_test_prob_svm = svm.predict_proba(X_test_flat)

    print(f"[SVM {identifier}] Val Accuracy:  {accuracy_score(y_val_final, y_val_pred_svm):.4f}")
    print(f"[SVM {identifier}] Val Precision: {precision_score(y_val_final, y_val_pred_svm, average='macro', zero_division=0):.4f}")

    print(f"[SVM {identifier}] Test Accuracy:  {accuracy_score(y_test_final, y_test_pred_svm):.4f}")
    print(f"[SVM {identifier}] Test Precision: {precision_score(y_test_final, y_test_pred_svm, average='macro', zero_division=0):.4f}")
    print()


    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test_final, y_test_pred_svm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"SVM Confusion Matrix (C={C}, gamma={gamma or 'scale'})")
    plt.savefig(os.path.join("graphs", f"svm_confusion_matrix_{identifier}.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    doCNN = True
    doMLP = False
    doSVM = False

    # performCNN(dropout=0.1, learning_rate=5e-4, k=7, p=3, identifier=17)
    performCNN(dropout=0.1, learning_rate=5e-4, k=3, p=2, c=16, identifier=20)

    if doCNN:
        performCNN(dropout=0, identifier=1)
        performCNN(dropout=0.1, identifier=2)
        performCNN(dropout=0.2, identifier=3)
        performCNN(dropout=0.3, identifier=4)
        performCNN(dropout=0.4, identifier=5)
        performCNN(dropout=0.5, identifier=6)
        performCNN(learning_rate=1e-4, identifier=7)
        performCNN(learning_rate=5e-4, identifier=8)
        performCNN(learning_rate=1e-3, identifier=9)
        performCNN(learning_rate=3e-3, identifier=10)
        performCNN(c=8, identifier=11)
        performCNN(c=16, identifier=12)
        performCNN(c=32, identifier=13)
        performCNN(c=64, identifier=14)
        performCNN(p=2, identifier=15)
        performCNN(p=3, identifier=16)
        performCNN(k=5, identifier=17)
        performCNN(k=7, identifier=18)
        performCNN(identifier=19)
    # performMLP(identifier=8, alpha=0.0005, learning_rate=0.0005, hidden_layer_sizes=(128,64))
        # performCNN(dropout=0, identifier=2)
        # performCNN(dropout=0.2, identifier=3)
        # performCNN(dropout=0.5, identifier=4)
        # performCNN(learning_rate=5e-4, identifier=5)
        # performCNN(learning_rate=3e-3, identifier=6)
        # performCNN(epochs=50, identifier=7)
        # performCNN(epochs=200, identifier=8)
        # performCNN(learning_rate=1e-4, identifier=9)
        # performCNN(dropout=0.35, learning_rate=5e-4, epochs=200, identifier=10)
        # performCNN(k=7, identifier=11)

    if doMLP:
        performMLP(identifier=1)
        performMLP(alpha=0.001, identifier=2)
        performMLP(alpha=0.0001, identifier=3)
        performMLP(learning_rate=0.005, identifier=4)
        performMLP(learning_rate=0.0005, identifier=5)
        performMLP(hidden_layer_sizes=(256, 128), identifier=6)
        performMLP(hidden_layer_sizes=(64, 32), identifier=7)

    if doSVM:
        performSVM(C=10, identifier=1)
        performSVM(C=0.1, identifier=2)
        performSVM(C=1, identifier=3)
        performSVM(C=100, identifier=4)
        performSVM(gamma=0.01, identifier=5)
        performSVM(gamma=0.001, identifier=6)
