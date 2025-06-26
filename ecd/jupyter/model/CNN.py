# ===============================
# 1. Imports
# ===============================
import sys
import os
sys.path.append(os.path.abspath("../../"))  # Adjust path to find 'bundle' folder
sys.path.append("/home/wghanem/GraduationProject/BrainWaveResearch/ecd/bundle")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from bundle.DataCraft import load_sentence_eeg_prob_data

# ===============================
# 2. Constants and Configs
# ===============================
NUM_CLASSES = 36  # 26 letters + 9 digits + 1 underscore
MODEL_SAVE_PATH = "trained_eegcnn_model_selected_channels.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

SELECTED_CHANNELS = [10, 33, 48, 50, 52, 55, 59, 61]  # Only 8 selected EEG channels

# ===============================
# 3. Dataset Class
# ===============================
class EEGDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = np.array(self.data[idx]["eeg_chunk"], dtype=np.float32)  # (31, 78, 64)
        chunk = chunk[:, :, SELECTED_CHANNELS]  # Now (31, 78, 8)
        chunk[30] *= 3.0  # Emphasize the prediction timestep
        label = self.label_encoder.transform([self.data[idx]["character"]])[0]
        return torch.tensor(chunk).unsqueeze(0), torch.tensor(label)  # Shape: (1, 31, 78, 8)

# ===============================
# 4. Squeeze-and-Excite Block
# ===============================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

# ===============================
# 5. EEGCNN Model
# ===============================
class EEGCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(EEGCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.se1 = SEBlock(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.se2 = SEBlock(128)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: (B, 1, 31, 78, 8)
        x = F.relu(self.bn1(self.conv1(x)))       # -> (B, 32, 31, 78, 8)
        x = F.relu(self.bn2(self.conv2(x)))       # -> (B, 64, 31, 78, 8)
        x = self.se1(x)
        x = F.relu(self.bn3(self.conv3(x)))       # -> (B, 128, 31, 78, 8)
        x = self.se2(x)
        x = self.pool(x).squeeze()                # -> (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))     # -> (B, 64)
        return self.fc2(x)                        # -> (B, NUM_CLASSES)

# ===============================
# 6. Load and Prepare Data
# ===============================
raw_data = load_sentence_eeg_prob_data()
if not raw_data:
    raise ValueError("Data failed to load. Check path or preprocessing.")

all_labels = [item["character"] for item in raw_data]
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

train_dataset = EEGDataset(train_data, label_encoder)
test_dataset = EEGDataset(test_data, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ===============================
# 7. Training Function
# ===============================
def train_model(model, loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}] Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1}/{epochs} complete. Total Loss: {total_loss:.4f}")
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")

# ===============================
# 8. Evaluation Function
# ===============================
def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    print(f"Test Accuracy: {correct / total:.2%}")

# ===============================
# 9. Train and Evaluate
# ===============================
model = EEGCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_model(model, train_loader, optimizer, criterion, epochs=50)
evaluate_model(model, test_loader)

# ===============================
# 10. Reload Trained Model (Optional)
# ===============================
model = EEGCNN()
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()
