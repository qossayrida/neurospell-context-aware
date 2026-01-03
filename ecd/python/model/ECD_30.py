# ===============================
# 1. Imports
# ===============================
import sys
import os
import pickle
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_sentence_eeg_prob_data(sentences_eeg_filepath):
    """Loads the final processed data list from a pickle file."""
    print(f"Attempting to load processed data from: {sentences_eeg_filepath}")
    if not os.path.exists(sentences_eeg_filepath):
        print(f"Error: File not found at {sentences_eeg_filepath}.")
        return None
    try:
        with open(sentences_eeg_filepath, "rb") as f:
            data = pickle.load(f)
        print(f"Successfully loaded processed data. Total samples: {len(data)}")
        if isinstance(data, list):
            return data
        else:
            print(f"Error: Loaded object is not a list (type: {type(data)}). Returning None.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred during loading processed data: {e}")
        return None

# ===============================
# 2. Constants and Configs
# ===============================
NUM_CLASSES = 36  # 26 letters + 9 digits + 1 underscore
MODEL_SAVE_PATH = "trained_eegcnn_model_selected_channels_set1.pth"

# Check CUDA availability and compatibility
if torch.cuda.is_available():
    try:
        # Test with actual Conv3d operation to catch sm_61 incompatibility
        test_conv = torch.nn.Conv3d(1, 1, kernel_size=3, padding=1).cuda()
        test_input = torch.zeros(1, 1, 4, 4, 4).cuda()
        _ = test_conv(test_input)
        del test_conv, test_input, _
        torch.cuda.empty_cache()
        DEVICE = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    except Exception as e:
        print(f"CUDA incompatible with this GPU: {str(e)[:100]}")
        print("Falling back to CPU (training will be slower but functional)")
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")
    print(f"Using device: CPU")

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
        chunk = np.array(self.data[idx]["eeg_chunk"], dtype=np.float32)[:30]  # (30, 78, 64) - Take first 30 windows
        chunk = chunk[:, :, SELECTED_CHANNELS]  # Now (30, 78, 8)
        label = self.label_encoder.transform([self.data[idx]["character"]])[0]
        return torch.tensor(chunk).unsqueeze(0), torch.tensor(label)  # Shape: (1, 30, 78, 8)

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
        # Input: (B, 1, 30, 78, 8)
        x = F.relu(self.bn1(self.conv1(x)))       # -> (B, 32, 30, 78, 8)
        x = F.relu(self.bn2(self.conv2(x)))       # -> (B, 64, 30, 78, 8)
        x = self.se1(x)
        x = F.relu(self.bn3(self.conv3(x)))       # -> (B, 128, 30, 78, 8)
        x = self.se2(x)
        x = self.pool(x).squeeze()                # -> (B, 128)
        x = self.dropout(F.relu(self.fc1(x)))     # -> (B, 64)
        return self.fc2(x)                        # -> (B, NUM_CLASSES)

# ===============================
# 6. Load and Prepare Data
# ===============================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

train_data = load_sentence_eeg_prob_data("../../data/sentences_eeg_train.pkl")
test_data = load_sentence_eeg_prob_data("../../data/sentences_eeg_val.pkl")

if not train_data:
    raise ValueError("Training data failed to load. Check path or preprocessing.")
if not test_data:
    raise ValueError("Test/Validation data failed to load. Check path or preprocessing.")

# Create label encoder from all data
all_labels = [item["character"] for item in train_data] + [item["character"] for item in test_data]
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

train_dataset = EEGDataset(train_data, label_encoder)
test_dataset = EEGDataset(test_data, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(test_dataset)}")

# ===============================
# 7. Training Function
# ===============================
def train_model(model, loader, optimizer, criterion, epochs=10):
    model.train()
    loss_history = []
    start_time = time.time()
    convergence_epoch = None
    convergence_threshold = 0.01  # Loss change threshold for convergence
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            if i % 10 == 0:
                print(f"[Epoch {epoch+1}] Batch {i+1}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / num_batches
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs} complete. Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
        # Check for convergence
        if convergence_epoch is None and epoch > 5:
            if abs(loss_history[-1] - loss_history[-2]) < convergence_threshold:
                convergence_epoch = epoch + 1
                convergence_time = time.time() - start_time
                print(f"*** Convergence detected at epoch {convergence_epoch} (Time: {convergence_time:.2f}s) ***")
        
        torch.cuda.empty_cache()

    total_training_time = time.time() - start_time
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    
    return loss_history, total_training_time, convergence_epoch

# ===============================
# 8. Evaluation Function
# ===============================
def evaluate_model(model, loader, test_data, label_encoder):
    model.eval()
    all_preds = []
    all_targets = []
    all_subjects = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Extract subject information from test data
    for item in test_data:
        all_subjects.append(item.get("subject", "Unknown"))
    
    # Overall metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"\nAll Test Data:")
    print(f"   Accuracy  = {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision = {precision:.4f}")
    print(f"   Recall    = {recall:.4f}")
    print(f"   F1-Score  = {f1:.4f}")
    
    # Per-subject metrics
    unique_subjects = sorted(set(all_subjects))
    print(f"\nPer-Subject Results ({len(unique_subjects)} subjects):")
    print("=" * 80)
    
    for subject in unique_subjects:
        subject_indices = [i for i, s in enumerate(all_subjects) if s == subject]
        if len(subject_indices) == 0:
            continue
            
        subject_preds = [all_preds[i] for i in subject_indices]
        subject_targets = [all_targets[i] for i in subject_indices]
        
        subj_acc = accuracy_score(subject_targets, subject_preds)
        subj_prec = precision_score(subject_targets, subject_preds, average='weighted', zero_division=0)
        subj_rec = recall_score(subject_targets, subject_preds, average='weighted', zero_division=0)
        subj_f1 = f1_score(subject_targets, subject_preds, average='weighted', zero_division=0)
        
        print(f"\nSubject {subject} ({len(subject_indices)} samples):")
        print(f"   Accuracy  = {subj_acc:.4f} ({subj_acc*100:.2f}%)")
        print(f"   Precision = {subj_prec:.4f}")
        print(f"   Recall    = {subj_rec:.4f}")
        print(f"   F1-Score  = {subj_f1:.4f}")
    
    return accuracy, precision, recall, f1

# ===============================
# 9. Plot Training Loss
# ===============================
def plot_loss_curve(loss_history, convergence_epoch=None):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    
    if convergence_epoch is not None:
        plt.axvline(x=convergence_epoch, color='r', linestyle='--', 
                   label=f'Convergence (Epoch {convergence_epoch})')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Loss vs Epoch', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = 'training_loss_curve.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nLoss curve saved to: {plot_path}")
    plt.close()

# ===============================
# 10. Main Execution
# ===============================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    model = EEGCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    loss_history, total_time, convergence_epoch = train_model(
        model, train_loader, optimizer, criterion, epochs=200
    )

    # Print training summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if convergence_epoch:
        print(f"Convergence Epoch: {convergence_epoch}")
    else:
        print("Convergence: Not detected (loss still decreasing)")
    print(f"Final Training Loss: {loss_history[-1]:.4f}")

    # Plot loss curve
    plot_loss_curve(loss_history, convergence_epoch)

    # Evaluate the model
    evaluate_model(model, test_loader, test_data, label_encoder)

    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION COMPLETE")
    print("=" * 80)

