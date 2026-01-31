#!/usr/bin/env python3
"""
Comprehensive EEG Model Training Script

Trains 36 models with different configurations:
- Type 1: EEG only (78 or 36 samples) 
- Type 2: EEG + Probability data (78+36 or 36+36)
- Window sizes: 78, 36
- Sensor configurations: 64, 16, 8 channels  
- Repetitions: 5, 10, 15

For each contributor (I, II), trains 18 models, totaling 36 models.
Each model outputs: .pth model, .txt results, .csv convergence data
"""

import sys
import os
import pickle
import time
import csv
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
try:
    from bundle.DataCraft import *
except ImportError:
    print("Warning: Could not import DataCraft. Make sure bundle module is available.")

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Sensor configurations
SENSOR_CONFIGS = {
    64: list(range(64)),  # All channels
    16: [9, 11, 13, 32, 34, 36, 49, 51, 53, 56, 57, 59, 60, 61],
    8: [10, 33, 48, 50, 52, 55, 59, 61]
}

# Training configurations
CONTRIBUTORS = ["I", "II"]  
WINDOW_SIZES = [78, 36]
REPETITIONS = [5, 10, 15]
SENSOR_COUNTS = [64, 16, 8]
DATA_TYPES = ["eeg_only", "eeg_with_prob"]

NUM_CLASSES = 36
PROB_WINDOW_SIZE = 36

print(f"Total models to train: {len(DATA_TYPES) * len(WINDOW_SIZES) * len(SENSOR_COUNTS) * len(REPETITIONS)} per contributor")
print(f"Total across all contributors: {len(CONTRIBUTORS) * len(DATA_TYPES) * len(WINDOW_SIZES) * len(SENSOR_COUNTS) * len(REPETITIONS)}")


class EEGOnlyDataset(Dataset):
    """Dataset for EEG data only (without probability matrix)"""
    def __init__(self, data, label_encoder, selected_channels, window_size):
        self.data = data
        self.label_encoder = label_encoder
        self.selected_channels = selected_channels
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get EEG data only (first window_size rows)
        eeg_data = self.data[idx]["eeg_with_prob"][:self.window_size, :]  # Shape: (window_size, 64)
        eeg_data = eeg_data[:, self.selected_channels]  # Shape: (window_size, n_channels)
        
        # Add dimensions for 3D CNN: (1, 1, window_size, n_channels)
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        label = self.label_encoder.transform([self.data[idx]["character"]])[0]
        return eeg_data, torch.tensor(label, dtype=torch.long)


class EEGWithProbDataset(Dataset):
    """Dataset for EEG data with probability matrix"""
    def __init__(self, data, label_encoder, selected_channels, window_size):
        self.data = data
        self.label_encoder = label_encoder
        self.selected_channels = selected_channels
        self.window_size = window_size
        self.total_size = window_size + PROB_WINDOW_SIZE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get full data (EEG + probability matrix)
        full_data = self.data[idx]["eeg_with_prob"]  # Shape: (window_size+36, 64)
        full_data = full_data[:, self.selected_channels]  # Shape: (window_size+36, n_channels)
        
        # Add dimensions for 3D CNN: (1, 1, window_size+36, n_channels) 
        full_data = torch.tensor(full_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        label = self.label_encoder.transform([self.data[idx]["character"]])[0]
        return full_data, torch.tensor(label, dtype=torch.long)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AdaptiveEEGCNN(nn.Module):
    def __init__(self, input_time_steps, input_channels, num_classes=NUM_CLASSES):
        super(AdaptiveEEGCNN, self).__init__()
        
        # Adaptive architecture based on input size
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.bn1 = nn.BatchNorm3d(16)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.bn2 = nn.BatchNorm3d(32)
        self.se1 = SEBlock(32)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.bn3 = nn.BatchNorm3d(64)
        self.se2 = SEBlock(64)
        
        # Adaptive pooling and fully connected layers
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 1))
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (B, 1, 1, time_steps, channels)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se2(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    """
    Train model and return training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    convergence_epoch = None
    convergence_threshold = 0.001
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_acc = accuracy_score(val_targets, val_preds)
        val_accuracies.append(val_acc)
        
        # Check for convergence
        if convergence_epoch is None and epoch > 10:
            if len(train_losses) >= 5:
                recent_losses = train_losses[-5:]
                loss_variance = np.var(recent_losses)
                if loss_variance < convergence_threshold:
                    convergence_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'final_val_accuracy': val_accuracies[-1],
        'convergence_epoch': convergence_epoch,
        'training_time': total_time
    }


def save_results(config, results, model_path, results_path, convergence_path):
    """
    Save model results to files
    """
    # Save convergence data to CSV
    convergence_data = {
        'epoch': list(range(1, len(results['train_losses']) + 1)),
        'train_loss': results['train_losses'],
        'val_accuracy': results['val_accuracies']
    }
    pd.DataFrame(convergence_data).to_csv(convergence_path, index=False)
    
    # Save detailed results to text file
    with open(results_path, 'w') as f:
        f.write(f"Model Configuration:\n")
        f.write(f"Contributor: {config['contributor']}\n")
        f.write(f"Data Type: {config['data_type']}\n")
        f.write(f"Window Size: {config['window_size']}\n")
        f.write(f"Sensor Count: {config['sensor_count']}\n")
        f.write(f"Repetitions: {config['repetitions']}\n")
        f.write(f"\nResults:\n")
        f.write(f"Final Validation Accuracy: {results['final_val_accuracy']:.4f}\n")
        f.write(f"Training Time: {results['training_time']:.2f} seconds\n")
        f.write(f"Convergence Epoch: {results['convergence_epoch']}\n")
        f.write(f"\nModel Path: {model_path}\n")
        f.write(f"Convergence Data: {convergence_path}\n")


def load_data_for_config(contributor, window_size, repetitions, data_type):
    """
    Load training and validation data for specific configuration
    """
    # Construct file paths - both types use the same files
    train_path = f"../../data/sentences_eeg_train_{contributor}_window{window_size}_{repetitions}_rep.pkl"
    val_path = f"../../data/sentences_eeg_val_{contributor}_window{window_size}_{repetitions}_rep.pkl"
    
    # Load data
    try:
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        
        print(f"Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        return train_data, val_data
        
    except FileNotFoundError as e:
        print(f"Data file not found: {e}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def main():
    """Main training function"""
    # Create results directory
    results_dir = "../../model/comprehensive_results"
    os.makedirs(results_dir, exist_ok=True)

    # Summary results
    all_results = []
    model_counter = 0
    total_models = len(CONTRIBUTORS) * len(DATA_TYPES) * len(WINDOW_SIZES) * len(SENSOR_COUNTS) * len(REPETITIONS)

    print(f"Starting training of {total_models} models...\n")

    for contributor in CONTRIBUTORS:
        for data_type in DATA_TYPES:
            for window_size in WINDOW_SIZES:
                for sensor_count in SENSOR_COUNTS:
                    for repetitions in REPETITIONS:
                        model_counter += 1
                        
                        config = {
                            'contributor': contributor,
                            'data_type': data_type,
                            'window_size': window_size,
                            'sensor_count': sensor_count,
                            'repetitions': repetitions
                        }
                        
                        print(f"\n{'='*80}")
                        print(f"Model {model_counter}/{total_models}")
                        print(f"Config: {config}")
                        print(f"{'='*80}")
                        
                        # Load data
                        train_data, val_data = load_data_for_config(
                            contributor, window_size, repetitions, data_type
                        )
                        
                        if train_data is None or val_data is None:
                            print(f"Skipping model {model_counter} due to data loading error")
                            continue
                        
                        # Create label encoder
                        all_labels = [item["character"] for item in train_data] + [item["character"] for item in val_data]
                        label_encoder = LabelEncoder()
                        label_encoder.fit(all_labels)
                        
                        # Get selected channels
                        selected_channels = SENSOR_CONFIGS[sensor_count]
                        
                        # Create datasets
                        if data_type == "eeg_only":
                            train_dataset = EEGOnlyDataset(train_data, label_encoder, selected_channels, window_size)
                            val_dataset = EEGOnlyDataset(val_data, label_encoder, selected_channels, window_size)
                            input_time_steps = window_size
                        else:  # eeg_with_prob
                            train_dataset = EEGWithProbDataset(train_data, label_encoder, selected_channels, window_size)
                            val_dataset = EEGWithProbDataset(val_data, label_encoder, selected_channels, window_size)
                            input_time_steps = window_size + PROB_WINDOW_SIZE
                        
                        # Create data loaders
                        batch_size = min(32, len(train_dataset) // 4)  # Adaptive batch size
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        
                        # Create model
                        model = AdaptiveEEGCNN(
                            input_time_steps=input_time_steps,
                            input_channels=sensor_count,
                            num_classes=NUM_CLASSES
                        ).to(DEVICE)
                        
                        # File paths
                        model_name = f"model_{contributor}_{data_type}_w{window_size}_s{sensor_count}_r{repetitions}"
                        model_path = os.path.join(results_dir, f"{model_name}.pth")
                        results_path = os.path.join(results_dir, f"{model_name}_results.txt")
                        convergence_path = os.path.join(results_dir, f"{model_name}_convergence.csv")
                        
                        # Train model
                        print(f"Training {model_name}...")
                        results = train_model(model, train_loader, val_loader, epochs=100)
                        
                        # Save model
                        torch.save(model.state_dict(), model_path)
                        
                        # Save results
                        save_results(config, results, model_path, results_path, convergence_path)
                        
                        # Add to summary
                        summary_result = config.copy()
                        summary_result.update({
                            'final_accuracy': results['final_val_accuracy'],
                            'convergence_epoch': results['convergence_epoch'],
                            'training_time': results['training_time'],
                            'model_path': model_path
                        })
                        all_results.append(summary_result)
                        
                        print(f"Completed {model_name} | Accuracy: {results['final_val_accuracy']:.4f}")
                        
                        # Clean up GPU memory
                        del model, train_loader, val_loader, train_dataset, val_dataset
                        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"{'='*80}")

    # Save summary CSV
    summary_df = pd.DataFrame(all_results)
    summary_path = os.path.join(results_dir, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"Summary saved to: {summary_path}")
    print(f"\nTotal models trained: {len(all_results)}")

    # Display summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for data_type in DATA_TYPES:
        for window_size in WINDOW_SIZES:
            for sensor_count in SENSOR_COUNTS:
                subset = summary_df[
                    (summary_df['data_type'] == data_type) & 
                    (summary_df['window_size'] == window_size) & 
                    (summary_df['sensor_count'] == sensor_count)
                ]
                if not subset.empty:
                    avg_acc = subset['final_accuracy'].mean()
                    print(f"{data_type} | Window {window_size} | {sensor_count} sensors: {avg_acc:.4f} avg accuracy")

    print(f"\nAll results saved to: {results_dir}")


if __name__ == "__main__":
    main()