#!/usr/bin/env python3
"""
Test script to verify training setup works before running all 36 models
"""

import sys
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Test configuration - single model
TEST_CONFIG = {
    'contributor': 'I',
    'data_type': 'eeg_only', 
    'window_size': 78,
    'sensor_count': 8,
    'repetitions': 5
}

SENSOR_CONFIGS = {
    64: list(range(64)),
    16: [9, 11, 13, 32, 34, 36, 49, 51, 53, 56, 57, 59, 60, 61],
    8: [10, 33, 48, 50, 52, 55, 59, 61]
}

def test_data_loading():
    """Test data loading"""
    print("Testing data loading...")
    
    config = TEST_CONFIG
    train_path = f"../../data/sentences_eeg_train_{config['contributor']}_window{config['window_size']}_{config['repetitions']}_rep.pkl"
    val_path = f"../../data/sentences_eeg_val_{config['contributor']}_window{config['window_size']}_{config['repetitions']}_rep.pkl"
    
    try:
        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(val_path, 'rb') as f:
            val_data = pickle.load(f)
        
        print(f"✓ Loaded {len(train_data)} training samples, {len(val_data)} validation samples")
        
        # Check data structure
        sample = train_data[0]
        print(f"✓ Sample keys: {list(sample.keys())}")
        print(f"✓ EEG data shape: {sample['eeg_with_prob'].shape}")
        print(f"✓ Character: {sample['character']}")
        
        return train_data, val_data
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None


class TestEEGOnlyDataset(torch.utils.data.Dataset):
    """Test dataset for EEG data only"""
    def __init__(self, data, label_encoder, selected_channels, window_size):
        self.data = data
        self.label_encoder = label_encoder
        self.selected_channels = selected_channels
        self.window_size = window_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eeg_data = self.data[idx]["eeg_with_prob"][:self.window_size, :]
        eeg_data = eeg_data[:, self.selected_channels]
        eeg_data = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        label = self.label_encoder.transform([self.data[idx]["character"]])[0]
        return eeg_data, torch.tensor(label, dtype=torch.long)


class TestEEGCNN(nn.Module):
    """Simple test model"""
    def __init__(self, input_time_steps, input_channels, num_classes=36):
        super(TestEEGCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 1), padding=(1, 1, 0))
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 1))
        self.fc1 = nn.Linear(16 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)


def test_dataset_and_model():
    """Test dataset and model creation"""
    print("\nTesting dataset and model...")
    
    # Load test data
    train_data, val_data = test_data_loading()
    if not train_data or not val_data:
        return False
    
    try:
        # Create label encoder
        all_labels = [item["character"] for item in train_data] + [item["character"] for item in val_data]
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        print(f"✓ Label encoder created with {len(label_encoder.classes_)} classes")
        
        # Create dataset
        config = TEST_CONFIG
        selected_channels = SENSOR_CONFIGS[config['sensor_count']]
        train_dataset = TestEEGOnlyDataset(train_data, label_encoder, selected_channels, config['window_size'])
        
        print(f"✓ Dataset created with {len(train_dataset)} samples")
        
        # Test data loading
        sample_input, sample_label = train_dataset[0]
        print(f"✓ Sample input shape: {sample_input.shape}")
        print(f"✓ Sample label: {sample_label}")
        
        # Create data loader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        batch_input, batch_labels = next(iter(train_loader))
        print(f"✓ Batch input shape: {batch_input.shape}")
        print(f"✓ Batch labels shape: {batch_labels.shape}")
        
        # Create model
        model = TestEEGCNN(
            input_time_steps=config['window_size'],
            input_channels=config['sensor_count']
        ).to(DEVICE)
        
        print(f"✓ Model created and moved to {DEVICE}")
        
        # Test forward pass
        batch_input = batch_input.to(DEVICE)
        with torch.no_grad():
            output = model(batch_input)
        print(f"✓ Forward pass successful, output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in dataset/model test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test a few training steps"""
    print("\nTesting training steps...")
    
    try:
        # Load data and setup
        train_data, val_data = test_data_loading()
        if not train_data:
            return False
            
        config = TEST_CONFIG
        all_labels = [item["character"] for item in train_data]
        label_encoder = LabelEncoder()
        label_encoder.fit(all_labels)
        
        selected_channels = SENSOR_CONFIGS[config['sensor_count']]
        train_dataset = TestEEGOnlyDataset(train_data[:100], label_encoder, selected_channels, config['window_size'])  # Use only 100 samples
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        model = TestEEGCNN(config['window_size'], config['sensor_count']).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("✓ Training setup complete")
        
        # Test training for a few steps
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            if i >= 3:  # Only test 3 batches
                break
                
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            print(f"✓ Training step {i+1}: Loss = {loss.item():.4f}")
        
        print("✓ Training steps successful!")
        return True
        
    except Exception as e:
        print(f"✗ Error in training test: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("EEG MODEL TRAINING - SETUP TEST")
    print("="*60)
    
    print(f"Test configuration: {TEST_CONFIG}")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_data_loading():
        tests_passed += 1
    
    if test_dataset_and_model():
        tests_passed += 1
        
    if test_training_step():
        tests_passed += 1
    
    print("\n" + "="*60)
    print(f"TESTS COMPLETED: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! Ready to run full training.")
        print("Run: python comprehensive_training.py")
    else:
        print("✗ Some tests failed. Please fix issues before full training.")
    print("="*60)


if __name__ == "__main__":
    main()