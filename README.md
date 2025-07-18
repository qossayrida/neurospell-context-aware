
# Context-Aware P300 Speller System Using EEG and Language Models

This project presents a hybrid Brain-Computer Interface (BCI) system that enhances character-level communication for individuals with severe motor impairments. The system combines electroencephalography (EEG) signals with context-aware predictions from a Large Language Model (LLM) to improve accuracy and speed in P300-based speller systems.



## Overview

- **Goal**: Enable users to spell characters using brain signals alone.
- **Method**: Integrate deep EEG signal decoding with language model predictions in a multimodal neural architecture.
- **Architecture**:
  - **EEG-based Character Decoder (ECD)**: A 3D CNN model that classifies spatio-temporal EEG responses.
  - **Language Model (LSTM)**: Predicts likely next characters based on textual context.
  - **Fusion Mechanism**: Combines both signals using dynamic language probability scaling.


## Dataset

- EEG data collected from two contributors using a 6x6 P300 speller matrix.
- Preprocessing steps include:
  - Butterworth bandpass filtering (0.1–20Hz)
  - Downsampling from 240Hz to 120Hz
  - Z-score normalization
- EEG windows of shape `(78, 64)` aligned with language model outputs.


## Technologies Used

- Python, PyTorch
- NumPy, SciPy, Matplotlib
- Scikit-learn
- EEG Signal Processing
- Deep Learning (3D CNN, LSTM)
- NLP (Character-level language modeling)


## Results

- EEG-only classification: ~75% accuracy  
- Language model (LSTM) top-3 accuracy: ~65%  
- Combined EEG + LLM model: **~90% character-level accuracy**  
- Central lobe electrodes (Cz, Fz, Pz, etc.) showed highest signal quality.


## Authors

- **Qossay Rida**  
- **Mohammad Zidan** 

Supervised by **Dr. Wasel Ghanem**  
Faculty of Engineering & Technology, Birzeit University, 2025


## Future Work

- Real-time deployment for clinical use  
- Adaptive interfaces that learn user preferences  
- Integration with other biosignals (e.g., EMG, eye-tracking)
