import numpy as np
import warnings
from sklearn.preprocessing import scale

contributor_selected = "I"
contributor_train_file_path = '../data/Contributor_' + contributor_selected + '_Train.mat'
contributor_test_file_path = '../data/Contributor_' + contributor_selected + '_Test.mat'
channel_name_file_path = '../data/channels.csv'
channels = [i for i in range(64)]
warnings.filterwarnings('ignore')
# %%
from scipy.io import loadmat
from scipy import signal
from bundle.DataCraft import *

data_train = loadmat(contributor_train_file_path)
signals_train = data_train['Signal']
flashing_train = data_train['Flashing']
stimulus_train = data_train['StimulusType']
word_train = data_train['TargetChar']
sampling_frequency = 240
repetitions = 15
recording_duration_train = (len(signals_train)) * (len(signals_train[0])) / (sampling_frequency * 60)
trials_train = len(word_train[0])

print("Train Data:")
print_data(signals_train, word_train, contributor_selected, sampling_frequency)

# %%
# Application of butterworth filter
b, a = signal.butter(4, [0.1 / sampling_frequency, 20 / sampling_frequency], 'bandpass')
for trial in range(trials_train):
    signals_train[trial, :, :] = signal.filtfilt(b, a, signals_train[trial, :, :], axis=0)

# Down-sampling of the signals from 240Hz to 120Hz
down_sampling_frequency = 120
SCALE_FACTOR = round(sampling_frequency / down_sampling_frequency)
sampling_frequency = down_sampling_frequency

print("# Samples of EEG signals before downsampling: {}".format(len(signals_train[0])))

signals_train = signals_train[:, 0:-1:SCALE_FACTOR, :]
flashing_train = flashing_train[:, 0:-1:SCALE_FACTOR]
stimulus_train = stimulus_train[:, 0:-1:SCALE_FACTOR]

print("# Samples of EEG signals after downsampling: {}".format(len(signals_train[0])))
# %%
# Number of EEG channels
N_CHANNELS = len(channels)
# Window duration after each flashing [ms]
WINDOW_DURATION = 650
# Number of samples of each window
WINDOW_SAMPLES = round(sampling_frequency * (WINDOW_DURATION / 1000))
# Number of samples for each character in trials
SAMPLES_PER_TRIAL = len(signals_train[0])

train_features = []
train_labels = []

count_positive = 0
count_negative = 0

for trial in range(trials_train):
    for sample in (range(SAMPLES_PER_TRIAL)):
        if (sample == 0) or (flashing_train[trial, sample - 1] == 0 and flashing_train[trial, sample] == 1):
            lower_sample = sample
            upper_sample = sample + WINDOW_SAMPLES
            window = signals_train[trial, lower_sample:upper_sample, :]
            # Features extraction
            train_features.append(window)
            # Labels extraction
            if stimulus_train[trial, sample] == 1:
                count_positive += 1
                train_labels.append(1)  # Class P300
            else:
                count_negative += 1
                train_labels.append(0)  # Class no-P300

# Get negative-positive classes ratio
train_ratio = count_negative / count_positive

# Convert lists to numpy arrays
train_features = np.array(train_features)
train_labels = np.array(train_labels)

# 3D Tensor shape (SAMPLES, 64, 78)
dim_train = train_features.shape
print("Features tensor shape: {}".format(dim_train))

# Data normalization Zi = (Xi - mu) / sigma
for pattern in range(len(train_features)):
    train_features[pattern] = scale(train_features[pattern], axis=0)

