import os
import pickle

def load_sentence_eeg_prob_data(sentences_eeg_filepath="../../data/sentences_eeg.pkl"):
    """Loads the final processed data list from a pickle file."""
    print(f"Attempting to load processed data from: {sentences_eeg_filepath}")
    if not os.path.exists(sentences_eeg_filepath):
        print(f"Error: File not found at {sentences_eeg_filepath}.")
        return None
    try:
        with open(sentences_eeg_filepath, "rb") as f:
            data = pickle.load(f)
        print("Successfully loaded processed data.")
        if isinstance(data, list):
            return data
        else:
            print(f"Error: Loaded object is not a list (type: {type(data)}). Returning None.")
            return None
    except Exception as e:
        print(f"An unexpected error occurred during loading processed data: {e}")
        return None