import os
import pickle

# remove this channel from the list
electrode_names_to_remove = ['Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6', 'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4',
                'Cp6', 'Af7', 'Af3', 'Afz', 'Af4', 'Af8', 'Ft7', 'Ft8', 'Tp7', 'Tp8', 'Po7', 'Po3', 'Poz', 'Po4', 'Po8']


def print_data(signals, word, contributor_selected, sampling_frequency):

    recording_duration= (len(signals)) * (len(signals[0])) / (sampling_frequency * 60)
    trials = len(word[0])


    print("{:<15} {:<20} {:<20} {:<10} {:<30}".format(
        "Contributor", "Sampling Freq. (Hz)", "Recording (min)", "Trials", "Spelled Word"
    ))
    print("=" * 110)

    # Break the Spelled Word into chunks of 30 characters
    word_chunks = [''.join(word)[i:i + 30] for i in range(0, len(''.join(word)), 30)]

    # Print the first line of the row
    print("{:<15} {:<20} {:<20} {:<10} {:<30}".format(
    contributor_selected,
    "{:.2f}".format(sampling_frequency),
    "{:.2f}".format(recording_duration),
    trials,
    word_chunks[0] if word_chunks else ""
    ))

    # Print the remaining lines for the Spelled Word, if any
    for chunk in word_chunks[1:]:
        print("{:<15} {:<20} {:<20} {:<10} {:<30}".format(
            "", "", "", "", chunk
        ))

    print()


# --- Helper Function to Load Sample Groups ---
def load_characters_eeg(characters_eeg_filepath="../../data/characters_eeg.pkl"):
    """Loads the grouped samples dictionary from a pickle file."""
    print(f"Attempting to load sample groups from: {characters_eeg_filepath}")
    if not os.path.exists(characters_eeg_filepath):
        print(f"Error: File not found at {characters_eeg_filepath}. Please run the grouping script first.")
        return None
    try:
        with open(characters_eeg_filepath, "rb") as f:
            loaded_groups = pickle.load(f)
        print("Successfully loaded sample groups dictionary.")
        if not isinstance(loaded_groups, dict):
            print(f"Error: Loaded object is not a dictionary (type: {type(loaded_groups)}). Returning None.")
            return None
        # Check if it contains data
        if not any(loaded_groups.values()):
             print("Warning: Loaded sample groups dictionary appears empty.")
        return loaded_groups
    except Exception as e:
        print(f"An unexpected error occurred during loading sample groups: {e}")
        return None

# --- Optional: Add a function to load the final data ---
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

def load_characters(characters_file_path="../../data/characters.txt"):
    """Load characters from the specified file and add a space character."""
    try:
        with open(characters_file_path, "r") as f:
            chars = f.read().strip()
        # Add space character to the set
        chars += " "
        print(f"Loaded {len(chars)} characters from {characters_file_path} (including added space)")
        return chars
    except FileNotFoundError:
        print(f"Warning: Characters file not found at {characters_file_path}. Using default character set.")
        # Default character set if file not found
        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789_ "