import os
import pickle
import numpy as np

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


def convert_probabilities_to_78x2(data):
    """
    Convert probability dictionaries to shape (78, 2) matrices.
    Each of the 36 characters gets two double values per row.

    Args:
        data: List of dictionaries containing character data

    Returns:
        List of dictionaries with probabilities converted to (78, 2) matrices
    """
    if not data:
        print("No data to convert.")
        return None

    print("Converting probability dictionaries to (78, 2) matrices...")

    # Get the list of characters in the vocabulary
    sample_item = next((item for item in data if "next_char_probabilities" in item), None)
    if not sample_item:
        print("Error: No items with next_char_probabilities found.")
        return None

    # Extract the vocabulary and sort it for consistency
    vocab = sorted(sample_item["next_char_probabilities"].keys())
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Check if we have enough rows to represent all characters
    if vocab_size > 78:
        print(f"Warning: Vocabulary size ({vocab_size}) exceeds target rows (78). Some characters will be omitted.")

    for item in data:
        if "next_char_probabilities" not in item:
            print(f"Warning: Item missing next_char_probabilities. Skipping.")
            continue

        # Create a (78, 2) matrix filled with zeros
        prob_matrix = np.zeros((78, 2))

        # Fill the matrix with probability values
        # Each character gets a row with two values
        for i, char in enumerate(vocab):
            if i < 78:  # Ensure we don't exceed the matrix dimensions
                # Get the probability for this character
                prob = item["next_char_probabilities"].get(char)

                # Set both values in the row to the probability
                prob_matrix[i*2, 0] = prob
                prob_matrix[i*2, 1] = prob
                prob_matrix[i*2+1, 0] = prob
                prob_matrix[i*2+1, 1] = prob

        # Add the probability matrix to the item as converted_data
        item["prob_chunk"] = prob_matrix

        # Remove the original next_char_probabilities to save space
        # del item["next_char_probabilities"]

    print(f"Converted {len(data)} items.")
    return data



def convert_probabilities_to_78x64(data):
    """
    Convert each 'next_char_probabilities' into a (78, 64) matrix
    where each row is filled with the same repeated probability pattern.
    Append this matrix to the existing 'eeg_chunk', resulting in 31 matrices.

    Args:
        data: List of dictionaries with 'next_char_probabilities' and 'eeg_chunk'

    Returns:
        Updated list with 'eeg_chunk' extended by one matrix per item
    """
    if not data:
        print("No data to convert.")
        return None

    print("Converting probability dictionaries to (78, 64) matrices...")

    # Find vocab keys from any valid item
    sample_item = next((item for item in data if "next_char_probabilities" in item), None)
    if not sample_item:
        print("Error: No items with next_char_probabilities found.")
        return None

    vocab = sorted(sample_item["next_char_probabilities"].keys())
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    if vocab_size != 37:
        print(f"Warning: Expected 37 probabilities, got {vocab_size}. Adjusting fill pattern accordingly.")

    for item in data:
        if "next_char_probabilities" not in item or "eeg_chunk" not in item:
            print(f"Warning: Skipping item with missing keys.")
            continue

        # Ensure eeg_chunk is a list
        if isinstance(item["eeg_chunk"], np.ndarray):
            item["eeg_chunk"] = list(item["eeg_chunk"])
        elif not isinstance(item["eeg_chunk"], list):
            continue

        if len(item["eeg_chunk"]) != 30:
            print(f"Skipping item with unexpected eeg_chunk length: {len(item['eeg_chunk'])}")
            continue

        # Create the repeated 64-length pattern
        probs = [item["next_char_probabilities"].get(char, 0.0) for char in vocab[:37]]
        repeated_row = (probs * ((64 // len(probs)) + 1))[:64]  # Repeat and truncate to 64

        # Create matrix (78 rows, each same as repeated_row)
        prob_matrix = np.tile(repeated_row, (78, 1)).astype(np.float32)

        # Append to eeg_chunk
        item["eeg_chunk"].append(prob_matrix)

    print(f"Successfully converted and extended {len(data)} items.")
    return data

