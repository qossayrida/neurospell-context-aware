import matplotlib.pyplot as plt


# Define the 6x6 matrix of characters
characters = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '0']
]


# Function to create an image of the matrix with highlighted rows and columns
def create_p300_image(highlight_row=None, highlight_col=None, title="P300 Speller Matrix"):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw the grid
    for i in range(7):
        ax.axhline(i, color='black', linewidth=1)
        ax.axvline(i, color='black', linewidth=1)

    # Fill the matrix
    for i in range(6):
        for j in range(6):
            ax.text(j + 0.5, 5 - i + 0.5, characters[i][j], ha='center', va='center',
                    fontsize=16, fontweight='bold', color='black')

    # Highlight row
    if highlight_row is not None:
        ax.add_patch(plt.Rectangle((0, 5 - highlight_row), 6, 1, color='yellow', alpha=0.5))

    # Highlight column
    if highlight_col is not None:
        ax.add_patch(plt.Rectangle((highlight_col, 0), 1, 6, color='cyan', alpha=0.5))

    # Set limits and title
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis('off')
    plt.title(title, fontsize=18)
    plt.show()


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