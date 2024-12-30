


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