# %% md
# Character Prediction with LSTM - Jupyter Notebook Version
# %%
# ========================
# 1. Imports
# ========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import string

import nltk

nltk.download('gutenberg')
from nltk.corpus import gutenberg

# %% md
# 1. Characters Setup
# %%
# Define character set (a-z, A-Z, 0-9)
all_chars = list(string.ascii_lowercase + string.ascii_uppercase + string.digits)
char2idx = {ch: idx for idx, ch in enumerate(all_chars)}
idx2char = {idx: ch for ch, idx in char2idx.items()}
vocab_size = len(all_chars)

print("Vocabulary size:", vocab_size)


# %% md
# 2. Model Definition
# %%
class CharPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super(CharPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        _, (hidden, _) = self.lstm(embed)
        hidden = hidden.squeeze(0)
        out = self.fc(hidden)
        return out


# %% md
# 3. Training Function
# %%
def train_model(model, dataset, epochs=5, seq_len=10, batch_size=64):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for i in range(0, len(dataset) - seq_len - 1, batch_size):
            inputs = []
            targets = []

            for b in range(batch_size):
                idx = i + b
                if idx + seq_len >= len(dataset) - 1:
                    break

                seq = dataset[idx: idx + seq_len]
                target = dataset[idx + seq_len]

                inputs.append([char2idx[ch] for ch in seq if ch in char2idx])
                targets.append(char2idx[target])

            if not inputs:
                continue

            inputs = torch.tensor(inputs)
            targets = torch.tensor(targets)

            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")


# %% md
# 4. Prediction Function
# %%
def predict_next_chars(model, sentence, top_k=5):
    model.eval()
    with torch.no_grad():
        input_seq = [char2idx[ch] for ch in sentence if ch in char2idx]
        if not input_seq:
            raise ValueError("Input sentence must contain at least one known character.")

        input_seq = torch.tensor(input_seq).unsqueeze(0)
        output = model(input_seq)
        probs = F.softmax(output, dim=-1).squeeze(0)

        top_probs, top_indices = torch.topk(probs, top_k)

        result = {}
        for prob, idx in zip(top_probs, top_indices):
            result[idx2char[idx.item()]] = round(prob.item(), 4)

        return result


# %% md
# 5. Save and Load Model
# %%
def save_model(model, path="../../model/api/char_predictor.pth"):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")


def load_model(path="../../model/api/char_predictor.pth"):
    model = CharPredictor(vocab_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}.")
    return model


# %% md
# 6. Train the Model
# %%

mode = "predict"  # <<< Change to "predict" when needed

if mode == "train":
    model = CharPredictor(vocab_size)

    # Load bigger dataset
    book_ids = [
        'austen-emma.txt',
        'bible-kjv.txt',
        'blake-poems.txt',
        'melville-moby_dick.txt',
        'shakespeare-macbeth.txt',
    ]

    # Merge all books
    text = ""
    for book_id in book_ids:
        text += gutenberg.raw(book_id)

    allowed_chars = set(string.ascii_lowercase + string.ascii_uppercase + string.digits)
    text = "".join(ch for ch in text if ch in allowed_chars)

    print(f"Training dataset size: {len(text)} characters.")

    # Train the model
    train_model(model, text, epochs=10)

    # Save the model
    save_model(model)

elif mode == "predict":
    # ========================
    # 8. Load & Predict
    # ========================
    model = load_model()

    # Provide test input
    test_sentence = "A DARK HAND HEL"
    result = predict_next_chars(model, test_sentence, top_k=5)

    # Display result
    print("Prediction Probabilities:")
    print(json.dumps(result, indent=2))

    # Show most likely completion
    best_char = max(result, key=result.get)
    completed_sentence = test_sentence + best_char
    print("Completed sentence:", completed_sentence)