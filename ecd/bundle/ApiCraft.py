import torch
import torch.nn as nn

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

def load_nlp_model(vocab_size, path="../../model/api/char_predictor.pth"):
    model = CharPredictor(vocab_size)
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"Model loaded from {path}.")
    return model

