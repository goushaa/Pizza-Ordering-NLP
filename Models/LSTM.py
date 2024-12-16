import torch.nn as nn
from config import *

class LSTM(nn.Module): 
    def __init__(self, vocab_size, label_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=DROPOUT)
        self.output = nn.Linear(hidden_dim*2, label_size)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)  # Shape: (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embeddings)
        output = self.output(lstm_out)  # Shape: (batch_size, seq_len, label_size)
        return output