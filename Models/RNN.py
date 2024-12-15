import torch.nn as nn
from config import *

class RNN(nn.Module): 
    def __init__(self, vocab_size, label_size=LABELS_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, label_size)

    def forward(self, tokens):
        embeddings = self.embedding(tokens) 
        rnn_out, _ = self.rnn(embeddings)
        output = self.output(rnn_out)
        return output