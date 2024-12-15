import torch.nn as nn
from config import *

class BLSTM(nn.Module): 
    def __init__(self, vocab_size, label_size=LABELS_SIZE, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(BLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=DROPOUT)
        self.fc_entity = nn.Linear(hidden_dim*2, label_size)
        self.fc_intent = nn.Linear(hidden_dim*2, label_size)

    def forward(self, tokens):
        embeddings = self.embedding(tokens)  # Shape: (batch_size, seq_len, embedding_dim)
        blstm_out, _ = self.blstm(embeddings)
        entity_logits = self.fc_entity(blstm_out)  # Shape: (batch_size, seq_len, label_size)
        intent_logits = self.fc_intent(blstm_out)  # Shape: (batch_size, seq_len, label_size)
        return entity_logits, intent_logits