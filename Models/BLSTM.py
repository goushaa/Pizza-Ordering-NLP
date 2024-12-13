import torch.nn as nn
from config import *

class BLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes=LABELS_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(BLSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional LSTM layer
        self.blstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=DROPOUT)
        
        # Linear layers
        self.linear_entity = nn.Linear(hidden_size * 2, n_classes)
        self.linear_intent = nn.Linear(hidden_size * 2, 5) ####CHECKKKKKKKKKK THISSSSS

    def forward(self, sentences):
        embeddings = self.embedding(sentences)
        
        blstm_out, _ = self.blstm(embeddings)
        
        entity_output = self.linear_entity(blstm_out)
        intent_output = self.linear_intent(blstm_out)

        return entity_output, intent_output