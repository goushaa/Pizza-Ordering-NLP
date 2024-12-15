import torch.nn as nn
from config import *

class CNN(nn.Module):
    def __init__(self, vocab_size, num_classes=LABELS_SIZE, embedding_dim=EMBEDDING_DIM, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS):
        super(CNN, self).__init__()
        
        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional Layer
        self.conv1d = nn.Conv1d(embedding_dim, 256, kernel_size=3, padding=1)
        
        # LSTM Layer
        self.lstm = nn.LSTM(256, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=DROPOUT)
        
        # Linear Layer
        self.linear = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, sentences):
        embeddings = self.embedding(sentences)
        
        # Convolutional Layer
        conv_out = self.conv1d(embeddings.permute(0, 2, 1))
        conv_out = nn.functional.relu(conv_out)
        
        # LSTM Layer
        lstm_out, _ = self.lstm(conv_out.permute(0, 2, 1))
        
        # Linear Layer
        output = self.linear(lstm_out)

        return output