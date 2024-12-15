import json
from torch.utils.data import Dataset
import torch
from config import *
from torch.nn.utils.rnn import pad_sequence
import random

def read_file(file_path, num_rows=1000000):
    data = []
    row_count = 0  # Counter to track the number of rows read

    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            if row_count >= num_rows:  # Stop once we reach the desired number of rows
                break
            try:
                # Strip whitespace and convert the line to a JSON object
                json_data = json.loads(line.strip())
                data.append(json_data)  # Append the JSON object to the list
                row_count += 1
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}")
                print(f"Error details: {e}")

    # Shuffle the data
    random.shuffle(data)

    return data

# Create a dataset class
class PizzaDataset(Dataset):
    def __init__(self, tokens, entities,intents, pad_idx):
        self.tokens = pad_sequence([torch.tensor(i) for i in tokens], batch_first=True, padding_value=pad_idx)
        self.entities = pad_sequence([torch.tensor(j) for j in entities], batch_first=True, padding_value=0)
        self.intents = pad_sequence([torch.tensor(j) for j in intents], batch_first=True, padding_value=0)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.entities[idx], self.intents[idx]
