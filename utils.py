import json
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import torch.optim as optim
from config import *
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

def read_file(file_path):
    data = []
    # Open the file and read it line by line
    with open(file_path, 'r') as file:
        for line in file:
            try:
                # Strip whitespace and convert the line to a JSON object
                json_data = json.loads(line.strip())
                data.append(json_data)  # Append the JSON object to the list
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}")
                print(f"Error details: {e}")
    return data

def data_loader(train_inputs, train_entities, train_intents, eval_inputs, eval_entities, eval_intents, batch_size=BATCH_SIZE):
    # Pad the sequences to the same length
    train_inputs = pad_sequences(train_inputs)
    train_entities = pad_sequences(train_entities)
    train_intents = pad_sequences(train_intents)

    eval_inputs = pad_sequences(eval_inputs)
    eval_entities = pad_sequences(eval_entities)
    eval_intents = pad_sequences(eval_intents)

    # # Convert inputs to PyTorch tensors
    # train_inputs = torch.tensor(train_inputs)
    # train_entities = torch.tensor(train_entities)
    # train_intents = torch.tensor(train_intents)

    # eval_inputs = torch.tensor(eval_inputs)
    # eval_entities = torch.tensor(eval_entities)
    # eval_intents = torch.tensor(eval_intents)

    # Create DataLoader for training data
    train_data = TensorDataset(train_inputs, train_entities, train_intents)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create DataLoader for validation data
    val_data = TensorDataset(eval_inputs, eval_entities, eval_intents)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    return train_dataloader, val_dataloader


def pad_sequences(sequences, max_length=None):
    # If no max_length is provided, pad to the length of the longest sequence
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    # Pad each sequence to max_length
    padded_sequences = [torch.tensor(seq) for seq in sequences]
    padded_sequences = [pad(seq, (0, max_length - len(seq))) for seq in padded_sequences]

    return torch.stack(padded_sequences)