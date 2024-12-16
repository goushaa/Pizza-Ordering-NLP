import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from preprocessing import *
from utils import *
from Models.BLSTM import *
from Models.LSTM import *
from Models.RNN import *
from Models.CNN import *
from Feature_Extraction.one_hot_vector import *
from Feature_Extraction.bag_of_words import *
from Feature_Extraction.trainable_embeddings import *
from Feature_Extraction.word_embedding_word2vec import *

def train_model(device, model, dataloader, criterion, optimizer, dataset_len, sentence_len):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_acc_entities = 0
        total_acc_intents = 0
        total_loss_entities = 0
        total_loss_intents = 0
        
        for tokens, entities, intents in dataloader:
            tokens, entities, intents = tokens.to(device), entities.to(device), intents.to(device)
            output_entities, output_intents = model(tokens)
            batch_loss_entities = criterion(output_entities.view(-1, output_entities.shape[-1]), entities.view(-1))
            batch_loss_intents = criterion(output_intents.view(-1, output_intents.shape[-1]), intents.view(-1))
            total_loss_entities += batch_loss_entities.item()
            total_loss_intents += batch_loss_intents.item()

            acc_entities = (output_entities.argmax(dim=-1) == entities).sum().item()
            acc_intents = (output_intents.argmax(dim=-1) == intents).sum().item()
            total_acc_entities += acc_entities
            total_acc_intents += acc_intents

            optimizer.zero_grad() 
            total_loss = batch_loss_entities + batch_loss_intents
            total_loss.backward()

            optimizer.step()

            
        epoch_loss_entities = total_loss_entities / dataset_len
        epoch_acc_entities = total_acc_entities / (dataset_len * sentence_len)
    
        epoch_loss_intents = total_loss_intents / dataset_len
        epoch_acc_intents = total_acc_intents / (dataset_len * sentence_len)        
        print(
            f'Epoch: {epoch + 1} \n'
            f'Train Loss (Entities): {epoch_loss_entities:.4f}\
            | Entities Accuracy: {epoch_acc_entities:.4f}\n'
            f'Train Loss (Intents): {epoch_loss_intents:.4f} \
            | Intents Accuracy: {epoch_acc_intents:.4f}'
        )


########################################################################################################################
def evaluate_model(device, model, dataloader, dataset_len):
    total_acc_entities = 0
    total_acc_intents = 0
    total_em_entities = 0
    total_em_intents = 0

    with torch.no_grad():
        for tokens, entities, intents in tqdm(dataloader):
            tokens, entities, intents = tokens.to(device), entities.to(device), intents.to(device)
            
            # Get predictions from the model
            output_entities, output_intents = model(tokens)
            
            # Predicted entities and intents
            pred_entities = torch.argmax(output_entities, dim=-1)
            pred_intents = torch.argmax(output_intents, dim=-1)

            # Remove padding 
            filtered_actual_entities = []
            filtered_actual_intents = []
            filtered_predicted_entities = []
            filtered_predicted_intents = []

            for batch_idx in range(tokens.shape[0]):  # Iterate over batch
                filtered_actual_entities.append([entity for token, entity in zip(tokens[batch_idx], entities[batch_idx]) if token.item() != PAD])
                filtered_actual_intents.append([intent for token, intent in zip(tokens[batch_idx], intents[batch_idx]) if token.item() != PAD])
                filtered_predicted_entities.append([entity for token, entity in zip(tokens[batch_idx], pred_entities[batch_idx]) if token.item() != PAD])
                filtered_predicted_intents.append([intent for token, intent in zip(tokens[batch_idx], pred_intents[batch_idx]) if token.item() != PAD])
            
            # Compute accuracy for entities (Exact Match)
            em_entities = sum(all(torch.tensor(pred) == torch.tensor(true)) for pred, true in zip(filtered_predicted_entities, filtered_actual_entities))
            total_em_entities += em_entities


            # Compute accuracy for intents (Exact Match)
            em_intents = sum(all(torch.tensor(pred) == torch.tensor(true)) for pred, true in zip(filtered_predicted_intents, filtered_actual_intents))
            total_em_intents += em_intents
            
            # Compute entity match with modulo sibling order (ignoring padding)
            for i in range(len(filtered_actual_entities)):
                true_entities = filtered_actual_entities[i]
                predicted_entities = filtered_predicted_entities[i]

                # Sort both to ignore sibling order but still check if the same set of entities are predicted
                if sorted(true_entities) == sorted(predicted_entities):
                    total_acc_entities += 1

            # Compute intent accuracy
            for i in range(len(filtered_actual_intents)):
                true_intents = filtered_actual_intents[i]
                predicted_intents = filtered_predicted_intents[i]

                # Sort both to ignore sibling order but still check if the same set of intents are predicted
                if sorted(true_intents) == sorted(predicted_intents):
                    total_acc_intents += 1

    # Calculate overall accuracy for entities and intents
    total_em_entities /= dataset_len
    total_em_intents /= dataset_len

    # Calculate Exact Match Accuracy (EM) for entities and intents
    total_acc_entities /= dataset_len
    total_acc_intents /= dataset_len

    print(f'\nExact Match Accuracy (Entities): {total_em_entities:.4f}')
    print(f'Exact Match Accuracy (Intents): {total_em_intents:.4f}')
    print(f'Accuracy with Modulo Sibling Order (Entities): {total_acc_entities:.4f}')
    print(f'Accuracy with Modulo Sibling Order (Intents): {total_acc_intents:.4f}')


########################################################################################################################
def train(device, model_name, data_size):
    ################################################### READING FILES ##################################################
    train_data = read_file("./Dataset/PIZZA_train.json", data_size)

    ################################################### PREPROCESSING ##################################################
    # 0-->train // 1-->test
    word_to_int = {}
    word_to_int["<UNK>"] = 0
    word_to_int["<PAD>"] = 1
    VOCAB_SIZE = 2
    train_tokens_tokenized, train_entities, train_intents, word_to_int, VOCAB_SIZE = preprocess_data(train_data, 0, "train.SRC", "train.TOP", VOCAB_SIZE,word_to_int)
    
    # Save dictionary to a file
    with open(os.path.join('./Dataset',"word_to_int_"+str(data_size)+".json"), "w") as file:
        json.dump(word_to_int, file, indent=2)

    # print(train_tokens_tokenized,train_entities,train_intents)
    # print(test_tokens_tokenized,test_entities,test_intents)

    ################################################# FEATURE EXTRACTION ################################################
    # one_hot_vectors --> features of each sentence in trained dataset
    # vocab_to_vector --> vector of each word
    #one_hot_vectors, vocab_to_vector = fit(train_tokens_tokenized)

    ###################################################### TRAINING #####################################################    
    # Create the DataLoader
    train_dataset = PizzaDataset(train_tokens_tokenized, train_entities,train_intents, PAD)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if model_name == 'blstm':
        model = BLSTM(VOCAB_SIZE)
    elif model_name == 'lstm':
        model = None
    else:
        model = None
    model.to(device) # Send model to `device` (GPU/CPU)

    # Specify loss function
    criterion = nn.CrossEntropyLoss()  # Ignore padding index ignore_index=PAD
    criterion = criterion.to(device)

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model(device, model, train_dataloader, criterion, optimizer, len(train_dataset), train_dataset[0][0].shape[0])
    torch.save(model, os.path.join('./Models/Saved_Models', model_name+str(data_size)+".pth"))


########################################################################################################################
def evaluate(device, model_name, data_size):
    model = torch.load(os.path.join('./Models/Saved_Models', model_name+str(data_size)+".pth"))

    # Load the dictionary from the file
    with open(os.path.join('./Dataset',"word_to_int_"+str(data_size)+".json"), "r") as file:
        word_to_int = json.load(file)
    VOCAB_SIZE = len(word_to_int)

    # Remove if needed
    test_data = read_file("./Dataset/test.json")
    test_tokens_tokenized, test_entities, test_intents = preprocess_data(test_data, 1, "train.SRC", "train.TOP", VOCAB_SIZE, word_to_int)
    test_dataset = PizzaDataset(test_tokens_tokenized, test_entities,test_intents, PAD)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    evaluate_model(device, model, test_dataloader, len(test_dataset))
    
    dev_data = read_file("./Dataset/PIZZA_dev.json")
    dev_tokens_tokenized, dev_entities, dev_intents = preprocess_data(dev_data, 1, "dev.SRC", "dev.TOP", VOCAB_SIZE, word_to_int)
    dev_dataset = PizzaDataset(dev_tokens_tokenized, dev_entities,dev_intents, PAD)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=True)
    evaluate_model(device, model, dev_dataloader, len(dev_dataset))  