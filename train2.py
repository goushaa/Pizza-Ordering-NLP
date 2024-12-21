import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from preprocessing2 import *
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
        total_acc = 0
        total_loss = 0
        
        for tokens, outputs in dataloader:
            tokens, outputs = tokens.to(device), outputs.to(device)
            prediction = model(tokens)

            batch_loss = criterion(prediction.view(-1, prediction.shape[-1]), outputs.view(-1))
            total_loss += batch_loss.item()

            total_acc += (prediction.argmax(dim=-1) == outputs).sum().item()

            optimizer.zero_grad() 
          
            batch_loss.backward()

            optimizer.step()

            
        epoch_loss = total_loss / dataset_len
        epoch_acc = total_acc / (dataset_len * sentence_len)
      
        print(
            f'Epoch: {epoch + 1} \n'
            f'Train Loss: {epoch_loss:.4f}\
            | Train Accuracy: {epoch_acc:.4f}\n'
        )


########################################################################################################################
def evaluate_model(device, model, dataloader, dataset_len):
    total_acc = 0
    total_em = 0
    total_basic_acc = 0 
    total_tokens = 0

    with torch.no_grad():
        for tokens, outputs in tqdm(dataloader):
            tokens, outputs = tokens.to(device), outputs.to(device)
            
            # Get predictions from the model
            prediction = model(tokens)

            # Predicted entities and intents
            pred_outputs = torch.argmax(prediction, dim=-1)
            # print("Actual:",outputs,"\n","Prediction:",pred_outputs)

            # Remove padding 
            filtered_actual_outputs = []
            filtered_predicted_outputs = []

            for batch_idx in range(tokens.shape[0]):  # Iterate over batch
                filtered_actual_outputs.append([entity for token, entity in zip(tokens[batch_idx], outputs[batch_idx]) if token.item() != PAD])
                filtered_predicted_outputs.append([intent for token, intent in zip(tokens[batch_idx], pred_outputs[batch_idx]) if token.item() != PAD])
            # print("Filtered Actual:",filtered_actual_outputs,"\n","Filtered Prediction:",filtered_predicted_outputs)

            for true, pred in zip(filtered_actual_outputs, filtered_predicted_outputs):
                for t, p in zip(true, pred):
                    if t == p:
                        total_basic_acc += 1
                total_tokens += len(true)
            # print("Basic acc:",total_basic_acc,"\nTotal Tokens:",total_tokens)

            # Compute accuracy for outputs (Exact Match)
            total_em += sum(all(torch.tensor(pred) == torch.tensor(true)) for pred, true in zip(filtered_predicted_outputs, filtered_actual_outputs))
            # print("EM acc:",total_em)

            # Compute entity match with modulo sibling order (ignoring padding)
            for i in range(len(filtered_actual_outputs)):
                true_outputs = filtered_actual_outputs[i]
                predicted_outputs = filtered_predicted_outputs[i]

                # Sort both to ignore sibling order but still check if the same set of outputs are predicted
                if sorted(true_outputs) == sorted(predicted_outputs):
                    total_acc += 1

    # Calculate overall accuracy for outputs
    total_em /= dataset_len

    # Calculate Exact Match Accuracy (EM) for outputs
    total_acc /= dataset_len

    # Calculate Basic Accuracy for outputs
    total_basic_acc /= total_tokens

    print(f'\nExact Match Accuracy: {total_em:.4f}')
    print(f'Accuracy with Modulo Sibling Order: {total_acc:.4f}')
    print(f'Basic Accuracy: {total_basic_acc:.4f}')


def evaluate_submission(device, model, dataloader, dataset_len):
    total_acc = 0
    total_em = 0
    total_basic_acc = 0 
    total_tokens = 0

    with torch.no_grad():
        for tokens, outputs in tqdm(dataloader):
            tokens, outputs = tokens.to(device), outputs.to(device)
            
            # Get predictions from the model
            prediction = model(tokens)

            # Predicted entities and intents
            pred_outputs = torch.argmax(prediction, dim=-1)
            # print("Actual:",outputs,"\n","Prediction:",pred_outputs)

            # for out, pred in zip(pred_outputs,outputs)



########################################################################################################################
def train(device, model_name, data_size):
    ################################################### READING FILES ##################################################
    train_data = read_file2("./Dataset/final_pizza_train.json")

    ################################################### PREPROCESSING ##################################################
    # 0-->train // 1-->test
    word_to_int = {}
    word_to_int["<UNK>"] = 0
    word_to_int["<PAD>"] = 1
    VOCAB_SIZE = 2
    train_tokens_tokenized, train_entities, train_intents, word_to_int, VOCAB_SIZE = preprocess_data(train_data, 0, "train.SRC", "train.TOP", VOCAB_SIZE, word_to_int)
    
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
    train_entities_dataset = PizzaDataset2(train_tokens_tokenized, train_entities, PAD)
    train_entities_dataloader = DataLoader(train_entities_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_intents_dataset = PizzaDataset2(train_tokens_tokenized,train_intents, PAD)
    train_intents_dataloader = DataLoader(train_intents_dataset, batch_size=BATCH_SIZE, shuffle=True)

    entities_model = LSTM(VOCAB_SIZE, ENTITIES_LABELS_SIZE)
    entities_model.to(device) # Send model to `device` (GPU/CPU)

    intents_model = LSTM(VOCAB_SIZE, INTENTS_LABELS_SIZE)
    intents_model.to(device) # Send model to `device` (GPU/CPU)


    # Specify loss function
    criterion = nn.CrossEntropyLoss()  # Ignore padding index ignore_index=PAD
    criterion = criterion.to(device)

    # Instantiate the optimizer
    entities_optimizer = torch.optim.Adam(entities_model.parameters(), lr=LEARNING_RATE)
    intents_optimizer = torch.optim.Adam(intents_model.parameters(), lr=LEARNING_RATE)

    print("******************* TRAIN ENTITIES *******************")
    train_model(device, entities_model, train_entities_dataloader, criterion, entities_optimizer, len(train_entities_dataset), train_entities_dataset[0][0].shape[0])
    print("******************* TRAIN INTENTS ********************")
    train_model(device, intents_model, train_intents_dataloader, criterion, intents_optimizer, len(train_intents_dataset), train_intents_dataset[0][0].shape[0])
    torch.save(entities_model, os.path.join('./Models/Saved_Models', model_name+"_entities_"+str(data_size)+".pth"))
    torch.save(intents_model, os.path.join('./Models/Saved_Models', model_name+"_intents_"+str(data_size)+".pth"))


########################################################################################################################
def evaluate(device, model_name, data_size):
    entities_model = torch.load(os.path.join('./Models/Saved_Models', model_name+"_entities_"+str(data_size)+".pth"))
    intents_model = torch.load(os.path.join('./Models/Saved_Models', model_name+"_intents_"+str(data_size)+".pth"))

    # Load the dictionary from the file
    with open(os.path.join('./Dataset',"word_to_int_"+str(data_size)+".json"), "r") as file:
        word_to_int = json.load(file)
    VOCAB_SIZE = len(word_to_int)

    # Remove if needed
    # test_data = read_file("./Dataset/test.json")
    # test_tokens_tokenized, test_entities, test_intents = preprocess_data(test_data, 1, "train.SRC", "train.TOP", VOCAB_SIZE, word_to_int)
    # test_entities_dataset = PizzaDataset2(test_tokens_tokenized, test_entities, PAD)
    # test_entities_dataloader = DataLoader(test_entities_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_intents_dataset = PizzaDataset2(test_tokens_tokenized,test_intents, PAD)
    # test_intents_dataloader = DataLoader(test_intents_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # print("******************* EVAL ENTITIES *******************")
    # evaluate_model(device, entities_model, test_entities_dataloader, len(test_entities_dataset))
    # print("******************* EVAL INTENTS ********************")
    # evaluate_model(device, intents_model, test_intents_dataloader, len(test_intents_dataset))
    
    dev_data = read_file("./Dataset/PIZZA_dev.json")
    dev_tokens_tokenized, dev_entities, dev_intents = preprocess_data(dev_data, 1, "dev.SRC", "dev.TOP", VOCAB_SIZE, word_to_int)
    dev_entities_dataset = PizzaDataset2(dev_tokens_tokenized, dev_entities, PAD)
    dev_entities_dataloader = DataLoader(dev_entities_dataset, batch_size=BATCH_SIZE, shuffle=True)
    dev_intents_dataset = PizzaDataset2(dev_tokens_tokenized,dev_intents, PAD)
    dev_intents_dataloader = DataLoader(dev_intents_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("******************* DEV ENTITIES *******************")
    evaluate_model(device, entities_model, dev_entities_dataloader, len(dev_entities_dataset))
    print("******************* DEV INTENTS ********************")
    evaluate_model(device, intents_model, dev_intents_dataloader, len(dev_intents_dataset))