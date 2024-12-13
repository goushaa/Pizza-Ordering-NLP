from preprocessing import *
from utils import *
from Feature_Extraction.one_hot_vector import *
import time
import torch
import numpy as np
import torch.nn as nn
from Models.BLSTM import *

def train(device, model, optimizer, loss_fn, train_dataloader, eval_dataloader, VOCAB_SIZE):
    # =======================================
    #               Train
    # =======================================
    entities_best_accuracy = 0
    entities_best_loss = float('inf')
    intents_best_accuracy = 0
    intents_best_loss = float('inf')

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Entities Loss':^10} | {'Entities Acc':^9}| {'Intents Loss':^10} | {'Intents Acc':^9} | {'Elapsed':^9}")
    print("-" * 60)

    for epoch_i in range(NUM_EPOCHS):
        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Load batch to GPU
            b_input_ids,  b_entities, b_intents = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            print("VOCAB_SIZE: ",VOCAB_SIZE)
            print(f"b_input_ids shape: {b_input_ids.shape}")
            print(b_input_ids.min(), b_input_ids.max())
            # Perform a forward pass
            entities_output, intents_output = model(b_input_ids)
         
            # Compute loss and accumulate the loss values
            entities_loss = loss_fn(entities_output.view(-1, entities_output.shape[-1]), b_entities.view(-1))
            intents_loss = loss_fn(intents_output.view(-1, intents_output.shape[-1]), b_intents.view(-1))
            total_loss += entities_loss.item() + intents_loss.item()
            
            # Perform a backward pass to calculate gradients
            entities_loss.backward()   #CHECKK THISSS

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Test
        # =======================================
        if eval_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            entities_loss, entities_accuracy, intents_loss, intents_accuracy = evaluate(model, eval_dataloader,loss_fn)

            # Track the best accuracy
            if entities_accuracy > entities_best_accuracy:
                entities_best_accuracy = entities_accuracy
                #torch.save(model.state_dict(), path)
            
            # Track the best loss
            if entities_loss < entities_best_loss:
                entities_best_loss = entities_loss
                #torch.save(model.state_dict(), LOSS_PATH)

            # Track the best accuracy
            if intents_accuracy > intents_best_accuracy:
                intents_best_accuracy = intents_accuracy
                #torch.save(model.state_dict(), path)
            
            # Track the best loss
            if intents_loss < intents_best_loss:
                intents_best_loss = intents_loss
                #torch.save(model.state_dict(), LOSS_PATH)
                
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | {entities_loss:^10.6f} | {entities_accuracy:^9.2f}| {intents_loss:^10.6f} | {intents_accuracy:^9.2f} | {time_elapsed:^9.2f}")

    print("\n")
    print(f"Training complete! Best accuracy: Entities --> {entities_best_accuracy:.2f}% //// Intents --> {intents_best_accuracy:.2f}%.")



def evaluate(model, val_dataloader, loss_fn):
    """
    After the completion of each training epoch, measure the model's performance on our validation set.
    """
    # Put the model into the evaluation mode.
    model.eval()

    # Tracking variables
    entities_loss = []
    intents_loss = []
    entities_accuracy = []
    intents_accuracy = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_entities, b_intents = tuple(t.to(device) for t in batch)

        # Filter out the padding value
        entities_without_pad = (b_entities != PAD)
        intents_without_pad = (b_intents != PAD)

        # Get the output
        with torch.no_grad():
            entities_output, intents_output = model(b_input_ids)

        # Compute loss
        loss = loss_fn(entities_output.view(-1, entities_output.shape[-1]), b_entities.view(-1))
        entities_loss.append(loss.item())
        loss = loss_fn(intents_output.view(-1, intents_output.shape[-1]), b_intents.view(-1))
        intents_loss.append(loss.item())

        # Get the predictions
        entities_prediction = entities_output.argmax(dim=2)
        intents_prediction = intents_output.argmax(dim=2)

        # Calculate the accuracy of entities
        correct_predictions = ((entities_prediction == b_entities) & entities_without_pad).sum().item()
        actual_predictions = entities_without_pad.sum().item()
        accuracy = (correct_predictions / actual_predictions) * 100
        entities_accuracy.append(accuracy)

        # Calculate the accuracy of intents
        correct_predictions = ((intents_prediction == b_intents) & intents_without_pad).sum().item()
        actual_predictions = intents_without_pad.sum().item()
        accuracy = (correct_predictions / actual_predictions) * 100
        intents_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    entities_loss = np.mean(entities_loss)
    entities_accuracy = np.mean(entities_accuracy)
    intents_loss = np.mean(intents_loss)
    intents_accuracy = np.mean(intents_accuracy)

    return entities_loss, entities_accuracy, intents_loss, intents_accuracy




def main(device):
    ################################################### READING FILES ###################################################
    # the file paths should be changed in kaggle
    train_data = read_file("./train.json") 
    test_data = read_file("test.json")
    dev_data = read_file("PIZZA_dev.json")

    ################################################### PREPROCESSING ###################################################
    # 0-->train // 1-->test
    train_tokens_tokenized, train_entities, train_intents, word_to_int, VOCAB_SIZE = preprocess_data(train_data, 0, "train.SRC", "train.TOP")
    test_tokens_tokenized, test_entities, test_intents = preprocess_data(test_data, 1, "train.SRC", "train.TOP", word_to_int)

    ################################################# FEATURE EXTRACTION #################################################
    # one_hot_vectors --> features of each sentence in trained dataset
    # vocab_to_vector --> vector of each word
    #one_hot_vectors, vocab_to_vector = fit(train_tokens_tokenized)

    ###################################################### TRAINING ######################################################
    train_dataloader, val_dataloader = data_loader(train_tokens_tokenized, train_entities, train_intents,test_tokens_tokenized, test_entities, test_intents, BATCH_SIZE)
    
    model = BLSTM(VOCAB_SIZE)
    # Send model to `device` (GPU/CPU)
    model.to(device)

    # Specify loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
    # Instantiate the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


    train(device, model, optimizer, loss_fn, train_dataloader, val_dataloader, VOCAB_SIZE)
    
    

# Test GPU
device = "cpu"
# device_type = None
# if torch.cuda.is_available():
#     device_type = "cuda"
#     device = torch.device(device_type)
#     print(f"There are {torch.cuda.device_count()} GPU(s) available.")
#     print("Device name:", torch.cuda.get_device_name(0))
# else:
#     device_type = "cpu"
#     device = torch.device(device_type)
#     print("No GPU available, using the CPU instead.")
    
main(device)