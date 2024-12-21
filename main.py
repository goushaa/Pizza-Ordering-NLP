import torch
import pandas as pd
from train2 import train, evaluate
from predict2 import predict, predict_JSON
from pizza.utils.semantic_matchers import *
from pizza.utils.entity_resolution import PizzaSkillEntityResolver
from reformat_results import *
from utils import *

MODEL_NAME = 'lstm'
DATA_SIZE = 100000

# 0 --> train // 1 --> predict
def main(type):
    # Test GPU
    device = None
    device_type = None
    if torch.cuda.is_available():
        device_type = "cuda"
        device = torch.device(device_type)
        print(f"There are {torch.cuda.device_count()} GPU(s) available.")
        print("Device name:", torch.cuda.get_device_name(0))
    else:
        device_type = "cpu"
        device = torch.device(device_type)
        print("No GPU available, using the CPU instead.")

    if type==0:
        train(device, MODEL_NAME, DATA_SIZE)
    elif type==1:
        evaluate(device, MODEL_NAME, DATA_SIZE)
    elif type==2:
        predict_JSON(device, MODEL_NAME, DATA_SIZE) # Predict a sentence in this path: "./Dataset/Input_Output/input.txt"
    elif type==3:
        resolver = PizzaSkillEntityResolver()
        dev_data = read_file2("./Dataset/PIZZA_dev.json")
        accuracy = 0
        i = 1
        for entry in dev_data:
            dev_SRC = entry["dev.SRC"]
            dev_TOP = entry["dev.TOP"]
            pred_JSON = predict(device, MODEL_NAME, DATA_SIZE, dev_SRC)
            #print(pred_JSON)
            pred_TOP = parse_tree(pred_JSON)
            accuracy += is_semantics_only_unordered_exact_match_post_ER_top_top(pred_TOP, dev_TOP, resolver)
            print("Done: ",i)
            i+=1

        print(accuracy/len(dev_data))
    else:
        # Load the CSV file
        df = pd.read_csv("./Dataset/test_set.csv")

        output = {}

        # Loop through the rows and store sentences in the array, indexed by id
        for index, row in df.iterrows():
            #sentences[row['id']] = row['order']
            pred_JSON = predict(device, MODEL_NAME, DATA_SIZE, row['order'])
            #print(pred_JSON)
            pred_TOP = parse_tree(pred_JSON)
            output[row['id']] = pred_TOP

     
        predictions_df = pd.DataFrame(list(output.items()), columns=['id', 'output'])

        # Save the updated DataFrame back to a CSV file
        predictions_df.to_csv("17.csv", index=False)
        

# 0 --> train // 1 --> evaluate dev_set // 2 --> predict a sentence
main(3)