import torch
from train import train, evaluate
from predict import predict

MODEL_NAME = 'blstm'
DATA_SIZE = 1000

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
    else:
        predict(device, MODEL_NAME, DATA_SIZE) # Predict a sentence in this path: "./Dataset/Input_Output/input.txt"


# 0 --> train // 1 --> evaluate dev_set // 2 --> predict a sentence
main(0)