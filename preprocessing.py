import json

def preprocess_pizza_data(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.readlines()
            
        for line in data:
            if line.strip():  # Check if the line is not empty
                pizza_order = json.loads(line)
                print("Source:", pizza_order["train.SRC"])
                print("EXR:", pizza_order["train.EXR"])
                print("TOP:", pizza_order["train.TOP"])
                print("TOP-DECOUPLED:", pizza_order["train.TOP-DECOUPLED"])
                print("-" * 50)  # Separator for readability
                
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

# Replace 'PIZZA_train.json' with the path to your file
preprocess_pizza_data('PIZZA_train.json')
