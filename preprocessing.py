import re
from transformers import AutoTokenizer
import json

with open('PIZZA_train_10.json', 'r') as file:
    data = json.load(file)


    def process_text(text):
        # Remove special characters and unnecessary punctuation
        cleaned_text = re.sub(r'[^\w\s]', ' ', text).lower()
        return ' '.join(cleaned_text.split())

    def map_words_to_entities(src_text, top_text):
        # Clean the source text
        cleaned_src = process_text(src_text)
        tokens = cleaned_src.split()
        
        # Initialize labels with 'NONE'
        labels = ['NONE'] * len(tokens)
        
        # Extract entity patterns from TOP
        patterns = [
            (r'\(REQUEST\s+([^\)]+)\)', 'REQUEST'),
            (r'\(NUMBER\s+([^\)]+)\)', 'NUMBER'),
            (r'\(SIZE\s+([^\)]+)\)', 'SIZE'),
            (r'\(TOPPING\s+([^\)]+)\)', 'TOPPING'),
            (r'\(QUANTITY\s+([^\)]+)\)', 'QUANTITY'),
            (r'\(STYLE\s+([^\)]+)\)', 'STYLE'),
            (r'\(DRINKTYPE\s+([^\)]+)\)', 'DRINKTYPE'),
            (r'\(CONTINERTYPE\s+([^\)]+)\)', 'CONTINERTYPE'),
            (r'\(VOLUME\s+([^\)]+)\)', 'VOlUME'),
        ]
        
        # Match each pattern and assign labels
        for pattern, entity in patterns:
            matches = re.finditer(pattern, top_text)
            for match in matches:
                value = process_text(match.group(1))
                for i, token in enumerate(tokens):
                    if token in value.split():
                        labels[i] = entity
        
        return tokens, labels

    # Process each entry
    for entry in data:
        train_SRC = entry['train.SRC']
        train_TOP = entry['train.TOP']
        
        tokens, labels = map_words_to_entities(train_SRC, train_TOP)
        
        print("Train SRC:", train_SRC)
        print("Tokens:", tokens)
        print("Labels:", labels)
        print()