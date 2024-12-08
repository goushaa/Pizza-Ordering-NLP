import re
import json

# Create entity and intent label mapping
label_map = {
    'NONE': 0,
    'BPIZZAORDER': 1, 'IPIZZAORDER': 2,
    'BDRINKORDER': 3, 'IDRINKORDER': 4,
    'BCOMPLEX_TOPPING': 5, 'ICOMPLEX_TOPPING': 6,
    'BNUMBER': 7, 'INUMBER': 8,
    'BSIZE': 9, 'ISIZE': 10,
    'BVOLUME': 11, 'IVOLUME': 12,
    'BCONTAINERTYPE': 13, 'ICONTAINERTYPE': 14,
    'BDRINKTYPE': 15, 'IDRINKTYPE': 16,
    'BSTYLE': 17, 'ISTYLE': 18,
    'BQUANTITY': 19, 'IQUANTITY': 20,
    'BTOPPING': 21, 'ITOPPING': 22
}
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
        entities = ['NONE'] * len(tokens)
        
        # Define entity patterns
        patterns = [
            (r'\(NUMBER\s+([^\)]+)\)', 'NUMBER'),
            (r'\(SIZE\s+([^\)]+)\)', 'SIZE'),
            (r'\(TOPPING\s+([^\)]+)\)', 'TOPPING'),
            (r'\(QUANTITY\s+([^\)]+)\)', 'QUANTITY'),
            (r'\(STYLE\s+([^\)]+)\)', 'STYLE'),
            (r'\(DRINKTYPE\s+([^\)]+)\)', 'DRINKTYPE'),
            (r'\(CONTAINERTYPE\s+([^\)]+)\)', 'CONTAINERTYPE'),
            (r'\(VOLUME\s+([^\)]+)\)', 'VOLUME'),
        ]
        
        # Match each pattern and assign B/I labels
        for pattern, entity in patterns:
            matches = re.finditer(pattern, top_text)
            for match in matches:
                value = process_text(match.group(1))
                value_tokens = value.split()
                entity_started = False
                
                for i, token in enumerate(tokens):
                    if token in value_tokens:
                        if not entity_started:  # Beginning of the entity
                            entities[i] = f'B{entity}'
                            entity_started = True
                            value_tokens.remove(token)  # Avoid duplicate matches
                        else:  # Continuation of the entity
                            entities[i] = f'I{entity}'
                            value_tokens.remove(token)
        
        return tokens, entities
  
    def extract_nested_block(text, start_pattern):
        """
        Extracts the block of text starting with `start_pattern` and handles nested parentheses.
        """
        start_index = text.find(start_pattern)
        if start_index == -1:
            return None  # Pattern not found

        stack = []
        end_index = start_index
        for i, char in enumerate(text[start_index:], start=start_index):
            if char == '(':
                stack.append('(')
            elif char == ')':
                stack.pop()
                if not stack:  # Stack is empty, block is fully closed
                    end_index = i + 1
                    break

        return text[start_index:end_index] if not stack else None


    def map_words_to_intents(src_text, top_text):
        # Clean the source text
        cleaned_src = process_text(src_text)
        tokens = cleaned_src.split()

        # Initialize labels with 'NONE'
        labels = ['NONE'] * len(tokens)

        # Define orders and their intents
        orders = {
            '(PIZZAORDER': 'PIZZAORDER',
            '(DRINKORDER': 'DRINKORDER'
        }

        # Process each order type
        for start_pattern, intent in orders.items():
            block = extract_nested_block(top_text, start_pattern)
            if block:
                order_words = process_text(block).split()
                intent_started = False
                for i, token in enumerate(tokens):
                    if token in order_words:
                        if not intent_started:
                            labels[i] = f'B{intent}'
                            intent_started = True
                        else:
                            labels[i] = f'I{intent}'
                        order_words.remove(token)

        # Extract COMPLEX_TOPPING block and override where applicable
        complextopping_block = extract_nested_block(top_text, '(COMPLEX_TOPPING')
        if complextopping_block:
            complextopping_words = process_text(complextopping_block).split()
            intent_started = False
            for i, token in enumerate(tokens):
                if token in complextopping_words:
                    if not intent_started:  # Beginning of the intent
                        labels[i] = 'BCOMPLEX_TOPPING'
                        intent_started = True
                    else:  # Continuation of the intent
                        labels[i] = 'ICOMPLEX_TOPPING'
                    complextopping_words.remove(token)
        return tokens, labels

    # Declare global variables
    word_to_int = {}
    current_int = 23  # Start from 23 to avoid conflict with label_map

    def tokens_to_integers(tokens):
        global word_to_int, current_int  # Access the global variables
        
        # Generate integers for each unique word
        int_tokens = []
        for token in tokens:
            if token not in word_to_int:
                word_to_int[token] = current_int
                current_int += 1
            int_tokens.append(word_to_int[token])
        
        return int_tokens

    # Process each entry
    for entry in data:
        train_SRC = entry['train.SRC']
        train_TOP = entry['train.TOP']
        
        tokens, entites = map_words_to_entities(train_SRC, train_TOP)
        tokens, intents = map_words_to_intents(train_SRC, train_TOP)
        # Convert entity and intent labels to integers using label_map
        entity_indices = [label_map[entity] for entity in entites]
        intent_indices = [label_map[intent] for intent in intents]
        print("Train SRC:", train_SRC)
        print("Tokens:", tokens)
        tokenized_input = tokens_to_integers(tokens = tokens)
        print("Tokenized Input:", tokenized_input)
        print("Entites:", entites)
        print("Entity Indices:", entity_indices)
        print("Intents:", intents)
        print("Intent Indices:", intent_indices)
        print()