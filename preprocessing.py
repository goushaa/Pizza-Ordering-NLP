import re
from nltk.tokenize import word_tokenize

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

########################################################################################################################
def process_text(text):
    # Remove special characters and unnecessary punctuation
    cleaned_text = re.sub(r'[^\w\s]', ' ', text).lower()
    return ' '.join(cleaned_text.split())

########################################################################################################################
def map_words_to_entities(tokens, top_text):
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
    
    return entities

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


def map_words_to_intents(tokens, top_text):
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
    return labels

########################################################################################################################
def train_tokens_to_integers(tokens, word_to_int, VOCAB_SIZE):
    # Generate integers for each unique word
    int_tokens = []
    for token in tokens:
        if token not in word_to_int:
            word_to_int[token] = VOCAB_SIZE
            VOCAB_SIZE += 1
        int_tokens.append(word_to_int[token])
    
    return int_tokens, VOCAB_SIZE

def test_tokens_to_integers(tokens, word_to_int):
    # Generate integers for each unique word
    int_tokens = []
    for token in tokens:
        if token not in word_to_int:
            int_tokens.append(0) #Needs Revision
        else:
            int_tokens.append(word_to_int[token])    
    return int_tokens

########################################################################################################################
def preprocess_data(data, type, srcKey, topKey, word_to_int = None): # Type: 0-->train // 1-->test
# Process each entry
    all_tokens_tokenized = []
    all_entities = []
    all_intents = []

    word_to_int = {}
    word_to_int["<UNK>"] = 0
    VOCAB_SIZE = 1
    
    for entry in data:
        train_SRC = entry[srcKey]
        train_TOP = entry[topKey]
        cleaned_src = process_text(train_SRC)
        tokens = word_tokenize(cleaned_src)
        entities = map_words_to_entities(tokens, train_TOP)
        intents = map_words_to_intents(tokens, train_TOP)

        # Convert entity and intent labels to integers using label_map
        entity_indices = [label_map[entity] for entity in entities]
        intent_indices = [label_map[intent] for intent in intents]
        if (type==0):
            tokenized_input, VOCAB_SIZE = train_tokens_to_integers(tokens, word_to_int, VOCAB_SIZE)
        else:
            tokenized_input = test_tokens_to_integers(tokens, word_to_int)

        # Append the arrays for the current entry to the containers
        all_tokens_tokenized.append(tokenized_input)
        all_entities.append(entity_indices)
        all_intents.append(intent_indices)

    if (type==0):
        return all_tokens_tokenized, all_entities, all_intents, word_to_int, VOCAB_SIZE
    else:
        return all_tokens_tokenized, all_entities, all_intents