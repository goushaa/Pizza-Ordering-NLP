from sklearn.preprocessing import OneHotEncoder
import numpy as np

def fit(tokens):
    # Flatten the list of tokens and create a vocabulary
    vocab = set(token for sentence in tokens for token in sentence)
    vocab.add("<UNK>")  # Add <UNK> token for unknown words
    vocab=sorted(vocab)

    # Flatten the tokens and fit the encoder
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(np.array(vocab).reshape(-1, 1))
    
    # Mapping vocabulary to one-hot vectors
    vocab_to_vector = {token: vector for token, vector in zip(vocab, encoded)}
    
    # Encode each sentence
    # Assign <UNK> vector for unknown words
    one_hot_vectors = [
        predict(vocab_to_vector, sentence)
        for sentence in tokens
    ]
    
    return one_hot_vectors, vocab_to_vector

def predict(vocab_to_vector, sentence_tokens):
    one_hot_vector = [vocab_to_vector.get(token, vocab_to_vector["<UNK>"]) for token in sentence_tokens]
    return one_hot_vector
