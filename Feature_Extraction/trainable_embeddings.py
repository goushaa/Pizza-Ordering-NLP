import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIMENSION = 8

# Training function
def fit(sentences, embedding_dim=EMBEDDING_DIMENSION, max_len=None):
    # Initialize tokenizer and handle unknown words
    tokenizer = Tokenizer(oov_token="<UNK>")
    tokenizer.fit_on_texts(sentences)

    # Convert texts to sequences and pad them
    sequences = tokenizer.texts_to_sequences(sentences)
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding="post")

    # Define a simple embedding model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        input_length=max_len, 
        trainable=True
    )

    model = tf.keras.Sequential([
        embedding_layer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, tokenizer, padded_sequences, max_len

# Prediction function
def predict(model, tokenizer, sentence, max_len):
    # Tokenize and pad the input sentence
    sequence = tokenizer.texts_to_sequences([sentence])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding="post")

    # Get embeddings from the model's embedding layer
    embedding_layer = model.layers[0]
    word_embeddings = embedding_layer(padded_sequence).numpy()[0]

    # Count unknown words
    unknown_token_index = tokenizer.word_index.get("<UNK>")
    unknown_count = sum(1 for idx in sequence[0] if idx == unknown_token_index)

    # Average the embeddings to get the sentence embedding
    sentence_embedding = np.mean(word_embeddings, axis=0)
    return sentence_embedding, unknown_count

# Example usage
train_sentences = [
    "pizza with balsamic glaze",
    "pepperoni and extra roasted green pepper",
    "roasted pizza with cheese",
    "balsamic glaze and roasted pepper",
]

test_sentence = "balsamic glaze with unknown words"

# Train the embedding layer and tokenize training data
model, tokenizer, padded_sequences, max_len = fit(train_sentences, embedding_dim=EMBEDDING_DIMENSION)

# Dummy labels for training
labels = np.array([1, 0, 1, 0])
model.fit(padded_sequences, labels, epochs=5, verbose=1)

# Predict embedding for a test sentence
sentence_embedding, unknown_count = predict(model, tokenizer, test_sentence, max_len)
print("Sentence Embedding:", sentence_embedding)
print("Unknown Words Count:", unknown_count)