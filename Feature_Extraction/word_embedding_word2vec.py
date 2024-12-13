from gensim.models import Word2Vec
import numpy as np

EMBEDDING_DIMENSION = 8

def fit(sentences, vector_size=EMBEDDING_DIMENSION, window=5, min_count=1, epochs=10):
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, sg=1, epochs=epochs)

    # Add an <UNK> token to the model manually if it doesn't exist
    if "<UNK>" not in model.wv:
        model.wv.add_vector("<UNK>", np.random.rand(vector_size))

    return model

def predict(model, sentence):
    word_embeddings = []
    unknown_count = 0

    for word in sentence:
        if word in model.wv:
            word_embeddings.append(model.wv[word])
        else:
            word_embeddings.append(model.wv["<UNK>"])
            unknown_count += 1

    # Average the word embeddings to get the sentence embedding
    if word_embeddings:
        sentence_embedding = np.mean(word_embeddings, axis=0)
    else:
        sentence_embedding = np.zeros(model.vector_size)

    print(unknown_count)
    return sentence_embedding

# Example usage:

# Training data (tokenized sentences)
train_sentences = [
    ["pizza", "with", "balsamic", "glaze"],
    ["pepperoni", "and", "extra", "roasted", "green", "pepper"]
]

# Test data (tokenized sentences)
test_sentences = [
    ["balsamic", "glaze", "with", "pepperoni"],
    ["roasted", "green", "pepper", "pizza"],
    ["unknown", "word", "here"]
]

# Train the Word2Vec model
model = fit(train_sentences, vector_size=EMBEDDING_DIMENSION, window=3, min_count=1, epochs=10)

# Generate sentence embeddings for test data
sentence_embedding = predict(model, test_sentences[2])
print(sentence_embedding)