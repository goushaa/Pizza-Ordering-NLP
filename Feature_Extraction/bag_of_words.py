from sklearn.feature_extraction.text import CountVectorizer

# Training function to build the vocabulary using CountVectorizer
def fit(texts):
    # Initialize the CountVectorizer with the option to handle unknown words
    vectorizer = CountVectorizer(token_pattern=r"[^\s]+")
    
    # Fit the vectorizer on the training data (build the vocabulary)
    X_train = vectorizer.fit_transform(texts)
    
    # Get the vocabulary (the words in the vocabulary)
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulary:", vocabulary)
    
    return vectorizer, vocabulary

# Testing function to convert sentences to Bag of Words vectors
def predict(texts, vectorizer):
    # Transform the test texts using the already fitted vectorizer
    X_test = vectorizer.transform(texts)
    
    # Convert the sparse matrix to dense format and convert to numpy array
    return X_test.toarray()

# Example corpus (list of sentences)
train_texts = ["pizza with balsamic glaze", "pepperoni and extra roasted green pepper"]
test_texts = ["balsamic balsamic with pepperoni", "roasted green pepper pizza", "unknown word here"]

# Train the model (build vocabulary and process tokens)
vectorizer, vocabulary = fit(train_texts)

# Convert the training data to Bag of Words vectors
train_bow_vectors = predict(train_texts, vectorizer)

# Convert the test data to Bag of Words vectors
test_bow_vectors = predict(test_texts, vectorizer)

# Print the results
print("\nTraining Bag of Words Vectors:")
for vector in train_bow_vectors:
    print(vector)

print("\nTest Bag of Words Vectors:")
for vector in test_bow_vectors:
    print(vector)
