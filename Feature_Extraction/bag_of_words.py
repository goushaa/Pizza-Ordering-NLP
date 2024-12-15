from sklearn.feature_extraction.text import CountVectorizer

def fit(texts):
    # Initialize the CountVectorizer with the option to handle unknown words
    vectorizer = CountVectorizer(token_pattern=r"[^\s]+")
    
    # Fit the vectorizer on the training data (build the vocabulary)
    X_train = vectorizer.fit_transform(texts)
    
    # Get the vocabulary (the words in the vocabulary)
    vocabulary = vectorizer.get_feature_names_out()
    print("Vocabulary:", vocabulary)
    
    return vectorizer, vocabulary


def predict(texts, vectorizer):
    # Transform the test texts using the already fitted vectorizer
    X_test = vectorizer.transform(texts)
    
    # Convert the sparse matrix to dense format and convert to numpy array
    return X_test.toarray()