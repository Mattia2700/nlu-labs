# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    vectorizers = load_vectorizers()
    for vectorizer_id, vectorizer in vectorizers.items():
        data = fetch_20newsgroups(subset="all", shuffle=True, random_state=42)
        data.data = vectorizer.fit_transform(data.data)
        test_vectorizer(vectorizer_id, data)
