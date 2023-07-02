# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    chars, words, sents = get_nltk_corpus("milton-paradise.txt")
    print_nltk_descriptive_statistics(chars, words, sents, manual=True)
    doc = load_spacy_model(chars)
    print_spacy_descriptive_statistics(doc, chars)
    print_nltk_descriptive_statistics(chars, manual=False)
    compare_lowercase_lexicons(chars, words, doc)
    compare_top_N_frequencies(chars, words, doc, n=5)