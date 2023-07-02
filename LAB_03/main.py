# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    test_set = [
        "william shakespeare was an english poet",
        "he was born in 1564",
        "he died in 1616",
    ]
    ngram_order = 3
    padded_ngrams_oov, flat_text_oov, lex = get_dataset_and_vocab(ngram_order)
    evaluate_my_model(ngram_order, lex, padded_ngrams_oov, flat_text_oov, test_set)
    padded_ngrams_oov, flat_text_oov, lex = get_dataset_and_vocab(ngram_order)
    evaluate_nltk_model(ngram_order, lex, padded_ngrams_oov, flat_text_oov, test_set)
