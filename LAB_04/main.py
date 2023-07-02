# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    trn_data, tst_data = setup_dataset()
    taggers = {
        "Unigram": NgramTagger(1, trn_data),
        "Bigram": NgramTagger(2, trn_data),
        "Trigram": NgramTagger(3, trn_data),
        "Backoff": NgramTagger(3, trn_data, backoff=NgramTagger(2, trn_data)),
        "Cutoff": NgramTagger(3, trn_data, cutoff=2),
    }
    best_ngram_acc = print_taggers_accuracy(taggers, tst_data)
    data, flatten_tst_data = setup_spacy_pos_tag(tst_data)
    spacy_acc = print_spacy_accuracy(data, flatten_tst_data)
    print_result(best_ngram_acc, spacy_acc)
