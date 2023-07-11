# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    nlp, crf, trn_sents, tst_sents = load_data()
    evaluate_features(nlp, crf, trn_sents, tst_sents)
    evaluate_features(nlp, crf, trn_sents, tst_sents, suffix=True)
    evaluate_features(nlp, crf, trn_sents, tst_sents, conll_tutorial=True)
    evaluate_features(nlp, crf, trn_sents, tst_sents, sorrounding_one=True)
    evaluate_features(nlp, crf, trn_sents, tst_sents, sorrounding_two=True)
