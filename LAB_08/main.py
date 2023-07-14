# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    data, labels = get_data()
    vectorizer, classifier, stratified_split = get_models()
    print("Only BOW features:", end="\t")
    vectors = train_and_eval(vectorizer, classifier, data, labels, stratified_split)

    data, _ = get_data(collocational=True)
    vectorizer, _, _ = get_models(collocational=True)
    print("Only collocational features:", end="\t")
    dvectors = train_and_eval(vectorizer, classifier, data, labels, stratified_split)

    uvectors = concatenate_vectors(vectors, dvectors)
    print("BOW + collocational features:", end="\t")
    evaluate_all(classifier, uvectors, labels, stratified_split)