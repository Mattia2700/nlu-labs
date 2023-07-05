# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *

if __name__ == "__main__":
    data, tagged_data = get_data()
    spacy_graphs = get_spacy_graphs(data)
    compare_dependency_graphs(spacy_graphs, tagged_data)
    stanza_graphs = get_stanza_graphs(data)
    compare_dependency_graphs(stanza_graphs, tagged_data)
