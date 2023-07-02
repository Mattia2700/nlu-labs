# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import math
from nltk.corpus import treebank
from nltk.tag import NgramTagger
from nltk.metrics import accuracy
from itertools import chain
import spacy
import en_core_web_sm
from spacy.tokenizer import Tokenizer

mapping_spacy_to_NLTK = {
    "ADJ": "ADJ",
    "ADP": "ADP",
    "ADV": "ADV",
    "AUX": "VERB",
    "CCONJ": "CONJ",
    "DET": "DET",
    "INTJ": "X",
    "NOUN": "NOUN",
    "NUM": "NUM",
    "PART": "PRT",
    "PRON": "PRON",
    "PROPN": "NOUN",
    "PUNCT": ".",
    "SCONJ": "CONJ",
    "SYM": "X",
    "VERB": "VERB",
    "X": "X",
}


def setup_dataset():
    total_size = len(treebank.tagged_sents())
    train_indx = math.ceil(total_size * 0.8)
    trn_data = treebank.tagged_sents(tagset="universal")[:train_indx]
    tst_data = treebank.tagged_sents(tagset="universal")[train_indx:]
    return trn_data, tst_data


def print_taggers_accuracy(taggers, tst_data):
    best_tagger_acc = 0
    for tagger_name, tagger in taggers.items():
        acc = tagger.accuracy(tst_data)
        print("{} Accuracy: {:6.4f}".format(tagger_name, acc))
        if acc > best_tagger_acc:
            best_tagger_acc = acc
    return best_tagger_acc


def setup_spacy_pos_tag(tst_data):
    nlp = en_core_web_sm.load()
    nlp.tokenizer = Tokenizer(nlp.vocab)  # Tokenize by whitespace
    data = []
    flatten_tst_data = list(chain.from_iterable(tst_data))
    for sent, _ in flatten_tst_data:
        doc = nlp(sent)
        data.append([(x.text, mapping_spacy_to_NLTK[x.pos_]) for x in doc])

    # flatten the list
    data = list(chain.from_iterable(data))
    return data, flatten_tst_data


def print_spacy_accuracy(data, flatten_tst_data):
    acc = accuracy(data, flatten_tst_data)
    print("Spacy Accuracy: {:6.4f}".format(acc))
    return acc


def print_result(ngram_acc, spacy_acc):
    print("NLTK: {:6.4f} SPACY: {:6.4f}".format(ngram_acc, spacy_acc))
