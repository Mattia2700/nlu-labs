# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import nltk
from pcfg import PCFG
# from nltk import Nonterminal
# from nltk.parse.generate import generate


def get_sentences():
    # return sentences with 10 words
    return [
        "he is a student at the University of Trento",
        "Paul wants to go to the cinema",
        "the balloon floats in the air",
    ]

def get_rules():
    return [
        'S -> NP VP [1.0]',

        'NP -> PRON [0.3]',
        'NP -> Det N [0.3]',
        'NP -> NP PP [0.2]',
        'NP -> N [0.2]',

        'PP -> P NP [0.5]',
        'PP -> P VP [0.3]',
        'PP -> P N [0.2]',

        'VP -> V NP [0.7]',
        'VP -> V PP [0.3]',

        'PRON -> "he" [1.0]',

        'Det -> "the" [0.6]',
        'Det -> "a" [0.4]',

        'N -> "cinema" [0.2]',
        'N -> "balloon" [0.2]',
        'N -> "air" [0.15]',
        'N -> "Paul" [0.15]',
        'N -> "student" [0.1]',
        'N -> "University" [0.1]',
        'N -> "Trento" [0.1]',

        'V -> "is" [0.3]',
        'V -> "wants" [0.25]',
        'V -> "go" [0.25]',
        'V -> "floats" [0.2]',

        'P -> "to" [0.35]',
        'P -> "at" [0.3]',
        'P -> "of" [0.2]',
        'P -> "in" [0.15]',
    ]

def create_grammar(rules):
    # nltk_grammar = nltk.PCFG.fromstring(rules)
    pcfg_grammar = PCFG.fromstring(rules)

    return pcfg_grammar

def generate_sentences(pcfg_grammar):
    # start = Nonterminal('S')

    # print("NLTK generate:\n")

    # for sent in generate(nltk_grammar, start=start, depth=5, n=10):
    #     print(" ".join(sent))

    # print("\nPCFG generate:\n")

    for sent in pcfg_grammar.generate(10):
        print(sent)