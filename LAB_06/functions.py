# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
from nltk.corpus import dependency_treebank
# Spacy version 
from nltk.parse.dependencygraph import DependencyGraph
from spacy.tokenizer import Tokenizer
import spacy
# Stanza version
import stanza
import spacy_stanza

from nltk.parse import DependencyEvaluator

def get_data():
    # get the last 100 sentences
    data = dependency_treebank.sents()[-100:]
    tagged_data = dependency_treebank.parsed_sents()[-100:]
    return data, tagged_data

def get_spacy_graphs(data):

    print("Computing Spacy graphs and evaluating...")

    # Load the spacy model
    nlp = spacy.load("en_core_web_sm")

    # Set up the conll formatter 
    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"DEPREL": {"nsubj": "subj"}}}

    # Add the formatter to the pipeline
    nlp.add_pipe("conll_formatter", config=config, last=True)
    # Split by white space
    nlp.tokenizer = Tokenizer(nlp.vocab)  

    spacy_graphs = []

    for sentence in data:    
        # Join the words to a sentence
        sentence = " ".join(sentence)
        # Parse the sentence
        doc = nlp(sentence)
        # Convert doc to a pandas object
        df = doc._.pandas
        # Select the columns accoroding to Malt-Tab format
        tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        # Get finally our the DepencecyGraph
        dp = DependencyGraph(tmp)
        spacy_graphs.append(dp)

    return spacy_graphs

def get_stanza_graphs(data):

    print("Computing Stanza graphs and evaluating...")

    # Download the stanza model if necessary
    stanza.download("en", verbose=False)

    # Set up the conll formatter 
    #tokenize_pretokenized used to tokenize by white space 
    nlp = spacy_stanza.load_pipeline("en", verbose=False, use_gpu=False, tokenize_pretokenized=True)

    config = {"ext_names": {"conll_pd": "pandas"},
            "conversion_maps": {"DEPREL": {"nsubj": "subj", "root":"ROOT"}}}

    # Add the formatter to the pipeline
    nlp.add_pipe("conll_formatter", config=config, last=True)

    stanza_graphs = []

    for sentence in data:
        # Join the words to a sentence
        sentence = " ".join(sentence)
        # Parse the sentence
        doc = nlp(sentence)
        # Convert doc to a pandas object
        df = doc._.pandas
        # Select the columns accoroding to Malt-Tab format
        tmp = df[["FORM", 'XPOS', 'HEAD', 'DEPREL']].to_string(header=False, index=False)
        # Get finally our the DepencecyGraph
        dp = DependencyGraph(tmp)
        stanza_graphs.append(dp)

    return stanza_graphs

def compare_dependency_graphs(graphs, tagged_data):

    de = DependencyEvaluator(graphs, tagged_data)
    las, uas = de.eval()

    print(f"LAS: {las}, UAS: {uas}")
    print()
