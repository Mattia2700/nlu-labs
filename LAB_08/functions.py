# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
import nltk
import numpy as np
try:
    from nltk.corpus import senseval
except ImportError:
    nltk.download('senseval')
    from nltk.corpus import senseval


def get_data(collocational = False):
    if collocational:
        data = [collocational_features(inst, pos=True, ngram=True) for inst in senseval.instances('interest.pos')]
    else:
        data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]

    lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]

    # encoding labels for multi-calss
    lblencoder = LabelEncoder()
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)
    
    return data, labels

def collocational_features(inst, pos = False, ngram = False):
    p = inst.position

    features = {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0]
    }

    if pos:
        features.update({
            "w-2_pos": 'NULL' if p < 2 else inst.context[p-2][1],
            "w-1_pos": 'NULL' if p < 1 else inst.context[p-1][1],
            "w+1_pos": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][1],
            "w+2_pos": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][1]
        })

    if ngram:
        features.update({
            "w-2:w-1": 'NULL' if p < 2 else inst.context[p-2][0] + " " + inst.context[p-1][0],
            "w-1:w+1": 'NULL' if p < 1 or len(inst.context) - 1 < p+1 else inst.context[p-1][0] + " " + inst.context[p+1][0],
            "w+1:w+2": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+1][0] + " " + inst.context[p+2][0]
        })
    
    return features

def get_models(collocational = False):
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    if collocational:
        return DictVectorizer(sparse=False), MultinomialNB(), stratified_split
    else:
        return CountVectorizer(), MultinomialNB(), stratified_split
            
def train_and_eval(vectorizer, classifier, data, labels, stratified_split):

    vectors = vectorizer.fit_transform(data)

    scores = cross_validate(classifier, vectors, labels, cv=stratified_split, scoring=['f1_micro'])

    print(sum(scores['test_f1_micro'])/len(scores['test_f1_micro']))
    return vectors

def concatenate_vectors(vectors, dvectors):
    return np.concatenate((vectors.toarray(), dvectors), axis=1)

def evaluate_all(classifier, uvectors, labels, stratified_split):
    # cross-validating classifier the usual way
    scores = cross_validate(classifier, uvectors, labels, cv=stratified_split, scoring=['f1_micro'])

    print(sum(scores['test_f1_micro'])/len(scores['test_f1_micro']))