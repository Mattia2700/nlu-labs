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
from collections import Counter
from nltk.metrics.scores import precision, recall, f_measure, accuracy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

MAPPING = {
    "interest_1": "interest.n.01",
    "interest_2": "interest.n.03",
    "interest_3": "pastime.n.01",
    "interest_4": "sake.n.01",
    "interest_5": "interest.n.05",
    "interest_6": "interest.n.04",
}


def get_data(collocational=False, encoded=True):
    nltk.download("senseval", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet_ic', quiet=True)
    if collocational:
        data = [
            collocational_features(inst, pos=True, ngram=True)
            for inst in nltk.corpus.senseval.instances("interest.pos")
        ]
    else:
        data = [
            " ".join([t[0] for t in inst.context])
            for inst in nltk.corpus.senseval.instances("interest.pos")
        ]

    lbls = [inst.senses[0] for inst in nltk.corpus.senseval.instances("interest.pos")]

    if encoded:
        # encoding labels for multi-calss
        lblencoder = LabelEncoder()
        lblencoder.fit(lbls)
        lbls = lblencoder.transform(lbls)

    return data, lbls


def collocational_features(inst, pos=False, ngram=False):
    p = inst.position

    features = {
        "w-2_word": "NULL" if p < 2 else inst.context[p - 2][0],
        "w-1_word": "NULL" if p < 1 else inst.context[p - 1][0],
        "w+1_word": "NULL" if len(inst.context) - 1 < p + 1 else inst.context[p + 1][0],
        "w+2_word": "NULL" if len(inst.context) - 1 < p + 2 else inst.context[p + 2][0],
    }

    if pos:
        features.update(
            {
                "w-2_pos": "NULL" if p < 2 else inst.context[p - 2][1],
                "w-1_pos": "NULL" if p < 1 else inst.context[p - 1][1],
                "w+1_pos": "NULL"
                if len(inst.context) - 1 < p + 1
                else inst.context[p + 1][1],
                "w+2_pos": "NULL"
                if len(inst.context) - 1 < p + 2
                else inst.context[p + 2][1],
            }
        )

    if ngram:
        features.update(
            {
                "w-2:w-1": "NULL"
                if p < 2
                else inst.context[p - 2][0] + " " + inst.context[p - 1][0],
                "w-1:w+1": "NULL"
                if p < 1 or len(inst.context) - 1 < p + 1
                else inst.context[p - 1][0] + " " + inst.context[p + 1][0],
                "w+1:w+2": "NULL"
                if len(inst.context) - 1 < p + 2
                else inst.context[p + 1][0] + " " + inst.context[p + 2][0],
            }
        )

    return features


def get_models(collocational=False):
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    if collocational:
        return DictVectorizer(sparse=False), MultinomialNB(), stratified_split
    else:
        return CountVectorizer(), MultinomialNB(), stratified_split


def train_and_eval(vectorizer, classifier, data, labels, stratified_split):
    vectors = vectorizer.fit_transform(data)

    scores = cross_validate(
        classifier, vectors, labels, cv=stratified_split, scoring=["f1_micro"]
    )

    print(sum(scores["test_f1_micro"]) / len(scores["test_f1_micro"]))
    return vectors


def concatenate_vectors(vectors, dvectors):
    return np.concatenate((vectors.toarray(), dvectors), axis=1)


def evaluate_all(classifier, uvectors, labels, stratified_split):
    # cross-validating classifier the usual way
    scores = cross_validate(
        classifier, uvectors, labels, cv=stratified_split, scoring=["f1_micro"]
    )

    print(sum(scores["test_f1_micro"]) / len(scores["test_f1_micro"]))


def original_lesk(
    context_sentence, ambiguous_word, pos=None, synsets=None, majority=False
):
    context_senses = get_sense_definitions(
        set(context_sentence) - set([ambiguous_word])
    )
    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None
    scores = []
    # print(synsets)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense


def lesk_similarity(
    context_sentence,
    ambiguous_word,
    similarity="resnik",
    pos=None,
    synsets=None,
    majority=True,
):
    context_senses = get_sense_definitions(
        set(context_sentence) - set([ambiguous_word])
    )

    if synsets is None:
        synsets = get_sense_definitions(ambiguous_word)[0][1]

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None

    scores = []

    # Here you may have some room for improvement
    # For instance instead of using all the definitions from the context
    # you pick the most common one of each word (i.e. the first)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)

    return best_sense


def preprocess(text):
    mapping = {
        "NOUN": nltk.corpus.wordnet.NOUN,
        "VERB": nltk.corpus.wordnet.VERB,
        "ADJ": nltk.corpus.wordnet.ADJ,
        "ADV": nltk.corpus.wordnet.ADV,
    }
    sw_list = stopwords.words("english")

    lem = WordNetLemmatizer()

    # tokenize, if input is text
    tokens = nltk.word_tokenize(text) if type(text) is str else text
    # pos-tag
    tagged = nltk.pos_tag(tokens, tagset="universal")
    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    # lemmatize
    tagged = [(w, lem.lemmatize(w, pos=p), p) for w, p in tagged]
    # unique the list
    tagged = list(set(tagged))
    return tagged


def get_test_set(skf, data, labels):
    X_test = []
    y_test = []
    for train_index, test_index in skf.split(data, labels):
        X_test.append([data[i] for i in test_index])
        y_test.append([labels[i] for i in test_index])

    return X_test, y_test


def eval_lesks(sentences, labels):
    f1s_or = []
    f1s_sim = []

    for s_group, l_group in zip(sentences, labels):
        refs = {k: set() for k in MAPPING.values()}
        hyps_original = {k: set() for k in MAPPING.values()}
        hyps_similarity = {k: set() for k in MAPPING.values()}

        # since WordNet defines more senses, let's restrict predictions
        synsets = []
        for ss in nltk.corpus.wordnet.synsets("interest", pos="n"):
            if ss.name() in MAPPING.values():
                defn = ss.definition()
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                synsets.append((ss, toks))

        for i, (sent, lbl) in enumerate(zip(s_group, l_group)):
            word = "interest" if "interest" in sent else "interests"
            hyp_original = original_lesk(
                sent, word, synsets=synsets, majority=True
            ).name()
            hyp_similarity = lesk_similarity(
                sent, word, pos="n", synsets=synsets, similarity="path", majority=True
            ).name()
            ref = MAPPING.get(lbl)

            # f1
            refs[ref].add(i)
            hyps_original[hyp_original].add(i)
            hyps_similarity[hyp_similarity].add(i)

        for cls_or, cls_sim in zip(hyps_original.keys(), hyps_similarity.keys()):
            f_or = f_measure(refs[cls_or], hyps_original[cls_or])
            f_sim = f_measure(refs[cls_sim], hyps_similarity[cls_sim])

            f_or = 0 if f_or is None else f_or
            f_sim = 0 if f_sim is None else f_sim

            f1s_or.append(f_or)
            f1s_sim.append(f_sim)

    print(
        sum(f1s_or) / len(f1s_or),
        "(original lesk)",
        sum(f1s_sim) / len(f1s_sim),
        "(lesk similarity)",
    )


def get_top_sense_sim(context_sense, sense_list, similarity):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    semcor_ic = nltk.corpus.wordnet_ic.ic("ic-semcor.dat")
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                scores.append((context_sense.path_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lch":
            try:
                scores.append((context_sense.lch_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                scores.append((context_sense.wup_similarity(ss), ss))
            except:
                scores.append((0, ss))
        elif similarity == "resnik":
            try:
                scores.append((context_sense.res_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                scores.append((context_sense.lin_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        elif similarity == "jiang":
            try:
                scores.append((context_sense.jcn_similarity(ss, semcor_ic), ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None
    val, sense = max(scores)
    return val, sense


def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)
    # let's get senses for each
    senses = [(w, nltk.corpus.wordnet.synsets(l, p)) for w, l, p in lemma_tags]

    # let's get their definitions
    definitions = []
    for raw_word, sense_list in senses:
        if len(sense_list) > 0:
            # let's tokenize, lowercase & remove stop words
            def_list = []
            for s in sense_list:
                defn = s.definition()
                # let's use the same preprocessing
                tags = preprocess(defn)
                toks = [l for w, l, p in tags]
                def_list.append((s, toks))
            definitions.append((raw_word, def_list))
    return definitions


def get_top_sense(words, sense_list):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    val, sense = max(
        (len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list
    )
    return val, sense
