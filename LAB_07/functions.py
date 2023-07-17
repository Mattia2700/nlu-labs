# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import spacy
from spacy.tokenizer import Tokenizer
from nltk.corpus import conll2002
from sklearn_crfsuite import CRF
import pandas as pd
import nltk

try:
    from conll import evaluate
except ImportError:
    # downlaod it from https://raw.githubusercontent.com/BrownFortress/NLU-2023-Labs/main/labs/conll.py
    import wget

    url = "https://raw.githubusercontent.com/BrownFortress/NLU-2023-Labs/main/labs/conll.py"
    wget.download(url)
    from conll import evaluate


def word2features(sent, i):
    word = sent[i][0]
    return {"bias": 1.0, "word.lower()": word.lower()}


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, pos, label in sent]


def sent2tokens(sent):
    return [token for token, pos, label in sent]


def sent2pos(sent):
    return [pos for token, pos, label in sent]


def load_data():
    if not spacy.util.is_package("es_core_news_sm"):
        spacy.cli.download("es_core_news_sm")
    nltk.download('conll2002', quiet=True)
    nlp = spacy.load("es_core_news_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab)  # to use white space tokenization

    crf = CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    trn_sents = conll2002.iob_sents("esp.train")
    tst_sents = conll2002.iob_sents("esp.testa")

    return nlp, crf, trn_sents, tst_sents


def sent2spacy_features(
    nlp,
    sent,
    suffix=False,
    conll_tutorial=False,
    sorrounding_one=False,
    sorrounding_two=False,
):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    pos = sent2pos(sent)
    feats = []
    for index, (token, pos) in enumerate(zip(spacy_sent, pos)):
        if suffix:
            token_feats = {
                "bias": 1.0,
                "word.lower()": token.lower_,
                "pos": token.pos_,
                "lemma": token.lemma_,
                "suffix": token.suffix_,
            }
        elif conll_tutorial:
            token_feats = {
                "bias": 1.0,
                "word.lower()": token.lower_,
                "word[-3:]": str(token)[-3:],
                "word[-2:]": str(token)[-2:],
                "word.isupper()": token.is_upper,
                "word.istitle()": token.is_title,
                "word.isdigit()": token.is_digit,
                "postag": pos,
                "postag[:2]": pos[:2],
            }
        else:
            token_feats = {
                "bias": 1.0,
                "word.lower()": token.lower_,
                "pos": token.pos_,
                "lemma": token.lemma_,
            }
        if sorrounding_one or sorrounding_two:
            if index > 0:
                token_feats.update(
                    {
                        "-1:word.lower()": spacy_sent[index - 1].lower_,
                        "-1:word.istitle()": spacy_sent[index - 1].is_title,
                        "-1:word.isupper()": spacy_sent[index - 1].is_upper,
                        "-1:postag": spacy_sent[index - 1].pos_,
                        "-1:postag[:2]": spacy_sent[index - 1].pos_[:2],
                    }
                )
            else:
                token_feats["BOS"] = True

            if index < len(spacy_sent) - 1:
                token_feats.update(
                    {
                        "+1:word.lower()": spacy_sent[index + 1].lower_,
                        "+1:word.istitle()": spacy_sent[index + 1].is_title,
                        "+1:word.isupper()": spacy_sent[index + 1].is_upper,
                        "+1:postag": spacy_sent[index + 1].pos_,
                        "+1:postag[:2]": spacy_sent[index + 1].pos_[:2],
                    }
                )
            else:
                token_feats["EOS"] = True

        if sorrounding_two:
            if index > 1:
                token_feats.update(
                    {
                        "-2:word.lower()": spacy_sent[index - 2].lower_,
                        "-2:word.istitle()": spacy_sent[index - 2].is_title,
                        "-2:word.isupper()": spacy_sent[index - 2].is_upper,
                        "-2:postag": spacy_sent[index - 2].pos_,
                        "-2:postag[:2]": spacy_sent[index - 2].pos_[:2],
                    }
                )
            else:
                token_feats["BOS"] = True

            if index < len(spacy_sent) - 2:
                token_feats.update(
                    {
                        "+2:word.lower()": spacy_sent[index + 2].lower_,
                        "+2:word.istitle()": spacy_sent[index + 2].is_title,
                        "+2:word.isupper()": spacy_sent[index + 2].is_upper,
                        "+2:postag": spacy_sent[index + 2].pos_,
                        "+2:postag[:2]": spacy_sent[index + 2].pos_[:2],
                    }
                )
            else:
                token_feats["EOS"] = True

        feats.append(token_feats)

    return feats


def evaluate_features(
    nlp,
    crf,
    trn_sents,
    tst_sents,
    suffix=False,
    conll_tutorial=False,
    sorrounding_one=False,
    sorrounding_two=False,
):
    if suffix:
        print("\t\tSUFFIX FEATURES")
        trn_feats = [sent2spacy_features(nlp, s, suffix=True) for s in trn_sents]
        tst_feats = [sent2spacy_features(nlp, s, suffix=True) for s in tst_sents]
    elif conll_tutorial:
        print("\t\tCONLL TUTORIAL FEATURES")
        trn_feats = [
            sent2spacy_features(nlp, s, conll_tutorial=True) for s in trn_sents
        ]
        tst_feats = [
            sent2spacy_features(nlp, s, conll_tutorial=True) for s in tst_sents
        ]
    elif sorrounding_one:
        print("\t\t[-1, +1] FEATURES")
        trn_feats = [
            sent2spacy_features(nlp, s, sorrounding_one=True) for s in trn_sents
        ]
        tst_feats = [
            sent2spacy_features(nlp, s, sorrounding_one=True) for s in tst_sents
        ]
    elif sorrounding_two:
        print("\t\t[-2, +2] FEATURES")
        trn_feats = [
            sent2spacy_features(nlp, s, sorrounding_two=True) for s in trn_sents
        ]
        tst_feats = [
            sent2spacy_features(nlp, s, sorrounding_two=True) for s in tst_sents
        ]
    else:
        print("\t\tBASELINE FEATURES")
        trn_feats = [sent2spacy_features(nlp, s) for s in trn_sents]
        tst_feats = [sent2spacy_features(nlp, s) for s in tst_sents]

    trn_label = [sent2labels(s) for s in trn_sents]

    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    pred = crf.predict(tst_feats)

    hyp = [
        [(tst_feats[i][j], t) for j, t in enumerate(tokens)]
        for i, tokens in enumerate(pred)
    ]

    results = evaluate(tst_sents, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient="index")
    pd_tbl.round(decimals=3)
    print(pd_tbl)
    print()
