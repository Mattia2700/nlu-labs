# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import nltk
import spacy
import subprocess
from collections import Counter


def get_nltk_corpus(model):
    nltk.download("gutenberg")
    (
        chars,
        words,
        sents,
    ) = (
        nltk.corpus.gutenberg.raw(model),
        nltk.corpus.gutenberg.words(model),
        nltk.corpus.gutenberg.sents(model),
    )
    return chars, words, sents


def load_spacy_model(chars):
    # subprocess.run("python -m spacy download en_core_web_sm", shell=True)
    if "en_core_web_sm" not in spacy.util.get_installed_models():
        subprocess.run("python -m spacy download en_core_web_sm", shell=True)
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(chars, disable=["tagger", "ner", "lemmatizer"])
    return doc


def print_nltk_descriptive_statistics(chars, words=None, sents=None, manual=False):
    if manual:
        print("Number of characters:", len(chars))
        print("Number of words:", len(words))
        print("Number of sentences:", len(sents))
        min_char_per_token, max_char_per_token, avg_char_per_token = (
            min(len(word) for word in words),
            max(len(word) for word in words),
            round(sum(len(word) for word in words) / len(words)),
        )
        print(
            "Minimum - Maximum - Average number of characters per token:",
            min_char_per_token,
            max_char_per_token,
            avg_char_per_token,
        )
        min_word_per_sentence, max_word_per_sentence, avg_word_per_sentence = (
            min(len(sent) for sent in sents),
            max(len(sent) for sent in sents),
            round(sum(len(sent) for sent in sents) / len(sents)),
        )
        print(
            "Minimum - Maximum - Average number of words per sentence:",
            min_word_per_sentence,
            max_word_per_sentence,
            avg_word_per_sentence,
        )
        (
            min_sentences_per_document,
            max_sentences_per_document,
            avg_sentences_per_document,
        ) = (
            min(len(sent) for sent in sents),
            max(len(sent) for sent in sents),
            round(sum(len(sent) for sent in sents) / len(sents)),
        )
        print(
            "Minimum - Maximum - Average number of sentences per document:",
            min_sentences_per_document,
            max_sentences_per_document,
            avg_sentences_per_document,
        )
    else:
        words = nltk.word_tokenize(chars)
        sents = nltk.sent_tokenize(chars)
        print("Number of characters:", len(chars))
        print("Number of words:", len(words))
        print("Number of sentences:", len(sents))
        min_char_per_token, max_char_per_token, avg_char_per_token = (
            min(len(word) for word in words),
            max(len(word) for word in words),
            round(sum(len(word) for word in words) / len(words)),
        )
        print(
            "Minimum - Maximum - Average number of characters per token:",
            min_char_per_token,
            max_char_per_token,
            avg_char_per_token,
        )
        min_word_per_sentence, max_word_per_sentence, avg_word_per_sentence = (
            min(len(sent) for sent in sents),
            max(len(sent) for sent in sents),
            round(sum(len(sent) for sent in sents) / len(sents)),
        )
        print(
            "Minimum - Maximum - Average number of words per sentence:",
            min_word_per_sentence,
            max_word_per_sentence,
            avg_word_per_sentence,
        )
        (
            min_sentences_per_document,
            max_sentences_per_document,
            avg_sentences_per_document,
        ) = (
            min(len(sent) for sent in sents),
            max(len(sent) for sent in sents),
            round(sum(len(sent) for sent in sents) / len(sents)),
        )
        print(
            "Minimum - Maximum - Average number of sentences per document:",
            min_sentences_per_document,
            max_sentences_per_document,
            avg_sentences_per_document,
        )


def print_spacy_descriptive_statistics(doc, chars):
    words = [token for token in doc]
    sents = [sent for sent in doc.sents]
    print("Number of characters:", len(chars))
    print("Number of words:", len(words))
    print("Number of sentences:", len(sents))
    min_char_per_token, max_char_per_token, avg_char_per_token = (
        min(len(word) for word in words),
        max(len(word) for word in words),
        round(sum(len(word) for word in words) / len(words)),
    )
    print(
        "Minimum - Maximum - Average number of characters per token:",
        min_char_per_token,
        max_char_per_token,
        avg_char_per_token,
    )
    min_word_per_sentence, max_word_per_sentence, avg_word_per_sentence = (
        min(len(sent) for sent in sents),
        max(len(sent) for sent in sents),
        round(sum(len(sent) for sent in sents) / len(sents)),
    )
    print(
        "Minimum - Maximum - Average number of words per sentence:",
        min_word_per_sentence,
        max_word_per_sentence,
        avg_word_per_sentence,
    )
    (
        min_sentences_per_document,
        max_sentences_per_document,
        avg_sentences_per_document,
    ) = (
        min(len(sent) for sent in sents),
        max(len(sent) for sent in sents),
        round(sum(len(sent) for sent in sents) / len(sents)),
    )
    print(
        "Minimum - Maximum - Average number of sentences per document:",
        min_sentences_per_document,
        max_sentences_per_document,
        avg_sentences_per_document,
    )


def compare_lowercase_lexicons(chars, words, doc):
    # nltk manual
    nltk_lowercase_manual_lexicon = set([word.lower() for word in words])
    # nltk automatic
    nltk_lowercase_automatic_lexicon = set(
        [word.lower() for word in nltk.word_tokenize(chars)]
    )
    # spacy
    spacy_lowercase_lexicon = set([token.lower_ for token in doc])

    print("Lexicon size (lowercase) - nltk manual:", len(nltk_lowercase_manual_lexicon))
    print(
        "Lexicon size (lowercase) - nltk automatic:",
        len(nltk_lowercase_automatic_lexicon),
    )
    print("Lexicon size (lowercase) - spacy:", len(spacy_lowercase_lexicon))


def compare_top_N_frequencies(chars, words, doc, n=5):
    def nbest(d, n=1):
        """
        get n max values from a dict
        :param d: input dict (values are numbers, keys are stings)# Counter(X) # Replace X with the word list of the corpus in lower case (see above))
        :param n: number of values to get (int)
        :return: dict of top n key-value pairs
        """
        return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

    nltk_manual_frequencies = Counter([word.lower() for word in words])
    nltk_manual_top_n = nbest(nltk_manual_frequencies, n)
    nltk_automatic_frequencies = Counter(
        [word.lower() for word in nltk.word_tokenize(chars)]
    )
    nltk_automatic_top_n = nbest(nltk_automatic_frequencies, n)
    spacy_frequencies = Counter([token.lower_ for token in doc])
    spacy_top_n = nbest(spacy_frequencies, n)

    print("Top", n, "most frequent words (lowercase) - nltk manual:", nltk_manual_top_n)
    print(
        "Top",
        n,
        "most frequent words (lowercase) - nltk automatic:",
        nltk_automatic_top_n,
    )
    print("Top", n, "most frequent words (lowercase) - spacy:", spacy_top_n)
