# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline, flatten
from nltk.corpus import gutenberg
from nltk.lm import Vocabulary
from itertools import chain
from nltk.lm import StupidBackoff


class MyBackoff(MLE):
    def __init__(self, order, vocab, alpha=0.4):
        super().__init__(order=order, vocabulary=vocab)
        self.alpha = alpha

    def unmasked_score(self, word, context=None):
        if context is None or len(context) == 0:
            score = self.counts[word] / len(self.vocab)
        else:
            score = super().unmasked_score(word, context)

            if score == 0:
                score = self.alpha * self.unmasked_score(word, context[1:])
        return score


def get_dataset_and_vocab(ngram_order):
    # Dataset
    macbeth_sents = [
        [w.lower() for w in sent] for sent in gutenberg.sents("shakespeare-macbeth.txt")
    ]
    macbeth_words = flatten(macbeth_sents)
    # Compute vocab
    lex = Vocabulary(macbeth_words, unk_cutoff=2)
    # Handeling OOV
    macbeth_oov_sents = [list(lex.lookup(sent)) for sent in macbeth_sents]
    padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(
        ngram_order, macbeth_oov_sents
    )
    return padded_ngrams_oov, flat_text_oov, lex


def evaluate_my_model(ngram_order, lex, padded_ngrams_oov, flat_text_oov, test_set):
    # Train the model
    my_backoff = MyBackoff(ngram_order, lex)
    my_backoff.fit(padded_ngrams_oov, flat_text_oov)
    # Compute PPL and entropy with OOV on test 1
    ngrams, flat_text = padded_everygram_pipeline(
        my_backoff.order, [lex.lookup(sent.split()) for sent in test_set]
    )
    ngrams = chain.from_iterable(ngrams)
    ppl = my_backoff.perplexity([x for x in ngrams if len(x) == my_backoff.order])
    print("My StupidBackoff PPL:", ppl)


def evaluate_nltk_model(ngram_order, lex, padded_ngrams_oov, flat_text_oov, test_set):
    # Train the model
    stupid_backoff = StupidBackoff(order=ngram_order, vocabulary=lex)
    stupid_backoff.fit(padded_ngrams_oov, flat_text_oov)
    # Compute PPL and entropu with OOV on test 1
    ngrams, flat_text = padded_everygram_pipeline(
        stupid_backoff.order, [lex.lookup(sent.split()) for sent in test_set]
    )
    ngrams = chain.from_iterable(ngrams)
    ppl = stupid_backoff.perplexity(
        [x for x in ngrams if len(x) == stupid_backoff.order]
    )
    print("NLTK StupidBackoff PPL:", ppl)
