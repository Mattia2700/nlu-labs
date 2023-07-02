# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
import warnings

warnings.filterwarnings("ignore")


def load_vectorizers():
    binary_vectorizer = CountVectorizer(binary=True)
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer_cutoff = TfidfVectorizer(
        stop_words="english", lowercase=True, min_df=2, max_df=0.5
    )
    tfidf_vectorizer_no_stopwords = TfidfVectorizer(stop_words="english")
    tfidf_vectorizer_no_lowercase = TfidfVectorizer(lowercase=False)
    vectorizers = {
        "CountVect": binary_vectorizer,
        "TF-IDF": tfidf_vectorizer,
        "CutOff": tfidf_vectorizer_cutoff,
        "WithoutStopWords": tfidf_vectorizer_no_stopwords,
        "NoLowercase": tfidf_vectorizer_no_lowercase,
    }
    return vectorizers


def test_vectorizer(experiment_id, data):
    # split the data into training and testing with stratified k-fold
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # choose scoring metrics
    scores = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    # choose classification algorithm & initialize it
    clf = LinearSVC(C=0.1, max_iter=10000)

    # cross-validate the model
    results = cross_validate(
        clf, data.data, data.target, cv=stratified_split, scoring=scores
    )

    # print results
    print(experiment_id, end=": ")
    for score in scores:
        print(
            "{:.3} ({})".format(
                sum(results["test_" + score]) / len(results["test_" + score]), score
            ),
            end="\t",
        )
    print()
