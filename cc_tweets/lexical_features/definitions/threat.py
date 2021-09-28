from collections import defaultdict
from typing import DefaultDict

from cc_tweets.experiment_configs import SUBSET_PKL_PATH
from cc_tweets.feature_utils import save_features, visualize_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.utils import load_pkl
from nltk.stem.snowball import SnowballStemmer

LEXICON = set(
    [
        "threat",
        "danger",
        "crisis",
        "destroy",
        "destruct",
        "catastrophe",
        "worsen",
        "worst",
    ]
)
stemmer = SnowballStemmer("english")
LEXICON = set(stemmer.stem(w) for w in LEXICON)


def _extract_econ_feature(tweets):
    id2count = {}
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in LEXICON:
                count += 1
        id2count[tweet["id"]] = count
    return id2count


register_feature(Feature("term.threat", _extract_econ_feature))


def _extract_word_closure(word):
    def _extract(tweets):
        id2count = {}
        for t in tweets:
            count = 0
            for stem in t["stems"]:
                if stem == word:
                    count = 1
            id2count[t["id"]] = count
        return id2count

    return _extract


for word in LEXICON:
    register_feature(Feature(f"term.threat.{word}", _extract_word_closure(word)))
