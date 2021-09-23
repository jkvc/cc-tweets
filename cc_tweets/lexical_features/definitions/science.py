from collections import defaultdict
from typing import DefaultDict

from cc_tweets.feature_utils import save_features, visualize_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.utils import load_pkl
from experiment_configs.base import SUBSET_PKL_PATH
from nltk.stem.snowball import SnowballStemmer

TERMS = set(
    [
        "science",
        "research",
        "scientist",
        "researcher",
        "study",
    ]
)
stemmer = SnowballStemmer("english")
TERMS = set(stemmer.stem(w) for w in TERMS)


def _extract_features(tweets):
    id2count = {}
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in TERMS:
                count += 1
        id2count[tweet["id"]] = count
    return id2count


register_feature(Feature("science", _extract_features))
