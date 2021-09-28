from collections import defaultdict
from typing import DefaultDict

from cc_tweets.experiment_configs import SUBSET_PKL_PATH
from cc_tweets.feature_utils import save_features, visualize_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.utils import load_pkl
from nltk.stem.snowball import SnowballStemmer

ECONOMY_WORDS = set(
    [
        "capitalism",
        "finance",
        "labor",
        "economy",
        "stock",
        "workforce",
        "market",
        "cost",
        "trade",
        "capital",
        "tax",
        "job",
        "growth",
        "money",
        "econ",
        "wealth",
        "bank",
        "financial",
        "utility",
        "investment",
        "tariff",
        "economist",
        "economics",
        "profit",
    ]
)
stemmer = SnowballStemmer("english")
ECONOMY_WORDS = set(stemmer.stem(w) for w in ECONOMY_WORDS)


def _extract_econ_feature(tweets):
    id2numeconomy = {}
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in ECONOMY_WORDS:
                count += 1
        id2numeconomy[tweet["id"]] = count
    return id2numeconomy


register_feature(Feature("term.economy", _extract_econ_feature))
