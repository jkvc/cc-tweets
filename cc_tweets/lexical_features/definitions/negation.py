import re

from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.utils import load_pkl
from cc_tweets.experiment_configs import SUBSET_PKL_PATH

NEGATION_REGEX = "not|n't|never|nor|no|nobody|nowhere|nothing|noone"


def _extract_features(tweets):

    id2numnegation = {}
    for tweet in tweets:
        id2numnegation[tweet["id"]] = len(
            list(re.finditer(NEGATION_REGEX, tweet["text"], re.IGNORECASE))
        )
    return id2numnegation


register_feature(Feature("negation", _extract_features))
