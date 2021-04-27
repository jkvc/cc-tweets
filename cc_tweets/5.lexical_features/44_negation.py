import re

from cc_tweets.experiment_config import SUBSET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.utils import load_pkl

NEGATION_REGEX = "not|n't|never|nor|no|nobody|nowhere|nothing|noone"
if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    id2numnegation = {}
    for tweet in tweets:
        id2numnegation[tweet["id"]] = len(
            list(re.finditer(NEGATION_REGEX, tweet["text"], re.IGNORECASE))
        )

    save_features(tweets, {"negation": id2numnegation}, "negation")
