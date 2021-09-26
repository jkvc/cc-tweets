from collections import defaultdict
from os.path import join

from cc_tweets.feature_utils import save_features, visualize_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.utils import load_pkl, save_json
from cc_tweets.viz import plot_horizontal_bars
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from nltk.stem.snowball import SnowballStemmer

NATURAL_DISASTER_WORDS = set(
    [
        "sinkhole",
        "tsunami",
        "erupt",
        "thunderstorm",
        "hail",
        "avalanche",
        "downpour",
        "heat wave",
        "disaster",
        "earthquake",
        "drought",
        "mudslide",
        "eruption",
        "bushfire",
        "catastrophe",
        "volcano",
        "blizzard",
        "fire",
        "landslide",
        "cyclone",
        "storm",
        "wildfire",
        "hurricane",
        "flood",
        "tornado",
    ]
)
stemmer = SnowballStemmer("english")
NATURAL_DISASTER_WORDS = set(stemmer.stem(w) for w in NATURAL_DISASTER_WORDS)


def _extract_features(tweets):
    id2count = {}
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in NATURAL_DISASTER_WORDS:
                count += 1
        id2count[tweet["id"]] = count
    return id2count


register_feature(Feature("disaster", _extract_features))
