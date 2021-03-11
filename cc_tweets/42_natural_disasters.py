from os.path import join

from config import DATA_DIR
from nltk.stem.snowball import SnowballStemmer

from cc_tweets.experiment_config import DATASET_NAME, DATASET_PKL_PATH
from cc_tweets.feature_utils import get_stats, save_features
from cc_tweets.utils import load_pkl, save_json

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


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    id2numdisaster = {}
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in NATURAL_DISASTER_WORDS:
                count += 1
        id2numdisaster[tweet["id"]] = count

    save_features(tweets, {"natural_disasters": id2numdisaster}, "42_natural_disasters")
