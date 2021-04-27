from collections import defaultdict
from os.path import join

from nltk.stem.snowball import SnowballStemmer

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.feature_utils import save_features, visualize_features
from cc_tweets.utils import load_pkl, save_json
from cc_tweets.viz import plot_horizontal_bars

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
    disaster2id2value = defaultdict(lambda: defaultdict(int))
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in NATURAL_DISASTER_WORDS:
                count += 1
                disaster2id2value[stem][tweet["id"]] += 1
        id2numdisaster[tweet["id"]] = count

    save_features(tweets, {"natural_disasters": id2numdisaster}, "natural_disasters")

    visualize_features(disaster2id2value, tweets, "natural_disasters")
