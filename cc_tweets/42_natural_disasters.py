from collections import Counter, OrderedDict
from datetime import datetime
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
from config import DATA_DIR
from nltk.stem.snowball import SnowballStemmer

from cc_tweets.utils import load_pkl, mkdir_overwrite, save_json

DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")

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
    tweets = load_pkl(PKL_PATH)

    id2numdisaster = {}
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in NATURAL_DISASTER_WORDS:
                count += 1
        id2numdisaster[tweet["id"]] = count
    save_json(
        id2numdisaster,
        join(DATA_DIR, DATASET_NAME, "42_natural_disaster_counts.json"),
    )

    stats = {}
    stats["mean_count"] = sum(id2numdisaster.values()) / len(id2numdisaster)
    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    stats["mean_count_dem"] = sum(id2numdisaster[t["id"]] for t in dem_tweets) / len(
        dem_tweets
    )
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    stats["mean_count_rep"] = sum(id2numdisaster[t["id"]] for t in rep_tweets) / len(
        rep_tweets
    )
    stats["mean_count_adjusted_for_dem_rep_imbalance"] = (
        stats["mean_count_dem"] + stats["mean_count_rep"]
    ) / 2
    save_json(
        stats,
        join(DATA_DIR, DATASET_NAME, "42_natural_disaster_stats.json"),
    )
