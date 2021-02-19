from collections import Counter, OrderedDict
from datetime import datetime
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
from config import DATA_DIR

from cc_tweets.utils import load_pkl, mkdir_overwrite

DATASET_NAME = "tweets_downsized100_unfiltered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")
SAVE_DIR = join(DATA_DIR, f"describe_{DATASET_NAME}")

# Fri Nov 30 19:41:04 +0000 2018
TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)

    mkdir_overwrite(SAVE_DIR)

    # count stances
    stances = dict(Counter(t["stance"] for t in tweets))
    plt.bar(stances.keys(), stances.values())
    plt.savefig(join(SAVE_DIR, "stances_count.png"))
    plt.clf()

    # count time
    times = [datetime.strptime(tweet["time"], TIME_FORMAT) for tweet in tweets]
    # aggregate by year
    yyyymms = [f"{t.year}" for t in times]
    agg_year = dict(Counter(yyyymms))
    agg_year = OrderedDict(sorted(agg_year.items()))
    pprint(agg_year)
    plt.bar(agg_year.keys(), agg_year.values())
    plt.savefig(join(SAVE_DIR, "agg_year.png"))
    plt.clf()
