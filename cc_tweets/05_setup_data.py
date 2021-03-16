from collections import Counter, OrderedDict
from datetime import datetime
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_pkl, mkdir_overwrite, save_json

# Fri Nov 30 19:41:04 +0000 2018
TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    mkdir_overwrite(DATASET_SAVE_DIR)

    # count stances
    stances = dict(Counter(t["stance"] for t in tweets))
    plt.bar(stances.keys(), stances.values())
    plt.savefig(join(DATASET_SAVE_DIR, "stances_count.png"))
    plt.clf()

    # count time
    times = [datetime.strptime(tweet["time"], TIME_FORMAT) for tweet in tweets]
    # aggregate by year
    yyyymms = [f"{t.year}" for t in times]
    agg_year = dict(Counter(yyyymms))
    agg_year = OrderedDict(sorted(agg_year.items()))
    pprint(agg_year)
    plt.bar(agg_year.keys(), agg_year.values())
    plt.savefig(join(DATASET_SAVE_DIR, "agg_year.png"))
    plt.clf()

    # various metrics
    metrics = {}
    metrics["total"] = len(tweets)
    metrics["stances"] = stances
    metrics["unique_tweeter"] = len(set(t["userid"] for t in tweets))
    metrics["mean_raw_len"] = sum(len(t["text"]) for t in tweets) / len(tweets)
    metrics["mean_num_stem"] = sum(len(t["stems"]) for t in tweets) / len(tweets)
    metrics["mean_num_hashtags"] = sum(len(t["hashtags"]) for t in tweets) / len(tweets)
    save_json(metrics, join(DATASET_SAVE_DIR, "metrics.json"))