import json
from collections import Counter, OrderedDict
from datetime import datetime
from glob import glob
from os import makedirs, mkdir
from os.path import exists, join
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
from cc_tweets.data_utils import get_ngrams, parse_raw_tweet
from cc_tweets.experiment_config import (
    DOWNSIZE_FACTOR,
    FILTER_UNK,
    SUBSET_PKL_PATH,
    SUBSET_WORKING_DIR,
)
from cc_tweets.utils import (
    ParallelHandler,
    load_pkl,
    mkdir_overwrite,
    save_json,
    save_pkl,
)
from config import DATA_DIR, RAW_DIR
from nltk.corpus import stopwords
from tqdm import tqdm

# Fri Nov 30 19:41:04 +0000 2018
TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


STOPWORDS = stopwords.words("english")


def process_tweets_from_raw(jsonl_path):
    cache_path = f"{jsonl_path}.{DOWNSIZE_FACTOR}.pkl"
    if exists(cache_path):
        return

    userid2stance_path = join(DATA_DIR, "userid2stance.pkl")
    userid2stance = load_pkl(userid2stance_path)

    all_tweet_data = []
    with open(jsonl_path) as f:
        lines = f.readlines()

    for i, line in enumerate(tqdm(lines, disable=True)):
        if i % DOWNSIZE_FACTOR != 0:
            continue

        tweet = json.loads(line)
        tweet_data = parse_raw_tweet(tweet)

        stance = userid2stance[tweet_data["userid"]]
        if stance == "unk" and FILTER_UNK:
            continue
        if tweet_data["lang"] != "en":
            # dont keep non english
            continue

        tweet_data["stance"] = stance
        all_tweet_data.append(tweet_data)

    save_pkl(all_tweet_data, cache_path)


def merge_tweets_from_processed_raw(jsonl_paths):
    cache_paths = [f"{jsonl_path}.{DOWNSIZE_FACTOR}.pkl" for jsonl_path in jsonl_paths]
    handler = ParallelHandler(load_pkl)
    all_tweets = handler.run(cache_paths, flatten=True)
    return all_tweets


def get_all_tweets_from_raw_tweets():
    all_jsonl_paths = sorted(glob(join(RAW_DIR, "tweets", "*.jsonl")))
    # process
    handler = ParallelHandler(process_tweets_from_raw)
    handler.run(all_jsonl_paths)
    # merge
    all_tweets = merge_tweets_from_processed_raw(all_jsonl_paths)
    # dedup
    id2tweets = {t["id"]: t for t in all_tweets}
    all_tweets = list(id2tweets.values())

    return all_tweets


def _save_stats(tweets):
    stats_dir = join(SUBSET_WORKING_DIR, "overall_stats")
    mkdir_overwrite(stats_dir)

    # count stances
    stances = dict(Counter(t["stance"] for t in tweets))
    plt.bar(stances.keys(), stances.values())
    plt.savefig(join(stats_dir, "stances_count.png"))
    plt.clf()

    # count time
    times = [datetime.strptime(tweet["time"], TIME_FORMAT) for tweet in tweets]
    # aggregate by year
    yyyymms = [f"{t.year}" for t in times]
    agg_year = dict(Counter(yyyymms))
    agg_year = OrderedDict(sorted(agg_year.items()))
    pprint(agg_year)
    plt.bar(agg_year.keys(), agg_year.values())
    plt.savefig(join(stats_dir, "agg_year.png"))
    plt.clf()

    # various metrics
    metrics = {}
    metrics["total"] = len(tweets)
    metrics["stances"] = stances
    metrics["unique_tweeter"] = len(set(t["userid"] for t in tweets))
    metrics["mean_raw_len"] = sum(len(t["text"]) for t in tweets) / len(tweets)
    metrics["mean_num_stem"] = sum(len(t["stems"]) for t in tweets) / len(tweets)
    metrics["mean_num_hashtags"] = sum(len(t["hashtags"]) for t in tweets) / len(tweets)
    save_json(metrics, join(stats_dir, "metrics.json"))


if __name__ == "__main__":
    makedirs(SUBSET_WORKING_DIR, exist_ok=True)

    if not exists(SUBSET_PKL_PATH):
        all_tweets = get_all_tweets_from_raw_tweets()
        save_pkl(all_tweets, SUBSET_PKL_PATH)
    else:
        all_tweets = load_pkl(SUBSET_PKL_PATH)

    counter = Counter(t["stance"] for t in all_tweets)
    print(counter)

    _save_stats(all_tweets)
