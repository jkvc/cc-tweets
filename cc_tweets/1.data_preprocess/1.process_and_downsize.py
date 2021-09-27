import json
import random
from collections import Counter, OrderedDict
from datetime import datetime
from glob import glob
from os import makedirs, mkdir
from os.path import exists, join
from posixpath import basename, dirname
from pprint import pprint
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cc_tweets.data_utils import parse_raw_tweet
from cc_tweets.experiment_configs import (
    DATA_SUBSET_SIZE,
    FILTER_UNK,
    SUBSET_PKL_PATH,
    SUBSET_WORKING_DIR,
)
from cc_tweets.utils import (
    ParallelHandler,
    load_json,
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

INVALIDATE_CACHE = False

_all_jsonl_paths = sorted(glob(join(RAW_DIR, "tweets", "*.jsonl")))
NUM_SAMPLES_TO_KEEP_PER_PROC = DATA_SUBSET_SIZE // len(_all_jsonl_paths) + 1


def process_tweets_from_raw(jsonl_path):
    # cache_path = f"{jsonl_path}.{DOWNSIZE_FACTOR}.pkl"

    cache_path = join(
        SUBSET_WORKING_DIR, "tweet_subset_cache", f"{basename(jsonl_path)}.pkl"
    )
    makedirs(dirname(cache_path), exist_ok=True)
    if not INVALIDATE_CACHE and exists(cache_path):
        return

    userid2stance_path = join(DATA_DIR, "followers_data", "userid2stance.pkl")
    userid2stance = load_pkl(userid2stance_path)

    all_tweet_data = []
    with open(jsonl_path) as f:
        lines = f.readlines()

    chosen_lines = random.sample(lines, min(len(lines), NUM_SAMPLES_TO_KEEP_PER_PROC))

    for line in tqdm(chosen_lines, disable=True):

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
    cache_paths = [
        join(SUBSET_WORKING_DIR, "tweet_subset_cache", f"{basename(jsonl_path)}.pkl")
        for jsonl_path in jsonl_paths
    ]
    handler = ParallelHandler(load_pkl)
    all_tweets = handler.run(cache_paths, flatten=True)
    return all_tweets


def dedup(all_tweets):
    tid2tweets = {}
    for t in all_tweets:
        tid = t["id"]
        if tid in tid2tweets:
            tid2tweets[tid]["retweeter_userids"].update(t["retweeter_userids"])
        else:
            tid2tweets[tid] = t
    for tid, t in tid2tweets.items():
        t["retweeter_userids"] = list(t["retweeter_userids"])
    return tid2tweets


def populate_followers_inplace(all_tweets):
    userid2numfollowers = load_json(
        join(DATA_DIR, "followers_data", "userid2numfollowers.json")
    )

    def get_num_follower(userid):
        return userid2numfollowers.get(userid, 0)

    for t in all_tweets:
        t["num_follower"] = get_num_follower(t["userid"])
        all_tweeter_userids = t["retweeter_userids"] + [t["userid"]]
        t["max_num_follower"] = max(
            get_num_follower(userid) for userid in all_tweeter_userids
        )


def get_all_tweets_from_raw_tweets():
    all_jsonl_paths = sorted(glob(join(RAW_DIR, "tweets", "*.jsonl")))
    # process
    handler = ParallelHandler(process_tweets_from_raw)
    handler.run(all_jsonl_paths)
    # merge
    all_tweets = merge_tweets_from_processed_raw(all_jsonl_paths)
    # dedup
    id2tweets = dedup(all_tweets)
    all_tweets = list(id2tweets.values())
    populate_followers_inplace(all_tweets)

    return all_tweets


def _save_stats(tweets):
    stats_dir = join(SUBSET_WORKING_DIR, "overall_stats")
    mkdir_overwrite(stats_dir)

    # count stances
    stances = dict(Counter(t["stance"] for t in tweets))
    ax = sns.barplot(
        x="y",
        y="x",
        data={
            "x": list(stances.keys()),
            "y": list(stances.values()),
        },
    )
    ax.set(xlabel="number of tweeter", ylabel="lean")
    plt.savefig(join(stats_dir, "stances_count.png"))
    plt.clf()

    # count time
    times = [datetime.strptime(tweet["time"], TIME_FORMAT) for tweet in tweets]
    # aggregate by year
    yyyymms = [f"{t.year}" for t in times]
    agg_year = dict(Counter(yyyymms))
    agg_year = OrderedDict(sorted(agg_year.items()))
    pprint(agg_year)
    ax = sns.barplot(
        x="y",
        y="x",
        data={
            "x": list(agg_year.keys()),
            "y": np.array(list(agg_year.values())) * 4,
        },
    )
    ax.set(xlabel="number of tweet", ylabel="year")
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
    metrics["mean_num_follower"] = sum((t["num_follower"]) for t in tweets) / len(
        tweets
    )
    metrics["mean_max_num_follower"] = sum(
        (t["max_num_follower"]) for t in tweets
    ) / len(tweets)
    save_json(metrics, join(stats_dir, "metrics.json"))


if __name__ == "__main__":
    makedirs(SUBSET_WORKING_DIR, exist_ok=True)

    if not exists(SUBSET_PKL_PATH) or INVALIDATE_CACHE:
        all_tweets = get_all_tweets_from_raw_tweets()
        save_pkl(all_tweets, SUBSET_PKL_PATH)
    else:
        all_tweets = load_pkl(SUBSET_PKL_PATH)

    counter = Counter(t["stance"] for t in all_tweets)
    print(counter)

    _save_stats(all_tweets)
