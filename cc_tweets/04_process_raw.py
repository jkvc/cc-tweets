import json
from collections import Counter
from glob import glob
from os.path import exists, join
from typing import List

from config import DATA_DIR, RAW_DIR
from nltk.corpus import stopwords
from tqdm import tqdm

from cc_tweets.data_utils import parse_raw_tweet
from cc_tweets.experiment_config import DATASET_PKL_PATH, DOWNSIZE_FACTOR, FILTER_UNK
from cc_tweets.utils import ParallelHandler, load_pkl, save_pkl

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


if __name__ == "__main__":
    if not exists(DATASET_PKL_PATH):
        all_tweets = get_all_tweets_from_raw_tweets()
        save_pkl(all_tweets, DATASET_PKL_PATH)
    else:
        all_tweets = load_pkl(DATASET_PKL_PATH)

    counter = Counter(t["stance"] for t in all_tweets)
    print(counter)
