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


def get_tweets_from_raw(jsonl_path):
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

        stance = userid2stance[tweet_data["id"]]
        if stance == "unk" and FILTER_UNK:
            continue
        if tweet_data["lang"] != "en":
            # dont keep non english
            continue

        all_tweet_data.append(tweet_data)
    return all_tweet_data


def get_all_tweets_from_raw_tweets():
    all_jsonl_paths = sorted(glob(join(RAW_DIR, "tweets", "*.jsonl")))
    handler = ParallelHandler(get_tweets_from_raw)
    all_tweets = handler.run(all_jsonl_paths, flatten=True)

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
