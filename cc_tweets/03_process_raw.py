import json
import re
import string
from collections import Counter
from glob import glob
from os.path import exists, join
from typing import List

import validators
from config import DATA_DIR, RAW_DIR
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from cc_tweets.utils import ParallelHandler, load_pkl, save_pkl

DOWNSIZE_FACTOR = 100
FILTER_UNK = False
SAVE_PATH = join(
    DATA_DIR,
    f"tweets_downsized{DOWNSIZE_FACTOR}{'_filtered' if FILTER_UNK else '_unfiltered'}.pkl",
)

STOPWORDS = stopwords.words("english")


def get_data_from_raw_tweet(tweet):
    if "retweeted_status" in tweet:
        # do the 'dereference' if it's a retweet
        return tweet["retweeted_status"]
    else:
        return tweet


def get_tokens(cleaned_text: str) -> List[str]:
    text = cleaned_text.lower()
    tokens = re.findall(r"[\w']+|[.,!?;]", text)
    tokens = [tok.replace("'", "") for tok in tokens]
    return tokens


def get_words(cleaned_text: str) -> List[str]:
    tokens = get_tokens(cleaned_text)
    return [
        tok
        for tok in tokens
        if (
            len(tok) > 0
            and tok not in STOPWORDS
            and not tok in string.punctuation
            and not tok.isdigit()
        )
    ]


def remove_urls(text: str) -> str:
    tokens = text.split()
    tokens = [tok for tok in tokens if not validators.url(tok)]
    return " ".join(tokens)


def get_stems(tokens: List[str]) -> List[str]:
    stemmer = SnowballStemmer("english")
    return [stemmer.stem(tok) for tok in tokens]


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
        tweet = get_data_from_raw_tweet(tweet)
        stance = userid2stance[tweet["user"]["id_str"]]
        if stance == "unk" and FILTER_UNK:
            continue

        text = tweet["full_text"]
        stems = get_stems(get_words(remove_urls(text)))

        tweet_data = {
            "id": tweet["id_str"],
            "likes": tweet["favorite_count"],
            "retweets": tweet["retweet_count"],
            "time": tweet["created_at"],
            "userid": tweet["user"]["id_str"],
            "username": tweet["user"]["screen_name"],
            "hashtags": tweet["entities"]["hashtags"],
            "text": text,
            "stems": stems,
            "stance": stance,
        }
        all_tweet_data.append(tweet_data)
    return all_tweet_data


def get_all_tweets_from_raw_tweets():
    all_jsonl_paths = sorted(glob(join(RAW_DIR, "tweets", "*.jsonl")))
    handler = ParallelHandler(get_tweets_from_raw)
    all_tweets = handler.run(all_jsonl_paths, flatten=True)
    return all_tweets


if __name__ == "__main__":
    if not exists(SAVE_PATH):
        all_tweets = get_all_tweets_from_raw_tweets()
        save_pkl(all_tweets, SAVE_PATH)
    else:
        all_tweets = load_pkl(SAVE_PATH)

    counter = Counter(t["stance"] for t in all_tweets)
    print(counter)
