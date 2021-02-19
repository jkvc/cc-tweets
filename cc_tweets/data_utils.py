from datetime import datetime
from typing import List

from cc_tweets.utils import read_txt_as_str_list


def get_ngrams(unigrams: List[str], n: int) -> List[str]:
    if len(unigrams) < n:
        return []
    grams = []
    for i in range(len(unigrams) - n + 1):
        gram = [unigrams[i + j] for j in range(n)]
        grams.append(" ".join(gram))
    return grams


# Fri Nov 30 19:41:04 +0000 2018
TIME_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def get_tweet_time(tweet):
    return datetime.strptime(tweet["time"], TIME_FORMAT)


def load_vocab2idx(txt_path):
    return {gram: i for i, gram in enumerate(read_txt_as_str_list(txt_path))}
