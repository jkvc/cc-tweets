import json
import re
import string
from collections import Counter
from datetime import datetime
from glob import glob
from os.path import exists, join
from typing import List

import validators
from config import DATA_DIR, RAW_DIR
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from cc_tweets.utils import ParallelHandler, load_pkl, read_txt_as_str_list, save_pkl

STOPWORDS = stopwords.words("english")


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


def parse_raw_tweet(tweet):
    tweet = get_data_from_raw_tweet(tweet)

    if "full_text" not in tweet:
        return None
    text = tweet["full_text"]
    toks = get_words(remove_urls(text))
    stems = get_stems(toks)
    lemmas = get_lemmas(toks)

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
        "lemmas": lemmas,
        "lang": tweet["lang"],
    }
    return tweet_data


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


def get_lemmas(tokens: List[str]) -> List[str]:
    lemmeatizer = WordNetLemmatizer()
    return [lemmeatizer.lemmatize(tok) for tok in tokens]
