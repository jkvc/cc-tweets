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

from cc_tweets.utils import (
    ParallelHandler,
    get_ngrams,
    load_pkl,
    save_pkl,
    write_str_list_as_txt,
)

SRC_DATASET_NAME = "tweets_downsized100_unfiltered"
PKL_PATH = join(DATA_DIR, f"{SRC_DATASET_NAME}.pkl")
TOP_N = 10000
NGRAM = 2
SAVE_PATH = join(DATA_DIR, f"vocab_{TOP_N}_{NGRAM}gram.txt")


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)
    counts = Counter()
    for t in tqdm(tweets):
        ngrams = get_ngrams(t["stems"], 2)
        counts.update(Counter(ngrams))

    count0token = sorted([(c, w) for w, c in counts.items()], reverse=True)
    vocab = [w for c, w in count0token[:TOP_N]]
    write_str_list_as_txt(vocab, SAVE_PATH)
