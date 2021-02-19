import json
import re
import string
from collections import Counter
from os.path import join

from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams
from cc_tweets.utils import load_pkl, write_str_list_as_txt

SRC_DATASET_NAME = "tweets_downsized100_unfiltered"
PKL_PATH = join(DATA_DIR, f"{SRC_DATASET_NAME}.pkl")
TOP_N = 3000
NGRAM = 1
SAVE_PATH = join(DATA_DIR, f"vocab_{TOP_N}_{NGRAM}gram.txt")


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)
    counts = Counter()
    for t in tqdm(tweets):
        ngrams = get_ngrams(t["stems"], NGRAM)
        counts.update(Counter(ngrams))

    count0token = sorted([(c, w) for w, c in counts.items()], reverse=True)
    vocab = [w for c, w in count0token[:TOP_N]]
    write_str_list_as_txt(vocab, SAVE_PATH)
