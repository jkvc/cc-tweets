import json
import re
import string
from collections import Counter, defaultdict
from os import makedirs
from os.path import join

from cc_tweets.data_utils import get_ngrams
from cc_tweets.utils import load_pkl, write_str_list_as_txt
from config import DATA_DIR
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from tqdm import tqdm

TOP_NS = [10000, 4000, 1000, 300]

TOK_TYPES = ["stems", "lemmas"]
NGRAMS = [1, 2, 3]
MIN_NUM_UNIQUE_TWEETER = 30

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    makedirs(join(SUBSET_WORKING_DIR, "vocab"), exist_ok=True)

    for tok_type in TOK_TYPES:
        for ngram in NGRAMS:
            ngram2tweeters = defaultdict(set)
            counts = Counter()
            for t in tqdm(tweets):
                ngrams = get_ngrams(t[tok_type], ngram)
                counts.update(Counter(ngrams))
                for w in ngrams:
                    ngram2tweeters[w].add(t["username"])

            count0token = sorted(
                [
                    (c, w)
                    for w, c in counts.items()
                    if len(ngram2tweeters[w]) >= MIN_NUM_UNIQUE_TWEETER
                ],
                reverse=True,
            )
            for top_n in TOP_NS:
                vocab = [w for c, w in count0token[:top_n]]
                save_path = join(
                    SUBSET_WORKING_DIR, "vocab", f"{tok_type}_{ngram}gram_{top_n}.txt"
                )
                write_str_list_as_txt(vocab, save_path)
