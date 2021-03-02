import json
import re
import string
from collections import Counter
from os.path import join

from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_pkl, write_str_list_as_txt

TOP_N = 4000

TOK_TYPES = ["stems", "lemmas"]
NGRAMS = [1, 2]

if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    for tok_type in TOK_TYPES:
        for ngram in NGRAMS:

            counts = Counter()
            for t in tqdm(tweets):
                ngrams = get_ngrams(t[tok_type], ngram)
                counts.update(Counter(ngrams))

            count0token = sorted([(c, w) for w, c in counts.items()], reverse=True)
            vocab = [w for c, w in count0token[:TOP_N]]
            save_path = join(DATASET_SAVE_DIR, f"vocab_{tok_type}_{ngram}gram.txt")
            write_str_list_as_txt(vocab, save_path)
