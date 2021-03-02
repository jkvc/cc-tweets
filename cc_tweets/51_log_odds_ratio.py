from collections import Counter, defaultdict
from os.path import join
from pprint import pprint
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import DATA_DIR
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm, trange

from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import ParallelHandler, load_pkl, save_json

STOPWORDS = stopwords.words("english")
MIN_WORD_COUNT = 20
MIN_WORD_LEN = 2
MIN_UNIQUE_USER = 100  # only keep words used by at least this many users


def scaled_lor(
    left_wc: Dict[str, int],
    right_wc: Dict[str, int],
    background_wc: Dict[str, int],
    min_word_count: int = MIN_WORD_COUNT,
) -> List[Tuple[float, str]]:

    n_left = sum(left_wc.values()) + 1
    n_right = sum(right_wc.values()) + 1
    n_bg = sum(background_wc.values()) + 1
    l_r_corpus_ratio = n_right / n_left
    n_left *= l_r_corpus_ratio

    lor = {}
    for w, f_w_left in left_wc.items():
        if f_w_left < min_word_count:
            continue
        if len(w) < MIN_WORD_LEN:
            continue
        if w in STOPWORDS:
            continue

        f_w_left *= l_r_corpus_ratio
        f_w_right = right_wc.get(w, min_word_count)
        f_w_bg = background_wc.get(w, min_word_count)

        l_numerator = f_w_left + f_w_bg
        l_denominator = n_left + n_bg - l_numerator
        l = np.log(l_numerator / l_denominator)

        r_numerator = f_w_right + f_w_bg
        r_denominator = n_right + n_bg - r_numerator
        r = np.log(r_numerator / r_denominator)

        variance = 1 / (f_w_left + f_w_bg) + 1 / (f_w_right + f_w_bg)

        lor_w = l - r
        z_score = lor_w / (np.sqrt(variance))
        lor[w] = z_score

    lor0w = [(lor, w) for w, lor in lor.items()]
    lor0w = sorted(lor0w, reverse=True)
    return lor0w


def get_topn_lors(dem_tweets, rep_tweets, tok_type, ngrams, top_n=50):
    dem_tok2count = defaultdict(int)
    rep_tok2count = defaultdict(int)
    tok2users = defaultdict(set)
    for t in tqdm(dem_tweets):
        for tok in get_ngrams(t[tok_type], ngrams):
            tok2users[tok].add(t["id"])
            dem_tok2count[tok] += 1
    for t in tqdm(rep_tweets):
        for tok in get_ngrams(t[tok_type], ngrams):
            tok2users[tok].add(t["id"])
            rep_tok2count[tok] += 1

    filtered_dem_tok2count = {
        tok: count
        for tok, count in dem_tok2count.items()
        if len(tok2users[tok]) >= MIN_UNIQUE_USER
    }
    filtered_rep_tok2count = {
        tok: count
        for tok, count in rep_tok2count.items()
        if len(tok2users[tok]) >= MIN_UNIQUE_USER
    }

    lor0w = scaled_lor(filtered_dem_tok2count, filtered_rep_tok2count, {})
    dem_topwords = [w for lor, w in lor0w][:top_n]
    rep_topwords = [w for lor, w in lor0w[::-1]][:top_n]
    return dem_topwords, rep_topwords


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]

    results = {}
    for tok_type, ngrams in [
        ("lemmas", 1),
        ("lemmas", 2),
        ("stems", 1),
        ("stems", 2),
    ]:
        print(tok_type, ngrams)
        dem_topwords, rep_topwords = get_topn_lors(
            dem_tweets, rep_tweets, tok_type, ngrams
        )
        results[f"{tok_type}_{ngrams}grams"] = {
            "dem": dem_topwords,
            "rep": rep_topwords,
        }
    save_json(results, join(DATASET_SAVE_DIR, "51_log_odds_ratio.json"))
