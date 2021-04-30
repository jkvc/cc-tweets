from collections import defaultdict
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_config import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.log_odds import scaled_lor
from cc_tweets.utils import load_pkl, mkdir_overwrite
from tqdm import tqdm

MIN_UNIQUE_USER = 100  # only keep words used by at least this many users


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
    tweets = load_pkl(SUBSET_PKL_PATH)

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]

    savedir = join(SUBSET_WORKING_DIR, "log_odds")
    mkdir_overwrite(savedir)

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

        df = pd.DataFrame()
        df["dem"] = dem_topwords
        df["rep"] = rep_topwords
        df.to_csv(join(savedir, f"{tok_type}_{ngrams}.csv"))
