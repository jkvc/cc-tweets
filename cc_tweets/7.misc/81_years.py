from collections import defaultdict
from os import makedirs
from os.path import join
from pprint import pprint

import numpy as np
import pandas as pd
import scipy.sparse
import statsmodels.api as sm
from config import RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_config import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import get_log_follower_features, get_log_retweets
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json


def howmany_year(tweets):
    numyear2count = defaultdict(int)
    for t in tqdm(tweets):
        bigrams = get_ngrams(t["toks"], 2)
        for gram in bigrams:
            a, b = gram.split(" ")
            if a.isdigit() and b in ["year", "years", "yrs"]:
                numyear2count[a] += 1
    return numyear2count


NAME0FUNC = [
    ("howmany_year", howmany_year),
]

if __name__ == "__main__":
    savedir = join(SUBSET_WORKING_DIR, "misc")
    makedirs(savedir, exist_ok=True)

    tweets = load_pkl(SUBSET_PKL_PATH)

    for name, func in NAME0FUNC:
        val2count = func(tweets)
        val0count = sorted(
            [(n, c) for n, c in val2count.items()], reverse=True, key=lambda x: x[1]
        )
        df = pd.DataFrame(val0count, columns=["val", "count"])
        df.to_csv(join(savedir, f"{name}.csv"), index=False)
