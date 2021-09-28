from collections import defaultdict
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.log_odds import get_topn_lors, scaled_lor
from cc_tweets.utils import load_pkl, mkdir_overwrite
from tqdm import tqdm

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]

    savedir = join(SUBSET_WORKING_DIR, "log_odds_partisan")
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

    print("stuff written to ", savedir)
