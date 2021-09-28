import fnmatch
import sys
from os import makedirs
from os.path import exists, join
from posixpath import dirname

import numpy as np
import scipy.sparse
import statsmodels.api as sm
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import get_log_follower_features, get_log_retweets
from cc_tweets.lexical_features.bank import get_all_feature_names, get_feature
from cc_tweets.regression_configs import get_regression_config
from cc_tweets.utils import load_pkl, save_json, save_pkl
from cc_tweets.viz import plot_horizontal_bars
from config import DATA_DIR
from tqdm import tqdm

# want to take a look at tweets that are
# 1. high or low dominance, but not in betweet (10 percentile)
# 2. high arousal negative HAN
# 3. has explicit threat feature
# 4. has at least 1000 followers
# 5. has at least 100 retweets

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    dominance_featdict = get_feature("vad.dominance").get_feature_dict(tweets)
    sentihan_featdict = get_feature("senti.HAN").get_feature_dict(tweets)
    threat_featdict = get_feature("term.threat").get_feature_dict(tweets)

    dominance_scores = sorted(list(dominance_featdict.values()))
    dominance_bottom_10_percentile_threash = dominance_scores[
        len(dominance_scores) // 10
    ]
    dominance_top_10_percentile_threash = dominance_scores[
        (len(dominance_scores) // 10) * 9
    ]

    selected_tweets = []
    for t in tweets:
        tid = t["id"]

        dominance_score = dominance_featdict[tid]
        if (
            dominance_score > dominance_bottom_10_percentile_threash
            and dominance_score < dominance_top_10_percentile_threash
        ):
            continue

        if sentihan_featdict[tid] == 0:
            continue

        if threat_featdict[tid] == 0:
            continue

        if t["retweets"] < 100:
            continue
        if t["max_num_follower"] < 1000:
            continue

        # if t["stance"] != "rep":
        #     continue

        selected_tweets.append(t)

    print(len(selected_tweets), "tweets remaining")

    for t in selected_tweets:
        print(t["retweets"])
        print(t["max_num_follower"])
        print(t["text"])
        input()
