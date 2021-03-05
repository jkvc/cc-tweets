from os.path import join
from pprint import pprint

import numpy as np
import scipy.sparse
from config import DATA_DIR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_json, load_pkl, read_txt_as_str_list, save_json

if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)
    userid2numfollowers = load_json(join(DATA_DIR, "userid2numfollowers.json"))
    mean_num_followers = sum(userid2numfollowers.values()) / len(userid2numfollowers)

    def _get_num_followers(userid):
        if userid in userid2numfollowers:
            return userid2numfollowers[userid]
        else:
            return mean_num_followers

    stats = {
        "median_likes_to_followers": (
            np.median(
                np.array(
                    [t["likes"] / (_get_num_followers(t["userid"]) + 1) for t in tweets]
                )
            )
        ),
        "median_retweets_to_followers": (
            np.median(
                np.array(
                    [
                        t["retweets"] / (_get_num_followers(t["userid"]) + 1)
                        for t in tweets
                    ]
                )
            )
        ),
    }
    save_json(stats, join(DATASET_SAVE_DIR, "71_engagement_stats.json"))
