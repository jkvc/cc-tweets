from os.path import join
from pprint import pprint

import numpy as np
import scipy.sparse
from config import DATA_DIR
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_json, load_pkl, read_txt_as_str_list

if __name__ == "__main__":
    feature_matrix = scipy.sparse.load_npz(join(DATASET_SAVE_DIR, "features.npz"))
    feature_names = read_txt_as_str_list(join(DATASET_SAVE_DIR, f"feature_names.txt"))
    feature_ids = read_txt_as_str_list(join(DATASET_SAVE_DIR, f"feature_ids.txt"))
    id2idx = {id: i for i, id in enumerate(feature_ids)}
    userid2numfollowers = load_json(join(DATA_DIR, "userid2numfollowers.json"))
    mean_num_followers = sum(userid2numfollowers.values()) / len(userid2numfollowers)

    tweets = load_pkl(DATASET_PKL_PATH)
    targets = np.zeros((len(tweets),))
    for tweet in tweets:
        if tweet["userid"] not in userid2numfollowers:
            num_followers = mean_num_followers
        else:
            num_followers = userid2numfollowers[tweet["userid"]]

        target = np.log(tweet["retweets"] + 2) / np.log(num_followers + 2)
        targets[id2idx[tweet["id"]]] = target

    reg = Ridge(fit_intercept=False)
    reg.fit(feature_matrix, targets)

    feature2coef = {}
    for i, feature_name in enumerate(feature_names):
        feature2coef[feature_name] = reg.coef_[i]

    pprint(feature2coef)
