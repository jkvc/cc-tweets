from os.path import join
from pprint import pprint

import numpy as np
import scipy.sparse
from config import DATA_DIR
from sklearn.linear_model import LinearRegression, Ridge

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_json, load_pkl, read_txt_as_str_list
from cc_tweets.viz import plot_feature_weights

if __name__ == "__main__":
    feature_matrix = scipy.sparse.load_npz(join(DATASET_SAVE_DIR, "features.npz"))
    feature_names = read_txt_as_str_list(join(DATASET_SAVE_DIR, f"feature_names.txt"))
    feature_ids = read_txt_as_str_list(join(DATASET_SAVE_DIR, f"feature_ids.txt"))
    id2idx = {id: i for i, id in enumerate(feature_ids)}
    userid2numfollowers = load_json(join(DATA_DIR, "userid2numfollowers.json"))
    mean_num_followers = sum(userid2numfollowers.values()) / len(userid2numfollowers)

    tweets = load_pkl(DATASET_PKL_PATH)

    followers = np.zeros((len(id2idx),))
    retweets = np.zeros((len(id2idx),))
    for t in tweets:
        id = t["id"]
        userid = t["userid"]
        idx = id2idx[id]
        followers[idx] = userid2numfollowers.get(userid, 0)
        retweets[idx] = t["retweets"]
    log_followers = np.log(followers + 1)
    log_retweets = np.log(retweets + 1)

    feature_names.append("log_followers")
    print(feature_matrix.shape)
    feature_matrix = scipy.sparse.hstack(
        [feature_matrix, np.expand_dims(log_followers, -1)]
    )

    for regname, regfactory in [
        (
            "linreg",
            lambda: LinearRegression(fit_intercept=False),
        ),
        (
            "ridge",
            lambda: Ridge(fit_intercept=False),
        ),
    ]:
        reg = regfactory()
        reg.fit(feature_matrix, log_retweets)

        feature2coef = {}
        for i, feature_name in enumerate(feature_names):
            feature2coef[feature_name] = reg.coef_[i]

        plot_feature_weights(
            feature2coef,
            join(DATASET_SAVE_DIR, f"73_{regname}.png"),
            title=f"{regname} on log retweets",
            xlim=(-0.3, 0.3),
        )
