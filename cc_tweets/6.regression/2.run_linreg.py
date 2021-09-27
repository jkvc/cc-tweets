# USAGE python3 script_name config_name
# for config name see ../regression_configs.py

import sys
from os import makedirs
from os.path import exists, join
from posixpath import dirname

import numpy as np
import scipy.sparse
import statsmodels.api as sm
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import get_log_follower_features, get_log_retweets
from cc_tweets.regression_configs import get_regression_config
from cc_tweets.utils import load_pkl, save_json, save_pkl
from cc_tweets.viz import plot_horizontal_bars
from config import DATA_DIR
from tqdm import tqdm

SEED = 0xDEADBEEF

CONFIG_NAME = sys.argv[1]
USE_FEATURE_MATRIX_CACHE = False


def _load_single_feature(tweets, path):
    id2val = load_pkl(path)
    f = np.array([id2val.get(t["id"], 0) for t in tweets])
    f = (f - f.mean()) / f.std()  # zscore
    f = scipy.sparse.csr_matrix(f)
    return f


def load_features(tweets, feature_names):
    features = []
    for name in tqdm(feature_names):
        features.append(
            _load_single_feature(
                tweets, join(SUBSET_WORKING_DIR, "feature_cache", f"{name}.pkl")
            )
        )
    return features


def get_linreg_inputs(config_name, config):
    matrix_cache_path = join(SUBSET_WORKING_DIR, "linreg_cache", f"{config_name}.npz")
    makedirs(dirname(matrix_cache_path), exist_ok=True)
    target_cache_path = join(
        SUBSET_WORKING_DIR, "linreg_cache", f"{config_name}.target.npz"
    )
    makedirs(dirname(target_cache_path), exist_ok=True)

    tweets = None

    # cache or build feature matrix
    if exists(matrix_cache_path) and USE_FEATURE_MATRIX_CACHE:
        feature_matrix = scipy.sparse.load_npz(matrix_cache_path)
    else:
        if tweets is None:
            tweets = load_pkl(SUBSET_PKL_PATH)
        features = load_features(tweets, config["features"])

        if config["const_bias"]:
            features.append(scipy.sparse.csr_matrix(np.ones((len(tweets),))))
        if config["log_follower_bias"]:
            features.append(
                get_log_follower_features(tweets, source="max_num_follower")
            )

        # stack and save
        feature_matrix = scipy.sparse.vstack(features).T
        scipy.sparse.save_npz(matrix_cache_path, feature_matrix)

        # save idx filters
        idxs_dem = [i for i, t in enumerate(tweets) if t["stance"] == "dem"]
        idxs_rep = [i for i, t in enumerate(tweets) if t["stance"] == "rep"]
        save_pkl(idxs_dem, join(SUBSET_WORKING_DIR, "linreg_cache", "filter_dem.pkl"))
        save_pkl(idxs_rep, join(SUBSET_WORKING_DIR, "linreg_cache", "filter_rep.pkl"))
        np.random.seed(SEED)
        subsampled_idxs_dem = np.random.choice(
            idxs_dem, size=len(idxs_rep), replace=False
        )
        idxs_balanced = idxs_rep + subsampled_idxs_dem.tolist()
        save_pkl(
            idxs_balanced,
            join(SUBSET_WORKING_DIR, "linreg_cache", "filter_balanced.pkl"),
        )

    # cache or build target vector
    if exists(target_cache_path) and USE_FEATURE_MATRIX_CACHE:
        target = scipy.sparse.load_npz(target_cache_path)
    else:
        if tweets is None:
            tweets = load_pkl(SUBSET_PKL_PATH)
        target = get_log_retweets(tweets)
        scipy.sparse.save_npz(target_cache_path, target)

    if tweets is not None:
        del tweets

    return feature_matrix, target


if __name__ == "__main__":
    print(CONFIG_NAME)
    config = get_regression_config(CONFIG_NAME)

    feature_matrix, target = get_linreg_inputs(CONFIG_NAME, config)

    savedir = join(SUBSET_WORKING_DIR, "linreg_out", CONFIG_NAME)
    makedirs(savedir, exist_ok=True)

    feature_matrix = feature_matrix.toarray()
    target = target.toarray().reshape(-1)

    for filter_name in ["balanced", "dem", "rep"]:
        idxs = load_pkl(
            join(SUBSET_WORKING_DIR, "linreg_cache", f"filter_{filter_name}.pkl")
        )
        filtered_feature_matrix = feature_matrix[idxs, :]
        filtered_targets = target[idxs]

        model = sm.OLS(filtered_targets, filtered_feature_matrix)
        fit = model.fit()

        input_names = config["features"]
        if config["const_bias"]:
            input_names.append("const_bias")
        if config["log_follower_bias"]:
            input_names.append("log_follower_bias")

        name2coef = {name: coef for name, coef in zip(input_names, fit.params)}
        name2std = {name: std for name, std in zip(input_names, fit.bse)}
        plot_horizontal_bars(
            name2coef,
            save_path=join(savedir, f"coef_{filter_name}.png"),
            title=f"LinReg coefficients | config [{CONFIG_NAME}] | filter [{filter_name}]",
            xlim=(-0.2, 0.2),
            yerr=fit.bse * 2,
            figsize=(8, 0.2 * feature_matrix.shape[1] + 1),
        )
        save_json(
            {"coef": name2coef, "std": name2std},
            join(savedir, f"coef_{filter_name}.json"),
        )

    print(savedir)

    # if name == "all":
    #     pred = fit.predict()
    #     err = (filtered_targets - pred) ** 2
    #     err0idx = sorted([(err, idx) for idx, err in enumerate(err)], reverse=True)
    #     most_inacc_idx = [i for _, i in err0idx[:30]]
    #     most_inacc = [
    #         {
    #             "idx": i,
    #             "pred": pred[i],
    #             "actual": filtered_targets[i],
    #             "sqerr": err[i],
    #             "tweet": tweets[i],
    #         }
    #         for i in most_inacc_idx
    #     ]
    #     save_json(most_inacc, join(savedir, f"inacc_{name}.json"))
