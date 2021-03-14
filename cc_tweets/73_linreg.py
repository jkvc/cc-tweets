from os import makedirs
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import statsmodels.api as sm

from cc_tweets.experiment_config import DATASET_SAVE_DIR
from cc_tweets.utils import read_txt_as_str_list, save_json
from cc_tweets.viz import plot_feature_weights

if __name__ == "__main__":
    savedir = join(DATASET_SAVE_DIR, "linreg")
    makedirs(savedir, exist_ok=True)

    regin_dir = join(DATASET_SAVE_DIR, "regression_inputs")
    feature_matrix = scipy.sparse.load_npz(join(regin_dir, "feature_matrix.npz"))
    feature_names = read_txt_as_str_list(join(regin_dir, f"feature_names.txt"))
    log_retweets = scipy.sparse.load_npz(join(regin_dir, f"log_retweets.npz"))
    log_retweets, feature_matrix = log_retweets.toarray(), feature_matrix.toarray()
    log_retweets = np.squeeze(log_retweets)

    model = sm.OLS(log_retweets, feature_matrix)
    fit = model.fit()

    name2coef = {name: coef for name, coef in zip(feature_names, fit.params)}
    name2std = {name: std for name, std in zip(feature_names, fit.bse)}
    plot_feature_weights(
        name2coef,
        save_path=join(savedir, "coef.png"),
        title="coefs: lin reg on log retweets",
        xlim=(-0.3, 0.3),
        yerr=fit.bse,
    )
    save_json({"coef": name2coef, "std": name2std}, join(savedir, "coef.json"))
