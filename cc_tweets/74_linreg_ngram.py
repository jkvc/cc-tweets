from collections import defaultdict
from os import makedirs
from os.path import join

import numpy as np
import pandas as pd
import scipy.sparse
import statsmodels.api as sm
from config import RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.feature_utils import get_log_follower_features, get_log_retweets
from cc_tweets.utils import load_pkl, read_txt_as_str_list

TOKTYPE = "stems"
NGRAM = 3

if __name__ == "__main__":
    savedir = join(DATASET_SAVE_DIR, "linreg_ngram")
    makedirs(savedir, exist_ok=True)

    tweets = load_pkl(DATASET_PKL_PATH)
    bidx2bigram = read_txt_as_str_list(
        join(DATASET_SAVE_DIR, "vocab", f"{TOKTYPE}_{NGRAM}gram_300.txt")
    )
    bigram2bidx = {bigram: i for i, bigram in enumerate(bidx2bigram)}

    feature_matrix = scipy.sparse.lil_matrix((len(bidx2bigram), len(tweets)))

    for tidx, tweet in enumerate(tqdm(tweets)):
        bigrams = get_ngrams(tweet["stems"], NGRAM)
        for bigram in bigrams:
            if bigram in bigram2bidx:
                bidx = bigram2bidx[bigram]
                feature_matrix[bidx, tidx] += 1

    features = []
    feature_names = []
    for bidx, bigram in enumerate(tqdm(bidx2bigram)):
        feature_names.append(bigram)
        f = feature_matrix[bidx].toarray()
        f = (f - f.mean()) / (f.std() + 1e-10)  # zscore
        f = scipy.sparse.csr_matrix(f)
        features.append(f)

    # add bias and follower features
    feature_names.append("bias")
    features.append(scipy.sparse.csr_matrix(np.ones((len(tweets),))))
    feature_names.append("log_followers")
    features.append(get_log_follower_features(tweets))

    # stack
    feature_matrix = scipy.sparse.vstack(features).T.toarray()
    log_retweets = get_log_retweets(tweets).T.toarray()
    print(feature_matrix.shape, log_retweets.shape)

    print("begin reg")
    model = sm.OLS(log_retweets, feature_matrix)
    fit = model.fit()
    print("  end reg")

    name2coef = {name: coef for name, coef in zip(feature_names, fit.params)}
    rows = [
        (name, coef)
        for name, coef in name2coef.items()
        if name not in ["log_followers", "bias"]
    ]
    df = pd.DataFrame(
        rows,
        columns=["name", "coef"],
    )
    df.to_csv(join(savedir, f"{TOKTYPE}_{NGRAM}gram.csv"))
