import re
from collections import Counter, defaultdict
from os import makedirs, replace
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import statsmodels.api as sm
from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from scipy.sparse.lil import lil_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.feature_utils import (
    get_log_follower_features,
    get_log_retweets,
    save_features,
)
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import (
    load_pkl,
    read_txt_as_str_list,
    save_json,
    write_str_list_as_txt,
)
from cc_tweets.viz import grouped_bars, plot_horizontal_bars

VAD_PATH = join(RESOURCES_DIR, "NRC-VAD-Lexicon-Aug2018Release", "OneFilePerDimension")
VAD_TO_ABBRV = {
    "valence": "v",
    "arousal": "a",
    "dominance": "d",
}

TOP_NUM_LEMMAS = 300
TYPE = "valence"


def load_vad2lemma2score():
    lemmatizer = WordNetLemmatizer()
    vad2lemma2score = defaultdict(dict)
    for vad, abbrv in VAD_TO_ABBRV.items():
        lines = read_txt_as_str_list(join(VAD_PATH, f"{abbrv}-scores.txt"))
        for line in lines:
            word, score = line.split("\t")
            lemma = lemmatizer.lemmatize(word)
            score = float(score)
            vad2lemma2score[vad][lemma] = score
    return dict(vad2lemma2score)


if __name__ == "__main__":
    savedir = join(DATASET_SAVE_DIR, f"linreg_vad.{TYPE}")
    makedirs(savedir, exist_ok=True)

    tweets = load_pkl(DATASET_PKL_PATH)
    lemma2score = load_vad2lemma2score()[TYPE]
    lemma2lidx = {lemma: i for i, lemma in enumerate(lemma2score)}
    lidx2lemma = {i: lemma for i, lemma in enumerate(lemma2score)}

    feature_matrix = np.zeros((len(tweets), len(lemma2score)))
    lemma_counts = np.zeros((len(lemma2score),))

    for tidx, tweet in enumerate(tqdm(tweets)):
        for lemma in tweet["lemmas"]:
            if lemma in lemma2score:
                score = lemma2score[lemma]
                lidx = lemma2lidx[lemma]
                feature_matrix[tidx, lidx] += score
                lemma_counts[lidx] += 1

    count0lidx = [(count, lidx) for lidx, count in enumerate(lemma_counts)]
    count0lidx = sorted(count0lidx, reverse=True)[:TOP_NUM_LEMMAS]
    filtered_lidx = [lidx for _, lidx in count0lidx]

    features = []
    feature_names = []
    for lidx in tqdm(filtered_lidx):
        lemma = lidx2lemma[lidx]
        feature_names.append(lemma)
        f = feature_matrix[:, lidx]
        f = (f - f.mean()) / f.std()  # zscore
        f = scipy.sparse.csr_matrix(f)
        features.append(f)

    # add bias and follower features
    feature_names.append("bias")
    features.append(scipy.sparse.csr_matrix(np.ones((len(tweets),))))
    feature_names.append("log_followers")
    features.append(get_log_follower_features(tweets))

    # stack and save
    feature_matrix = scipy.sparse.vstack(features).T.toarray()
    # write_str_list_as_txt(feature_names, join(savedir, f"feature_names.txt"))
    # scipy.sparse.save_npz(join(savedir, "feature_matrix.npz"), feature_matrix)

    log_retweets = get_log_retweets(tweets).T.toarray()
    # scipy.sparse.save_npz(join(savedir, "log_retweets.npz"), log_retweets)

    print(feature_matrix.shape, log_retweets.shape)

    model = sm.OLS(log_retweets, feature_matrix)
    fit = model.fit()

    name2coef = {name: coef for name, coef in zip(feature_names, fit.params)}
    abscoef0name = sorted(
        [(abs(coef), name) for name, coef in name2coef.items()], reverse=True
    )
    coef0name0vadscore0count = [
        (name2coef[name], name, lemma2score[name], lemma_counts[lemma2lidx[name]])
        for _, name in abscoef0name
        if name not in ["log_followers", "bias"]
    ]
    pprint(coef0name0vadscore0count)
