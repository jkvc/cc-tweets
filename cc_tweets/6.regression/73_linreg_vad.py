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

from cc_tweets.experiment_config import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import get_log_follower_features, get_log_retweets
from cc_tweets.utils import load_pkl, read_txt_as_str_list

VAD_PATH = join(RESOURCES_DIR, "NRC-VAD-Lexicon-Aug2018Release", "OneFilePerDimension")
VAD_TO_ABBRV = {
    "valence": "v",
    "arousal": "a",
    "dominance": "d",
}

TOP_NUM_LEMMAS = 500


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
    savedir = join(SUBSET_WORKING_DIR, "linreg_vad")
    makedirs(savedir, exist_ok=True)

    tweets = load_pkl(SUBSET_PKL_PATH)
    lemma2score = load_vad2lemma2score()["valence"]
    lemma2lidx = {lemma: i for i, lemma in enumerate(lemma2score)}
    lidx2lemma = {i: lemma for i, lemma in enumerate(lemma2score)}

    feature_matrix = scipy.sparse.lil_matrix((len(lemma2score), len(tweets)))
    lemma_counts = np.zeros((len(lemma2score),))

    for tidx, tweet in enumerate(tqdm(tweets)):
        for lemma in tweet["lemmas"]:
            if lemma in lemma2score:
                score = lemma2score[lemma]
                lidx = lemma2lidx[lemma]
                feature_matrix[lidx, tidx] += score
                lemma_counts[lidx] += 1

    count0lidx = [(count, lidx) for lidx, count in enumerate(lemma_counts)]
    count0lidx = sorted(count0lidx, reverse=True)[:TOP_NUM_LEMMAS]
    filtered_lidx = [lidx for _, lidx in count0lidx]

    features = []
    feature_names = []
    for lidx in tqdm(filtered_lidx):
        lemma = lidx2lemma[lidx]
        feature_names.append(lemma)
        f = feature_matrix[lidx].toarray()
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

    model = sm.OLS(log_retweets, feature_matrix)
    fit = model.fit()

    name2coef = {name: coef for name, coef in zip(feature_names, fit.params)}
    type2lemma2score = load_vad2lemma2score()
    rows = [
        (
            coef,
            name,
            lemma_counts[lemma2lidx[name]],
            type2lemma2score["valence"][name],
            type2lemma2score["arousal"][name],
            type2lemma2score["dominance"][name],
        )
        for name, coef in name2coef.items()
        if name not in ["log_followers", "bias"]
    ]
    df = pd.DataFrame(
        rows,
        columns=["coef", "name", "count", "valence", "arousal", "dominance"],
    )
    df.to_csv(join(savedir, "coefs.csv"))
