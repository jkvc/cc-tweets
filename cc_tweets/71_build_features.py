from os import makedirs
from os.path import join
from pprint import pprint

import numpy as np
import scipy.sparse
from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import (
    load_json,
    load_pkl,
    read_txt_as_str_list,
    save_pkl,
    write_str_list_as_txt,
)


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
                tweets, join(DATASET_SAVE_DIR, "features", f"{name}.pkl")
            )
        )
    return features


def load_vocab_feature(tweets, prefix):
    vocab_feature_dir = join(DATASET_SAVE_DIR, f"{prefix}_features")
    vocab = read_txt_as_str_list(join(vocab_feature_dir, "_names.txt"))
    feature_names = [f"{prefix}: {w}" for w in vocab]
    features = [
        _load_single_feature(tweets, join(vocab_feature_dir, f"{w}.pkl"))
        for w in tqdm(vocab, desc=f"vocab feature {prefix}")
    ]
    return feature_names, features


def get_log_follower_features(tweets):
    userid2numfollowers = load_json(join(DATA_DIR, "userid2numfollowers.json"))
    followers = np.array([userid2numfollowers.get(t["userid"], 0) for t in tweets])
    log_followers = np.log(followers + 1)
    return scipy.sparse.csr_matrix(log_followers)


def get_log_retweets(tweets):
    retweets = np.array([t["retweets"] for t in tweets])
    log_retweets = np.log(retweets + 1)
    return scipy.sparse.csr_matrix(log_retweets)


if __name__ == "__main__":
    savedir = join(DATASET_SAVE_DIR, "regression_inputs")
    makedirs(savedir, exist_ok=True)

    tweets = load_pkl(DATASET_PKL_PATH)

    # specifically defined features
    feature_names = [
        "economy",
        "emolex_anger",
        "emolex_anticipation",
        "emolex_disgust",
        "emolex_fear",
        "emolex_joy",
        "emolex_sadness",
        "emolex_surprise",
        "emolex_trust",
        "natural_disasters",
        "negation",
        "p_aoc",
        "p_attenborough",
        "p_bernie",
        "p_biden",
        "p_clinton",
        "p_gore",
        "p_inslee",
        "p_nye",
        "p_obama",
        "p_pruitt",
        "p_thunberg",
        "p_trump",
        "subj_combined",
        # "subj_strong",
        # "subj_weak",
        "vad_arousal",
        "vad_dominance",
        "vad_valence",
        "vader_compound",
        # "vader_neg",
        # "vader_neu",
        # "vader_pos",
        "mfd_vice_authority",
        "mfd_vice_fairness",
        "mfd_vice_harm",
        "mfd_vice_loyalty",
        "mfd_vice_purity",
        "mfd_virtue_authority",
        "mfd_virtue_fairness",
        "mfd_virtue_harm",
        "mfd_virtue_loyalty",
        "mfd_virtue_purity",
    ]
    features = load_features(tweets, feature_names)

    # add vocab features
    # for prefix in ["2gram"]:
    #     ns, fs = load_vocab_feature(tweets, prefix)
    #     feature_names.extend(ns)
    #     features.extend(fs)

    # add bias and follower features
    feature_names.append("bias")
    features.append(scipy.sparse.csr_matrix(np.ones((len(tweets),))))
    feature_names.append("log_followers")
    features.append(get_log_follower_features(tweets))

    # stack and save
    feature_matrix = scipy.sparse.vstack(features).T
    write_str_list_as_txt(feature_names, join(savedir, f"feature_names.txt"))
    scipy.sparse.save_npz(join(savedir, "feature_matrix.npz"), feature_matrix)

    log_retweets = get_log_retweets(tweets)
    scipy.sparse.save_npz(join(savedir, "log_retweets.npz"), log_retweets)

    idxs_dem = [i for i, t in enumerate(tweets) if t["stance"] == "dem"]
    idxs_rep = [i for i, t in enumerate(tweets) if t["stance"] == "rep"]
    save_pkl(idxs_dem, join(savedir, "idxs_dem.pkl"))
    save_pkl(idxs_rep, join(savedir, "idxs_rep.pkl"))
