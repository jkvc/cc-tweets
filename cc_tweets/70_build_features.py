from os import makedirs
from os.path import join
from pprint import pprint

import numpy as np
import scipy.sparse
from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams, load_vocab2idx
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_json, load_pkl, save_pkl, write_str_list_as_txt


def load_features(tweets, feature_names):
    features = []
    for name in tqdm(feature_names):
        id2val = load_pkl(join(DATASET_SAVE_DIR, "features", f"{name}.pkl"))
        f = np.array([id2val[t["id"]] for t in tweets])
        f = (f - f.mean()) / f.std()  # zscore
        f = scipy.sparse.csr_matrix(f)
        features.append(f)
    return features


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

    feature_names.append("bias")
    features.append(scipy.sparse.csr_matrix(np.ones((len(tweets),))))

    feature_names.append("log_followers")
    features.append(get_log_follower_features(tweets))

    feature_matrix = scipy.sparse.vstack(features).T
    write_str_list_as_txt(feature_names, join(savedir, f"feature_names.txt"))
    scipy.sparse.save_npz(join(savedir, "feature_matrix.npz"), feature_matrix)

    log_retweets = get_log_retweets(tweets)
    scipy.sparse.save_npz(join(savedir, "log_retweets.npz"), log_retweets)

    idxs_dem = [i for i, t in enumerate(tweets) if t["stance"] == "dem"]
    idxs_rep = [i for i, t in enumerate(tweets) if t["stance"] == "rep"]
    save_pkl(idxs_dem, join(savedir, "idxs_dem.pkl"))
    save_pkl(idxs_rep, join(savedir, "idxs_rep.pkl"))
