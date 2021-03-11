from os.path import join
from pprint import pprint

import numpy as np
import scipy.sparse
from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams, load_vocab2idx
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_json, load_pkl, write_str_list_as_txt

# def get_features_single_layer(json_path, ids):
#     id2val = load_json(json_path)
#     feature = np.array([id2val[id] for id in tqdm(ids)])
#     feature = (feature - feature.mean()) / feature.std()
#     return scipy.sparse.csr_matrix(feature)


# def get_features_double_layer(name_prefix, json_path, ids):
#     id2feat2val = load_json(json_path)
#     feature_names = list(id2feat2val[ids[0]].keys())
#     features = []
#     for name in feature_names:
#         features.append(scipy.sparse.csr_matrix([id2feat2val[id][name] for id in ids]))
#     feature_names = [f"{name_prefix}_{name}" for name in feature_names]
#     return feature_names, features


# def build_vocab_feature(toktype, ngram, topn, tweets, ids):
#     vocab2idx = load_vocab2idx(
#         join(DATASET_SAVE_DIR, f"vocab_{toktype}_{ngram}gram_{topn}.txt")
#     )
#     id2idx = {id: idx for idx, id in enumerate(ids)}

#     feats = np.zeros((len(vocab2idx), len(tweets)))
#     for t in tqdm(tweets):
#         ididx = id2idx[t["id"]]

#         for tok in get_ngrams(t[toktype], ngram):
#             if tok in vocab2idx:
#                 feats[vocab2idx[tok]][ididx] += 1

#     idx2vocab = {i: tok for tok, i in vocab2idx.items()}
#     featnames = [
#         f"_{toktype}_{ngram}gram {idx2vocab[i]}" for i in range(len(idx2vocab))
#     ]
#     return featnames, scipy.sparse.csr_matrix(feats)


def load_features(ids, feature_names):
    features = []
    for name in tqdm(feature_names):
        id2val = load_pkl(join(DATASET_SAVE_DIR, "features", f"{name}.pkl"))
        f = np.array([id2val[id] for id in ids])
        f = (f - f.mean()) / f.std()
        f = scipy.sparse.csr_matrix(f)
        features.append(f)
    return features


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    ids = [t["id"] for t in tweets]

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
    features = load_features(ids, feature_names)

    feature_names.append("bias")
    features.append(scipy.sparse.csr_matrix(np.ones((len(ids),))))

    feature_matrix = scipy.sparse.vstack(features).T
    print(feature_matrix.shape)

    write_str_list_as_txt(feature_names, join(DATASET_SAVE_DIR, f"feature_names.txt"))
    write_str_list_as_txt(ids, join(DATASET_SAVE_DIR, f"feature_ids.txt"))

    scipy.sparse.save_npz(join(DATASET_SAVE_DIR, "features.npz"), feature_matrix)
