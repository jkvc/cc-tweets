import json
import random
import re
import string
from collections import Counter
from os.path import join
from pprint import pprint

import numpy as np
import scipy.sparse
from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams, load_vocab2idx
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_json, load_pkl, save_pkl, write_str_list_as_txt


def get_features_single_layer(json_path, ids):
    id2val = load_json(json_path)
    feature = scipy.sparse.csr_matrix([id2val[id] for id in ids])
    return feature


def get_features_double_layer(name_prefix, json_path, ids):
    id2feat2val = load_json(json_path)
    feature_names = list(id2feat2val[ids[0]].keys())
    features = []
    for name in feature_names:
        features.append(scipy.sparse.csr_matrix([id2feat2val[id][name] for id in ids]))
    feature_names = [f"{name_prefix}_{name}" for name in feature_names]
    return feature_names, features


def build_vocab_feature(toktype, ngram, topn, tweets, ids):
    vocab2idx = load_vocab2idx(
        join(DATASET_SAVE_DIR, f"vocab_{toktype}_{ngram}gram_{topn}.txt")
    )
    id2idx = {id: idx for idx, id in enumerate(ids)}

    feats = np.zeros((len(vocab2idx), len(tweets)))
    for t in tqdm(tweets):
        ididx = id2idx[t["id"]]

        for tok in get_ngrams(t[toktype], ngram):
            if tok in vocab2idx:
                feats[vocab2idx[tok]][ididx] += 1

    idx2vocab = {i: tok for tok, i in vocab2idx.items()}
    featnames = [
        f"_{toktype}_{ngram}gram {idx2vocab[i]}" for i in range(len(idx2vocab))
    ]
    return featnames, scipy.sparse.csr_matrix(feats)


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    ids = [t["id"] for t in tweets]

    feature_names = []
    features = []

    # feature_names.append("bias")
    # features.append(np.ones((len(ids),)))

    feature_names.append("num_natural_disaster")
    features.append(
        get_features_single_layer(
            join(DATASET_SAVE_DIR, "42_natural_disaster_counts.json"), ids
        )
    )
    feature_names.append("num_economy")
    features.append(
        get_features_single_layer(join(DATASET_SAVE_DIR, "43_economy_counts.json"), ids)
    )
    feature_names.append("num_negation")
    features.append(
        get_features_single_layer(
            join(DATASET_SAVE_DIR, "44_negation_counts.json"), ids
        )
    )
    for strength in ["strong", "weak"]:
        feature_names.append(f"num_{strength}_subjectivity")
        features.append(
            get_features_single_layer(
                join(DATASET_SAVE_DIR, f"45_subjectivity_{strength}_subj.json"), ids
            )
        )

    ns, fs = get_features_double_layer(
        "vader_score", join(DATASET_SAVE_DIR, "52_vader.json"), ids
    )
    feature_names.extend(ns)
    features.extend(fs)

    ns, fs = get_features_double_layer(
        "nrc_emolex_score", join(DATASET_SAVE_DIR, "53_nrc_emolex_scores.json"), ids
    )
    feature_names.extend(ns)
    features.extend(fs)

    ns, fs = get_features_double_layer(
        "nrc_vad_score", join(DATASET_SAVE_DIR, "54_nrc_vad_scores.json"), ids
    )
    feature_names.extend(ns)
    features.extend(fs)

    ns, fs = get_features_double_layer(
        "mfd_moral_count", join(DATASET_SAVE_DIR, "61_mfd.json"), ids
    )
    feature_names.extend(ns)
    features.extend(fs)

    ns, fs = build_vocab_feature("stems", 1, 300, tweets, ids)
    feature_names.extend(ns)
    features.extend(fs)

    ns, fs = build_vocab_feature("stems", 2, 300, tweets, ids)
    feature_names.extend(ns)
    features.extend(fs)

    feature_matrix = scipy.sparse.vstack(features).T
    # print(feature_matrix)
    print(feature_matrix.shape)
    # pprint(feature_names)

    write_str_list_as_txt(feature_names, join(DATASET_SAVE_DIR, f"feature_names.txt"))
    write_str_list_as_txt(ids, join(DATASET_SAVE_DIR, f"feature_ids.txt"))

    # sparse_matrix = scipy.sparse.csc_matrix(feature_matrix)
    scipy.sparse.save_npz(join(DATASET_SAVE_DIR, "features.npz"), feature_matrix)
