from collections import Counter, defaultdict
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
from cc_tweets.polarization.textual import calc_dem_rep_polarization
from cc_tweets.utils import load_json, load_pkl, read_txt_as_str_list, save_json
from cc_tweets.viz import plot_horizontal_bars
from config import DATA_DIR
from experiment_configs.base import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from nltk.stem.snowball import SnowballStemmer

NUM_TRIALS = 10
VOCAB_FILE = join(SUBSET_WORKING_DIR, "vocab", "stems_2gram_4000.txt")

NUM_CLUSTERS = 10
SAVE_JSON_PATH = join(
    SUBSET_WORKING_DIR,
    "clustering",
    f"{NUM_CLUSTERS}clusters",
    "intra_topic_pol.json",
)
SAVE_PNG_PATH = join(
    SUBSET_WORKING_DIR,
    "clustering",
    f"{NUM_CLUSTERS}clusters",
    "intra_topic_pol.png",
)

if __name__ == "__main__":
    cluster_names = read_txt_as_str_list(
        join(
            SUBSET_WORKING_DIR,
            "clustering",
            f"{NUM_CLUSTERS}clusters",
            "cluster_names.txt",
        )
    )
    vocab2idx = {gram: i for i, gram in enumerate(read_txt_as_str_list(VOCAB_FILE))}

    if not exists(SAVE_JSON_PATH):
        tweets = load_pkl(SUBSET_PKL_PATH)
        assignments = load_pkl(
            join(
                SUBSET_WORKING_DIR,
                "clustering",
                f"{NUM_CLUSTERS}clusters",
                "cluster_assignments.pkl",
            )
        )

        clusteridx2tweets = [[] for _ in range(NUM_CLUSTERS)]
        for tweet, clusteridx in zip(tweets, assignments):
            clusteridx2tweets[clusteridx].append(tweet)

        clusteridx2pol = []
        for tweets in clusteridx2tweets:
            clusteridx2pol.append(
                calc_dem_rep_polarization(tweets, vocab2idx, NUM_TRIALS)
            )

        save_json(clusteridx2pol, SAVE_JSON_PATH)
    else:
        clusteridx2pol = load_json(SAVE_JSON_PATH)

    print(clusteridx2pol)
    clustername2pol = {
        f"{i}: {cluster_names[i]}": clusteridx2pol[i]
        for i in reversed(range(NUM_CLUSTERS - 1))
    }
    plot_horizontal_bars(
        clustername2pol,
        SAVE_PNG_PATH,
        "intra topic leave-out estimator",
        xlim=(min(clustername2pol.values()) - 0.03, max(clustername2pol.values())),
        figsize=(7, 4),
    )
