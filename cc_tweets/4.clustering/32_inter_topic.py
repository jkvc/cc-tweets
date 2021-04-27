from collections import Counter, defaultdict
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
from config import DATA_DIR
from nltk.stem.snowball import SnowballStemmer

from cc_tweets.experiment_config import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.utils import load_json, load_pkl, read_txt_as_str_list, save_json

NUM_CLUSTERS = 10
SAVE_JSON_PATH = join(
    SUBSET_WORKING_DIR,
    "sif",
    f"{NUM_CLUSTERS}clusters",
    "in_cluster_stances.json",
)
SAVE_PNG_PATH = join(
    SUBSET_WORKING_DIR,
    "sif",
    f"{NUM_CLUSTERS}clusters",
    "in_cluster_stances.png",
)

if __name__ == "__main__":
    cluster_names = read_txt_as_str_list(
        join(
            SUBSET_WORKING_DIR,
            "sif",
            f"{NUM_CLUSTERS}clusters",
            "cluster_names.txt",
        )
    )

    if not exists(SAVE_JSON_PATH):
        tweets = load_pkl(SUBSET_PKL_PATH)
        assignments = load_pkl(
            join(
                SUBSET_WORKING_DIR,
                "sif",
                f"{NUM_CLUSTERS}clusters",
                "cluster_assignments.pkl",
            )
        )
        clusteridx2stance2count = [defaultdict(int) for _ in range(NUM_CLUSTERS)]
        for tweet, clusteridx in zip(tweets, assignments):
            clusteridx2stance2count[clusteridx][tweet["stance"]] += 1
        stance2count = Counter([t["stance"] for t in tweets])

        clusteridx2info = {}
        for i in range(NUM_CLUSTERS):
            clusteridx2info[i] = {
                "name": cluster_names[i],
                "counts": clusteridx2stance2count[i],
                "props": {
                    "dem": clusteridx2stance2count[i]["dem"] / stance2count["dem"],
                    "rep": clusteridx2stance2count[i]["rep"] / stance2count["rep"],
                },
            }

        save_json(clusteridx2info, SAVE_JSON_PATH)
    else:
        clusteridx2info = load_json(SAVE_JSON_PATH)

    figs, axs = plt.subplots(
        nrows=2, ncols=NUM_CLUSTERS // 2, figsize=(NUM_CLUSTERS * 2, 8)
    )
    axs = axs.flatten()
    for i in range(NUM_CLUSTERS):
        ax = axs[i]
        labels = ["dem", "rep"]
        props = np.array(
            [clusteridx2info[str(i)]["props"][stance] for stance in labels]
        )
        ax.pie(props, labels=labels, autopct="%1.1f%%", normalize=True)
        ax.axis("equal")
        ax.set_title(f"{i}: {cluster_names[i]}")
    plt.savefig(SAVE_PNG_PATH)
    plt.clf()
