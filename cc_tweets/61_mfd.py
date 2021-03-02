import re
from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json
from cc_tweets.viz import grouped_bars

DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")

MFD_PATH = join(RESOURCES_DIR, "MFD", "MFD2.0.csv")


def load_mfd():
    lemmatizer = WordNetLemmatizer()

    df = pd.read_csv(MFD_PATH)
    valencefoundation2lemmas = {}
    for i, row in df.iterrows():
        valence = row["valence"]
        foundation = row["foundation"]
        vf = f"{valence}_{foundation}"
        word = row["word"]
        lemma = lemmatizer.lemmatize(word)
        if vf not in valencefoundation2lemmas:
            valencefoundation2lemmas[vf] = set()
        valencefoundation2lemmas[vf].add(lemma)
    return valencefoundation2lemmas


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)
    valencefoundation2lemmas = load_mfd()

    id2vf2count = {}
    for tweet in tqdm(tweets):
        id2vf2count[tweet["id"]] = {vf: 0 for vf in valencefoundation2lemmas.keys()}
        for lemma in tweet["lemmas"]:
            if lemma in AFFECT_IGNORE_LEMMAS:
                continue
            for vf, vf_lemmas in valencefoundation2lemmas.items():
                if lemma in vf_lemmas:
                    id2vf2count[tweet["id"]][vf] += 1
    id2vf2count = dict(id2vf2count)

    save_json(
        id2vf2count,
        join(DATA_DIR, DATASET_NAME, "61_mfd.json"),
    )

    stats = {}
    for vf in valencefoundation2lemmas.keys():
        stats[f"mean_{vf}"] = sum(count[vf] for count in id2vf2count.values()) / len(
            id2vf2count
        )

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    for lean, tweets in [("dem", dem_tweets), ("rep", rep_tweets)]:
        partisan_stats = {}
        for vf in valencefoundation2lemmas.keys():
            partisan_stats[f"mean_{vf}"] = sum(
                id2vf2count[t["id"]][vf] for t in tweets
            ) / len(tweets)
        stats[lean] = partisan_stats

    for vf in valencefoundation2lemmas.keys():
        stats[f"mean_{vf}_adjusted_for_dem_rep_imbalance"] = (
            stats["dem"][f"mean_{vf}"] + stats["rep"][f"mean_{vf}"]
        ) / 2

    save_json(
        stats,
        join(DATA_DIR, DATASET_NAME, "61_mfd_stats.json"),
    )

    vfs = list(valencefoundation2lemmas.keys())
    lean2vfs = {}
    for lean in ["dem", "rep"]:
        lean2vfs[lean] = [round(stats[lean][f"mean_{vf}"], 3) for vf in vfs]
    fig, ax = plt.subplots(figsize=(15, 5))
    fig, ax = grouped_bars(fig, ax, vfs, lean2vfs)
    ax.set_ylabel("mean count per tweet")
    plt.title("valence_foundation v lean")
    plt.savefig(
        join(DATA_DIR, DATASET_NAME, "61_mfd_stats.png"),
    )
