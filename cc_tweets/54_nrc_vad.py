import re
from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json
from cc_tweets.viz import grouped_bars

VAD_PATH = join(RESOURCES_DIR, "NRC-VAD-Lexicon-Aug2018Release", "OneFilePerDimension")
VAD_TO_ABBRV = {
    "valence": "v",
    "arousal": "a",
    "dominance": "d",
}


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
    tweets = load_pkl(DATASET_PKL_PATH)
    vad2lemma2score = load_vad2lemma2score()

    id2vad2sumscore = {}
    for tweet in tqdm(tweets):
        id2vad2sumscore[tweet["id"]] = {}
        for vad in VAD_TO_ABBRV:
            id2vad2sumscore[tweet["id"]][vad] = 0
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                score = vad2lemma2score[vad].get(lemma, 0)
                id2vad2sumscore[tweet["id"]][vad] += score
    save_json(
        id2vad2sumscore,
        join(DATASET_SAVE_DIR, "54_nrc_vad_scores.json"),
    )

    stats = {}
    for name in VAD_TO_ABBRV:
        stats[f"mean_{name}"] = sum(
            scores[name] for scores in id2vad2sumscore.values()
        ) / len(id2vad2sumscore)

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    for lean, tweets in [("dem", dem_tweets), ("rep", rep_tweets)]:
        partisan_stats = {}
        for name in VAD_TO_ABBRV:
            partisan_stats[f"mean_{name}"] = sum(
                id2vad2sumscore[t["id"]][name] for t in tweets
            ) / len(tweets)
        stats[lean] = partisan_stats

    for name in VAD_TO_ABBRV:
        stats[f"mean_{name}_adjusted_for_dem_rep_imbalance"] = (
            stats["dem"][f"mean_{name}"] + stats["rep"][f"mean_{name}"]
        ) / 2

    save_json(
        stats,
        join(DATASET_SAVE_DIR, "54_nrc_vad_stats.json"),
    )

    vads = list(VAD_TO_ABBRV.keys())
    lean2series = {}
    for lean in ["dem", "rep"]:
        lean2series[lean] = [round(stats[lean][f"mean_{vad}"], 3) for vad in vads]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig, ax = grouped_bars(fig, ax, vads, lean2series)
    ax.set_ylabel("mean sum scores per tweet")
    plt.title("NRC VAD sum scores v lean")
    plt.savefig(
        join(DATASET_SAVE_DIR, "54_nrc_vad_stats.png"),
    )
