import re
from collections import defaultdict
from os.path import join

from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json

DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")

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
    tweets = load_pkl(PKL_PATH)
    vad2lemma2score = load_vad2lemma2score()

    id2vad2sumscore = {}
    for tweet in tqdm(tweets):
        id2vad2sumscore[tweet["id"]] = {}
        for vad in VAD_TO_ABBRV:
            id2vad2sumscore[tweet["id"]][vad] = 0
            for lemma in tweet["lemmas"]:
                score = vad2lemma2score[vad].get(lemma, 0)
                id2vad2sumscore[tweet["id"]][vad] += score
    save_json(
        id2vad2sumscore,
        join(DATA_DIR, DATASET_NAME, "54_nrc_vad_scores.json"),
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
        join(DATA_DIR, DATASET_NAME, "54_nrc_vad_stats.json"),
    )
