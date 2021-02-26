import re
from collections import defaultdict
from os.path import join

from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json

DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")

EMOLEX_PATH = join(
    RESOURCES_DIR, "NRC-Emotion-Intensity-Lexicon-v1", "OneFilePerEmotion"
)
EMOLEX_EMOS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def load_emolex_emo2lemma2score():
    lemmatizer = WordNetLemmatizer()
    emo2lemma2score = defaultdict(dict)
    for emo in EMOLEX_EMOS:
        lines = read_txt_as_str_list(join(EMOLEX_PATH, f"{emo}-scores.txt"))
        for line in lines:
            word, score = line.split("\t")
            lemma = lemmatizer.lemmatize(word)
            score = float(score)
            emo2lemma2score[emo][lemma] = score
    return dict(emo2lemma2score)


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)
    emo2lemma2score = load_emolex_emo2lemma2score()

    id2emo2sumscore = {}
    for tweet in tqdm(tweets):
        id2emo2sumscore[tweet["id"]] = {}
        for emo in EMOLEX_EMOS:
            id2emo2sumscore[tweet["id"]][emo] = 0
            for lemma in tweet["lemmas"]:
                score = emo2lemma2score[emo].get(lemma, 0)
                id2emo2sumscore[tweet["id"]][emo] += score
    save_json(
        id2emo2sumscore,
        join(DATA_DIR, DATASET_NAME, "53_nrc_emolex_scores.json"),
    )

    stats = {}
    for name in EMOLEX_EMOS:
        stats[f"mean_{name}"] = sum(
            scores[name] for scores in id2emo2sumscore.values()
        ) / len(id2emo2sumscore)

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    for lean, tweets in [("dem", dem_tweets), ("rep", rep_tweets)]:
        partisan_stats = {}
        for name in EMOLEX_EMOS:
            partisan_stats[f"mean_{name}"] = sum(
                id2emo2sumscore[t["id"]][name] for t in tweets
            ) / len(tweets)
        stats[lean] = partisan_stats

    for name in EMOLEX_EMOS:
        stats[f"mean_{name}_adjusted_for_dem_rep_imbalance"] = (
            stats["dem"][f"mean_{name}"] + stats["rep"][f"mean_{name}"]
        ) / 2

    save_json(
        stats,
        join(DATA_DIR, DATASET_NAME, "53_nrc_emolex_stats.json"),
    )
