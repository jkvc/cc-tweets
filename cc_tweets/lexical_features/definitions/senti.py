import random
from collections import defaultdict
from os import makedirs
from os.path import join
from posixpath import dirname

import pandas as pd
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.log_odds import get_topn_lors
from cc_tweets.utils import ParallelHandler, load_pkl, save_pkl
from config import RESOURCES_DIR
from genericpath import exists
from sentistrength import PySentiStr
from tqdm import tqdm

BINS = ["LAP", "HAP", "LAN", "HAN", "NEU"]


def get_bin_name(pos_score, neg_score):
    if pos_score == 2:
        return "LAP"
    if pos_score >= 3:
        return "HAP"
    if neg_score == -2:
        return "LAN"
    if neg_score <= -3:
        return "HAN"
    if pos_score == 1 and neg_score == -1:
        return "NEU"


def is_in_bin(bin_name, pos_score, neg_score):
    # Low Arousal Positive: posts with positive 2
    # High Arousal Positive: posts with positive intensity 3, 4, or 5
    # Low Arousal Negative: posts with negative 2
    # High Arousal Negative: posts with negative intensity 3, 4, or 5
    # Neutral: posts with positive 1 AND negative 1

    if bin_name == "LAP":
        return pos_score == 2
    if bin_name == "HAP":
        return pos_score >= 3
    if bin_name == "LAN":
        return neg_score == -2
    if bin_name == "HAN":
        return neg_score <= -3
    if bin_name == "NEU":
        return pos_score == 1 and neg_score == -1
    raise ValueError()


def closure(binname):
    def _extract(tweets):
        senti = PySentiStr()
        senti.setSentiStrengthPath(
            join(RESOURCES_DIR, "sentistrength", "SentiStrength.jar")
        )
        senti.setSentiStrengthLanguageFolderPath(
            join(RESOURCES_DIR, "sentistrength", "data")
        )
        tweet_texts = [t["text"] for t in tweets]

        scores = senti.getSentiment(tweet_texts, score="dual")

        id2val = defaultdict(int)
        for tweet, score in zip(tweets, scores):
            pos_score, neg_score = score
            val = 1 if is_in_bin(binname, pos_score, neg_score) else 0
            id2val[tweet["id"]] = val
        return id2val

    return _extract


for binname in BINS:
    register_feature(Feature(f"senti.{binname}", closure(binname)))


if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    senti = PySentiStr()
    senti.setSentiStrengthPath(
        join(RESOURCES_DIR, "sentistrength", "SentiStrength.jar")
    )
    senti.setSentiStrengthLanguageFolderPath(
        join(RESOURCES_DIR, "sentistrength", "data")
    )

    tweet_texts = [t["text"] for t in tweets]
    scores = senti.getSentiment(tweet_texts, score="dual")

    cache_path = join(SUBSET_WORKING_DIR, "senti_out", "bin2tweets.pkl")
    makedirs(dirname(cache_path), exist_ok=True)
    if not exists(cache_path):
        bin2tweets = defaultdict(list)
        for tweet, score in zip(tweets, scores):
            binname = get_bin_name(*score)
            bin2tweets[binname].append(tweet)
        save_pkl(bin2tweets, cache_path)
    else:
        bin2tweets = load_pkl(cache_path)

    for name, (leftbins, rightbins) in [
        ("han_v_rest", (["HAN"], ["LAN", "LAP", "HAP", "NEU"])),
        ("han_v_lan", (["HAN"], ["LAN"])),
        ("pos_v_neg", (["LAP", "HAP"], ["LAN", "HAN"])),
        ("neu_v_rest", (["NEU"], ["HAN", "LAN", "LAP", "HAP"])),
    ]:
        left = []
        right = []
        for bin in leftbins:
            left.extend(bin2tweets[bin])
        for bin in rightbins:
            right.extend(bin2tweets[bin])

        for toktype, ngrams in [
            ("lemmas", 1),
            ("stems", 2),
        ]:
            print(toktype, ngrams)
            left_topwords, right_topwords = get_topn_lors(left, right, toktype, ngrams)
            df = pd.DataFrame()
            df["left"] = left_topwords
            df["right"] = right_topwords
            df.to_csv(
                join(SUBSET_WORKING_DIR, "senti_out", f"{name}.{ngrams}{toktype}.csv")
            )
