import re
from collections import Counter, defaultdict
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, mkdir_overwrite, read_txt_as_str_list, save_json
from cc_tweets.viz import grouped_bars
from config import DATA_DIR, RESOURCES_DIR
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

VAD_PATH = join(RESOURCES_DIR, "NRC-VAD-Lexicon-Aug2018Release", "OneFilePerDimension")
VAD_TO_ABBRV = {
    "valence": "v",
    "arousal": "a",
    "dominance": "d",
}

# todo vad bins
# _VAD_BIN_CUTOFFS = {}
# for vad in VAD_TO_ABBRV:
#     scores = sorted(list(name2id2score[vad].values()))
#     c1 = scores[int(len(scores) / 3)]
#     c2 = scores[int(len(scores) / 3 * 2)]
#     _VAD_BIN_CUTOFFS[vad] = (c1, c2)
# print(_VAD_BIN_CUTOFFS)

# for tweet in tqdm(tweets):
#     for vad, (c1, c2) in _VAD_BIN_CUTOFFS.items():
#         score = name2id2score[vad][tweet["id"]]
#         if score < c1:
#             name2id2score[f"{vad}_neg"][tweet["id"]] = 1
#         elif score >= c1 and score < c2:
#             name2id2score[f"{vad}_neu"][tweet["id"]] = 1
#         elif score >= c2:
#             name2id2score[f"{vad}_pos"][tweet["id"]] = 1
#         else:
#             raise ValueError()


def load_vad2lemma2score():
    lemmatizer = WordNetLemmatizer()
    vad2lemma2score = defaultdict(dict)
    for vad, abbrv in VAD_TO_ABBRV.items():
        lines = read_txt_as_str_list(join(VAD_PATH, f"{abbrv}-scores.txt"))
        for line in lines:
            word, score = line.split("\t")
            lemma = lemmatizer.lemmatize(word)
            score = float(score)
            vad2lemma2score[vad][lemma] = score - 0.5  # map [0,1] to [-.5, 5]
    return dict(vad2lemma2score)


def vad_top_n_tweets(tweets, name2id2score, vad, max_or_min, n=3000):
    score0ids = sorted(
        [(score, id) for id, score in name2id2score[vad].items()],
        reverse=max_or_min == "max",
    )
    score0ids = score0ids[:n]
    ids = [id for score, id in score0ids[:n]]
    scores = [score for score, id in score0ids[:n]]
    id2tweets = {t["id"]: t for t in tweets}
    tweets = [id2tweets[id] for id in ids]
    return tweets


def closure(v):
    def _extract(tweets):
        vad2lemma2score = load_vad2lemma2score()
        id2score = defaultdict(float)
        for tweet in tweets:
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                id2score[tweet["id"]] += vad2lemma2score[v].get(lemma, 0)
        return id2score

    return _extract


for v in VAD_TO_ABBRV:
    register_feature(Feature(f"vad.{v}", closure(v)))


# _VAD_BIN_CUTOFFS = {
#     "valence": (-2, 2),
#     "arousal": (-1, 1),
#     "dominance": (-1, 1),
# }

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    vad2lemma2score = load_vad2lemma2score()

    name2id2score = defaultdict(lambda: defaultdict(float))
    vad2stance2lemma2score = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    has_vad_ids = set()
    for tweet in tqdm(tweets):
        for vad in VAD_TO_ABBRV:
            name2id2score[vad][tweet["id"]] = 0
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue

                if lemma in vad2lemma2score[vad]:
                    score = vad2lemma2score[vad][lemma]
                    name2id2score[vad][tweet["id"]] += score
                    vad2stance2lemma2score[vad][tweet["stance"]][lemma] += score
                    has_vad_ids.add(tweet["id"])

    _VAD_BIN_CUTOFFS = {}
    for vad in VAD_TO_ABBRV:
        scores = sorted(list(name2id2score[vad].values()))
        c1 = scores[int(len(scores) / 3)]
        c2 = scores[int(len(scores) / 3 * 2)]
        _VAD_BIN_CUTOFFS[vad] = (c1, c2)
    print(_VAD_BIN_CUTOFFS)

    for tweet in tqdm(tweets):
        for vad, (c1, c2) in _VAD_BIN_CUTOFFS.items():
            score = name2id2score[vad][tweet["id"]]
            if score < c1:
                name2id2score[f"{vad}_neg"][tweet["id"]] = 1
            elif score >= c1 and score < c2:
                name2id2score[f"{vad}_neu"][tweet["id"]] = 1
            elif score >= c2:
                name2id2score[f"{vad}_pos"][tweet["id"]] = 1
            else:
                raise ValueError()

    name2id2score = {f"vad_{k}": v for k, v in name2id2score.items()}
    name2id2score["vad_present"] = {
        t["id"]: 1 if t["id"] in has_vad_ids else 0 for t in tweets
    }
    save_features(tweets, name2id2score, "nrc_vad")

    # case study, we want high dominance and low arousal
    mkdir_overwrite(join(SUBSET_WORKING_DIR, "vad_case"))
    d = vad_top_n_tweets(tweets, name2id2score, "vad_dominance", "max")
    a = vad_top_n_tweets(tweets, name2id2score, "vad_arousal", "min")
    ids = list({t["id"] for t in d} | {t["id"] for t in a})
    df = pd.DataFrame()
    df["id"] = ids
    df["dominance"] = [name2id2score["vad_dominance"][id] for id in ids]
    df["arousal"] = [name2id2score["vad_arousal"][id] for id in ids]
    df["d-a"] = df["dominance"] - df["arousal"]
    id2tweet = {t["id"]: t for t in tweets}
    df["text"] = [id2tweet[id]["text"] for id in ids]
    df.to_csv(join(SUBSET_WORKING_DIR, "vad_case", "d-a.max.csv"), index=False)

    d = vad_top_n_tweets(tweets, name2id2score, "vad_dominance", "min")
    a = vad_top_n_tweets(tweets, name2id2score, "vad_arousal", "max")
    ids = list({t["id"] for t in d} | {t["id"] for t in a})
    df = pd.DataFrame()
    df["id"] = ids
    df["dominance"] = [name2id2score["vad_dominance"][id] for id in ids]
    df["arousal"] = [name2id2score["vad_arousal"][id] for id in ids]
    df["d-a"] = df["dominance"] - df["arousal"]
    id2tweet = {t["id"]: t for t in tweets}
    df["text"] = [id2tweet[id]["text"] for id in ids]
    df.to_csv(join(SUBSET_WORKING_DIR, "vad_case", "d-a.min.csv"), index=False)

    # save top vad words for each stance
    vad2stance2toplemma = {}
    for vad in VAD_TO_ABBRV:
        stance2lemma2score = vad2stance2lemma2score[vad]
        stance2toplemma = {}
        for stance, lemma2score in stance2lemma2score.items():
            score0lemma = sorted(
                [(score, lemma) for lemma, score in lemma2score.items()], reverse=True
            )
            top_lemmas = [lemma for _, lemma in score0lemma[:30]]
            stance2toplemma[stance] = top_lemmas
        vad2stance2toplemma[vad] = stance2toplemma
    save_json(
        vad2stance2toplemma,
        join(SUBSET_WORKING_DIR, "feature_stats", "nrc_vad_toplemma.json"),
    )
