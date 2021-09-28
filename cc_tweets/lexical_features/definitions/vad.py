import re
from collections import Counter, defaultdict
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import (
    load_pkl,
    mkdir_overwrite,
    read_txt_as_str_list,
    save_json,
    unzip,
)
from cc_tweets.viz import grouped_bars
from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

VAD_PATH = join(RESOURCES_DIR, "NRC-VAD-Lexicon-Aug2018Release", "OneFilePerDimension")
VAD_TO_ABBRV = {
    "valence": "v",
    "arousal": "a",
    "dominance": "d",
}
SCORE_THRESHS = {
    "valence": (0.02109090909090909, 0.07905555555555555),
    "arousal": (-0.03983333333333334, 0.002400000000000002),
    "dominance": (0.012875000000000008, 0.05874999999999998),
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
            found = set()
            for lemma in tweet["lemmas"]:
                if lemma in found:
                    continue
                found.add(lemma)
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                id2score[tweet["id"]] += vad2lemma2score[v].get(lemma, 0)
        return id2score

    return _extract


for v in VAD_TO_ABBRV:
    register_feature(Feature(f"vad.{v}", closure(v)))


def bin_closure(v, polarity):
    def _extract(tweets):
        vad2lemma2score = load_vad2lemma2score()
        id2count = defaultdict(int)
        for t in tweets:
            found = set()
            scores = 0
            for lemma in t["lemmas"]:
                if lemma in found:
                    continue
                found.add(lemma)
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                score = vad2lemma2score[v].get(lemma, 0)
                scores += score
                if polarity == "pos" and score > SCORE_THRESHS[v][1]:
                    id2count[t["id"]] += 1
                if polarity == "neg" and score < SCORE_THRESHS[v][0]:
                    id2count[t["id"]] += 1
                if (
                    polarity == "neu"
                    and score <= SCORE_THRESHS[v][1]
                    and score >= SCORE_THRESHS[v][0]
                ):
                    id2count[t["id"]] += 1

                # if polarity == "pos" and scores > SCORE_THRESHS[v][1]:
                #     id2count[t["id"]] = 1
                # if polarity == "neg" and scores < SCORE_THRESHS[v][0]:
                #     id2count[t["id"]] = 1
                # if (
                #     polarity == "neu"
                #     and scores <= SCORE_THRESHS[v][1]
                #     and scores >= SCORE_THRESHS[v][0]
                # ):
                #     id2count[t["id"]] = 1

        return id2count

    return _extract


for v in VAD_TO_ABBRV:
    for polarity in ["pos", "neg", "neu"]:
        register_feature(Feature(f"vad.{v}.{polarity}", bin_closure(v, polarity)))


VAD_TERMS = {
    "valence": {
        "pos": {
            "new",
            "food",
            "natural",
            "family",
            "protection",
            "forward",
            "friend",
            "raise",
            "progress",
            "thanks",
            "thank",
            "happy",
            "love",
            "living",
            "belief",
            "life",
            "protecting",
            "party",
            "growth",
            "proud",
            "woman",
            "education",
            "opportunity",
            "great",
            "rich",
            "nature",
            "strong",
            "free",
            "rest",
            "leadership",
            "agree",
            "priority",
            "clear",
            "good",
            "view",
            "together",
            "kind",
            "powerful",
            "child",
            "generation",
            "future",
            "profit",
            "truth",
            "live",
            "win",
            "health",
            "hope",
            "interesting",
            "home",
            "benefit",
        },
        "neu": {
            "like",
            "fossil",
            "government",
            "planet",
            "earth",
            "impact",
            "go",
            "real",
            "science",
            "get",
            "think",
            "human",
            "stop",
            "take",
            "emission",
            "water",
            "action",
            "time",
            "year",
            "way",
            "must",
            "fuel",
            "plan",
            "today",
            "scientist",
            "see",
            "help",
            "issue",
            "say",
            "report",
            "world",
            "people",
            "one",
            "state",
            "make",
            "right",
            "work",
            "carbon",
            "believe",
            "thing",
            "know",
            "weather",
            "want",
            "day",
            "would",
            "oil",
            "even",
            "policy",
            "need",
            "let",
        },
        "neg": {
            "devastating",
            "worry",
            "ill",
            "war",
            "fear",
            "lie",
            "waste",
            "kill",
            "destruction",
            "destroy",
            "denying",
            "fake",
            "danger",
            "inequality",
            "dead",
            "damage",
            "dying",
            "idiot",
            "blame",
            "disaster",
            "stupid",
            "killing",
            "hoax",
            "lose",
            "bad",
            "hate",
            "fight",
            "dangerous",
            "wrong",
            "scam",
            "worse",
            "ignore",
            "catastrophe",
            "poverty",
            "catastrophic",
            "loss",
            "extinction",
            "protest",
            "ban",
            "disease",
            "ignorant",
            "problem",
            "wildfire",
            "destroying",
            "drought",
            "death",
            "threat",
            "pollution",
            "gun",
            "die",
        },
    },
    "arousal": {
        "pos": {
            "devastating",
            "war",
            "hurricane",
            "storm",
            "violence",
            "quickly",
            "kill",
            "impact",
            "destruction",
            "destructive",
            "attack",
            "destroy",
            "revolution",
            "fighting",
            "excited",
            "danger",
            "conflict",
            "toxic",
            "endangered",
            "accelerate",
            "action",
            "challenge",
            "disaster",
            "killing",
            "bomb",
            "hit",
            "suffer",
            "wild",
            "scary",
            "crime",
            "fight",
            "awesome",
            "dangerous",
            "threaten",
            "nuclear",
            "threatening",
            "abuse",
            "urgent",
            "catastrophe",
            "battle",
            "catastrophic",
            "protest",
            "urgency",
            "threatened",
            "emergency",
            "wildfire",
            "running",
            "threat",
            "gun",
            "alarming",
        },
        "neu": {
            "like",
            "fossil",
            "government",
            "planet",
            "new",
            "earth",
            "go",
            "real",
            "science",
            "get",
            "think",
            "human",
            "stop",
            "take",
            "emission",
            "time",
            "life",
            "year",
            "way",
            "must",
            "fuel",
            "today",
            "great",
            "scientist",
            "see",
            "help",
            "issue",
            "say",
            "report",
            "world",
            "people",
            "one",
            "state",
            "make",
            "right",
            "carbon",
            "good",
            "believe",
            "thing",
            "know",
            "future",
            "weather",
            "want",
            "day",
            "would",
            "oil",
            "even",
            "policy",
            "need",
            "let",
        },
        "neg": {
            "peaceful",
            "sitting",
            "basic",
            "silent",
            "holy",
            "calm",
            "sleeping",
            "yoga",
            "cloud",
            "standard",
            "pacific",
            "natural",
            "chair",
            "library",
            "passive",
            "sleep",
            "desk",
            "tree",
            "librarian",
            "water",
            "simply",
            "vegetable",
            "quietly",
            "slow",
            "cotton",
            "simple",
            "grey",
            "modest",
            "paper",
            "blanket",
            "asleep",
            "grape",
            "simpleton",
            "house",
            "geography",
            "common",
            "book",
            "turtle",
            "seed",
            "canvas",
            "blue",
            "quiet",
            "pasture",
            "dummy",
            "normal",
            "pray",
            "minimal",
            "meadow",
            "orange",
            "straightforward",
        },
    },
    "dominance": {
        "pos": {
            "smart",
            "leading",
            "military",
            "government",
            "increase",
            "corporation",
            "congress",
            "excellent",
            "industrial",
            "progress",
            "admin",
            "chief",
            "effort",
            "positive",
            "federal",
            "worldwide",
            "safety",
            "expert",
            "greatest",
            "director",
            "majority",
            "researcher",
            "proud",
            "rich",
            "minister",
            "strong",
            "force",
            "effective",
            "leadership",
            "politics",
            "activist",
            "strategy",
            "general",
            "priority",
            "political",
            "responsibility",
            "battle",
            "parliament",
            "judge",
            "god",
            "project",
            "powerful",
            "politician",
            "worth",
            "knowledge",
            "win",
            "innovation",
            "wealth",
            "governor",
            "major",
        },
        "neu": {
            "like",
            "fossil",
            "planet",
            "new",
            "earth",
            "impact",
            "real",
            "science",
            "get",
            "think",
            "human",
            "stop",
            "take",
            "emission",
            "water",
            "action",
            "time",
            "year",
            "way",
            "must",
            "fuel",
            "today",
            "fight",
            "scientist",
            "see",
            "help",
            "issue",
            "say",
            "report",
            "world",
            "people",
            "one",
            "state",
            "make",
            "right",
            "carbon",
            "good",
            "believe",
            "thing",
            "know",
            "future",
            "weather",
            "threat",
            "want",
            "day",
            "would",
            "even",
            "policy",
            "need",
            "let",
        },
        "neg": {
            "idiocy",
            "pointless",
            "unsafe",
            "lacking",
            "ineffective",
            "le",
            "lost",
            "mistake",
            "blunder",
            "litter",
            "casualty",
            "weakness",
            "bogus",
            "abandoned",
            "absent",
            "weak",
            "empty",
            "weakened",
            "idiot",
            "lowest",
            "lazy",
            "slow",
            "failure",
            "nonexistent",
            "sad",
            "vague",
            "pity",
            "hopeless",
            "depressed",
            "fool",
            "ash",
            "foolish",
            "coward",
            "low",
            "junk",
            "vacuum",
            "frog",
            "pathetic",
            "shy",
            "void",
            "small",
            "sick",
            "fragile",
            "tiny",
            "meaningless",
            "boring",
            "unstable",
            "defeated",
            "miserably",
            "poor",
        },
    },
}


def term_closure(term):
    lemmatizer = WordNetLemmatizer()
    term_lemma = lemmatizer.lemmatize(term)

    def _extract(tweets):
        id2count = defaultdict(int)
        for t in tweets:
            for lemma in t["lemmas"]:
                if lemma == term_lemma:
                    id2count[t["id"]] += 1
        return id2count

    return _extract


for v in VAD_TO_ABBRV:
    for polarity in ["pos", "neg", "neu"]:
        for term in VAD_TERMS[v][polarity]:
            register_feature(
                Feature(f"vad.term.{v}.{polarity}.{term}", term_closure(term))
            )


# _VAD_BIN_CUTOFFS = {
#     "valence": (-2, 2),
#     "arousal": (-1, 1),
#     "dominance": (-1, 1),
# }


if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    vad2lemma2score = load_vad2lemma2score()
    cat2selectlemmas = {}

    for vad, lemma2score in vad2lemma2score.items():
        highs = []
        lows = []
        mids = []

        for lemma, score in lemma2score.items():
            if score >= 0.35:
                highs.append(lemma)
            elif score <= -0.35:
                lows.append(lemma)
            else:
                mids.append(lemma)
        cat2selectlemmas[f"{vad}.pos"] = highs
        cat2selectlemmas[f"{vad}.neu"] = mids
        cat2selectlemmas[f"{vad}.neg"] = lows

    for filter in ["dem", "rep", "all"]:
        print(filter)
        if filter == "all":
            ts = tweets
        else:
            ts = [t for t in tweets if t["stance"] == filter]
            continue

        result = {}

        for cat, select_lemmas in cat2selectlemmas.items():
            select_lemmas = set(select_lemmas)
            counts = defaultdict(int)
            for t in ts:
                found = set()
                for lemma in t["lemmas"]:
                    if lemma in AFFECT_IGNORE_LEMMAS:
                        continue
                    if lemma not in select_lemmas:
                        continue
                    if lemma in found:
                        continue
                    found.add(lemma)
                    counts[lemma] += 1
            # print(cat)
            count0lemma = sorted(
                [(count, lemma) for lemma, count in counts.items()], reverse=True
            )
            # print(", ".join(unzip(count0lemma)[1][:15]))
            # print(unzip(count0lemma)[1][:30])

            vad, polarity = cat.split(".")
            if vad not in result:
                result[vad] = {}
            result[vad][polarity] = {w for w in unzip(count0lemma)[1][:50]}

        print(result)

    tweets = load_pkl(SUBSET_PKL_PATH)
    vad2lemma2score = load_vad2lemma2score()

    name2id2score = defaultdict(lambda: defaultdict(float))
    # vad2stance2lemma2score = defaultdict(
    #     lambda: defaultdict(lambda: defaultdict(float))
    # )
    # has_vad_ids = set()
    for tweet in tqdm(tweets):
        for vad in VAD_TO_ABBRV:
            name2id2score[vad][tweet["id"]] = 0
            found = set()
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                if lemma in found:
                    continue
                found.add(lemma)
                if lemma in vad2lemma2score[vad]:
                    score = vad2lemma2score[vad][lemma]
                    name2id2score[vad][tweet["id"]] += score
                    # vad2stance2lemma2score[vad][tweet["stance"]][lemma] += score
                    # has_vad_ids.add(tweet["id"])
            if len(found):
                name2id2score[vad][tweet["id"]] /= len(found)

    _VAD_BIN_CUTOFFS = {}
    for vad in VAD_TO_ABBRV:
        scores = sorted(list(name2id2score[vad].values()))
        c1 = scores[int(len(scores) / 3)]
        c2 = scores[int(len(scores) / 3 * 2)]
        _VAD_BIN_CUTOFFS[vad] = (c1, c2)
    print(_VAD_BIN_CUTOFFS)

#     for tweet in tqdm(tweets):
#         for vad, (c1, c2) in _VAD_BIN_CUTOFFS.items():
#             score = name2id2score[vad][tweet["id"]]
#             if score < c1:
#                 name2id2score[f"{vad}_neg"][tweet["id"]] = 1
#             elif score >= c1 and score < c2:
#                 name2id2score[f"{vad}_neu"][tweet["id"]] = 1
#             elif score >= c2:
#                 name2id2score[f"{vad}_pos"][tweet["id"]] = 1
#             else:
#                 raise ValueError()

#     name2id2score = {f"vad_{k}": v for k, v in name2id2score.items()}
#     name2id2score["vad_present"] = {
#         t["id"]: 1 if t["id"] in has_vad_ids else 0 for t in tweets
#     }
#     save_features(tweets, name2id2score, "nrc_vad")

#     # case study, we want high dominance and low arousal
#     mkdir_overwrite(join(SUBSET_WORKING_DIR, "vad_case"))
#     d = vad_top_n_tweets(tweets, name2id2score, "vad_dominance", "max")
#     a = vad_top_n_tweets(tweets, name2id2score, "vad_arousal", "min")
#     ids = list({t["id"] for t in d} | {t["id"] for t in a})
#     df = pd.DataFrame()
#     df["id"] = ids
#     df["dominance"] = [name2id2score["vad_dominance"][id] for id in ids]
#     df["arousal"] = [name2id2score["vad_arousal"][id] for id in ids]
#     df["d-a"] = df["dominance"] - df["arousal"]
#     id2tweet = {t["id"]: t for t in tweets}
#     df["text"] = [id2tweet[id]["text"] for id in ids]
#     df.to_csv(join(SUBSET_WORKING_DIR, "vad_case", "d-a.max.csv"), index=False)

#     d = vad_top_n_tweets(tweets, name2id2score, "vad_dominance", "min")
#     a = vad_top_n_tweets(tweets, name2id2score, "vad_arousal", "max")
#     ids = list({t["id"] for t in d} | {t["id"] for t in a})
#     df = pd.DataFrame()
#     df["id"] = ids
#     df["dominance"] = [name2id2score["vad_dominance"][id] for id in ids]
#     df["arousal"] = [name2id2score["vad_arousal"][id] for id in ids]
#     df["d-a"] = df["dominance"] - df["arousal"]
#     id2tweet = {t["id"]: t for t in tweets}
#     df["text"] = [id2tweet[id]["text"] for id in ids]
#     df.to_csv(join(SUBSET_WORKING_DIR, "vad_case", "d-a.min.csv"), index=False)

#     # save top vad words for each stance
#     vad2stance2toplemma = {}
#     for vad in VAD_TO_ABBRV:
#         stance2lemma2score = vad2stance2lemma2score[vad]
#         stance2toplemma = {}
#         for stance, lemma2score in stance2lemma2score.items():
#             score0lemma = sorted(
#                 [(score, lemma) for lemma, score in lemma2score.items()], reverse=True
#             )
#             top_lemmas = [lemma for _, lemma in score0lemma[:30]]
#             stance2toplemma[stance] = top_lemmas
#         vad2stance2toplemma[vad] = stance2toplemma
#     save_json(
#         vad2stance2toplemma,
#         join(SUBSET_WORKING_DIR, "feature_stats", "nrc_vad_toplemma.json"),
#     )
