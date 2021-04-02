import re
from collections import Counter, defaultdict
from os.path import join

import matplotlib.pyplot as plt
from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.feature_utils import save_features
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

    name2id2score = defaultdict(lambda: defaultdict(float))
    vad2stance2lemma2score = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    for tweet in tqdm(tweets):
        for vad in VAD_TO_ABBRV:
            name2id2score[vad][tweet["id"]] = 0
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                score = vad2lemma2score[vad].get(lemma, 0)
                name2id2score[vad][tweet["id"]] += score
                vad2stance2lemma2score[vad][tweet["stance"]][lemma] += score

    name2id2score = {f"vad_{k}": v for k, v in name2id2score.items()}

    save_features(tweets, name2id2score, "nrc_vad")

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
        join(DATASET_SAVE_DIR, "feature_stats", "nrc_vad_toplemma.json"),
    )
