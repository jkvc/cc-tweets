import re
from collections import defaultdict
from os.path import join

import matplotlib.pyplot as plt
import pandas as pd
from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from experiment_configs.base import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import save_features
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json
from cc_tweets.viz import grouped_bars

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
    tweets = load_pkl(SUBSET_PKL_PATH)
    valencefoundation2lemmas = load_mfd()

    name2id2count = defaultdict(lambda: defaultdict(float))
    for tweet in tqdm(tweets):
        for vf, vf_lemmas in valencefoundation2lemmas.items():
            name2id2count[vf][tweet["id"]] = 0
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                if lemma in vf_lemmas:
                    name2id2count[vf][tweet["id"]] += 1

    name2id2count = {f"mfd_{k}": v for k, v in name2id2count.items()}
    save_features(tweets, name2id2count, "mfd")
