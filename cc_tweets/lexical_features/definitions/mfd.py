import re
from collections import defaultdict
from os.path import join

import pandas as pd
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json
from cc_tweets.viz import grouped_bars
from config import DATA_DIR, RESOURCES_DIR
from experiment_configs.base import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

MFD_PATH = join(RESOURCES_DIR, "MFD", "MFD2.0.csv")


lemmatizer = WordNetLemmatizer()
df = pd.read_csv(MFD_PATH)
_valencefoundation2lemmas = {}
for i, row in df.iterrows():
    valence = row["valence"]
    foundation = row["foundation"]
    vf = f"{valence}_{foundation}"
    word = row["word"]
    lemma = lemmatizer.lemmatize(word)
    if vf not in _valencefoundation2lemmas:
        _valencefoundation2lemmas[vf] = set()
    _valencefoundation2lemmas[vf].add(lemma)


for vfname, vflemmas in _valencefoundation2lemmas.items():

    def _extract_features(tweets):
        id2count = defaultdict(int)
        for tweet in tweets:
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                if lemma in vflemmas:
                    id2count[tweet["id"]] += 1
        return id2count

    register_feature(Feature(f"mfd.{vfname}", _extract_features))
