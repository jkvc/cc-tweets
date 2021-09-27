import re
from collections import defaultdict
from os.path import join

import pandas as pd
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, get_feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json, unzip
from cc_tweets.viz import grouped_bars
from config import DATA_DIR, RESOURCES_DIR
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


def closure(vfname):
    def _extract_features(tweets):
        vflemmas = _valencefoundation2lemmas[vfname]
        id2count = defaultdict(int)
        for tweet in tweets:
            found_words = set()
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                if lemma in vflemmas:
                    if lemma not in found_words:
                        id2count[tweet["id"]] += 1
                        found_words.add(lemma)
        return id2count

    return _extract_features


for vfname in _valencefoundation2lemmas:
    register_feature(Feature(f"mfd.{vfname}", closure(vfname)))

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    cat_names = [
        "vice_authority",
        "virtue_authority",
        "vice_fairness",
        "virtue_fairness",
        "vice_harm",
        "virtue_harm",
        "vice_loyalty",
        "virtue_loyalty",
        "vice_purity",
        "virtue_purity",
    ]

    for name in cat_names:
        dem_wcs = defaultdict(int)
        rep_wcs = defaultdict(int)
        cat_lemmas = _valencefoundation2lemmas[name]
        for t in tweets:
            found_words = set()
            for lemma in t["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                if lemma in cat_lemmas:
                    if lemma not in found_words:
                        if t["stance"] == "dem":
                            dem_wcs[lemma] += 1
                        elif t["stance"] == "rep":
                            rep_wcs[lemma] += 1
                        found_words.add(lemma)
        print("-" * 69)
        print(name)
        print("dem")
        print(
            ",".join(
                unzip(
                    sorted(
                        [(count, word) for word, count in dem_wcs.items()], reverse=True
                    )
                )[1][:5]
            )
        )
        print("rep")
        print(
            ",".join(
                unzip(
                    sorted(
                        [(count, word) for word, count in rep_wcs.items()], reverse=True
                    )
                )[1][:5]
            )
        )
