from collections import defaultdict

from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.experiment_config import SUBSET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.utils import load_pkl

PEOPLE = [
    "bernie",
    "obama",
    "ocasio-cortez",
    "aoc",
    "inslee",
    "clinton",
    "pruitt",
    "trump",
    "biden",
    "thunberg",
    "gore",
    "attenborough",
    "nye",
]
lemmatizer = WordNetLemmatizer()

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    name2namelemma = {name: lemmatizer.lemmatize(name) for name in PEOPLE}

    name2id2indicator = defaultdict(lambda: defaultdict(int))
    for tweet in tqdm(tweets):
        for name, namelemma in name2namelemma.items():
            for lemma in tweet["lemmas"]:
                if lemma == namelemma:
                    name2id2indicator[f"p_{name}"][tweet["id"]] = 1

    save_features(tweets, name2id2indicator, "people")
