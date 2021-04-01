from collections import defaultdict
from os.path import join

from config import DATA_DIR
from nltk.stem.snowball import SnowballStemmer

from cc_tweets.experiment_config import DATASET_PKL_PATH
from cc_tweets.utils import load_pkl, save_json

PEOPLE = {
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
}
stemmer = SnowballStemmer("english")
PEOPLE = set(stemmer.stem(w) for w in PEOPLE)

if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    id2numeconomy = {}
    name2id2value = defaultdict(lambda: defaultdict(int))
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in ECONOMY_WORDS:
                count += 1
                name2id2value[stem][tweet["id"]] += 1

        id2numeconomy[tweet["id"]] = count

    save_features(tweets, {"economy": id2numeconomy}, "economy")
    visualize_features(name2id2value, tweets, "economy")
