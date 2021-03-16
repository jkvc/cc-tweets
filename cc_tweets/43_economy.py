from nltk.stem.snowball import SnowballStemmer

from cc_tweets.experiment_config import DATASET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.utils import load_pkl

ECONOMY_WORDS = set(
    [
        "capitalism",
        "finance",
        "labor",
        "economy",
        "stock",
        "workforce",
        "market",
        "cost",
        "trade",
        "capital",
        "tax",
        "job",
        "growth",
        "money",
        "econ",
        "wealth",
        "bank",
        "financial",
        "utility",
        "investment",
        "tariff",
        "economist",
        "economics",
        "profit",
    ]
)
stemmer = SnowballStemmer("english")
ECONOMY_WORDS = set(stemmer.stem(w) for w in ECONOMY_WORDS)

if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    id2numeconomy = {}
    for tweet in tweets:
        count = 0
        for stem in tweet["stems"]:
            if stem in ECONOMY_WORDS:
                count += 1
        id2numeconomy[tweet["id"]] = count

    save_features(tweets, {"economy": id2numeconomy}, "economy")
