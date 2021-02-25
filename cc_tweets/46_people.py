from os.path import join

from config import DATA_DIR
from nltk.stem.snowball import SnowballStemmer

from cc_tweets.utils import load_pkl, save_json

DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")


PEOPLE = {
    "politics": set(
        [
            "bernie",
            "obama",
            "ocasio-cortez",
            "aoc",
            "inslee",
            "clinton",
            "pruitt",
            "trump",
            "biden",
            "gore",
        ]
    ),
    "activism": set(
        [
            "thunberg",
            "gore",
            "attenborough",
        ]
    ),
    # "science": set(
    #     [
    #         "nye",
    #     ]
    # ),
}

if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)

    # id2numeconomy = {}
    # for tweet in tweets:
    #     count = 0
    #     for stem in tweet["stems"]:
    #         if stem in ECONOMY_WORDS:
    #             count += 1
    #     id2numeconomy[tweet["id"]] = count
    # save_json(
    #     id2numeconomy,
    #     join(DATA_DIR, DATASET_NAME, "43_economy_counts.json"),
    # )

    # stats = {}
    # stats["mean_count"] = sum(id2numeconomy.values()) / len(id2numeconomy)
    # dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    # stats["mean_count_dem"] = sum(id2numeconomy[t["id"]] for t in dem_tweets) / len(
    #     dem_tweets
    # )
    # rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    # stats["mean_count_rep"] = sum(id2numeconomy[t["id"]] for t in rep_tweets) / len(
    #     rep_tweets
    # )
    # stats["mean_count_adjusted_for_dem_rep_imbalance"] = (
    #     stats["mean_count_dem"] + stats["mean_count_rep"]
    # ) / 2
    # save_json(
    #     stats,
    #     join(DATA_DIR, DATASET_NAME, "43_economy_stats.json"),
    # )
