import re
from os.path import join

from config import DATA_DIR

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.utils import load_pkl, save_json

NEGATION_REGEX = "not|n't|never|nor|no|nobody|nowhere|nothing|noone"
if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    id2numnegation = {}
    for tweet in tweets:
        id2numnegation[tweet["id"]] = len(
            list(re.finditer(NEGATION_REGEX, tweet["text"], re.IGNORECASE))
        )
    save_json(
        id2numnegation,
        join(DATASET_SAVE_DIR, "44_negation_counts.json"),
    )

    stats = {}
    stats["mean_count"] = sum(id2numnegation.values()) / len(id2numnegation)
    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    stats["mean_count_dem"] = sum(id2numnegation[t["id"]] for t in dem_tweets) / len(
        dem_tweets
    )
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    stats["mean_count_rep"] = sum(id2numnegation[t["id"]] for t in rep_tweets) / len(
        rep_tweets
    )
    stats["mean_count_adjusted_for_dem_rep_imbalance"] = (
        stats["mean_count_dem"] + stats["mean_count_rep"]
    ) / 2
    save_json(
        stats,
        join(DATASET_SAVE_DIR, "44_negation_stats.json"),
    )
