from collections import defaultdict
from os.path import join

from config import DATA_DIR

from cc_tweets.experiment_config import DATASET_NAME
from cc_tweets.utils import save_json, save_pkl


def get_stats(tweets, name2id2value):
    name2sum = defaultdict(float)
    name2stance2sum = defaultdict(lambda: defaultdict(float))
    name2stance2tweetcount = defaultdict(lambda: defaultdict(float))

    for t in tweets:
        for name, id2value in name2id2value.items():
            value = id2value[t["id"]]
            name2sum[name] += value
            stance = t["stance"]
            if stance in ["rep", "dem"]:
                name2stance2sum[name][stance] += value
                name2stance2tweetcount[name][stance] += 1

    results = {}
    for name in name2id2value:
        results[name] = {}
        results[name]["sum"] = name2sum[name]
        results[name]["mean"] = name2sum[name] / len(tweets)
        results[name]["partisan"] = {}
        results[name]["partisan"]["dem"] = (
            name2stance2sum[name]["dem"] / name2stance2tweetcount[name]["dem"]
        )
        results[name]["partisan"]["rep"] = (
            name2stance2sum[name]["rep"] / name2stance2tweetcount[name]["rep"]
        )
        results[name]["partisan"]["mean"] = (
            results[name]["partisan"]["dem"] + results[name]["partisan"]["rep"]
        ) / 2
    return results


def save_features(tweets, name2id2value, source_name):
    for name, id2value in name2id2value.items():
        save_pkl(id2value, join(DATA_DIR, DATASET_NAME, "features", f"{name}.pkl"))

    stats = get_stats(tweets, name2id2value)
    save_json(
        stats,
        join(DATA_DIR, DATASET_NAME, "feature_stats", f"{source_name}.json"),
    )
