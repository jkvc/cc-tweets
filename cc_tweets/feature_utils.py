from collections import Counter, defaultdict
from os import makedirs
from os.path import join

from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_NAME
from cc_tweets.utils import save_json, save_pkl


def get_stats(tweets, name2id2value):
    name2sum = defaultdict(float)
    name2stance2sum = defaultdict(lambda: defaultdict(float))
    id2stance = {t["id"]: t["stance"] for t in tweets}
    stance2count = Counter([t["stance"] for t in tweets])

    for name, id2value in name2id2value.items():
        for id, value in id2value.items():
            name2sum[name] += value
            stance = id2stance[id]
            if stance in ["rep", "dem"]:
                name2stance2sum[name][stance] += value

    results = {}
    for name in name2id2value:
        results[name] = {}
        results[name]["sum"] = name2sum[name]
        results[name]["mean"] = name2sum[name] / len(tweets)
        results[name]["partisan"] = {}
        results[name]["partisan"]["dem"] = (
            name2stance2sum[name]["dem"] / stance2count["dem"]
        )
        results[name]["partisan"]["rep"] = (
            name2stance2sum[name]["rep"] / stance2count["rep"]
        )
        results[name]["partisan"]["mean"] = (
            results[name]["partisan"]["dem"] + results[name]["partisan"]["rep"]
        ) / 2
    return results


def save_features(
    tweets,
    name2id2value,
    source_name,
    save_features_dir=join(DATA_DIR, DATASET_NAME, "features"),
    save_feature_stats_dir=join(DATA_DIR, DATASET_NAME, "feature_stats"),
):
    makedirs(save_features_dir, exist_ok=True)
    makedirs(save_feature_stats_dir, exist_ok=True)
    for name, id2value in name2id2value.items():
        save_pkl(id2value, join(save_features_dir, f"{name}.pkl"))

    stats = get_stats(tweets, name2id2value)
    save_json(
        stats,
        join(save_feature_stats_dir, f"{source_name}.json"),
    )
