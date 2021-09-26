from collections import Counter, defaultdict
from os import makedirs
from os.path import join

import numpy as np
import scipy
from cc_tweets.experiment_configs import SUBSET_NAME, SUBSET_WORKING_DIR
from tqdm import tqdm

from cc_tweets.utils import load_json, save_json, save_pkl
from cc_tweets.viz import plot_grouped_bars, plot_horizontal_bars


def get_follower_features(tweets, source="num_follower"):
    assert source in ["num_follower", "max_num_follower"]
    followers = np.array([t[source] for t in tweets])
    return scipy.sparse.csr_matrix(followers)


def get_log_follower_features(tweets, source="num_follower"):
    assert source in ["num_follower", "max_num_follower"]
    followers = np.array([t[source] for t in tweets])
    log_followers = np.log(followers + 1)
    return scipy.sparse.csr_matrix(log_followers)


def get_retweets(tweets):
    retweets = np.array([t["retweets"] for t in tweets])
    return scipy.sparse.csr_matrix(retweets)


def get_log_retweets(tweets):
    retweets = np.array([t["retweets"] for t in tweets])
    log_retweets = np.log(retweets + 1)
    return scipy.sparse.csr_matrix(log_retweets)


def get_stats(tweets, name2id2value):
    name2sum = defaultdict(float)
    name2max = defaultdict(float)
    name2min = defaultdict(float)
    name2stance2sum = defaultdict(lambda: defaultdict(float))
    id2stance = {t["id"]: t["stance"] for t in tweets}
    stance2count = Counter([t["stance"] for t in tweets])

    for name, id2value in name2id2value.items():
        for id, value in id2value.items():
            name2sum[name] += value
            name2max[name] = max(name2max[name], value)
            name2min[name] = min(name2min[name], value)
            stance = id2stance[id]
            if stance in ["rep", "dem"]:
                name2stance2sum[name][stance] += value

    results = {}
    for name in name2id2value:
        results[name] = {}
        results[name]["sum"] = name2sum[name]
        results[name]["max"] = name2max[name]
        results[name]["min"] = name2min[name]
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


def get_single_stat(tweets, id2value):
    return get_stats(tweets, {"a": id2value})["a"]


def save_features(
    tweets,
    name2id2value,
    source_name,
    save_features_dir=join(SUBSET_WORKING_DIR, "features"),
    save_feature_stats_dir=join(SUBSET_WORKING_DIR, "feature_stats"),
):
    makedirs(save_features_dir, exist_ok=True)
    for name, id2value in name2id2value.items():
        save_pkl(id2value, join(save_features_dir, f"{name}.pkl"))
    save_stats(tweets, name2id2value, source_name, save_feature_stats_dir)


def save_stats(
    tweets,
    name2id2value,
    source_name,
    save_feature_stats_dir=join(SUBSET_WORKING_DIR, "feature_stats"),
):
    makedirs(save_feature_stats_dir, exist_ok=True)
    stats = get_stats(tweets, name2id2value)
    save_json(
        stats,
        join(save_feature_stats_dir, f"{source_name}.json"),
    )

    # lean
    x_labels = list(stats.keys())
    name2series = {
        lean: [stats[sname]["partisan"][lean] for sname in x_labels]
        for lean in ["dem", "rep"]
    }
    plot_grouped_bars(
        x_labels,
        name2series,
        source_name,
        join(save_feature_stats_dir, f"{source_name}.png"),
    )

    # # partisanship
    # name2partisanship = {
    #     sname: stats[sname]["partisan"]["dem"] - stats[sname]["partisan"]["rep"]
    #     for sname in x_labels
    # }
    # plot_horizontal_bars(
    #     name2partisanship,
    #     join(save_feature_stats_dir, f"{source_name}.partisan.png"),
    #     title=source_name,
    # )


def visualize_features(
    name2id2value,
    tweets,
    source_name,
    save_feature_stats_dir=join(SUBSET_WORKING_DIR, "feature_stats"),
):
    # individual feat horizontal bar
    name2value = {
        name: sum(id2value.values()) for name, id2value in name2id2value.items()
    }
    name2value = {k: v for k, v in sorted(name2value.items(), key=lambda x: x[1])}
    plot_horizontal_bars(
        name2value,
        join(save_feature_stats_dir, f"{source_name}.indiv.png"),
        title=f"{source_name} individual",
    )

    # partisanship horizontal bars
    stats = get_stats(tweets, name2id2value)
    name2partisanvalue = {
        name: -stats[name]["partisan"]["dem"] + stats[name]["partisan"]["rep"]
        for name in name2value
    }
    plot_horizontal_bars(
        name2partisanvalue,
        join(save_feature_stats_dir, f"{source_name}.partisan.png"),
        title=f"{source_name} partisan",
    )
