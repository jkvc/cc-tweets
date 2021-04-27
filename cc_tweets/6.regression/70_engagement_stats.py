from os import makedirs
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from config import DATA_DIR

from cc_tweets.experiment_config import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.utils import load_json, load_pkl, read_txt_as_str_list, save_json, unzip

SAVE_DIR = join(SUBSET_WORKING_DIR, "engagement")

if __name__ == "__main__":
    makedirs(SAVE_DIR, exist_ok=True)

    tweets = load_pkl(SUBSET_PKL_PATH)
    userid2numfollowers = load_json(join(DATA_DIR, "userid2numfollowers.json"))
    mean_num_followers = sum(userid2numfollowers.values()) / len(userid2numfollowers)

    def _get_num_followers(userid):
        if userid in userid2numfollowers:
            return userid2numfollowers[userid]
        else:
            return 0

    stats = {
        "num_tweets": len(tweets),
        "median_likes_to_followers": (
            np.median(
                np.array(
                    [t["likes"] / (_get_num_followers(t["userid"]) + 1) for t in tweets]
                )
            )
        ),
        "median_retweets_to_followers": (
            np.median(
                np.array(
                    [
                        t["retweets"] / (_get_num_followers(t["userid"]) + 1)
                        for t in tweets
                    ]
                )
            )
        ),
    }

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(16, 16))

    # n_retweet v follower
    follower0rts = [(_get_num_followers(t["userid"]), t["retweets"]) for t in tweets]
    followers, rts = unzip(follower0rts)
    followers, rts = np.array(followers), np.array(rts)
    axes[0][0].scatter(followers, rts, marker=".")
    axes[0][0].set_title("retweets v followers")
    axes[0][0].set_xlabel("followers")
    axes[0][0].set_ylabel("rts")
    c1, c0 = np.polyfit(followers, rts, 1)
    linreg_fn = np.poly1d([c1, c0])
    linespace = np.linspace(followers.min(), followers.max(), 20)
    axes[0][0].plot(linespace, linreg_fn(linespace), color="red")
    preds = linreg_fn(followers)
    stats["rt"] = {
        "c0": c0,
        "c1": c1,
        "higher": int((rts >= preds).sum()),
        "lower": int((rts < preds).sum()),
    }

    # log n_retweet v follower
    log_followers = np.log(np.array(followers) + 1)
    log_rts = np.log(np.array(rts) + 1)
    axes[0][1].scatter(log_followers, log_rts, marker=".")
    axes[0][1].set_title("log retweets v log followers")
    axes[0][1].set_xlabel("log followers")
    axes[0][1].set_ylabel("log rts")
    c1, c0 = np.polyfit(log_followers, log_rts, 1)
    linreg_fn = np.poly1d([c1, c0])
    linespace = np.linspace(log_followers.min(), log_followers.max(), 20)
    axes[0][1].plot(linespace, linreg_fn(linespace), color="red")
    preds = linreg_fn(log_followers)
    stats["log_rt"] = {
        "c0": c0,
        "c1": c1,
        "higher": int((log_rts >= preds).sum()),
        "lower": int((log_rts < preds).sum()),
    }

    # n_like v follower
    follower0likes = [(_get_num_followers(t["userid"]), t["likes"]) for t in tweets]
    followers, likes = unzip(follower0likes)
    followers, likes = np.array(followers), np.array(likes)
    axes[1][0].scatter(followers, likes, marker=".")
    axes[1][0].set_title("likes v followers")
    axes[1][0].set_xlabel("followers")
    axes[1][0].set_ylabel("likes")
    c1, c0 = np.polyfit(followers, likes, 1)
    linreg_fn = np.poly1d([c1, c0])
    linespace = np.linspace(followers.min(), followers.max(), 20)
    axes[1][0].plot(linespace, linreg_fn(linespace), color="red")
    preds = linreg_fn(followers)
    stats["likes"] = {
        "c0": c0,
        "c1": c1,
        "higher": int((likes >= preds).sum()),
        "lower": int((likes < preds).sum()),
    }

    # log n_like v follower
    log_followers = np.log(np.array(followers) + 1)
    log_likes = np.log(np.array(likes) + 1)
    axes[1][1].scatter(log_followers, log_likes, marker=".")
    axes[1][1].set_title("log likes v log followers")
    axes[1][1].set_xlabel("log followers")
    axes[1][1].set_ylabel("log likes")
    c1, c0 = np.polyfit(log_followers, log_likes, 1)
    linreg_fn = np.poly1d([c1, c0])
    linespace = np.linspace(log_followers.min(), log_followers.max(), 20)
    axes[1][1].plot(linespace, linreg_fn(linespace), color="red")
    preds = linreg_fn(log_followers)
    stats["log_likes"] = {
        "c0": c0,
        "c1": c1,
        "higher": int((log_likes >= preds).sum()),
        "lower": int((log_likes < preds).sum()),
    }

    plt.savefig(join(SAVE_DIR, "stats.png"))
    save_json(stats, join(SAVE_DIR, "stats.json"))

    plt.clf()
    fig, axes = plt.subplots(ncols=2, figsize=(10, 7))
    axes[0].hist(followers)
    axes[0].set_title("followers")
    axes[1].hist(log_followers)
    axes[1].set_title("log (follower + 1)")
    plt.savefig(join(SAVE_DIR, "followers_hist.png"))
