from collections import Counter
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.log_odds import scaled_lor
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS, AFFECT_IGNORE_STEMS
from cc_tweets.utils import load_pkl, mkdir_overwrite, unzip

ENGAGEMENT_TYPES = ["retweets", "likes"]
MIN_NUM_FOLLOWER = 30


def get_ratio(t, engagement_type):
    return np.log(t[engagement_type] + 1) / np.log(t["max_num_follower"] + 1)
    # return (t[engagement_type] + 1) / (t["max_num_follower"] + 1)


def _experiment(savedir, tweets):
    # savedir = join(SUBSET_WORKING_DIR, "log_odds_engagement")
    makedirs(savedir, exist_ok=True)

    # tweets = load_pkl(SUBSET_PKL_PATH)

    for engagement_type in ENGAGEMENT_TYPES:
        print(">>", engagement_type)

        ratios = np.array(
            [
                get_ratio(t, engagement_type)
                for t in tweets
                if t["max_num_follower"] >= MIN_NUM_FOLLOWER
            ]
        )
        plt.clf()
        plt.hist(ratios, bins=50)
        plt.savefig(join(savedir, f"hist.{engagement_type}.png"))

        median = np.median(ratios)

        for toktype, ngram in [("lemmas", 1), ("stems", 2)]:
            hi_engagement_wc, lo_engagement_wc = Counter(), Counter()
            hi_ratio0tweet, lo_ratio0tweet = [], []

            for t in tweets:
                if t["max_num_follower"] < MIN_NUM_FOLLOWER:
                    continue
                ratio = get_ratio(t, engagement_type)

                if ratio > median:
                    hi_ratio0tweet.append((ratio, t))
                else:
                    lo_ratio0tweet.append((ratio, t))

                wc = hi_engagement_wc if ratio > median else lo_engagement_wc
                ignores = (
                    AFFECT_IGNORE_LEMMAS if toktype == "lemmas" else AFFECT_IGNORE_STEMS
                )
                toks = [tok for tok in t[toktype] if tok not in ignores]
                wc.update(get_ngrams(toks, ngram))

            lor0w = scaled_lor(hi_engagement_wc, lo_engagement_wc, {})
            hi_engagement_topwords = [w for lor, w in lor0w][:100]
            lo_engagement_topwords = [w for lor, w in lor0w[::-1]][:100]
            # print(hi_engagement_topwords)
            # print(lo_engagement_topwords)

            scores, words = unzip(lor0w)
            df = pd.DataFrame()
            df["word"] = words
            df["score"] = scores
            df.to_csv(
                join(savedir, f"{engagement_type}.{toktype}.{ngram}.csv"), index=False
            )


if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    _experiment(join(SUBSET_WORKING_DIR, "log_odds_engagement", "all"), tweets)
    _experiment(
        join(SUBSET_WORKING_DIR, "log_odds_engagement", "dem"),
        [t for t in tweets if t["stance"] == "dem"],
    )
    _experiment(
        join(SUBSET_WORKING_DIR, "log_odds_engagement", "rep"),
        [t for t in tweets if t["stance"] == "rep"],
    )

    print("stuff written to ", join(SUBSET_WORKING_DIR, "log_odds_engagement"))
