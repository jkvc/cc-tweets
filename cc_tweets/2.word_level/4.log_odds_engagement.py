from collections import Counter
from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cc_tweets.experiment_config import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.log_odds import scaled_lor
from cc_tweets.utils import load_pkl, mkdir_overwrite, unzip

ENGAGEMENTS = ["retweets", "likes"]

if __name__ == "__main__":
    savedir = join(SUBSET_WORKING_DIR, "log_odds_engagement")
    makedirs(savedir, exist_ok=True)

    tweets = load_pkl(SUBSET_PKL_PATH)

    for engagement_type in ENGAGEMENTS:
        print(">>", engagement_type)

        ratios = np.array(
            [
                np.log(t[engagement_type] + 1) / np.log(t["max_num_follower"])
                for t in tweets
                if t["max_num_follower"] > 10
            ]
        )
        plt.clf()
        plt.hist(ratios, bins=50)
        plt.savefig(join(savedir, f"hist.{engagement_type}.png"))

        median = np.median(ratios)

        hi_engagement_wc, lo_engagement_wc = Counter(), Counter()
        for t in tweets:
            if t["max_num_follower"] <= 10:
                continue
            ratio = np.log(t[engagement_type] + 1) / np.log(t["max_num_follower"])
            wc = hi_engagement_wc if ratio > median else lo_engagement_wc
            wc.update(t["lemmas"])

        lor0w = scaled_lor(hi_engagement_wc, lo_engagement_wc, {})
        hi_engagement_topwords = [w for lor, w in lor0w][:100]
        lo_engagement_topwords = [w for lor, w in lor0w[::-1]][:100]
        print(hi_engagement_topwords)
        print(lo_engagement_topwords)

        scores, words = unzip(lor0w)
        df = pd.DataFrame()
        df["word"] = words
        df["score"] = scores
        df.to_csv(join(savedir, f"{engagement_type}.csv"), index=False)
