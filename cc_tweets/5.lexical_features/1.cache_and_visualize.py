import random
from os import makedirs
from os.path import join
from posixpath import dirname

import numpy as np
import pandas as pd
import seaborn as sbn
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.lexical_features.bank import get_all_feature_names, get_feature
from cc_tweets.utils import load_json, load_pkl
from cc_tweets.viz import plot_grouped_bars
from tqdm import tqdm

BAR_PLOTS_GROUPS = {
    "terms": [
        "disaster",
        "economy",
        "negation",
        "science",
    ],
    "emo": [
        "emo.anger",
        "emo.anticipation",
        "emo.disgust",
        "emo.fear",
        "emo.joy",
        "emo.sadness",
        "emo.surprise",
        "emo.trust",
    ],
    "mfd": [
        "mfd.vice_authority",
        "mfd.virtue_authority",
        "mfd.vice_fairness",
        "mfd.virtue_fairness",
        "mfd.vice_harm",
        "mfd.virtue_harm",
        "mfd.vice_loyalty",
        "mfd.virtue_loyalty",
        "mfd.vice_purity",
        "mfd.virtue_purity",
    ],
    "senti": [
        "senti.LAP",
        "senti.HAP",
        "senti.LAN",
        "senti.HAN",
        "senti.NEU",
    ],
    "subj": [
        "subj.combined",
        "subj.strong",
        "subj.weak",
    ],
    "vad": [
        "vad.dominance",
        "vad.arousal",
        "vad.valence",
    ],
    "vader": [
        "vder.compound",
        "vder.neg",
        "vder.neu",
        "vder.pos",
    ],
}

PAIRPLOT_FEATURES = [
    "emo.anger",
    "emo.anticipation",
    "emo.disgust",
    "emo.fear",
    "emo.joy",
    "emo.sadness",
    "emo.surprise",
    "emo.trust",
    # "mfd.vice_authority",
    # "mfd.vice_fairness",
    # "mfd.vice_harm",
    # "mfd.vice_loyalty",
    # "mfd.vice_purity",
    # "mfd.virtue_authority",
    # "mfd.virtue_fairness",
    # "mfd.virtue_harm",
    # "mfd.virtue_loyalty",
    # "mfd.virtue_purity",
    # "senti.LAP",
    # "senti.HAP",
    # "senti.LAN",
    # "senti.HAN",
    # "senti.NEU",
    # "subj.combined",
    # "subj.strong",
    # "subj.weak",
    "vad.dominance",
    "vad.arousal",
    "vad.valence",
    "vder.compound",
    "vder.neg",
    "vder.neu",
    "vder.pos",
]
PAIRPLOT_SAMPLE_SIZE = 250


if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    tweet_ids = [t["id"] for t in tweets]
    pairplot_tweet_ids = random.sample(tweet_ids, PAIRPLOT_SAMPLE_SIZE)
    pairplot_feature_vecs = []

    for name in get_all_feature_names():
        print(name)
        feature = get_feature(name)
        feature_dict = feature.get_feature_dict(tweets)
        if name in PAIRPLOT_FEATURES:
            pairplot_feature_vecs.append(
                np.array([feature_dict[tid] for tid in pairplot_tweet_ids])
            )
            del feature_dict

    for groupname, featnames in BAR_PLOTS_GROUPS.items():
        name2stats = {}
        for name in featnames:
            s = load_json(join(SUBSET_WORKING_DIR, "feature_stats", f"{name}.json"))
            name2stats[name] = s

        save_path = join(SUBSET_WORKING_DIR, "feature_viz", f"{groupname}.png")
        makedirs(dirname(save_path), exist_ok=True)

        x_labels = list(name2stats.keys())
        name2series = {
            lean: [name2stats[sname]["partisan"][lean] for sname in x_labels]
            for lean in ["dem", "rep"]
        }

        plot_grouped_bars(featnames, name2series, groupname, save_path=save_path)

    # pairplot feature colinearity
    subsample_idxs = np.random.choice(
        list(range(len(tweets))), replace=False, size=1000
    )
    df = pd.DataFrame()
    for name, f in zip(PAIRPLOT_FEATURES, pairplot_feature_vecs):
        # f = f.squeeze()[subsample_idxs]
        df[name] = f
    plot = sbn.pairplot(df, markers="+", corner=True)
    plot.savefig(join(SUBSET_WORKING_DIR, "feature_viz", "_pairplot.png"))
