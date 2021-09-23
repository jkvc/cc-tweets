from os import makedirs
from os.path import join
from posixpath import dirname

from cc_tweets.lexical_features.bank import get_all_feature_names, get_feature
from cc_tweets.utils import load_json, load_pkl, save_json, save_pkl
from cc_tweets.viz import plot_grouped_bars, plot_horizontal_bars
from experiment_configs.base import SUBSET_PKL_PATH, SUBSET_WORKING_DIR

VISUALIZATION_GROUPING = {
    "wordcounts": ["disaster", "economy", "negation"],
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
        "mfd.vice_fairness",
        "mfd.vice_harm",
        "mfd.vice_loyalty",
        "mfd.vice_purity",
        "mfd.virtue_authority",
        "mfd.virtue_fairness",
        "mfd.virtue_harm",
        "mfd.virtue_loyalty",
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
        "vder.neg",
        "vder.neu",
        "vder.pos",
    ],
}


if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    for name in get_all_feature_names():
        print(name)
        feature = get_feature(name)
        feature.cache_features(tweets)

    for groupname, featnames in VISUALIZATION_GROUPING.items():
        name2stats = {}
        for name in featnames:
            s = load_json(join(SUBSET_WORKING_DIR, "feature_stats", f"{name}.json"))
            name2stats[name] = s

        # name2series = {}
        # for name in featnames:
        #     stats = load_json(join(SUBSET_WORKING_DIR, "feature_stats", f"{name}.json"))
        #     name2series[name] = {
        #         lean: stats["partisan"][lean] for lean in ["dem", "rep"]
        #     }
        # print(name2series)
        save_path = join(SUBSET_WORKING_DIR, "feature_viz", f"{groupname}.png")
        makedirs(dirname(save_path), exist_ok=True)

        x_labels = list(name2stats.keys())
        name2series = {
            lean: [name2stats[sname]["partisan"][lean] for sname in x_labels]
            for lean in ["dem", "rep"]
        }

        plot_grouped_bars(featnames, name2series, groupname, save_path=save_path)
