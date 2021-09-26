from os import makedirs, replace
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import statsmodels.api as sm
from cc_tweets.feature_utils import get_log_retweets, get_retweets
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json
from cc_tweets.viz import plot_horizontal_bars
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR

SEED = 0xDEADBEEF

FEATURE_FILTER = None
FEATURE_FILTER = [
    "economy",
    "emolex_anger",
    "emolex_anticipation",
    "emolex_disgust",
    "emolex_fear",
    "emolex_joy",
    "emolex_sadness",
    "emolex_surprise",
    "emolex_trust",
    # "natural_disasters",
    # "negation",
    # "p_aoc",
    # "p_attenborough",
    # "p_bernie",
    # "p_biden",
    # "p_clinton",
    # "p_gore",
    # "p_inslee",
    # "p_nye",
    # "p_obama",
    # "p_pruitt",
    # "p_thunberg",
    # "p_trump",
    # "subj_combined",
    # "subj_strong",
    # "subj_weak",
    "vad_arousal",
    "vad_dominance",
    "vad_valence",
    # "vad_present",
    # --
    # "vad_arousal_neg",
    # "vad_arousal_neu",
    # "vad_arousal_pos",
    # "vad_dominance_neg",
    # "vad_dominance_neu",
    # "vad_dominance_pos",
    # "vad_valence_neg",
    # "vad_valence_neu",
    # "vad_valence_pos",
    # --
    # "vader_compound",
    "vader_neg",
    "vader_neu",
    "vader_pos",
    # "mfd_vice_authority",
    # "mfd_vice_fairness",
    # "mfd_vice_harm",
    # "mfd_vice_loyalty",
    # "mfd_vice_purity",
    # "mfd_virtue_authority",
    # "mfd_virtue_fairness",
    # "mfd_virtue_harm",
    # "mfd_virtue_loyalty",
    # "mfd_virtue_purity",
    # "senti_HAN",
    # "senti_HAP",
    # "senti_LAN",
    # "senti_LAP",
    # "senti_NEU",
    # "senti_RAW",
    # --
    "bias",
    "log_followers",
    # "followers",
]

if __name__ == "__main__":
    savedir = join(SUBSET_WORKING_DIR, "linreg")
    makedirs(savedir, exist_ok=True)

    tweets = load_pkl(SUBSET_PKL_PATH)

    regin_dir = join(SUBSET_WORKING_DIR, "regression_inputs")
    feature_matrix = scipy.sparse.load_npz(join(regin_dir, "feature_matrix.npz"))
    feature_names = read_txt_as_str_list(join(regin_dir, f"feature_names.txt"))
    if FEATURE_FILTER is not None:
        fn2idx = {fn: i for i, fn in enumerate(feature_names)}
        filtered_feature_idxs = [fn2idx[fn] for fn in FEATURE_FILTER]
        feature_names = [feature_names[i] for i in filtered_feature_idxs]
        feature_matrix = feature_matrix[:, filtered_feature_idxs]
    feature_matrix = feature_matrix.toarray()

    # log_retweets = scipy.sparse.load_npz(join(regin_dir, f"log_retweets.npz"))
    # log_retweets = log_retweets.toarray()
    # log_retweets = np.squeeze(log_retweets)
    # targets = log_retweets
    targets = np.squeeze(get_log_retweets(tweets).toarray())

    name0idxs = [("all", None)]
    idxs_dem = load_pkl(join(regin_dir, "idxs_dem.pkl"))
    idxs_rep = load_pkl(join(regin_dir, "idxs_rep.pkl"))
    name0idxs.extend([("dem", idxs_dem), ("rep", idxs_rep)])
    np.random.seed(SEED)
    subsampled_idxs_dem = np.random.choice(idxs_dem, size=len(idxs_rep), replace=False)
    idxs_balanced = idxs_rep + subsampled_idxs_dem.tolist()
    name0idxs.append(("balanced", idxs_balanced))

    science_words = ["science", "research", "scientist", "researcher", "study"]
    science_idxs = []
    for i, t in enumerate(tweets):
        if any(w in t["lemmas"] for w in science_words):
            science_idxs.append(i)
    name0idxs.append(("science", science_idxs))

    for name, idxs in name0idxs:
        if idxs == None:
            filtered_feature_matrix = feature_matrix
            filtered_targets = targets
        else:
            filtered_feature_matrix = feature_matrix[idxs, :]
            filtered_targets = targets[idxs]

        model = sm.OLS(filtered_targets, filtered_feature_matrix)
        fit = model.fit()

        name2coef = {name: coef for name, coef in zip(feature_names, fit.params)}
        name2std = {name: std for name, std in zip(feature_names, fit.bse)}
        plot_horizontal_bars(
            name2coef,
            save_path=join(savedir, f"coef_{name}.png"),
            title=f"coefs: lin reg on log retweets: {name}",
            xlim=(-0.2, 0.2),
            yerr=fit.bse * 2,
            figsize=(7, 8),
        )
        save_json(
            {"coef": name2coef, "std": name2std}, join(savedir, f"coef_{name}.json")
        )

        if name == "all":
            pred = fit.predict()
            err = (filtered_targets - pred) ** 2
            err0idx = sorted([(err, idx) for idx, err in enumerate(err)], reverse=True)
            most_inacc_idx = [i for _, i in err0idx[:30]]
            most_inacc = [
                {
                    "idx": i,
                    "pred": pred[i],
                    "actual": filtered_targets[i],
                    "sqerr": err[i],
                    "tweet": tweets[i],
                }
                for i in most_inacc_idx
            ]
            save_json(most_inacc, join(savedir, f"inacc_{name}.json"))
