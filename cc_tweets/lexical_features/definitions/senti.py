import random
from collections import defaultdict
from os.path import join

from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.utils import ParallelHandler, load_pkl
from config import RESOURCES_DIR
from experiment_configs.base import SUBSET_PKL_PATH
from sentistrength import PySentiStr
from tqdm import tqdm

BINS = ["LAP", "HAP", "LAN", "HAN", "NEU"]


def is_in_bin(bin_name, pos_score, neg_score):
    # Low Arousal Positive: posts with positive 2
    # High Arousal Positive: posts with positive intensity 3, 4, or 5
    # Low Arousal Negative: posts with negative 2
    # High Arousal Negative: posts with negative intensity 3, 4, or 5
    # Neutral: posts with positive 1 AND negative 1

    if bin_name == "LAP":
        return pos_score == 2
    if bin_name == "HAP":
        return pos_score >= 3
    if bin_name == "LAN":
        return neg_score == -1
    if bin_name == "HAN":
        return neg_score <= -3
    if bin_name == "NEU":
        return pos_score == 1 and neg_score == -1
    raise ValueError()


def closure(binname):
    def _extract(tweets):
        senti = PySentiStr()
        senti.setSentiStrengthPath(
            join(RESOURCES_DIR, "sentistrength", "SentiStrength.jar")
        )
        senti.setSentiStrengthLanguageFolderPath(
            join(RESOURCES_DIR, "sentistrength", "data")
        )
        tweet_texts = [t["text"] for t in tweets]

        scores = senti.getSentiment(tweet_texts, score="dual")

        id2val = defaultdict(int)
        for tweet, score in zip(tweets, scores):
            pos_score, neg_score = score
            val = 1 if is_in_bin(binname, pos_score, neg_score) else 0
            id2val[tweet["id"]] = val
        return id2val

    return _extract


for binname in BINS:
    register_feature(Feature(f"senti.{binname}", closure(binname)))


# if __name__ == "__main__":
#     tweets = load_pkl(SUBSET_PKL_PATH)

#     senti = PySentiStr()
#     senti.setSentiStrengthPath(
#         join(RESOURCES_DIR, "sentistrength", "SentiStrength.jar")
#     )
#     senti.setSentiStrengthLanguageFolderPath(
#         join(RESOURCES_DIR, "sentistrength", "data")
#     )

#     tweet_texts = [t["text"] for t in tweets]
#     scores = senti.getSentiment(tweet_texts, score="dual")

#     name2id2score = defaultdict(dict)
#     bin2tweets = defaultdict(list)
#     for tweet, score in zip(tweets, scores):
#         id = tweet["id"]
#         pos_score, neg_score = score
#         bins = get_senti_bins(pos_score, neg_score)
#         for bin in bins:
#             name2id2score[f"senti_{bin}"][id] = 1
#             bin2tweets[bin].append(tweet)
#         name2id2score[f"senti_RAW"][id] = pos_score + neg_score

#     name2id2score = {
#         k: name2id2score[k]
#         for k in [
#             "senti_HAN",
#             "senti_LAN",
#             "senti_NEU",
#             "senti_LAP",
#             "senti_HAP",
#             "senti_RAW",
#         ]
#     }
#     save_features(tweets, name2id2score, "senti")

#     _PRINT_N_SAMPLE_PER_BIN = 10
#     for bin, tweets in bin2tweets.items():
#         print("\n\n")
#         print("-" * 50)
#         print(bin)
#         print("-" * 50)
#         random.shuffle(tweets)
#         for t in tweets[:_PRINT_N_SAMPLE_PER_BIN]:
#             print(t["text"])
#             print("-" * 50)
