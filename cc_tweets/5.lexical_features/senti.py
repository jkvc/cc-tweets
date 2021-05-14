import random
from collections import defaultdict
from os.path import join

from cc_tweets.experiment_config import SUBSET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.utils import ParallelHandler, load_pkl
from config import RESOURCES_DIR
from sentistrength import PySentiStr
from tqdm import tqdm


def get_senti_bins(pos_score, neg_score):
    # Low Arousal Positive: posts with positive 2
    # High Arousal Positive: posts with positive intensity 3, 4, or 5
    # Low Arousal Negative: posts with negative 2
    # High Arousal Negative: posts with negative intensity 3, 4, or 5
    # Neutral: posts with positive 1 AND negative 1
    bins = []
    if pos_score == 2:
        bins.append("LAP")
    if pos_score >= 3:
        bins.append("HAP")
    if neg_score == -2:
        bins.append("LAN")
    if neg_score <= -3:
        bins.append("HAN")
    if pos_score == 1 and neg_score == -1:
        bins.append("NEU")
    return bins


if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    senti = PySentiStr()
    senti.setSentiStrengthPath(
        join(RESOURCES_DIR, "sentistrength", "SentiStrength.jar")
    )
    senti.setSentiStrengthLanguageFolderPath(
        join(RESOURCES_DIR, "sentistrength", "data")
    )

    tweet_texts = [t["text"] for t in tweets]
    scores = senti.getSentiment(tweet_texts, score="dual")

    name2id2score = defaultdict(dict)
    bin2tweets = defaultdict(list)
    for tweet, score in zip(tweets, scores):
        id = tweet["id"]
        pos_score, neg_score = score
        bins = get_senti_bins(pos_score, neg_score)
        for bin in bins:
            name2id2score[f"senti_{bin}"][id] = 1
            bin2tweets[bin].append(tweet)
        name2id2score[f"senti_RAW"][id] = pos_score + neg_score

    name2id2score = {
        k: name2id2score[k]
        for k in [
            "senti_HAN",
            "senti_LAN",
            "senti_NEU",
            "senti_LAP",
            "senti_HAP",
            "senti_RAW",
        ]
    }
    save_features(tweets, name2id2score, "senti")

    _PRINT_N_SAMPLE_PER_BIN = 10
    for bin, tweets in bin2tweets.items():
        print("\n\n")
        print("-" * 50)
        print(bin)
        print("-" * 50)
        random.shuffle(tweets)
        for t in tweets[:_PRINT_N_SAMPLE_PER_BIN]:
            print(t["text"])
            print("-" * 50)
