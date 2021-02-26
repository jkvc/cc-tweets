import re
from os.path import join

from config import DATA_DIR
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm import tqdm

from cc_tweets.utils import ParallelHandler, load_pkl, save_json

DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")

SID = SentimentIntensityAnalyzer()


def get_id0vaderscores(id, text):
    # {
    #     "neg": 0.075,
    #     "neu": 0.709,
    #     "pos": 0.216,
    #     "compound": 0.7003
    # }
    return id, SID.polarity_scores(text)


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)

    sid = SentimentIntensityAnalyzer()

    handler = ParallelHandler(get_id0vaderscores)
    id0vader2scores = handler.run([(tweet["id"], tweet["text"]) for tweet in tweets])
    id2vader2scores = {id: scores for id, scores in id0vader2scores}

    save_json(
        id2vader2scores,
        join(DATA_DIR, DATASET_NAME, "52_vader.json"),
    )

    stats = {}
    score_names = ["pos", "neu", "neg", "compound"]

    for name in score_names:
        stats[f"mean_{name}"] = sum(
            scores[name] for scores in id2vader2scores.values()
        ) / len(id2vader2scores)

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    for lean, tweets in [("dem", dem_tweets), ("rep", rep_tweets)]:
        partisan_stats = {}
        for name in score_names:
            partisan_stats[f"mean_{name}"] = sum(
                id2vader2scores[t["id"]][name] for t in tweets
            ) / len(tweets)
        stats[lean] = partisan_stats

    for name in score_names:
        stats[f"mean_{name}_adjusted_for_dem_rep_imbalance"] = (
            stats["dem"][f"mean_{name}"] + stats["rep"][f"mean_{name}"]
        ) / 2

    save_json(
        stats,
        join(DATA_DIR, DATASET_NAME, "52_vader_stats.json"),
    )
