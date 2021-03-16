from collections import defaultdict

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from cc_tweets.experiment_config import DATASET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.utils import ParallelHandler, load_pkl

ANALYZER = SentimentIntensityAnalyzer()


def get_id0vaderscores(id, text):
    return id, ANALYZER.polarity_scores(text)


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    sid = SentimentIntensityAnalyzer()

    handler = ParallelHandler(get_id0vaderscores)
    id0vadername2scores = handler.run(
        [(tweet["id"], tweet["text"]) for tweet in tweets]
    )

    score_names = ["pos", "neu", "neg", "compound"]
    name2id2score = defaultdict(dict)
    for id, vadername2scores in id0vadername2scores:
        for vadername, score in vadername2scores.items():
            name2id2score[f"vader_{vadername}"][id] = score

    save_features(tweets, name2id2score, "vader")
