from collections import defaultdict
from multiprocessing import cpu_count

from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.utils import ParallelHandler, load_pkl
from experiment_configs.base import SUBSET_PKL_PATH
from nltk.sentiment.vader import SentimentIntensityAnalyzer

SCORE_NAMES = ["pos", "neu", "neg", "compound"]
ANALYZER = SentimentIntensityAnalyzer()


def get_score(tid, text, score_name):
    return tid, ANALYZER.polarity_scores(text)[score_name]


def extract_vader(tweets, score_name):
    params = [(t["id"], t["text"], score_name) for t in tweets]
    handler = ParallelHandler(get_score)
    results = handler.run(params, quiet=True)

    id2v = {}
    for tid, score in results:
        id2v[tid] = score
    return id2v


def closure(name):
    def _extract(tweets):
        return extract_vader(tweets, name)

    return _extract


for name in SCORE_NAMES:
    register_feature(Feature(f"vder.{name}", closure(name)))
