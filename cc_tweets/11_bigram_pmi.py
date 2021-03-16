from collections import Counter, defaultdict
from os.path import join

import nltk
from config import DATA_DIR
from nltk.collocations import BigramCollocationFinder
from nltk.stem.snowball import SnowballStemmer
from numpy import sign
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams
from cc_tweets.experiment_config import (
    DATASET_NAME,
    DATASET_PKL_PATH,
    DATASET_SAVE_DIR,
    DOWNSIZE_FACTOR,
)
from cc_tweets.feature_utils import get_stats, save_features
from cc_tweets.utils import load_pkl, mkdir_overwrite, save_json, write_str_list_as_txt

DOWNSIZEFACTOR2MINOCC = {
    100: 100,
    10: 500,
}
MIN_OCCURRENCE = DOWNSIZEFACTOR2MINOCC[DOWNSIZE_FACTOR]
SAVE_DIR = join(DATASET_SAVE_DIR, "bigram_features")


def count_ngrams(tokens, ngrams, n):
    unfiltered_gram2count = Counter(get_ngrams(tokens, n))
    filtered_gram2count = {
        gram: unfiltered_gram2count[gram]
        for gram in ngrams
        if gram in unfiltered_gram2count
    }
    return filtered_gram2count


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    tokens = []
    for t in tweets:
        tokens.extend(t["stems"])
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(MIN_OCCURRENCE)

    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram0score = finder.score_ngrams(bigram_measures.pmi)
    bigram0score = sorted(bigram0score, key=lambda x: x[1], reverse=True)
    significant_bigrams = [
        " ".join(bigram) for bigram, score in bigram0score if score >= 0.5
    ]

    mkdir_overwrite(SAVE_DIR)
    write_str_list_as_txt(
        significant_bigrams,
        join(SAVE_DIR, "_bigrams.txt"),
    )

    # build features
    id2bigram2count = {
        t["id"]: count_ngrams(t["stems"], significant_bigrams, 2) for t in tqdm(tweets)
    }
    bigram2id2count = {gram: {} for gram in significant_bigrams}
    for t in tqdm(tweets):
        id = t["id"]
        for gram in significant_bigrams:
            if gram in id2bigram2count[id]:
                bigram2id2count[gram][id] = id2bigram2count[id][gram]

    save_features(
        tweets,
        bigram2id2count,
        "11_bigram_pmi",
        join(DATA_DIR, DATASET_NAME, "bigram_features"),
    )
