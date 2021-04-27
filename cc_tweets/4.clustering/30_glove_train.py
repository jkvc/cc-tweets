import operator
from collections import Counter
from os import makedirs
from os.path import dirname, join

import numpy as np
import pandas as pd
from config import DATA_DIR
from mittens import GloVe
from tqdm import tqdm

from cc_tweets.data_utils import load_vocab2idx
from cc_tweets.experiment_config import SUBSET_PKL_PATH, SUBSET_WORKING_DIR, EMB_DIM
from cc_tweets.utils import load_pkl

TOKEN_TYPE = "stems"
VOCAB_SIZE = 4000
VOCAB_PATH = join(SUBSET_WORKING_DIR, "vocab", f"{TOKEN_TYPE}_1gram_{VOCAB_SIZE}.txt")

SAVE_EMB_PATH = join(SUBSET_WORKING_DIR, "glove", f"glove.{EMB_DIM}.csv")


def build_coocc_matrix(tweets, vocab2idx):
    coocc = np.zeros((len(vocab2idx), len(vocab2idx)))
    for tweet in tqdm(tweets, desc="build coocc"):
        stems = tweet[TOKEN_TYPE]
        word_counts = Counter(stems)
        bow_sorted = sorted(
            word_counts.items(),
            key=operator.itemgetter(1),
            reverse=True,
        )
        for i, (stem, count1) in enumerate(bow_sorted):
            for (context, count2) in bow_sorted[i:]:
                if stem not in vocab2idx or context not in vocab2idx:
                    continue
                coocc[vocab2idx[stem], vocab2idx[context]] += count1 * count2
                if context != stem:
                    coocc[vocab2idx[context], vocab2idx[stem]] += count1 * count2
    return coocc


if __name__ == "__main__":
    makedirs(dirname(SAVE_EMB_PATH), exist_ok=True)

    tweets = load_pkl(SUBSET_PKL_PATH)
    vocab2idx = load_vocab2idx(VOCAB_PATH)
    coocc = build_coocc_matrix(tweets, vocab2idx)

    del tweets

    glove_model = GloVe(EMB_DIM, max_iter=10000, learning_rate=0.1)
    embeddings = glove_model.fit(coocc)
    emb_df = pd.DataFrame(embeddings, index=list(vocab2idx.keys()))
    emb_df.to_csv(SAVE_EMB_PATH, sep="\t")
