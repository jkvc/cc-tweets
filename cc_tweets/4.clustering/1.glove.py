import operator
from collections import Counter
from os import makedirs
from os.path import dirname, join

import numpy as np
import pandas as pd
from cc_tweets.clustering.glove import build_coocc_matrix_from_tweets_fixed_vocab
from cc_tweets.data_utils import load_vocab2idx
from cc_tweets.experiment_configs import EMB_DIM, SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.utils import load_pkl
from mittens import GloVe

TOKEN_TYPE = "stems"
VOCAB_SIZE = 4000
VOCAB_PATH = join(SUBSET_WORKING_DIR, "vocab", f"{TOKEN_TYPE}_1gram_{VOCAB_SIZE}.txt")

SAVE_PATH = join(SUBSET_WORKING_DIR, "clustering", f"glove.{EMB_DIM}.csv")

if __name__ == "__main__":
    makedirs(dirname(SAVE_PATH), exist_ok=True)

    # load
    tweets = load_pkl(SUBSET_PKL_PATH)
    vocab2idx = load_vocab2idx(VOCAB_PATH)

    # build
    coocc = build_coocc_matrix_from_tweets_fixed_vocab(
        tweets, vocab2idx, token_type=TOKEN_TYPE
    )

    # release
    del tweets

    # train
    glove_model = GloVe(EMB_DIM, max_iter=10000, learning_rate=0.1)
    embeddings = glove_model.fit(coocc)
    emb_df = pd.DataFrame(embeddings, index=list(vocab2idx.keys()))

    # save
    emb_df.to_csv(SAVE_PATH, sep="\t")
    print("glove embedding written to", SAVE_PATH)
