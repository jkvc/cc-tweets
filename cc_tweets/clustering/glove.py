import operator
from collections import Counter

import numpy as np
from experiment_configs.base import EMB_DIM, SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from tqdm import tqdm


def build_coocc_matrix_from_tweets_fixed_vocab(tweets, vocab2idx, token_type):
    coocc = np.zeros((len(vocab2idx), len(vocab2idx)))
    for tweet in tqdm(tweets, desc="build coocc"):
        stems = tweet[token_type]
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
