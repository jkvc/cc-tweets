from collections import Counter, defaultdict
from os import makedirs
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cc_tweets.data_utils import get_ngrams
from experiment_configs.base import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.log_odds import scaled_lor
from cc_tweets.utils import load_pkl, mkdir_overwrite, unzip


def _hash(t):
    return tuple(sorted(list(set(t["lemmas"]))))


if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    d = defaultdict(list)
    for t in tweets:
        d[_hash(t)].append(t)
    print(sorted((len(v), k) for k, v in d.items())[-20:])

    pprint(
        d[
            (
                "adapt",
                "announce",
                "around",
                "change",
                "climate",
                "girl",
                "help",
                "need",
                "support",
                "vulnerable",
                "woman",
                "world",
            )
        ]
    )
