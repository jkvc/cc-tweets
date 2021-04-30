from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams

STOPWORDS = stopwords.words("english")
MIN_WORD_COUNT = 20
MIN_WORD_LEN = 2


def scaled_lor(
    left_wc: Dict[str, int],
    right_wc: Dict[str, int],
    background_wc: Dict[str, int],
    min_word_count: int = MIN_WORD_COUNT,
) -> List[Tuple[float, str]]:

    n_left = sum(left_wc.values()) + 1
    n_right = sum(right_wc.values()) + 1
    n_bg = sum(background_wc.values()) + 1
    l_r_corpus_ratio = n_right / n_left
    n_left *= l_r_corpus_ratio

    lor = {}
    for w, f_w_left in left_wc.items():
        if f_w_left < min_word_count:
            continue
        if w not in right_wc or right_wc[w] < min_word_count:
            continue
        if len(w) < MIN_WORD_LEN:
            continue
        if w in STOPWORDS:
            continue

        f_w_left *= l_r_corpus_ratio
        f_w_right = right_wc.get(w, min_word_count)
        f_w_bg = background_wc.get(w, min_word_count)

        l_numerator = f_w_left + f_w_bg
        l_denominator = n_left + n_bg - l_numerator
        l = np.log(l_numerator / l_denominator)

        r_numerator = f_w_right + f_w_bg
        r_denominator = n_right + n_bg - r_numerator
        r = np.log(r_numerator / r_denominator)

        variance = 1 / (f_w_left + f_w_bg) + 1 / (f_w_right + f_w_bg)

        lor_w = l - r
        z_score = lor_w / (np.sqrt(variance))
        lor[w] = z_score

    lor0w = [(lor, w) for w, lor in lor.items()]
    lor0w = sorted(lor0w, reverse=True)
    return lor0w
