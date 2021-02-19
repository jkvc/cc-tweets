import argparse
import gc
import json
import random
import re
import string
import sys
from collections import Counter
from glob import glob
from os.path import exists, join
from typing import List

import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
import validators
from config import DATA_DIR, RAW_DIR
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

from cc_tweets.utils import (
    ParallelHandler,
    get_ngrams,
    load_pkl,
    read_txt_as_str_list,
    save_pkl,
    write_str_list_as_txt,
)

SRC_DATASET_NAME = "tweets_downsized10_filtered"
PKL_PATH = join(DATA_DIR, f"{SRC_DATASET_NAME}.pkl")
NUM_TRIALS = 10


def build_speaker_bigram_matrix(tweets, vocab2idx, speaker2idx):
    # speaker, phrase
    m_t = np.zeros((len(speaker2idx), len(vocab2idx)))
    for tweet in tqdm(tweets, "build m_t"):
        grams = get_ngrams(tweet["stems"], 2)
        speaker_idx = speaker2idx[tweet["userid"]]
        for gram in grams:
            if gram in vocab2idx:
                vocab_idx = vocab2idx[gram]
                m_t[speaker_idx][vocab_idx] += 1
    return m_t


def get_speaker2stance(tweets):
    return {t["userid"]: t["stance"] for t in tweets}


def leaveout_some_speakers(speakers, m_t, speaker2idx, speaker2stance):
    seq = []
    for speaker in tqdm(speakers):
        i = speaker2idx[speaker]
        # if speaker2stance[speaker] == "dem":
        #     continue
        c_i = m_t[i]  # (vocabsize, )
        if c_i.sum() == 0:
            continue
        q_i = c_i / (c_i.sum())
        other_dem_speakers = [
            o
            for other_speaker, o in speaker2idx.items()
            if speaker2stance[other_speaker] == "dem" and o != i
        ]
        c_noti_L = m_t[other_dem_speakers].sum(axis=0)
        other_rep_speakers = [
            o
            for other_speaker, o in speaker2idx.items()
            if speaker2stance[other_speaker] == "rep" and o != i
        ]
        c_noti_R = m_t[other_rep_speakers].sum(axis=0)
        q_noti_L = (c_noti_L) / (c_noti_L.sum() + np.finfo(float).eps)
        q_noti_R = (c_noti_R) / (c_noti_R.sum() + np.finfo(float).eps)
        rho_noti = q_noti_R / (q_noti_R + q_noti_L + np.finfo(float).eps)
        seq.append(q_i.dot(rho_noti))
    return seq


def leave_out_estimator(m_t, speaker2idx, speaker2stance):
    c_rep_speakers_sum = m_t[
        [
            o
            for other_speaker, o in speaker2idx.items()
            if speaker2stance[other_speaker] == "rep"
        ]
    ].sum(axis=0)
    c_dem_speakers_sum = m_t[
        [
            o
            for other_speaker, o in speaker2idx.items()
            if speaker2stance[other_speaker] == "dem"
        ]
    ].sum(axis=0)
    c_all_speakers_sum = m_t.sum(axis=0)

    # left side of eq on orig paper is republican leaning -_-
    left_side_seq = [0.5]
    for speaker, i in tqdm(speaker2idx.items()):
        if speaker2stance[speaker] == "dem":
            continue
        c_i = m_t[i]  # (vocabsize, )
        if c_i.sum() == 0:
            continue
        q_i = c_i / (c_i.sum())

        c_noti_L = c_all_speakers_sum - c_rep_speakers_sum - c_i + np.finfo(float).eps
        c_noti_R = c_all_speakers_sum - c_dem_speakers_sum - c_i + np.finfo(float).eps

        q_noti_L = (c_noti_L) / (c_noti_L.sum())
        q_noti_R = (c_noti_R) / (c_noti_R.sum())
        rho_noti = q_noti_R / ((q_noti_R + q_noti_L))
        left_side_seq.append(q_i.dot(rho_noti))
    left_side_mean = sum(left_side_seq) / len(left_side_seq)

    right_side_seq = [0.5]
    for speaker, i in tqdm(speaker2idx.items()):
        if speaker2stance[speaker] == "rep":
            continue
        c_i = m_t[i]  # (vocabsize, )
        if c_i.sum() == 0:
            continue
        q_i = c_i / (c_i.sum())

        c_noti_L = c_all_speakers_sum - c_rep_speakers_sum - c_i + np.finfo(float).eps
        c_noti_R = c_all_speakers_sum - c_dem_speakers_sum - c_i + np.finfo(float).eps

        q_noti_L = (c_noti_L) / (c_noti_L.sum())
        q_noti_R = (c_noti_R) / (c_noti_R.sum())
        rho_noti = q_noti_R / ((q_noti_R + q_noti_L))
        right_side_seq.append(q_i.dot(1 - rho_noti))
    right_side_mean = sum(right_side_seq) / len(right_side_seq)

    left_right_mean = (left_side_mean + right_side_mean) / 2
    return left_right_mean


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)
    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]
    print(len(dem_tweets), len(rep_tweets))

    pols = []
    for trial in range(NUM_TRIALS):
        print("trial", trial)

        # subsample dem tweets for class balance
        sampled_dem_tweets = random.sample(dem_tweets, len(rep_tweets))
        balanced_tweets = sampled_dem_tweets + rep_tweets

        vocab2idx = {
            gram: i
            for i, gram in enumerate(
                read_txt_as_str_list(join(DATA_DIR, "vocab_3000_2gram.txt"))
            )
        }
        speaker2idx = {
            userid: i
            for i, userid in enumerate(set(t["userid"] for t in balanced_tweets))
        }
        speaker2stance = get_speaker2stance(balanced_tweets)

        m_t = build_speaker_bigram_matrix(balanced_tweets, vocab2idx, speaker2idx)
        pol = leave_out_estimator(m_t, speaker2idx, speaker2stance)
        print(pol)
        pols.append(pol)

    avgpol = sum(pols) / len(pols)
    print(">> avgpol", avgpol)
