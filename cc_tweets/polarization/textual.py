import random

import numpy as np
from tqdm import tqdm

from cc_tweets.data_utils import get_ngrams


def build_speaker_wordcount_matrix(tweets, vocab2idx, speaker2idx):
    m_t = np.zeros((len(speaker2idx), len(vocab2idx)))  # speaker, phrase
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


def calc_dem_rep_polarization(tweets, vocab2idx, n_trials_avg):

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]

    pols = []
    for trial in range(n_trials_avg):
        print("trial", trial)

        # subsample dem tweets for class balance
        sampled_dem_tweets = random.sample(dem_tweets, len(rep_tweets))
        balanced_tweets = sampled_dem_tweets + rep_tweets

        speaker2idx = {
            userid: i
            for i, userid in enumerate(set(t["userid"] for t in balanced_tweets))
        }
        speaker2stance = get_speaker2stance(balanced_tweets)

        m_t = build_speaker_wordcount_matrix(balanced_tweets, vocab2idx, speaker2idx)
        pol = leave_out_estimator(m_t, speaker2idx, speaker2stance)
        print(pol)
        pols.append(pol)

    avgpol = sum(pols) / len(pols)
    return avgpol
