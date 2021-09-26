import gc
from collections import Counter, OrderedDict
from os import makedirs
from os.path import exists, join
from pprint import pprint
from random import shuffle

import numpy as np
import pandas as pd
from cc_tweets.experiment_configs import EMB_DIM, SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.utils import load_pkl, save_json, save_pkl
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm, trange

EMB_PATH = join(SUBSET_WORKING_DIR, "clustering", f"glove.{EMB_DIM}.csv")

NUM_CLUSTERS = [7, 10, 20]
SAVE_RESULTS_SUPERDIR = join(SUBSET_WORKING_DIR, "clustering")

MAXLEN = 1000
PRINT_EVERY = 5000


def get_word_weights(files, a=1e-3):
    # get word frequencies
    vectorizer = CountVectorizer(decode_error="ignore")
    counts = vectorizer.fit_transform(files)
    total_freq = np.sum(counts, axis=0).T  # aggregate frequencies over all files
    N = np.sum(total_freq)
    weighted_freq = a / (a + total_freq / N)
    gc.collect()
    # dict with words and their weights
    return dict(zip(vectorizer.get_feature_names(), weighted_freq))


def sentences2idx(sentences, words2index, words2weight):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i (0 means no word at the location)
    """
    # print(sentences[0].split())
    maxlen = min(max([len(s.split()) for s in sentences]), MAXLEN)
    print("maxlen", maxlen)
    n_samples = len(sentences)
    print("samples", n_samples)
    x = np.zeros((n_samples, maxlen)).astype("int32")
    w = np.zeros((n_samples, maxlen)).astype("float32")
    x_mask = np.zeros((n_samples, maxlen)).astype("float32")
    for idx, s in enumerate(tqdm(sentences)):
        split = s.split()
        indices = []
        weightlist = []
        for word in split:
            if word in words2index:
                indices.append(words2index[word])
                if word not in words2weight:
                    weightlist.append(0.000001)
                else:
                    weightlist.append(words2weight[word])
        length = min(len(indices), maxlen)
        x[idx, :length] = indices[:length]
        w[idx, :length] = weightlist[:length]
        x_mask[idx, :length] = [1.0] * length
    del sentences
    gc.collect()
    return x, x_mask, w


def get_weighted_average(We, x, m, w, dim):
    """
    Compute the weighted average vectors
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in sentence i
    :param w: w[i, :] are the weights for the words in sentence i
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    print("Getting weighted average...")
    n_samples = x.shape[0]
    emb = np.zeros((n_samples, dim)).astype("float32")

    for i in trange(n_samples):
        stacked = []
        for idx, j in enumerate(x[i, :]):
            if m[i, idx] != 1:
                stacked.append(np.zeros(dim))
            else:
                stacked.append(We.values[j, :])
        vectors = np.stack(stacked)
        # emb[i,:] = w[i,:].dot(vectors) / np.count_nonzero(w[i,:])
        nonzeros = np.sum(m[i, :])
        emb[i, :] = np.divide(
            w[i, :].dot(vectors),
            np.sum(m[i, :]),
            out=np.zeros(dim),
            where=nonzeros != 0,
        )  # where there is a word
    del x
    del w
    gc.collect()
    return emb


def compute_pc(X, npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    print("Computing principal components...")
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    print("Removing principal component...")
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


def SIF_embedding(We, x, m, w, rmpc, dim):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params.rmpc: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = np.nan_to_num(get_weighted_average(We, x, m, w, dim))
    if rmpc > 0:
        emb = remove_pc(emb, rmpc)
    return emb


def generate_embeddings(docs, all_data, model, words2idx, dim, rmpc=1):
    """
    :param docs: list of strings (i.e. docs), based on which to do the tf-idf weighting.
    :param all_data: dataframe column / list of strings (all tweets)
    :param model: pretrained word vectors
    :param vocab: a dictionary, words['str'] is the indices of the word 'str'
    :param dim: dimension of embeddings
    :param rmpc: number of principal components to remove
    :return:
    """
    print(dim)

    print("Getting word weights...")
    word2weight = get_word_weights(docs)
    # load sentences
    print("Loading sentences...")
    x, m, w = sentences2idx(
        all_data, words2idx, word2weight
    )  # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    print("Creating embeddings...")
    return SIF_embedding(
        model, x, m, w, rmpc, dim
    )  # embedding[i,:] is the embedding for sentence i


if __name__ == "__main__":
    makedirs(SAVE_RESULTS_SUPERDIR, exist_ok=True)
    embeddings_save_path = join(SAVE_RESULTS_SUPERDIR, "sif_embeddings.pkl")

    tweets = load_pkl(SUBSET_PKL_PATH)
    if not exists(embeddings_save_path):
        vectors = pd.read_csv(EMB_PATH, sep="\t", index_col=0)
        vocab2idx = {w: i for i, w in enumerate(vectors.index.values)}

        all_texts = [" ".join(t["stems"]) for t in tweets]
        embeddings = generate_embeddings(
            all_texts, all_texts, vectors, vocab2idx, EMB_DIM
        )

        # normalize, so cosine distance is same as euclidean distance
        length = np.sqrt((embeddings ** 2).sum(axis=1)) + np.finfo(float).eps
        embeddings = embeddings / np.expand_dims(length, axis=-1)

        save_pkl(embeddings, embeddings_save_path)
    else:
        embeddings = load_pkl(embeddings_save_path)

    for num_cluster in NUM_CLUSTERS:
        print(">>", num_cluster, "clusters")
        save_results_dir = join(SAVE_RESULTS_SUPERDIR, f"{num_cluster}clusters")
        makedirs(save_results_dir, exist_ok=True)

        kmeans = KMeans(n_clusters=num_cluster, random_state=0)
        assigned_clusters = kmeans.fit_predict(embeddings)

        save_pkl(assigned_clusters, join(save_results_dir, "cluster_assignments.pkl"))

        # save 50 sample tweets per cluster
        cluster2texts = {i: [] for i in range(num_cluster)}
        for ci, tweet in zip(assigned_clusters, tweets):
            cluster2texts[ci].append(tweet["text"])
        for ci, texts in cluster2texts.items():
            shuffle(texts)
            texts = texts[:50]
            cluster2texts[ci] = texts
        save_json(cluster2texts, join(save_results_dir, "samples.json"))

        counts = OrderedDict(sorted(Counter(assigned_clusters.tolist()).items()))
        pprint(counts)
        save_json(counts, join(save_results_dir, "counts.json"))
