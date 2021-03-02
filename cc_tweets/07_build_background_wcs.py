# import json
# from collections import defaultdict
# from glob import glob
# from os.path import join

# from config import RAW_DIR
# from twarc import Twarc

# from cc_tweets.credentials import AUTHS, Auth
# from cc_tweets.data_utils import get_ngrams, parse_raw_tweet
# from cc_tweets.utils import ParallelHandler, mkdir_overwrite, read_txt_as_str_list

# SAVE_DIR = join(RAW_DIR, "random_tweets")


# MIN_COUNT = 50


# def get_wcs_from_jsonl(jsonl_path):
#     unigram_wcs = defaultdict(int)
#     bigram_wcs = defaultdict(int)

#     lines = read_txt_as_str_list(jsonl_path)
#     for line in lines[:100]:
#         tweet = json.loads(line)
#         tweet = parse_raw_tweet(tweet)
#         if tweet is None:
#             continue

#         print("!")
#         lemmas = tweet["lemmas"]
#         for lemma in lemmas:
#             unigram_wcs[lemma] += 1
#         for bigram in get_ngrams(lemmas, 2):
#             bigram_wcs[bigram] += 1

#     return unigram_wcs, bigram_wcs


# if __name__ == "__main__":
#     jsonl_paths = sorted(glob(join(SAVE_DIR, "*jsonl")))

#     handler = ParallelHandler(get_wcs_from_jsonl)
#     results = handler.run(jsonl_paths[:5])
#     print(results)

#     all_unigram_wcs = defaultdict(int)
#     all_bigram_wcs = defaultdict(int)
#     for unigram_wcs, bigram_wcs in results:
#         for gram, count in unigram_wcs.items():
#             all_unigram_wcs[gram] += count
#         for gram, count in bigram_wcs.items():
#             all_bigram_wcs[gram] += count
#     filtered_unigram_wcs = {
#         gram: count for gram, count in all_unigram_wcs.items() if count >= MIN_COUNT
#     }
#     filtered_bigram_wcs = {
#         gram: count for gram, count in all_bigram_wcs.items() if count >= MIN_COUNT
#     }

#     print(len(filtered_unigram_wcs), len(filtered_bigram_wcs))
