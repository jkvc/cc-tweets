# from os.path import join
# from pprint import pprint

# import matplotlib.pyplot as plt
# import numpy as np
# import scipy.sparse
# from config import DATA_DIR
# from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
# from tqdm import tqdm

# from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
# from cc_tweets.utils import load_json, load_pkl, read_txt_as_str_list, save_json

# if __name__ == "__main__":
#     feature_matrix = scipy.sparse.load_npz(join(DATASET_SAVE_DIR, "features.npz"))
#     feature_names = read_txt_as_str_list(join(DATASET_SAVE_DIR, f"feature_names.txt"))
#     feature_ids = read_txt_as_str_list(join(DATASET_SAVE_DIR, f"feature_ids.txt"))
#     id2idx = {id: i for i, id in enumerate(feature_ids)}
#     userid2numfollowers = load_json(join(DATA_DIR, "userid2numfollowers.json"))
#     mean_num_followers = sum(userid2numfollowers.values()) / len(userid2numfollowers)

#     engagement_stats = load_json(join(DATASET_SAVE_DIR, "71_engagement_stats.json"))

#     tweets = load_pkl(DATASET_PKL_PATH)
#     targets = np.zeros((len(tweets),))
#     for tweet in tweets:
#         if tweet["userid"] not in userid2numfollowers:
#             num_followers = mean_num_followers
#         else:
#             num_followers = userid2numfollowers[tweet["userid"]]

#         expected_num_followers_from_likes = (
#             tweet["likes"] / engagement_stats["median_likes_to_followers"]
#         )
#         expected_num_followers_from_retweets = (
#             tweet["retweets"] / engagement_stats["median_retweets_to_followers"]
#         )

#         target = (
#             1
#             if (
#                 (
#                     expected_num_followers_from_likes
#                     + expected_num_followers_from_retweets
#                 )
#                 / 2
#                 > num_followers
#             )
#             else 0
#         )
#         targets[id2idx[tweet["id"]]] = target

#     reg = LogisticRegression(fit_intercept=False, max_iter=5000)
#     reg.fit(feature_matrix, targets)

#     feature2coef = {}
#     for i, feature_name in enumerate(feature_names):
#         feature2coef[feature_name] = reg.coef_[0][i]

#     pprint(feature2coef)
#     save_json(feature2coef, join(DATASET_SAVE_DIR, "72_logreg_feature2coef.json"))

#     # non vocab features
#     non_vocab_feature2coef = {
#         k: v for k, v in feature2coef.items() if not k.startswith("_")
#     }
#     save_json(
#         non_vocab_feature2coef,
#         join(DATASET_SAVE_DIR, "72_logreg_non_vocab_feature2coef.json"),
#     )
#     non_vocab_feature2coef_sorted = {
#         feat: coef
#         for feat, coef in sorted(
#             [(k, v) for k, v in non_vocab_feature2coef.items()],
#             key=lambda x: x[1],
#             reverse=True,
#         )
#     }
#     save_json(
#         non_vocab_feature2coef_sorted,
#         join(DATASET_SAVE_DIR, "72_logreg_non_vocab_feature2coef_sorted.json"),
#     )

#     plt.clf()
#     fig, ax = plt.subplots(figsize=(7, 7))
#     y_pos = np.arange(len(non_vocab_feature2coef))
#     ax.barh(
#         y_pos,
#         non_vocab_feature2coef.values(),
#         align="center",
#     )
#     ax.yaxis.tick_right()
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(non_vocab_feature2coef.keys())
#     ax.set_title("non vocab feature coefs")
#     plt.subplots_adjust(left=0.1, right=0.6)
#     plt.savefig(join(DATASET_SAVE_DIR, "72_logreg_non_vocab_feature2coef.png"))
#     plt.clf()

#     plt.clf()
#     fig, ax = plt.subplots(figsize=(7, 7))
#     y_pos = np.arange(len(non_vocab_feature2coef_sorted))
#     ax.barh(
#         y_pos,
#         non_vocab_feature2coef_sorted.values(),
#         align="center",
#     )
#     ax.yaxis.tick_right()
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels(non_vocab_feature2coef_sorted.keys())
#     ax.set_title("non vocab feature coefs")
#     plt.subplots_adjust(left=0.1, right=0.6)
#     plt.savefig(join(DATASET_SAVE_DIR, "72_logreg_non_vocab_feature2coef_sorted.png"))
#     plt.clf()

#     # vocab features
#     vocab_feature2coef = {k: v for k, v in feature2coef.items() if k.startswith("_")}
#     save_json(
#         vocab_feature2coef,
#         join(DATASET_SAVE_DIR, "72_logreg_vocab_feature2coef.json"),
#     )
#     vocab_feature2coef_sorted = {
#         feat: coef
#         for feat, coef in sorted(
#             [(k, v) for k, v in vocab_feature2coef.items()],
#             key=lambda x: x[1],
#             reverse=True,
#         )
#     }
#     save_json(
#         vocab_feature2coef_sorted,
#         join(DATASET_SAVE_DIR, "72_logreg_vocab_feature2coef_sorted.json"),
#     )

#     # top vocab features
#     TOPN = 30

#     vocab_feature0coef_sorted = [(k, v) for k, v in vocab_feature2coef_sorted.items()]
#     vocab_feature0coef_sorted_top = vocab_feature0coef_sorted[:TOPN]
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(7, 7))
#     y_pos = np.arange(len(vocab_feature0coef_sorted_top))
#     ax.barh(
#         y_pos,
#         [v for _, v in vocab_feature0coef_sorted_top],
#         align="center",
#     )
#     ax.yaxis.tick_right()
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels([k for k, v in vocab_feature0coef_sorted_top])
#     ax.set_title("vocab feature coefs top")
#     plt.subplots_adjust(left=0.1, right=0.6)
#     plt.savefig(join(DATASET_SAVE_DIR, "72_logreg_vocab_feature2coef_sorted_top.png"))
#     plt.clf()

#     vocab_feature0coef_sorted_bot = vocab_feature0coef_sorted[-TOPN:]
#     plt.clf()
#     fig, ax = plt.subplots(figsize=(7, 7))
#     y_pos = np.arange(len(vocab_feature0coef_sorted_bot))
#     ax.barh(
#         y_pos,
#         [v for _, v in vocab_feature0coef_sorted_bot],
#         align="center",
#     )
#     ax.yaxis.tick_right()
#     ax.set_yticks(y_pos)
#     ax.set_yticklabels([k for k, v in vocab_feature0coef_sorted_bot])
#     ax.set_title("vocab feature coefs bottom")
#     plt.subplots_adjust(left=0.1, right=0.6)
#     plt.savefig(join(DATASET_SAVE_DIR, "72_logreg_vocab_feature2coef_sorted_bot.png"))
#     plt.clf()
