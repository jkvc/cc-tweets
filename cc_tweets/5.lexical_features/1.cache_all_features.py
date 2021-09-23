from cc_tweets.lexical_features.bank import get_all_feature_names, get_feature
from cc_tweets.utils import load_pkl
from experiment_configs.base import SUBSET_PKL_PATH

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)

    for name in get_all_feature_names():
        print(name)
        feature = get_feature(name)
        feature.cache_features(tweets)
