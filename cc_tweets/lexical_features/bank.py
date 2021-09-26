from dataclasses import dataclass, field
from os import makedirs
from os.path import join
from posixpath import dirname
from typing import Any, Callable, Dict, List, Optional

from cc_tweets.feature_utils import get_single_stat, get_stats
from cc_tweets.utils import load_pkl, save_json, save_pkl
from config import WORKING_DIR
from cc_tweets.experiment_configs import (
    DATA_SUBSET_SIZE,
    FILTER_UNK,
    SUBSET_PKL_PATH,
    SUBSET_WORKING_DIR,
)
from genericpath import exists


@dataclass
class Feature:
    name: str
    extractor: Callable[[List[Dict[str, Any]]], Dict[str, float]]

    def cache_features(
        self, tweets, return_feature_dict=False
    ) -> Optional[Dict[str, float]]:
        feature_cache_path = join(
            SUBSET_WORKING_DIR, "feature_cache", f"{self.name}.pkl"
        )
        makedirs(dirname(feature_cache_path), exist_ok=True)
        if exists(feature_cache_path):
            if return_feature_dict:
                return load_pkl(feature_cache_path)
            else:
                return None

        feature_dict = self.extractor(tweets)
        save_pkl(feature_dict, feature_cache_path)

        save_stats_path = join(SUBSET_WORKING_DIR, "feature_stats", f"{self.name}.json")
        makedirs(dirname(save_stats_path), exist_ok=True)
        stats = get_single_stat(tweets, feature_dict)
        save_json(stats, save_stats_path)

        return feature_dict

    def get_feature_dict(self, tweets) -> Dict[str, float]:
        return self.cache_features(tweets, return_feature_dict=True)


_FEATURES: Dict[str, Feature] = {}


def register_feature(feature: Feature):
    name = feature.name
    assert name not in _FEATURES
    _FEATURES[name] = feature


def get_all_feature_names():
    return sorted(list(_FEATURES.keys()))


def get_feature(name):
    assert name in _FEATURES
    return _FEATURES[name]
