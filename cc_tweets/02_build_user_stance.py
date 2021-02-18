import json
from collections import Counter
from glob import glob
from os.path import exists, join

from config import DATA_DIR, RAW_DIR
from tqdm import tqdm

from cc_tweets.utils import ParallelHandler, load_pkl, read_txt_as_str_list, save_pkl


def get_data_from_raw_tweet(tweet):
    if "retweeted_status" in tweet:
        # do the 'dereference' if it's a retweet
        return tweet["retweeted_status"]
    else:
        return tweet


def get_userids_from_jsonl(jsonl_path):
    userids = []
    with open(jsonl_path) as f:
        lines = f.readlines()
    for line in lines:
        tweet = json.loads(line)
        tweet = get_data_from_raw_tweet(tweet)
        userid = tweet["user"]["id_str"]
        userids.append(userid)
    return userids


def get_all_userids_from_raw_tweets():
    all_jsonl_paths = sorted(glob(join(RAW_DIR, "tweets", "*.jsonl")))
    handler = ParallelHandler(get_userids_from_jsonl)
    userids = list(set(handler.run(all_jsonl_paths, flatten=True)))
    return userids


def load_profile2followers():
    def _load(profiles_path):
        profiles = set(read_txt_as_str_list(profiles_path))
        profile2followers = {}
        for profile in profiles:
            followers_path = join(DATA_DIR, "followers", f"{profile}.txt")
            if exists(followers_path):
                profile2followers[profile] = set(read_txt_as_str_list(followers_path))
        return profile2followers

    dem_p2fs = _load(join(DATA_DIR, "profiles", "profiles_dem_unordered.txt"))
    rep_p2fs = _load(join(DATA_DIR, "profiles", "profiles_rep_unordered.txt"))
    return dem_p2fs, rep_p2fs


def get_user_stance(dem_p2fs, rep_p2fs, userid):
    num_dem_following = 0
    for followers in dem_p2fs.values():
        if userid in followers:
            num_dem_following += 1
    num_rep_following = 0
    for followers in rep_p2fs.values():
        if userid in followers:
            num_rep_following += 1
    stance = "unk"
    if num_dem_following > num_rep_following:
        stance = "dem"
    if num_rep_following > num_dem_following:
        stance = "rep"
    return stance


if __name__ == "__main__":
    userid2stance_path = join(DATA_DIR, "userid2stance.pkl")

    if not exists(userid2stance_path):
        print("get userids")
        userids = get_all_userids_from_raw_tweets()
        print("load p2fs")
        dem_p2fs, rep_p2fs = load_profile2followers()

        print("build user stance")
        userid2stance = {}
        for userid in tqdm(userids):
            userid2stance[userid] = get_user_stance(dem_p2fs, rep_p2fs, userid)
        save_pkl(userid2stance, userid2stance_path)
    else:
        userid2stance = load_pkl(userid2stance_path)

    counter = Counter(userid2stance.values())
    print(counter)
