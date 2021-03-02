import json
import os
from glob import glob
from os.path import exists, join

from config import RAW_DIR
from twarc import Twarc

from cc_tweets.credentials import AUTHS, Auth
from cc_tweets.utils import (
    ParallelHandler,
    mkdir_overwrite,
    read_txt_as_str_list,
    write_str_list_as_txt,
)

NUM_TWEETS_TO_SAMPLE = 5000000
SAVE_DIR = join(RAW_DIR, "random_tweets")


def sample(i, auth: Auth):
    t = Twarc(
        consumer_key=auth.consumer_key,
        consumer_secret=auth.consumer_secret,
        access_token=auth.access_token,
        access_token_secret=auth.access_token_secret,
    )
    save_path = join(SAVE_DIR, f"{i:02}.jsonl")

    num_sampled = 0
    with open(save_path, "w") as f:
        iter = t.sample()
        while True:
            tweet = next(iter)
            f.write(json.dumps(tweet))
            f.write("\n")
            num_sampled += 1
            if num_sampled > (NUM_TWEETS_TO_SAMPLE / len(AUTHS)):
                break


if __name__ == "__main__":
    mkdir_overwrite(SAVE_DIR)

    handler = ParallelHandler(sample)
    params = [(i, auth) for i, auth in enumerate(AUTHS)]
    handler.run(params, num_procs=len(AUTHS))
