import os
from glob import glob
from os.path import basename, exists, join

from config import RAW_DIR

from cc_tweets.utils import write_str_list_as_txt

TWARC_HYDRATE_CMD_BASE = "twarc hydrate {} --log {} > {}"
LOG_PATH = join(RAW_DIR, "cc_tweets", "log.txt")

if __name__ == "__main__":
    tweet_ids_json_paths = sorted(glob(join(RAW_DIR, "ids", "*txt")))

    for p in tweet_ids_json_paths:
        if exists(join(RAW_DIR, "_completed", basename(p))):
            print("skip", basename(p))
            continue

        save_filename = basename(p).replace(".txt", ".jsonl")
        print("-- hydrate", save_filename)
        cmd = TWARC_HYDRATE_CMD_BASE.format(
            p,
            LOG_PATH,
            join(RAW_DIR, "tweets", save_filename),
        )

        assert os.system(cmd) == 0
        print("-- done   ", save_filename)

        write_str_list_as_txt(
            ["yay"],
            join(RAW_DIR, "_completed", basename(p)),
        )
