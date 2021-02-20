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

NUM_ID_PER_FILE = 10000


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def hydrate_chunk(id_chunk_path, save_path, auth: Auth):
    complete_mark = save_path + ".complete"
    if exists(complete_mark):
        print(">> skip", save_path)
        return

    ids = read_txt_as_str_list(id_chunk_path)

    t = Twarc(
        consumer_key=auth.consumer_key,
        consumer_secret=auth.consumer_secret,
        access_token=auth.access_token,
        access_token_secret=auth.access_token_secret,
    )
    tweets = t.hydrate(ids)

    with open(save_path, "w") as f:
        for tweet in tweets:
            f.write(json.dumps(tweet))
            f.write("\n")

    write_str_list_as_txt(["."], complete_mark)


if __name__ == "__main__":

    if not exists(join(RAW_DIR, "tweet_ids")):
        all_ids = []
        raw_id_paths = glob(join(RAW_DIR, "raw_ids", "climate_id.txt.*"))
        for p in raw_id_paths:
            all_ids.extend(read_txt_as_str_list(p))
        print(len(all_ids))

        id_chunks = list(chunks(all_ids, NUM_ID_PER_FILE))
        mkdir_overwrite(join(RAW_DIR, "tweet_ids"))
        for i, chunk in enumerate(id_chunks):
            write_str_list_as_txt(chunk, join(RAW_DIR, "tweet_ids", f"{i:04}.txt"))

    if not exists(join(RAW_DIR, "tweets")):
        os.mkdir(join(RAW_DIR, "tweets"))

    tweet_ids_json_paths = sorted(glob(join(RAW_DIR, "tweet_ids", "*txt")))
    params = []
    for i, p in enumerate(tweet_ids_json_paths):
        params.append(
            (
                p,
                join(RAW_DIR, "tweets", f"{i:04}.jsonl"),
                AUTHS[i % len(AUTHS)],
            )
        )

    handler = ParallelHandler(hydrate_chunk)
    handler.run(params, num_procs=len(AUTHS))

    # hydrate_chunk(
    #     join(RAW_DIR, "tweet_ids", "000.txt"),
    #     join(RAW_DIR, "tweets", "000.jsonl"),
    #     AUTHS[0],
    # )

    # tweet_ids_json_paths = sorted(glob(join(RAW_DIR, "ids", "*txt")))

    # for p in tweet_ids_json_paths:
    #     if exists(join(RAW_DIR, "_completed", basename(p))):
    #         print("skip", basename(p))
    #         continue

    #     save_filename = basename(p).replace(".txt", ".jsonl")
    #     print("-- hydrate", save_filename)
    #     cmd = TWARC_HYDRATE_CMD_BASE.format(
    #         p,
    #         LOG_PATH,
    #         join(RAW_DIR, "tweets", save_filename),
    #     )

    #     assert os.system(cmd) == 0
    #     print("-- done   ", save_filename)

    #     write_str_list_as_txt(
    #         ["yay"],
    #         join(RAW_DIR, "_completed", basename(p)),
    #     )
