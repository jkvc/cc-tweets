from os.path import exists, join
from time import sleep

from config import DATA_DIR
from twarc import Twarc

from cc_tweets.credentials import AUTHS, Auth
from cc_tweets.utils import (
    ParallelHandler,
    load_pkl,
    read_txt_as_str_list,
    save_json,
    write_str_list_as_txt,
)


def count_followers(userids, auth):
    t = Twarc(
        consumer_key=auth.consumer_key,
        consumer_secret=auth.consumer_secret,
        access_token=auth.access_token,
        access_token_secret=auth.access_token_secret,
    )
    profiles = list(t.user_lookup(userids))
    userid0numfollowers = [(p["id"], p["followers_count"]) for p in profiles]
    return userid0numfollowers


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    userid2stance = load_pkl(join(DATA_DIR, "userid2stance.pkl"))
    userids = list(userid2stance.keys())

    userid_chunks = list(chunks(userids, 100))
    params = [(chunk, AUTHS[i % len(AUTHS)]) for i, chunk in enumerate(userid_chunks)]
    handler = ParallelHandler(count_followers)
    userid0numfollowers = handler.run(params, num_procs=len(AUTHS), flatten=True)

    userid2numfollowers = {
        userid: numfollowers for userid, numfollowers in userid0numfollowers
    }
    save_json(userid2numfollowers, join(DATA_DIR, "userid2numfollowers.json"))
