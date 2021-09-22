from os.path import exists, join
from time import sleep

from cc_tweets.credentials import AUTHS, Auth
from cc_tweets.utils import ParallelHandler, read_txt_as_str_list, write_str_list_as_txt
from config import DATA_DIR, RESOURCES_DIR
from twarc import Twarc


def is_complete(username):
    username = username.lower()
    return exists(join(DATA_DIR, "followers", f"{username}.txt"))


def scrape_follower(username: str, auth: Auth):
    try:
        t = Twarc(
            consumer_key=auth.consumer_key,
            consumer_secret=auth.consumer_secret,
            access_token=auth.access_token,
            access_token_secret=auth.access_token_secret,
        )
        followers = list(t.follower_ids(username))

        write_str_list_as_txt(followers, join(DATA_DIR, "followers", f"{username}.txt"))
        print(username)
    except KeyboardInterrupt:
        return
    except BaseException as e:
        print(e)
        pass
    sleep(1)


if __name__ == "__main__":
    profiles = read_txt_as_str_list(join(RESOURCES_DIR, "profiles", "profiles_all.txt"))
    print(len(profiles))
    profiles = [user.lower() for user in profiles if not is_complete(user)]
    print(len(profiles))

    params = [(profile, AUTHS[i % len(AUTHS)]) for i, profile in enumerate(profiles)]

    handler = ParallelHandler(scrape_follower)
    handler.run(params, num_procs=len(AUTHS))
