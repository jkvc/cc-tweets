from dataclasses import dataclass
from os.path import exists, join
from time import sleep

from config import DATA_DIR
from twarc import Twarc

from cc_tweets.utils import ParallelHandler, read_txt_as_str_list, write_str_list_as_txt


@dataclass
class Auth:
    consumer_key: str
    consumer_secret: str
    access_token: str
    access_token_secret: str


AUTHS = []
# kevinehc
AUTHS.append(
    Auth(
        consumer_key="f9snXUToTChcwIlBFfHXl71Ik",
        consumer_secret="9NmQEQUMlufTouONasnd6AbxnuGZIKYUsZQvK6MqWXQbY82RXf",
        access_token="1322734672430862336-Wf1tB2JXpIzPJt0WmvhMDrBFjl1wF2",
        access_token_secret="LCGQOZ8q2EPFjQDif6Kt1DtLypeAXYnTpwvlXn9rZpCui",
    )
)
AUTHS.append(
    Auth(
        consumer_key="M8djP1OPoPQo0YbKQHBe78S9N",
        consumer_secret="5mtyNRMmdnAtdIClZQJJDSw6oAXcyIxXPkLdZ6DpXQA0IkyF3y",
        access_token="1322734672430862336-27qRIRlpuGc4pD9holU1QMvyOXCNjh",
        access_token_secret="155qCZ8ZW9UKWVfwfmLYwVvQ7PkvcDTG9aVlWmq2qSzgf",
    )
)
AUTHS.append(
    Auth(
        consumer_key="aLpPKPqkLTi34FP08csFr5dm6",
        consumer_secret="FzjN5N7VVhc6P8Iukul2hoGRahu09UqaKY3cu8Z3MGyzKPmYVF",
        access_token="1322734672430862336-gQ2CrATrpRghqd6Wtxs0c6754QJgAO",
        access_token_secret="imUc6lEfJuQBnYM8N7kEu5TWs0YDKm2vRuYWXsy4xGVWi",
    )
)
AUTHS.append(
    Auth(
        consumer_key="zQ9ikPHurxjcdSIyCKrZeyor0",
        consumer_secret="BLsDnUP4xV3CsxeJIBcLjICW9lgFJBKpIJNsJcNBPf9rWcAYqU",
        access_token="1322734672430862336-e2KNHr0DwKjpMLpWQ26xLwhR6gpxlv",
        access_token_secret="foujS2lSMd3F1j1w8WKjpvydLZo0VKKuo9k1JXGwmfxGX",
    )
)
AUTHS.append(
    Auth(
        consumer_key="O6VCnEOrgPIEqy6Qhuhh6OBpK",
        consumer_secret="ChaP4fe9O1zHUCIhGRG2rHGFvGp4CMgI7QtI2Gpoz7Iy8nW0oi",
        access_token="1322734672430862336-I4ySBmVxIWEK38XeiATNMdZgUs0nnR",
        access_token_secret="7oDuXWDwtP97eTZWyMR6IgYrghwKXJh6eXbDiViNzgXRP",
    )
)
AUTHS.append(
    Auth(
        consumer_key="nrpnwN1fwujyqM9uwcC24Ff8i",
        consumer_secret="bnX6OP5h4xHBm2URw6ZOl5RT4W6po5wSFqW7yckFFR2sClTMyN",
        access_token="1322734672430862336-pSvG34m8y0vXERDIOyOWxgKEPXipwP",
        access_token_secret="FYh1OxO5SUHYp0oSjRm3sRidH7oX39jGdBBKUs002pd7l",
    )
)
AUTHS.append(
    Auth(
        consumer_key="mdBwRuYPADf1dZt6oWTOuSLwY",
        consumer_secret="cPyJk3pW8AtEn7G7HkFyn9WmRlgSS3lMzFzWgaChYsc1dVPaXm",
        access_token="1322734672430862336-8bR1NUd0MJcbJj3Rcj1PZ3sOBTpTuY",
        access_token_secret="F6RfGfusA6VTbG46r8DudMj22tQFtsspfh6PUSni5v52l",
    )
)
AUTHS.append(
    Auth(
        consumer_key="8ifPVVOraO4EbW6PEVRUMHHVT",
        consumer_secret="yhHzbpjjtzHLrkqZFHmhlbUqn8nCNAuYPSFqCsjJwCwYYtwbKs",
        access_token="1322734672430862336-3fsGRwPyiSxUdhAlrufKqWsw7HUV2V",
        access_token_secret="Ngs7XgwRFpJhyaEuKYUmezFfPXk77caOPNuKxz1RxC7vS",
    )
)
AUTHS.append(
    Auth(
        consumer_key="W7xb4DxtwfOVuxqgatm2g3kTu",
        consumer_secret="xipZ4kYUTembaTquj9ROKIB5UwFfehKdrJOVnTyA0qY44msqLK",
        access_token="1322734672430862336-i7WZROBmdO1K2yxXjsmXMC381DUexa",
        access_token_secret="u6FPxkORJiwjwJwjeorz9JuBwAwUcvxbz3ZtswP6gr9cy",
    )
)
AUTHS.append(
    Auth(
        consumer_key="wi1rSbeI0WjZvYIuBUSgN0M8e",
        consumer_secret="AAWGkzGWVolh2rfWlSYDazSWbwSRRVrj7TItcHwxXUglOomPhB",
        access_token="1322734672430862336-3qAuWUlcIDfcll5BTg1BHWrWVhOFBe",
        access_token_secret="IXcQhyGlfONyDE6rWt4p7nJ8gA4bowYpwHADf1YLxLeUN",
    )
)
# kevina760
AUTHS.append(
    Auth(
        consumer_key="M9y1EgPVuJTi06WPAZzoTCHXW",
        consumer_secret="JQnXlkGOnfK0u7xzU8npkFPXNDt7WhNx8a7IPxFO1y0zvRgcLm",
        access_token="2705233580-ZKR0BFGICXBMoH8V4iqR36MgphE1XG9wiZxWGYo",
        access_token_secret="NRdTSb6jh82R8pfZtUG35EHqhlkkiXKY4ZO6HlEui67vj",
    )
)
AUTHS.append(
    Auth(
        consumer_key="EYPwLApEvRN2TltaYILdJ5HNV",
        consumer_secret="kjD9nH4aRyFc3eugR1IDxB27QSJWK1TVUktoGR0ImohPgKUnoI",
        access_token="2705233580-NZD68a0wJfrjUTgyUVfGKaBnDesAVavwyMxDcYm",
        access_token_secret="ASWRguUWvrpb5oEfRYDqQI6q89ZrD6R2NGnYuCe5n03QO",
    )
)
AUTHS.append(
    Auth(
        consumer_key="fIiYEOj8Kyu3lOnq14PLy3hYp",
        consumer_secret="8UlTBUmkxngSeiRDXzNSU900HAzyNSXxIdq1DMtzUA1AQICheC",
        access_token="2705233580-62r4BAYo78IDPzHZrvN7l0iUXawRFmzb9kwk5ZE",
        access_token_secret="miwVxpnbmBSzgvNak3MeBUAZbUh1b3rU315cIUlwM7M95",
    )
)
AUTHS.append(
    Auth(
        consumer_key="veJXGIHvf7pAvHRV4M6gOtmYw",
        consumer_secret="kRCzmAYOMe4QqGAWV2QtuYGggQTnXLUJJk1ZWXpEwMxw4xKIq9",
        access_token="2705233580-EqSIlz61lxEtorDRe3MiybQ901tfivQzUb1Prj4",
        access_token_secret="9lMzWv4x9co2FtmSCFfwvqtjLitrqmFXhl18DeazvcPox",
    )
)
AUTHS.append(
    Auth(
        consumer_key="WX2tPHgeJz7yaaZd2VZOS0Vo7",
        consumer_secret="kHWT3tv4cYPywNAzF1ZzX7vPit1GRJrz68GMpNQR0ZYxnIXDXv",
        access_token="2705233580-gOx7wysaMXqg8WBeIEEo0E7MgIuozbkmbeNwNYx",
        access_token_secret="BTr7noMrHsCmk25YsGw4DmrxEKegxIePt29m8E7eCApAo",
    )
)
AUTHS.append(
    Auth(
        consumer_key="C5PYhZGN1QdD2kxjebrUHCJGU",
        consumer_secret="Bn2E2NhtgE5oOAPnKGETKErzlUbwcB7HkhBXIlvrCppJwTjlaX",
        access_token="2705233580-Ihl4QAxBqlcqlAwTf5sQQaKkeHm0oq92e8M5Zj3",
        access_token_secret="Q9M2jVs8zJ1ri6nr7LwLdZV6LoVdSJ0KEG0iD0NdX0dzi",
    )
)


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
    profiles = read_txt_as_str_list(join(DATA_DIR, "profiles", "profiles_all.txt"))
    print(len(profiles))
    profiles = [user.lower() for user in profiles if not is_complete(user)]
    print(len(profiles))

    params = [(profile, AUTHS[i % len(AUTHS)]) for i, profile in enumerate(profiles)]

    handler = ParallelHandler(scrape_follower)
    handler.run(params, num_procs=len(AUTHS))
