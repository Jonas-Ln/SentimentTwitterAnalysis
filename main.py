# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI
from collections import Counter

from twython import TwythonStreamer

API_KEY = 'iuxe5bKOyRn7ADhApwrm099sF'
API_SECRET_KEY = 'iATtcNWUrUU0ZKJsNDX1aqJCeiSFvilOZPti7Z2n8h1BTKBQHv'
BEARER_TOKEN = 'AAAAAAAAAAAAAAAAAAAAAHXzlQEAAAAANkQ3N%2FYaOriXbxxnltspRk05ViA%3DqraNJ7TnVaAhvrKzgtbXCib79VTN5ktf8qzsVSIiK3xtJWfGYd'
ACCCESS_TOKEN = '841711430000861186-mE2Mayr8D0qVE9ZGKwgDBCnvJ78EOn0'
ACCCESS_TOKEN_SECRET = 'behr8EbbOCMDub22qs19Fv6v7ze5Pac96VJ044QL1AUoN'
#OAUTH_TOKEN = 'liddpgAAAAABlfN1AAABhfYm3U0'
#OAUTH_TOKEN_SECRET = 'LvoCYSpMWKy3a4Cib8Eg62aQxazFDGeT'
tweets = []

class Twitter_Streamer(TwythonStreamer):
    def on_success (self, data):
        # Was tun mit den Daten von Twitter ?
        if data.get('lang') == 'ger':
            tweets.append(data)
            print(f"received tweet #{len(tweets)}")
        #Genug gesammelt:
        if len(tweets) >= 100:
            self.disconnect()

    def on_error (self, status_code, data,headers=None):
        print(status_code,data)
        self.disconnect()


def mai():

    stream = Twitter_Streamer(API_KEY,API_SECRET_KEY,ACCCESS_TOKEN,ACCCESS_TOKEN_SECRET)
    #stream.statuses.filter(track='data')
    stream.statuses.filter(track='twitter')
    #stream.statuses.sample()
    top_hashtags = Counter(hastag['text'].lower() for tweet in tweets
                           for hastag in tweet['entities']['hashtags'])
    print(top_hashtags.most_common(10))
    print(tweets)
import requests
import os
import json

# To set your enviornment variables in your terminal run the following line:
# export 'BEARER_TOKEN'='<your_bearer_token>'
#bearer_token = os.environ.get("BEARER_TOKEN")


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r


def get_rules():
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream/rules", auth=bearer_oauth
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))
    return response.json()


def delete_all_rules(rules):
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload
    )
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    print(json.dumps(response.json()))


def set_rules(delete):
    # You can adjust the rules if needed
    sample_rules = [
        #{"value": "data", "tag": "data"},
        {"value": "data-science", "tag": "data-science"},
    ]
    payload = {"add": sample_rules}
    response = requests.post(
        "https://api.twitter.com/2/tweets/search/stream/rules",
        auth=bearer_oauth,
        json=payload,
    )
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    print(json.dumps(response.json()))


def get_stream(set):
    response = requests.get(
        "https://api.twitter.com/2/tweets/search/stream", auth=bearer_oauth, stream=True,
    )
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    for response_line in response.iter_lines():
        if response_line:
            json_response = json.loads(response_line)
            print(json.dumps(json_response, indent=4, sort_keys=True))


def main():
    rules = get_rules()
    delete = delete_all_rules(rules)
    set = set_rules(delete)
    get_stream(set)


if __name__ == "__main__":
    main()