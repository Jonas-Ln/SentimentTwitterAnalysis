# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import snscrape.modules.twitter as twitter
import os
import pickle
from Neural_Network import Neural_Network
cwd = os.getcwd()

filename_Features = '/Word_Features.txt'

filename_NN = '/Jonas_Neural_Net.txt'

filename_NB = '/NaiveBayes.txt'

def main():

    hashtag,origin,until,since = get_inputs()
    query = ((f'{hashtag}' if hashtag != '' else f'')
             + (f'(from:{origin})' if origin != '' else f'')
                + (f'until:{until}' if until != '' else f'')
                + (f'since:{since}' if since != '' else f'')
                + f'lang:en')

    print(query)
    tweets= []
    limit = 100
    for tweet in twitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date, tweet.user.username, tweet.rawContent])

    df = pd.DataFrame(tweets, columns=['Date','User','Text'])
    print(df)



def Classify_Tweet(x_test_NN):

    with open(cwd + filename_Features, 'rb') as f: #filename_Features is global_V
        load_features = pickle.load(f)

    with open(cwd + filename_NN, 'rb') as f: #filename_NN is global_V
        load_params = pickle.load(f)

    W1 = load_params.get('W1')
    b1 = load_params.get('b1')
    W2 = load_params.get('W2')
    b2 = load_params.get('b2')

    Neural_Network(len(load_features)).make_a_prediction(x_test_NN, W1, b1, W2, b2)

    return



def get_inputs():
    hashtag = []

    hashtag.append('#'+ input("Bitte Hashtag eingeben "))# Hier mehrere möglich da ein Thema N Hashtags haben kann
    origin = input("Hier einen bestimmten User eintragen: ")
    until = input("Ende des Zeitraums eingeben: ")
    since = input("Beginn des Zeitraums eingeben(YYYY-MM-DD): ")

    return hashtag,origin,until,since


if __name__ == "__main__":
    main()