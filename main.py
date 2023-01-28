# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import snscrape.modules.twitter as twitter


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



def get_inputs():
    hashtag = []

    hashtag.append('#'+ input("Bitte Hashtag eingeben "))# Hier mehrere möglich da ein Thema N Hashtags haben kann
    origin = input("Hier einen bestimmten User eintragen: ")
    until = input("Ende des Zeitraums eingeben: ")
    since = input("Beginn des Zeitraums eingeben: ")

    return hashtag,origin,until,since


if __name__ == "__main__":
    main()