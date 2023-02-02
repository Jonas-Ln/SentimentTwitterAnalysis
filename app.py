


import pandas as pd
import numpy as np
import snscrape.modules.twitter as twitter
import os
import pickle
from Neural_Network import Neural_Network
from NLP_Preprocessing_Sentiment import NLP_Transformer
cwd = os.getcwd()

filename_Features = '/Word_Features.txt'

filename_NN = '/Jonas_Neural_Net.txt'

filename_NB = '/NaiveBayes.txt'

def streamlit_webpage():
    import streamlit as st
    import warnings

    warnings.filterwarnings("ignore")

    st.title("FOM - SentimentAnalysis on Twitter Data")
    st.markdown("MatrNr.: 605169 Name: Jonas Lenz", unsafe_allow_html=True)

    hashtag = []

    hashtag.append('#' + st.text_input("Bitte Hashtag eingeben: ", "#Python"))  # Hier mehrere möglich da ein Thema N Hashtags haben kann
    origin = st.text_input("Hier einen bestimmten User eintragen: ", "Username")
    until = st.text_input("Ende des Zeitraums eingeben: ", 'YYYY-MM-DD')
    since = st.text_input("Beginn des Zeitraums eingeben: " ,'YYYY-MM-DD')

    text_input = st.text_input("Hier Beispielhaft einen Text Klassifizieren: ",)

    query = ((f'{hashtag}' if hashtag != '' else f'')
                 + (f'(from:{origin})' if origin != '' else f'')
                    + (f'until:{until}' if until != '' else f'')
                    + (f'since:{since}' if since != '' else f'')
                    + f'lang:en')
    result = ""
    if st.button("Click here to Predict the Sentiment of the filtered Tweets"):
        tweets= []
        if text_input == '':
            limit = 1000
            for tweet in twitter.TwitterSearchScraper(query).get_items():
                if len(tweets) == limit:
                    break
                else:
                    tweets.append([tweet.date, tweet.user.username, tweet.rawContent])

            df = pd.DataFrame(tweets, columns=['Date','User','Text'])

            #get text values from tweets and normalize them
            tweets = df.Text.values

        else:
            tweets.append(text_input)

        tweets_norm = [text_preprocessing(tweet) for tweet in tweets]

        # load the features from the Classifier
        with open(cwd + filename_Features, 'rb') as f: #filename_Features is global_V
            load_features = pickle.load(f)

        # Compare Feature with each tweet and put it into a Matrix for NN
        feature_matrix = [(export_features(load_features, data)) for data in tweets_norm]
        # For input into Neural Network model
        feature_matrix_NN = np.array(feature_matrix).T

        y_pred, proba = Classify_Tweet(feature_matrix_NN,len(load_features))

        st.balloons()

        st.success(f'{proba} / Positive Tweets: {list(y_pred).count(1)} Negative Tweets: {list(y_pred).count(0)}')



def export_features(most_words,features):
    feature_array = []
    for i in range(0,len(most_words)):
        feature_array.append(1 if most_words[i] in features else 0)
    return feature_array

def text_preprocessing(x):
    # NLP - Textnormalisierung
        # Initialize as Class
        # regex Sonderzeichen
    text = NLP_Transformer(x).regex_replace_tweet()
        # tokens aus text # Long word Autocorrect, Lemmatize
    text = NLP_Transformer(text).word_to_stem()
    return text

def Classify_Tweet(classify_tweets_NN, feature_length):

    #Load model
    with open(cwd + filename_NN, 'rb') as f: #filename_NN is global_V
        load_params = pickle.load(f)

    W1 = load_params.get('W1')
    b1 = load_params.get('b1')
    W2 = load_params.get('W2')
    b2 = load_params.get('b2')

    y_pred, y_pred_proba = Neural_Network(feature_length).make_a_prediction(classify_tweets_NN, W1, b1, W2, b2)

    #Output string
    if sum(y_pred_proba[0])/len(y_pred_proba[0]) > 0.5:
        proba = f'Der Negative Tweet-Score beträgt: {round((sum(y_pred_proba[0])/len(y_pred_proba[0]))*100,4)}% Negativ'
    if sum(y_pred_proba[1])/len(y_pred_proba[1])> 0.5:
        proba = f'Der Positive Tweet-Score beträgt: {round((sum(y_pred_proba[1])/len(y_pred_proba[1]))*100,4)} % Positiv'

    return y_pred, proba

if __name__ == "__main__":
    streamlit_webpage()