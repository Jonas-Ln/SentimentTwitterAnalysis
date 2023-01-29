
# Dataset Source: https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset
# Prelabeled Data of Tweets from Kaggle to Train the Model 1,6 Mio Rows
# Emoticons are Removed Prior --> Automatically Classified
# Language: English

# Data Preprocessing:
# Columns: 6 Columns
# 1 - Label: Negativ = 0 Neutral = 2 Positive = 4
# 2 - the id of the tweet
# 3 - the date of the tweet Format: (Sat May 16 23:58:44 UTC 2009)
# 4 - the query (lyx). If there is no query, then this value is NO_QUERY. --> Contains no Query
# 5 - the user that tweeted
# 6 - the text of the tweet

# Format ISO-8859-1 ANSI delimiter ','
import pandas as pd
import re
import collections
from autocorrect import Speller
spell = Speller(lang='en')
import nltk #NaturalLanguageToolKit
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, PrecisionRecallDisplay,RocCurveDisplay
import matplotlib.pyplot as plt
from nltk.classify import NaiveBayesClassifier,accuracy
from nltk.metrics.scores import (f_measure,precision,recall)
import tqdm
lemma = WordNetLemmatizer()

# Load Data:
DATA_PATH = '/Users/jonaslenz/pythonProject/SentimentTwitterAnalysis/Training_Data/training_1600000_processed_noemoticon.csv'

# Text Preprocessing as Class
# first option 1 Hot Embedding but too many Dimensions
# second option TF-IDF representation

class NLP_Transformer:
    def __init__(self, text):
        self.text = text

    def regex_replace_tweet(self: str) -> str:
        # 1-2 Regex Replace extra signs @ # etc.
        # 3-4 Remove Links
        # 5 remove numbers
        # 6 remove Sonderzeichen except . ! ?
        regex_list = [r'#[A-Za-z0-9_]+',r'@[A-Za-z0-9_]+',r'http\S+',r'www. \S+',r'[0-9]',r'/[^A-Za-z0-9\!\?\.]/',r"'",r'-']
        for regex in regex_list:
            self.text = re.sub(regex,'',self.text)
        return self.text.lower()

    def word_to_stem(self: str) -> str:
        # split into words
        self.text = word_tokenize(self.text)
        tweet = []
        output = ' '
        for word in self.text:
            if len(word) > 2:
                # Reduce length of long written words (not easy to interpret)
                _ = re.compile(r'(.)\1{2,}')
                word = _.sub(r'\1\1', word)
                word = spell(word)
                # get word stem
                word = lemma.lemmatize(word)
                tweet.append(word)
                # join together for output
        #output = output.join(tweet)

        return tweet


def load_transform_df ():
    df = pd.read_csv(DATA_PATH,delimiter=',',encoding='ISO-8859-1')

    df.columns = ['Sentiment', 'id', 'date', 'query', 'user', 'text']
    df = df[['Sentiment','text']].sample(n=100000, random_state=0)
    # Change Sentiment 0 negative and 1 Positive
    x = df.text.values
    y = df.Sentiment.replace(4,1)
    # Split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

    return x_train, x_test, y_train, y_test

def text_preprocessing(x):
    # NLP - Textnormalisierung
        # Initialize as Class
        # regex Sonderzeichen
    text = NLP_Transformer(x).regex_replace_tweet()
        # tokens aus text # Long word Autocorrect, Lemmatize
    text = NLP_Transformer(text).word_to_stem()
    return text

def feature_data(x, y):
    words = [text_preprocessing(row) for row in x]
    y = list(y)
    return words, y

def export_features(most_words,features):
    feature_array = []
    for i in range(0,len(most_words)):
        feature_array.append(1 if most_words[i] in features else 0)
    return feature_array

def main():
    x_train, x_test, y_train, y_test  = load_transform_df()

    x_train_new, y_train = feature_data(x_train, y_train)
    x_test_new, y_test = feature_data(x_test, y_test)
    feature_extract = FreqDist(sum([word for word in x_train_new], []))
    #feature_extract = FreqDist(sum([word.split(' ') for word in x_train], []))
    print(len(feature_extract))
    most_used_words = list(feature_extract)[:10000]
    #df_muw = pd.DataFrame(, columns= most_used_words)

    x_train_feature = [(export_features(most_used_words, data)) for data in x_train_new]
    x_test_feature = [(export_features(most_used_words, data)) for data in x_test_new]


    # Export test and train Data

    #Model for test
    model = MultinomialNB()
    model.fit(x_train_feature,y_train)

    y_pred = model.predict(x_test_feature)
    y_pred_proba = model.predict_proba(x_test_feature)
    #y_dec = model.decision_function(x_test_feature)
    print(y_pred)
    print(y_pred_proba)
    #decision_function
    print(accuracy_score(y_pred,y_test))
    print(precision_score(y_pred, y_test, average="macro"))
    print(recall_score(y_pred, y_test, average="macro"))
    print(f1_score(y_pred, y_test, average="macro"))

    PrecisionRecallDisplay.from_predictions(y_test, y_pred_proba[:, 1],color="darkorange").plot()
    RocCurveDisplay.from_predictions(y_test,y_pred_proba[:, 1],color="darkorange",)
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.show()


    #print(precision_recall_fscore_support(test_set, nb_classifier, average='weighted'))

if __name__ == "__main__":
    main()



