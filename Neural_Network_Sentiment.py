
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
from autocorrect import Speller
spell = Speller(lang='en')
import nltk #NaturalLanguageToolKit
#nltk.download('punkt')
#nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier,accuracy
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
            #print(self.text.lower())
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
    feature_data = list(zip(words, y))
    return feature_data, words

def export_features(most_words,features):
    dict = {}
    for word in most_words:
        dict['contains({})'.format(word)] = (word in set(features))
    return dict

def main():
    x_train, x_test, y_train, y_test  = load_transform_df()

    train, single_words_train = feature_data(x_train, y_train)
    test, _ = feature_data(x_test, y_test)
    feature_extract = FreqDist(sum([word for word in single_words_train], []))
    #feature_extract = FreqDist(sum([word.split(' ') for word in x_train], []))
    most_used_words = list(feature_extract)[:400]

    train_set = [(export_features(most_used_words, data),y) for (data,y) in train]
    test_set = [(export_features(most_used_words, data),y) for (data,y) in test]

    #Model
    nb_classifier = NaiveBayesClassifier.train(train_set)

    print(accuracy(nb_classifier,test_set))


if __name__ == "__main__":
    main()



