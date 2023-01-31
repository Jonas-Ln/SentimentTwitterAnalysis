
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
import os




lemma = WordNetLemmatizer()

cwd = os.getcwd()

#Output Model & Fearues
filename_Features = 'Word_Features.csv'

filename_NN = 'Jonas_Neural_Net.sav'

filename_NB = 'NaiveBayes.sav'

# Load Data:
DATA_PATH = cwd + '/training_1600000_processed_noemoticon.csv'



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
                if (len(word) > 2 or word == '..') and word not in ['the','and']:
                    tweet.append(word)
                # join together for output
        #output = output.join(tweet)

        return tweet


import numpy as np


# Input of x_train and test_data
class Neural_Network:
    def __init__(self):
        np.random.seed(2)


    def forward_propagation(self, W1, b1, W2, b2, X):
        Z1 = W1.dot(X) + b1
        A1 = self.ReLU(Z1)
        Z2 = W2.dot(A1) + b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def back_propagation(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        m = Y.size
        #encode_y
        one_h_endoding = np.zeros((Y.max()+1, m))
        one_h_endoding[Y, np.arange(m)] = 1
        #encode_y = np.zeros((Y.size, Y.max()+1))
        #enocde_y[np.arange(Y.size),Y] = 1
        dZ2 = 2 * (A2 - one_h_endoding)
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2,1)
        dZ1 = W2.T.dot(dZ2) * self.derivative_ReLU(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1,1)
        return dW1, db1, dW2, db2

    def derivative_ReLU(self, Z):
        return Z > 0

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp = np.exp(Z-np.max(Z))
        return exp / exp.sum(axis=0)

    def initialize_parameters(self):
        W1 = np.random.normal(size = (10, 2000)) * np.sqrt(1./2000)
        b1 = np.random.normal(size = (10, 1)) * np.sqrt(1./10)
        W2 = np.random.normal(size = (2, 10)) * np.sqrt(1./12)
        b2 = np.random.normal(size = (2, 1)) * np.sqrt(1./2000)
        #print(W1, b1, W2, b2)
        return W1, b1, W2, b2

    def update_parameters(self, W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * np.reshape(db1,(10,1))
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * np.reshape(db2,(2,1))

        return W1, b1, W2, b2

    def gradient_descent(self, X, Y, iterations, alpha):
        size, m = X.shape
        W1, b1, W2, b2 = self.initialize_parameters()
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_propagation(W1, b1, W2, b2, X)
            dW1, db1, dW2, db2 = self.back_propagation(Z1, A1, Z2, A2,W1, W2, X, Y)
            W1, b1, W2, b2 = self.update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                print("Iteration: ", i)
                print("Accuracy ", self.get_accuracy(self.get_prediction(A2), Y))
                print(self.get_prediction(A2), Y)

        return W1, b1, W2, b2

    def get_prediction(self, A2):
        return np.argmax(A2, 0)


    def get_accuracy(self, prediction, Y):
        return np.sum(prediction == Y) / Y.size

    def make_a_prediction(self, X, W1, b1, W2, b2):
        _,_,_, A2 = self.forward_propagation(W1,b1,W2,b2,X)
        predictions = self.get_prediction(A2)
        return predictions, self.softmax(A2)
    '''
    def test_prediction(self, index, W1, b1, W2, b2):
        datensatz = X_train[:, index, None]'''

def load_transform_df ():
    df = pd.read_csv(DATA_PATH,delimiter=',',encoding='ISO-8859-1')

    df.columns = ['Sentiment', 'id', 'date', 'query', 'user', 'text']
    df = df[['Sentiment','text']].sample(n=6000, random_state=0)

# Wenn daten ungleich gebalanced evtl aufslpitten zu 50/50:
    #df_neg = df[df['Sentiment']== 0].sample(n=50, random_state=0)
    #df_pos = df[df['Sentiment']== 4].sample(n=50, random_state=0)
    #df = pd.concat([df_pos, df_neg])
    #df = df.sample(frac=1)

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

    most_used_words = list(feature_extract)[:2000]
    #df_muw = pd.DataFrame(, columns= most_used_words)
    print(most_used_words)

    x_train_feature = [(export_features(most_used_words, data)) for data in x_train_new]
    x_test_feature = [(export_features(most_used_words, data)) for data in x_test_new]

    x_train_NN = np.array(x_train_feature).T
    y_train_NN = np.array(y_train)

    x_test_NN = np.array(x_test_feature).T
    y_test_NN = np.array(y_test)


    W1, b1, W2, b2 = Neural_Network().gradient_descent(x_train_NN, y_train_NN, 1500, 0.2)

    # Test Prediction:
    y_pred, y_pred_proba = Neural_Network().make_a_prediction(x_test_NN, W1, b1, W2, b2)

    accuracy = Neural_Network().get_accuracy(y_pred, y_test_NN)

    print(accuracy)
    # Export test and train Data

    print(precision_score(y_pred, y_test_NN, average="macro"))
    print(recall_score(y_pred, y_test_NN, average="macro"))
    print(f1_score(y_pred, y_test_NN, average="macro"))

    print(y_pred_proba)
    y_pred_proba = y_pred_proba[1]
    print(y_pred)
    print(y_pred_proba)
    print(y_test_NN)
    print(1-sum(y_pred_proba)/len(y_pred_proba)) #Prob that the Whole Dataset is Negative
    PrecisionRecallDisplay.from_predictions(y_test_NN, y_pred_proba, color="blue").plot()
    RocCurveDisplay.from_predictions(y_test_NN, y_pred_proba, color="blue", )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.show()
'''
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
'''




    #print(precision_recall_fscore_support(test_set, nb_classifier, average='weighted'))

if __name__ == "__main__":
    main()



