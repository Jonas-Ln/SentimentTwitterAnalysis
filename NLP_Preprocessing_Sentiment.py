# Made by Jonas Lenz

# Dataset Source: https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset
# Prelabeled Data of Tweets from Kaggle to Train the Model 1 Mio Rows
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
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, PrecisionRecallDisplay,RocCurveDisplay
import matplotlib.pyplot as plt
import os
cwd = os.getcwd()
import numpy as np
import pickle
# Own Model
from Neural_Network import Neural_Network

# Initialize Object for later use
lemma = WordNetLemmatizer()



#Output Model & Fearues
filename_Features = '/Word_Features.txt'

filename_NN = '/Jonas_Neural_Net.txt'

filename_NB = '/NaiveBayes.txt'

# Load Data:
DATA_PATH = cwd + '/Training_Data/training_1600000_processed_noemoticon.csv'



# Text Preprocessing as Class
# First option: 1 Hot Embedding but too many Dimensions
# Second option: TF-IDF representation used

class NLP_Transformer:
    def __init__(self, text):
        self.text = text

    def regex_replace_tweet(self: str) -> str:
        # 1-2 Regex Replace extra signs @ # etc.
        # 3-4 Remove Links
        # 5 remove numbers
        # 6 + 7 remove Sonderzeichen except . ! ?
        regex_list = [r'#[A-Za-z0-9_]+',r'@[A-Za-z0-9_]+',r'http\S+',r'www. \S+',r'[0-9]',r'/[^A-Za-z0-9\!\?\.]/',r"'",r'-']
        for regex in regex_list:
            self.text = re.sub(regex,'',self.text)

        return self.text.lower()

    def word_to_stem(self: str) -> str:
        # split into words
        self.text = word_tokenize(self.text)
        tweet = []
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

        return tweet


# Here begin the Definition of the Functions

# At the Bottom there is the Main() Function who will make use of all the others
def load_transform_df ():

    # load the Dataset
    df = pd.read_csv(DATA_PATH,delimiter=',',encoding='ISO-8859-1')

    # Get the Dimensions of the Data into a DataFrame
    df.columns = ['Sentiment', 'id', 'date', 'query', 'user', 'text']
    # 500.000 Tweets Sampled for this Model
    df = df[['Sentiment','text']].sample(n=500000, random_state=0)

    # Incase for a need Split the Data into a 50/50 Split:
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
    # Regex Sonderzeichen replacement
    text = NLP_Transformer(x).regex_replace_tweet()
    # tokens from text -> Long word Autocorrect -> Extracting the Word Stem from the Normalized word
    text = NLP_Transformer(text).word_to_stem()

    # -> Text Normalzed
    return text

def feature_data(x, y):
    # Preprocess each row in the descriptive Data given by X
    words = [text_preprocessing(row) for row in x]
    y = list(y)
    return words, y

def export_features(most_words,features):
    # 1 if the Word from the Features is in the Tweet and 0 if not
    feature_array = []
    for i in range(0,len(most_words)):
        feature_array.append(1 if most_words[i] in features else 0)
    return feature_array


def main():
    # Number of Dimensions:
    feature_size = 3500

    # Get Train and Test Sets from Function
    x_train, x_test, y_train, y_test  = load_transform_df()

    # Extract all Word Stems from all Rows in the Data
    x_train_new, y_train = feature_data(x_train, y_train)
    x_test_new, y_test = feature_data(x_test, y_test)

    # TF-IDF - Term Frequency Inverse Document Frequency
    # Using the Frequency of Terms to classify them as Relevant
    # (Faster than including every Word (Ca. 80.000))
    feature_extract = FreqDist(sum([word for word in x_train_new], []))

    most_used_words = list(feature_extract)[:feature_size]
    #df_muw = pd.DataFrame(, columns= most_used_words)
    feature_size = len(most_used_words)
    print(most_used_words)

    # Transform all Text from train and test data into Features from the TF-ID List
    x_train_feature = [(export_features(most_used_words, data)) for data in x_train_new]
    x_test_feature = [(export_features(most_used_words, data)) for data in x_test_new]

    # Shape the Train set so that Matrix operations can be applied on them
    x_train_NN = np.array(x_train_feature).T
    y_train_NN = np.array(y_train)

    # Shape the Test set so that Matrix operations can be applied on them
    x_test_NN = np.array(x_test_feature).T
    y_test_NN = np.array(y_test)

    # Train the Model from Class Neural Network
    W1, b1, W2, b2 = Neural_Network(feature_size).gradient_descent(x_train_NN, y_train_NN, 2000, 0.2)

    # Test Prediction:
    y_pred, y_pred_proba = Neural_Network(feature_size).make_a_prediction(x_test_NN, W1, b1, W2, b2)

    accuracy = Neural_Network(feature_size).get_accuracy(y_pred, y_test_NN)

    print(accuracy)

    # All KPIs
    # Macro accounts for label imbalance
    # Precision
    print(precision_score(y_pred, y_test_NN, average="macro"))
    # Recall
    print(recall_score(y_pred, y_test_NN, average="macro"))
    # F1-Score
    print(f1_score(y_pred, y_test_NN, average="macro"))

    # Display Measures in a Plot
    PrecisionRecallDisplay.from_predictions(y_test_NN, y_pred_proba, color="blue").plot()
    RocCurveDisplay.from_predictions(y_test_NN, y_pred_proba, color="blue", )
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.show()

    # Make a file for the Features or overwrite the current
    if (os.path.exists(cwd + filename_Features)): #filename_NN is global_V
        os.remove(cwd + filename_Features)
    else:
        print("File Does Not Exists")
    with open(cwd + filename_Features, "wb") as f:
        pickle.dump(most_used_words,f)

    # Make a file for the Model Parameters or overwrite the current
    dict_params= {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}
    if (os.path.exists(cwd + filename_NN)): #filename_NN is global_V
        os.remove(cwd + filename_NN)
    else:
        print("File Does Not Exists")
    with open(cwd + filename_NN, "wb") as f:
        pickle.dump(dict_params,f)

############################################################################
    # Took 48 Hours to Train this Neural Network with the above Parameter


#Optional for Naivebayes Classifier:
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

if __name__ == "__main__":
    main()



