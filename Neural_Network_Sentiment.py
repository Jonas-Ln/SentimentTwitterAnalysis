
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
from autocorrect import Speller(lang='en')
import nltk #NaturalLanguageToolKit
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer()

# Load Data:
DATA_PATH = '/Users/jonaslenz/pythonProject/SentimentTwitterAnalysis/Training_Data/training_1600000_processed_noemoticon.csv'

# Text Preprocessing as Class
# first option 1 Hot Embedding but too many Dimensions
# second option TF-IDF representation

class NLP_Transformer:
    def __init__(self,text):
        self.text = text

    #reduce length of long written words (not easy to interpret)
    def reduce_wordlength(self):
        return re.compile(r'(.)\1{2,}').sub(r'\1\1',self)

    def Preprocessing(self):
        self.lower()
        #Regex Replace extra signs @ # etc.
        self.re.sub(r'#[A-Za-z0-9_]+','')
        self.re.sub(r'@[A-Za-z0-9_]+','')
        #Remove Links
        self.re.sub(r'')


        self.remove_regex()
        self.remove_regex()
        self.remove_regex()
        self.remove_regex()

    def tokenizing(self):

    def lemmatize(self):

    def




def load_transform_df ():
    df = pd.read_csv(DATA_PATH,delimiter=',',encoding='ISO-8859-1', nrows=10)
    df.columns = ['Sentiment', 'id', 'date', 'query', 'user', 'text']
    #Nur relevante Spalten: y zu vorhersagende Variable = Sentiment, x Predictor = text
    df_for_Sentiment = df[['Sentiment', 'text']]
    return df_for_Sentiment

def transform_text_for_nlp


    print(df.head(5))



def main():
    load_transform_df()


if __name__ == "__main__":
    main()



