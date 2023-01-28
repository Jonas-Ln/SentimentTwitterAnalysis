
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


# Load Data:
DATA_PATH = '/Users/jonaslenz/pythonProject/SentimentTwitterAnalysis/Training_Data/training_1600000_processed_noemoticon.csv'

class twitter_text_for_Nlp():

    def remove_regex(self):
        self.



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



