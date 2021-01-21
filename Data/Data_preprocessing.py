import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.base import TransformerMixin ,BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
#from StockTwits_preprocesses import clean,clean2,split_count,remove_emoji,correct_words,correct_words_2,remove_emoji2,replace_emoji_text,emoji_message
from tensorflow.keras.preprocessing.sequence import pad_sequences
from statistics import variance,mean
from wordcloud import WordCloud, STOPWORDS
import datetime
import arrow

class DatasetExtractor(BaseEstimator, TransformerMixin):

    """Extractor class that loads multiple Tweet files and creates a single unified file."""

    def __init__(self,COMBINDED_DATASET,DATA_DIRECTORY):
        self.COMBINDED_DATASET=COMBINDED_DATASET
        self.DATA_DIRECTORY=DATA_DIRECTORY


    def transform(self):
        return self.hot_load()

    def hot_load(self):
        """Loads the pre-combined file if exists otherwise load all the files"""
        combined_file_path = f'{self.DATA_DIRECTORY}/{self.COMBINDED_DATASET}'
        print(combined_file_path)
        if os.path.isfile(combined_file_path):
            print('File Exists.Reloaded.')
            return pd.read_csv(combined_file_path, index_col=0)
        print('Loading Files..')
        combined_dataset = self.load_data()
        combined_dataset=combined_dataset.dropna()
        #combined_dataset=combined_dataset.drop_duplicates(subset=['id'])
        combined_dataset.to_csv(combined_file_path)
        return combined_dataset

    def load_data(self):
        """Loads multiple index  StockTwits files and returns a Single Pandas data frame"""
        combined_dataset = pd.DataFrame()
        for file_name in os.listdir(path=self.DATA_DIRECTORY):
            try:
                id = self.extract_id_new(file_name)
                df = pd.read_csv(f'{self.DATA_DIRECTORY}/{file_name}')
                df['id'] = id
                combined_dataset = combined_dataset.append(df, ignore_index=True)
            except :
                print('entra al ecxept')
                print(file_name)
                continue

        return combined_dataset

    def extract_id_new(self, file_name):
        """Helper method that extracts the Index name  from the file name"""
        file_name= file_name.split('.')[0]
        id=file_name.split('_')[0]
        return id

com_path = 'total_news.csv'
path = 'C:/Users/56979/PycharmProjects/TimeGAN/Data/DOWJONES/HD'
dataset = DatasetExtractor(DATA_DIRECTORY=path, COMBINDED_DATASET=com_path).transform()
dataset=dataset.dropna()

class DatasetGroup():
    def __init__(self,dataset):
        self.data=dataset

    def per_datetime(self):
        num_per_datetime = pd.DataFrame(self.data['datetime'].value_counts())
        num_per_datetime.reset_index(inplace=True)
        num_per_datetime.rename(columns={'index': 'datetime', "datetime": 'Cantidad de frases'}, inplace=True)
        sorted_values = num_per_datetime.sort_values(by=['datetime'])
        sorted_short = sorted_values.tail(10)
        return sorted_values,sorted_short
    def per_id(self):
        num_per_id = pd.DataFrame(self.data['id'].value_counts())
        num_per_id.reset_index(inplace=True)
        num_per_id.rename(columns={'index': 'ID', "id": 'Cantidad de frases'}, inplace=True)
        print(num_per_id)
        return num_per_id

    def per_sentiment(self):
        num_per_sentiment = pd.DataFrame(self.data['global_sentiment'].value_counts())
        num_per_sentiment.reset_index(inplace=True)
        #num_per_sentiment.rename(columns={'index': 'sentiment', "sentiment": 'Cantidad StockTwits'}, inplace=True)
        print(num_per_sentiment)
        return num_per_sentiment

    def muestra(self,dataset):
        for s in dataset.values:
            print(s)

    def sentiment_per_id(self):
        group=pd.DataFrame(self.data.groupby(['id']))
        total=pd.DataFrame()
        for g in group.values:
            num_per_sentiment = pd.DataFrame(g[1]['global_sentiment'].value_counts())
            num_per_sentiment.reset_index(inplace=True)
            #num_per_sentiment.rename(columns={'index': 'sentiment', "sentiment": 'Cantidad StockTwits'}, inplace=True)
            #num_per_sentiment['sentiment']=num_per_sentiment['sentiment'].apply(lambda x: x+g[0])
            total=total.append(num_per_sentiment)
        return total

    def sentiment_per_date(self):
        self.data['datetime'] = self.data['datetime'].apply(lambda x: arrow.get(x).date())
        self.data.rename(columns={'datetime':'date'},inplace=True)
        data_news=self.data.groupby(['date'])
        group_per_date=pd.DataFrame(columns=['datetime','avg_sentiment_mean','avg_sentiment_variance'])
        dates,avg_sentiment_mean,avg_sentiment_variance=[],[],[]
        for data in data_news:
            dates.append(data[0])
            #print(time)
            avg_sentiment=data[1]['avg_sentiment'].values
            print(avg_sentiment)
            avg_mean=mean(avg_sentiment)
            print(avg_mean)
            avg_sentiment_mean.append(avg_mean)
            if len(avg_sentiment)>1:
                avg_sentiment_variance.append(variance(avg_sentiment))
            else:
                avg_sentiment_variance.append(0.0)

        group_per_date['datetime']=dates
        group_per_date['avg_sentiment_mean']=avg_sentiment_mean
        group_per_date['avg_sentiment_variance']=avg_sentiment_variance
        path_save='C:/Users/56979/PycharmProjects/TimeGAN/Data/DOWJONES/HD/group_per_date.csv'
        group_per_date.to_csv(path_save)
        return group_per_date
        #return data_news

#groupi=DatasetGroup(dataset=dataset)
#sentiment_dataframe=groupi.sentiment_per_date()
news_path='C:/Users/56979/PycharmProjects/TimeGAN/Data/DOWJONES/HD/group_per_date.csv'
sentiment_dataframe=pd.read_csv(news_path)
#val=val[val['Cantidad de frases']==1]
price_path='C:/Users/56979/PycharmProjects/TimeGAN/Data/Prices/HD.csv'
price_dataframe=pd.read_csv(price_path)


def combine_price_and_sentiment(sentimentFrame, priceFrame):
    """
        receive sentimentFrame as (date, sentiment, message) indexed by date and sentiment
        and priceFrame as (Date, Opening Price, Closing Price, Volume) and return a combined
        frame as (sentiment_calculated_bullish, sentiment_calculated_bearish,
        sentiment_actual_previous, tweet_volume_change, cash_volume, label)
    """
    priceFrame['date']=priceFrame['datetime'].apply(lambda x: arrow.get(x).date())
    sentimentFrame['date']=sentimentFrame['datetime'].apply(lambda x: arrow.get(x).date())
    sentimentFrame=sentimentFrame.sort_values(by=['date'])
    priceFrame=priceFrame.sort_values(by=['date'])
    dataFrame = pd.DataFrame()
    dates,Open,Close,High,Low,Vol,AdjClose,avg_mean,avg_variance=[],[],[],[],[],[],[],[],[]
    for i in priceFrame.index:
        price_row=priceFrame.iloc[i]
        date=price_row['date']
        open = price_row['Open']
        close = price_row['Close']
        high = price_row['High']
        low = price_row['Low']
        vol = price_row['Volume']
        adjC = price_row['AdjClose']
        #print(open)
        Open.append(open)
        Close.append(close)
        High.append(high)
        Low.append(low)
        Vol.append(vol)
        AdjClose.append(adjC)
        dates.append(date)
        #print(price_date)
        sentiment_row=sentimentFrame[sentimentFrame['date'] == date]
        #print(price_row)
        if sentiment_row.empty:
            avg_mean.append(0)
            avg_variance.append(0)
        else:
            print('entra en el else')
            mean=sentiment_row['avg_sentiment_mean'].values[0]
            var=sentiment_row['avg_sentiment_variance'].values[0]
            print(mean)
            avg_mean.append(mean)
            avg_variance.append(var)

    dataFrame['Date']=dates
    dataFrame['Open']=Open
    dataFrame['High']=High
    dataFrame['Low']=Low
    dataFrame['Close']=Close
    dataFrame['Pct_change_raw']=avg_variance
    dataFrame['Compound_multiplied_row']=avg_mean
    dataFrame=dataFrame.set_index('Date')
    stock_name='HD'
    save_dir='C:/Users/56979/PycharmProjects/TimeGAN/input'
    file_name='results_{}_new.csv'.format(stock_name)
    save_file_path = f'{save_dir}/{file_name}'
    dataFrame.to_csv(save_file_path)
    return dataFrame



full_dataframe=combine_price_and_sentiment(sentimentFrame=sentiment_dataframe,priceFrame=price_dataframe)
print(full_dataframe)