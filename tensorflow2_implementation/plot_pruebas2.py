import pandas as pd
import mplfinance as mpf
from tensorflow.keras.models import load_model
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from yahoo_fin import *
from yahoo_fin.stock_info import *
import arrow

def set_datetime(today,type):
    #utc = pytz.utc
    #print('este es today')
    #print(today)
    if type=='stocktwits':
        #print('stocktwits')
        utc_datetime = arrow.get(today)
        utc_datetime=utc_datetime.to('local').to('utc')
        #print(utc_datetime.utcoffset())
        #today=today
    #tz = pytz.timezone('America/New_York')
    #today.replace(tzinfo=tz)
    #timestamp = tz.localize(today, is_dst=True)
    elif type=='price':
        #print('prices')
        utc_datetime = arrow.get(today).shift(hours=-5)
        #print(utc_datetime.utcoffset())
        #date_time_obj = utc_datetime.to('America/New_York')
    date_time_obj = utc_datetime.to('America/Santiago')
    #print(date_time_obj.datetime)
    today=date_time_obj.datetime
    tt = today.timetuple()
    date_val = datetime.datetime(year=tt.tm_year, month=tt.tm_mon, day=tt.tm_mday, hour=tt.tm_hour, minute=tt.tm_min)
    return date_val
        ###obtengo valores de un stock  por minuto
def get_yfinance_m(stock, period):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        val = yf.Ticker(stock)
        val_historical = val.history(period=period, interval="1m")
        return val_historical[features]


def get_database_data(SYMBOL ,DIR_PATH = 'C:/Users/56979/PycharmProjects/TimeGAN/Data/Prices'):
    search_path =DIR_PATH +'/{}.csv'.format(SYMBOL)
    index_column =0
    dataframe =pd.read_csv(search_path ,index_col=index_column ,parse_dates=True)
    features =['Open','High','Low','Close','Volume']
    data =dataframe[features]
    data.index = pd.to_datetime(data.index ,utc=True)
    return data

def get_sintetic_data(saved_path='C:/Users/56979/PycharmProjects/TimeGAN/tensorflow2_implementation/TimeGAN900/experiment_00/synthetic_data'):
    #Recrea exactamente el mismo modelo solo desde el archivo
    new_model =load_model(saved_path)
    return new_model

import numpy as np
import datetime
def make_random_data():
    while True:
        yield np.random.uniform(low=0, high=1, size=(60, 5))

def prepare_data( seq_len=20,n_seq=5,plot_data=False, batch_size=5):
    # PREPARE DATA
    batch_size = batch_size
    import yfinance as yf


    ###obtengo valores de un stock  por minuto
    def get_yfinance_m(stock, period):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        val = yf.Ticker(stock)
        val_historical = val.history(period=period, interval="1m")
        val_historical = val_historical.tail(45)
        val_historical['datetime'] = val_historical.index
        val_historical['datetime'] = val_historical['datetime'].apply(lambda x: set_datetime(today=x, type='price'))
        time_index = val_historical['datetime'].values
        return val_historical[features],time_index

    ###obtengo valores de un stock  por dia
    def get_yfinance_d(stock_name):
        features = ['open', 'high', 'low', 'close', 'volume']
        # val = yf.Ticker(stock_name)
        # val_historical = val.history(period=max)
        val_historical = get_data(stock_name)
        val_historical['datetime']=val_historical.index
        val_historical['datetime'] = val_historical['datetime'].apply(lambda x: set_datetime(today=x, type='price'))
        time_index = val_historical['datetime'].values
        # print(val_historical)
        return val_historical[features],time_index

    def get_database_data(SYMBOL, DIR_PATH='C:/Users/56979/PycharmProjects/TimeGAN/Data/Prices'):
        search_path = DIR_PATH + '/{}.csv'.format(SYMBOL)
        index_column = 1
        dataframe = pd.read_csv(search_path, index_col=index_column, parse_dates=True)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = dataframe[features]
        data.index = pd.to_datetime(data.index, utc=True)
        return data

    # self.df=get_database_data(SYMBOL='AAPL')
    df,time_index = get_yfinance_m(stock='AAPL', period='1d')

    print(df.tail(5))
    #df=get_yfinance_d(stock_name='AAPL')
    #df=df.tail(180)
    #print(df.head(20))
    if plot_data:
        # candle_data=candle_format(data=self.df)
        mpf.plot(df, type='candle')

    ##Normalizan los datos
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df).astype(np.float32)
    time_data = []
    ##Dividen los datos en secuencias
    seq_len = seq_len
    data = []
    for i in range(len(df) - 2 * seq_len):
        data.append(scaled_data[i:i + 2 * seq_len])
        time_data.append(time_index[i:i + 2 * seq_len])

    data_X = []
    data_Z = []
    for seq in data:
        data_Z.append(seq[:seq_len])
        data_X.append(seq[seq_len:])

    ##Numero de secuencias
    n_windows = len(data)
    print('n_windows:{}'.format(n_windows))

    #print(time_data)
    ##Generador de data
    z_series = (tf.data.Dataset.from_tensor_slices(data_Z).batch(batch_size))
    z_series_iter = iter(z_series.repeat())
    x_series = (tf.data.Dataset.from_tensor_slices(data_X).batch(batch_size))
    x_series_iter = iter(x_series.repeat())

    return z_series_iter,x_series_iter,scaler,time_data

# dir_path='C:/Users/56979/PycharmProjects/TimeGAN/tensorflow2_implementation/TimeGAN90'
# gen_dat=get_database_data(SYMBOL='AAPL',DIR_PATH=dir_path)
#mpf.plot(gen_dat, type='candle',mav=(3,6,9),volume=True)
#df=get_yfinance_m(stock='AAPL',period='1d')

#
# ##Normalizan los datos
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(df).astype(np.float32)
# print(scaled_data)
# ##Dividen los datos en secuencias
# seq_len=60
# n_seq=5
# data = []
# print(len(df) - seq_len)
# for i in range(len(df) - seq_len):
#     data.append(scaled_data[i:i + seq_len])
#
# ##Numero de secuencias
# n_windows = len(data)
# print(n_windows)
# ##Generador de data
# real_series = (tf.data.Dataset
#                        .from_tensor_slices(data)
#                        .shuffle(buffer_size=n_windows)
#                        .batch(60))
# real_series_iter = iter(real_series.repeat())
#
random_series = iter(tf.data.Dataset
                          .from_generator(make_random_data, output_types=tf.float32)
                          .batch(5)
                          .repeat())
n_seq=5
seq_len=20
z_iter,x_iter,scaler,t_data=prepare_data(seq_len=seq_len,n_seq=n_seq)
synthetic_data=get_sintetic_data()
Z_ = next(z_iter)
Z_r= next(random_series)
#print('Este es el embebido')
#print(Z_)
R_=next(x_iter)
#print('Este es el real')
#print(R_)
d = synthetic_data(Z_r)
d_2=synthetic_data(Z_r)
#print(d)
prev_data=np.array(Z_)
real_data=np.array(R_)
#print(real_data.shape)
generated_data = np.array(d)
generated_data_2=np.array(d_2)
#print(generated_data.shape)
#print(len(generated_data))
generated_data = (scaler.inverse_transform(generated_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
generated_data_2 = (scaler.inverse_transform(generated_data_2.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
real_data = (scaler.inverse_transform(real_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
prev_data = (scaler.inverse_transform(prev_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))



for i in range(len(generated_data)):
    #print(' sequence:{}'.format(i))
    generated=generated_data[i,:,:]
    generated_2=generated_data_2[i,:,:]
    prev = prev_data[i, :, :]
    ###suma delta
    first_gen = generated[0]
    print('fist gen:')
    print(first_gen)
    last_prev = prev[-1]
    print('last prev:')
    print(last_prev)
    delta_price = last_prev[3] - first_gen[0]
    delta_vol = last_prev[-1] - first_gen[-1]
    print(delta_vol)
    # generated= np.stack(generated)
    # print(generated)
    ###a cada valor de precio (velas japonesas), se le suma el delta
    # generated_prices=generated[:,:-1]+delta_price
    generated = np.append(generated[:, :-1] + delta_price, generated[:, 4:], axis=1)
    generated_2=np.append(generated_2[:, :-1] + delta_price, generated_2[:, 4:], axis=1)
    print(generated_2)
    generated = np.append(prev, generated, axis=0)
    generated_2=np.append(prev, generated_2, axis=0)
    if i<=len(real_data)-1:
        real=real_data[i,:,:]

        real=np.append(prev,real,axis=0)

        arr=np.array(t_data[i])
        didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
        gen_df = pd.DataFrame(data=generated, index=didx, columns=["Open", "High","Low","Close","Volume"])
        gen2_df = pd.DataFrame(data=generated, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
        #print(gen_df)
        real_df=pd.DataFrame(data=real, index=didx, columns=["Open", "High","Low","Close","Volume"])
        #mpf.plot(real_dat, type='candle',mav=(3,6,9),volume=True)
        fig = mpf.figure(style='yahoo',figsize=(14,16))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        av1 = fig.add_subplot(3,2,5,sharex=ax1)
        av2 = fig.add_subplot(3,2,6,sharex=ax2)

        mpf.plot(gen_df,type='candle',ax=ax1,volume=av1,mav=(10,20),axtitle='generated data')
        mpf.plot(gen2_df,type='candle',ax=ax2,volume=av2,mav=(10,20),axtitle='generated 2 data')

        mpf.show()
    else:
        arr = np.array(t_data[i])
        didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
        gen_df = pd.DataFrame(data=generated, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
        # print(gen_df)
        #real_df = pd.DataFrame(data=real, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
        # mpf.plot(real_dat, type='candle',mav=(3,6,9),volume=True)
        fig = mpf.figure(style='yahoo', figsize=(14, 16))
        ax1 = fig.add_subplot(2, 2, 1)
        #ax2 = fig.add_subplot(2, 2, 2)
        av1 = fig.add_subplot(2, 2, 2, sharex=ax1)
        #av2 = fig.add_subplot(3, 2, 6, sharex=ax2)

        mpf.plot(gen_df, type='candle', ax=ax1, volume=av1, mav=(10, 20), axtitle='generated data')
        #mpf.plot(real_df, type='candle', ax=ax2, volume=av2, mav=(10, 20), axtitle='real data')

        mpf.show()
    #mpf.plot(df, type='candle',volume=True)