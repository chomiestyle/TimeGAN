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

def get_sintetic_data(saved_path='C:/Users/56979/PycharmProjects/TimeGAN/tensorflow2_implementation/TimeGAN_future2_noise_past3000/experiment_00/synthetic_data'):
    #Recrea exactamente el mismo modelo solo desde el archivo
    new_model =load_model(saved_path)
    return new_model

import numpy as np
import datetime
def make_random_data():
    while True:
        yield np.random.uniform(low=0, high=1, size=(20, 5))


def prepare_data( seq_len=20,n_seq=5,plot_data=False, batch_size=5):
    # PREPARE DATA
    batch_size = batch_size
    import yfinance as yf


    ###obtengo valores de un stock  por minuto
    def get_yfinance_m(stock, period,n_values=None):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        val = yf.Ticker(stock)
        val_historical = val.history(period=period, interval="1m")
        if n_values!=None:
            val_historical = val_historical.tail(n_values)
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

    def get_database_data(SYMBOL, DIR_PATH='C:/Users/56979/PycharmProjects/TimeGAN/Data/Prices',n_values=None):
        search_path = DIR_PATH + '/{}.csv'.format(SYMBOL)
        index_column = 1
        dataframe = pd.read_csv(search_path, index_col=index_column, parse_dates=True)
        if n_values!=None:
            dataframe= dataframe.tail(n_values)
        dataframe.index = pd.to_datetime(dataframe.index, utc=True)
        dataframe['datetime'] = dataframe.index
        dataframe['datetime'] = dataframe['datetime'].apply(lambda x: set_datetime(today=x, type='price'))
        time_index = dataframe['datetime'].values
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = dataframe[features]

        return data,time_index

    # self.df=get_database_data(SYMBOL='AAPL')
    df,time_index = get_yfinance_m(stock='AAPL', period='1d',n_values=45)
    #saved_minute_path='C:/Users/56979/PycharmProjects/TimeGAN/Data/DOWJONES/1_minute'
    #df,time_index=get_database_data(SYMBOL='AAPL',DIR_PATH=saved_minute_path,n_values=45)
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
    def batch_noise_structure(n_batch,prev,seq_len):
        final_total=[]
        for i in range(n_batch):
            mean_1 = np.mean(prev[i, :, 0], dtype=np.float64)
            sigma_1 = np.std(prev[i, :, 0], dtype=np.float64)
            mean_2 = np.mean(prev[i, :, 1], dtype=np.float64)
            sigma_2 = np.std(prev[i, :, 1], dtype=np.float64)
            mean_3 = np.mean(prev[i, :, 2], dtype=np.float64)
            sigma_3 = np.std(prev[i, :, 2], dtype=np.float64)
            mean_4 = np.mean(prev[i, :, 3], dtype=np.float64)
            sigma_4 = np.std(prev[i, :, 3], dtype=np.float64)
            mean_5 = np.mean(prev[i, :, 4], dtype=np.float64)
            sigma_5 = np.std(prev[i, :, 4], dtype=np.float64)
            random_noise_1 = np.random.normal(loc=0.0, scale=sigma_1, size=(seq_len, 1))
            random_noise_2 = np.random.normal(loc=0.0, scale=sigma_2, size=(seq_len, 1))
            random_noise_3 = np.random.normal(loc=0.0, scale=sigma_3, size=(seq_len, 1))
            random_noise_4 = np.random.normal(loc=0.0, scale=sigma_4, size=(seq_len, 1))
            random_noise_5 = np.random.normal(loc=0.0, scale=sigma_5, size=(seq_len, 1))
            random_noise = np.append(random_noise_1, random_noise_2, axis=1)
            print(random_noise)
            random_noise2 = np.append(random_noise_3, random_noise_4, axis=1)
            # print(random_noise2)
            random_noise3 = np.append(random_noise, random_noise2, axis=1)
            # print(random_noise3)
            random_final = np.append(random_noise3, random_noise_5, axis=1)
            final_total.append(random_final)
        return np.array(final_total)

    ##Numero de secuencias
    n_windows = len(data)
    print('n_windows:{}'.format(n_windows))

    prev_series = (tf.data.Dataset.from_tensor_slices(data_Z).batch(batch_size))
    prev_series_iter = iter(prev_series.repeat())
    def make_noise_data():
        while True:
            prev_data=next(prev_series_iter)
            prev=np.array(prev_data)
            random_final=batch_noise_structure(n_batch=5,prev=prev,seq_len=20)
            print(random_final)
            noised_data=prev_data+random_final
            yield noised_data

    z_series_iter = iter(tf.data.Dataset
                              .from_generator(make_noise_data, output_types=tf.float32)
                              .repeat())

    #print(time_data)
    ##Generador de data

    x_series = (tf.data.Dataset.from_tensor_slices(data_X).batch(batch_size))
    x_series_iter = iter(x_series.repeat())

    return z_series_iter,x_series_iter,prev_series_iter,scaler,time_data

def seq_sinthetic_data(n,synthetic_data,z_iter):
    array=[]
    for i in range(n):
        Z_ = next(z_iter)
        d_1 = synthetic_data(Z_)
        generated_data_1 = np.array(d_1)
        array.append(generated_data_1)
    total_data=array[0]
    for data in array[1:]:
        total_data=total_data+data

    mean_data=total_data/len(array)
    return mean_data
def fib_sinthetic_data(n,synthetic_data,z_iter):
    input=next(z_iter)
    print('esta es el noised data')
    print(input)
    for i in range(n):
        input=synthetic_data(input)

    return np.array(input)

random_series = iter(tf.data.Dataset
                          .from_generator(make_random_data, output_types=tf.float32)
                          .batch(5)
                          .repeat())
n_seq=5
seq_len=20
z_iter,x_iter,prev_iter,scaler,t_data=prepare_data(seq_len=seq_len,n_seq=n_seq)
model_path_1= 'C:/Users/56979/PycharmProjects/TimeGAN/tensorflow2_implementation/TimeGAN_future2_noise_past13000/experiment_00/synthetic_data'
model_path_2= '/tensorflow2_implementation/past_noised_data/TimeGAN_future2_noise_past3000/experiment_00/synthetic_data'
model_path_3= '/tensorflow2_implementation/past_noised_data/TimeGAN_future2_noise_past3000/experiment_00/synthetic_data'
model_path_4= '/tensorflow2_implementation/past_noised_data/TimeGAN_future2_noise_past3000/experiment_00/synthetic_data'
synthetic_data_1=get_sintetic_data(saved_path=model_path_1)
#synthetic_data_2=get_sintetic_data(saved_path=model_path_2)
#synthetic_data_3=get_sintetic_data(saved_path=model_path_3)
#synthetic_data_4=get_sintetic_data(saved_path=model_path_4)
#synthetic_data=get_sintetic_data()
Z_ = next(prev_iter)

#print(Z_)

R_=next(x_iter)

#generated_data=seq_sinthetic_data(n=20,synthetic_data=synthetic_data_1,z_iter=z_iter)
generated_data=fib_sinthetic_data(n=1,synthetic_data=synthetic_data_1,z_iter=z_iter)
#Z_r1= next(z_iter)
#Z_r2=next(z_iter)
#Z_r3=next(z_iter)
#print('Este es el embebido')
#print(Z_)

#print('Este es el real')
#print(R_)
# d_1 = synthetic_data_1(Z_)
# d_2 = synthetic_data_2(Z_r1)
# d_3 = synthetic_data_3(Z_r2)
# d_4=synthetic_data_4(Z_r3)
# generated_data_1 = np.array(d_1)
# generated_data_2 = np.array(d_2)
# generated_data_3 = np.array(d_3)
# generated_data_4 = np.array(d_4)
# generated_gan_noised=(generated_data_2+generated_data_3+generated_data_4)/3
# generated_data=(generated_data_1+generated_gan_noised)/2
#generated_data=generated_data_1
#print(d)
prev_data=np.array(Z_)
real_data=np.array(R_)
#print(real_data.shape)


#print(generated_data.shape)
#print(len(generated_data))
generated_data = (scaler.inverse_transform(generated_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
real_data = (scaler.inverse_transform(real_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))
prev_data = (scaler.inverse_transform(prev_data.reshape(-1,n_seq)).reshape(-1, seq_len, n_seq))



for i in range(len(generated_data)):
    #print(' sequence:{}'.format(i))
    generated=generated_data[i,:,:]
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
    generated = np.append(generated[:, :-1]+delta_price , generated[:, 4:]+delta_vol, axis=1)
    print(generated)
    generated = np.append(prev, generated, axis=0)
    if i<=len(real_data)-1:
        real=real_data[i,:,:]

        real=np.append(prev,real,axis=0)

        arr=np.array(t_data[i])
        didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
        gen_df = pd.DataFrame(data=generated, index=didx, columns=["Open", "High","Low","Close","Volume"])
        #print(gen_df)
        real_df=pd.DataFrame(data=real, index=didx, columns=["Open", "High","Low","Close","Volume"])
        #mpf.plot(real_dat, type='candle',mav=(3,6,9),volume=True)
        fig = mpf.figure(style='yahoo',figsize=(14,16))
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        av1 = fig.add_subplot(3,2,5,sharex=ax1)
        av2 = fig.add_subplot(3,2,6,sharex=ax2)

        mpf.plot(gen_df,type='candle',ax=ax1,volume=av1,mav=(10,20),axtitle='generated data')
        mpf.plot(real_df,type='candle',ax=ax2,volume=av2,mav=(10,20),axtitle='real data')

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



