
import pandas as pd
import mplfinance as mpf

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np




def prepare_data( seq_len=3,n_seq=5,plot_data=True, batch_size=1):
    # PREPARE DATA
    batch_size = batch_size
    import yfinance as yf

    ###obtengo valores de un stock  por minuto
    def get_yfinance_m(stock, period):
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        val = yf.Ticker(stock)
        val_historical = val.history(period=period, interval="1m")
        return val_historical[features]

    def get_database_data(SYMBOL, DIR_PATH='C:/Users/56979/PycharmProjects/TimeGAN/Data/Prices'):
        search_path = DIR_PATH + '/{}.csv'.format(SYMBOL)
        index_column = 1
        dataframe = pd.read_csv(search_path, index_col=index_column, parse_dates=True)
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = dataframe[features]
        data.index = pd.to_datetime(data.index, utc=True)
        return data

    # self.df=get_database_data(SYMBOL='AAPL')
    df = get_yfinance_m(stock='AAPL', period='1d')
    print(df.head(20))
    if plot_data:
        # candle_data=candle_format(data=self.df)
        mpf.plot(df, type='candle')

    ##Normalizan los datos
    #scaler = MinMaxScaler()
    #scaled_data = scaler.fit_transform(df).astype(np.float32)
    scaled_data=np.array(df)
    time_index = df.index.values
    print(time_index)
    time_data = []
    ##Dividen los datos en secuencias
    seq_len = seq_len
    data = []
    for i in range(len(df) - 2*seq_len):
        data.append(scaled_data[i:i + 2*seq_len])
        time_data.append(time_index[i:i + 2*seq_len])

    data_X=[]
    data_Z=[]
    for seq in data:
        data_Z.append(seq[:seq_len])
        data_X.append(seq[seq_len:])


    ##Numero de secuencias
    n_windows = len(data)
    print('n_windows:{}'.format(n_windows))

    ##Generador de data
    z_series = (tf.data.Dataset.from_tensor_slices(data_Z).batch(batch_size))
    z_series_iter = iter(z_series.repeat())
    x_series = (tf.data.Dataset.from_tensor_slices(data_X).batch(batch_size))
    x_series_iter = iter(x_series.repeat())

    return z_series_iter,x_series_iter,time_data


iterator_z,iterator_x,t_data=prepare_data()
first=next(iterator_z)
second=next(iterator_x)
print('first')
print(first)
print('time data')
print(t_data[0])
print('second')
print(second)

