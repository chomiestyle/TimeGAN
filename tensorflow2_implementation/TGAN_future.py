import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, RNN, GRUCell, Input
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import matplotlib.dates as mdates

#PREPARE DATA
# seq_len = 24
# n_seq = 6
# batch_size = 128
# ##Design parameters
# hidden_dim = 24
# num_layers = 3

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

class TGAN():

    def __init__(self,hidden_dim=20,num_layer=3,seq_len=20,n_seq=5,train_step=1,save_directory='time_gan'):
        self.hidden_dim=hidden_dim
        self.num_layers=num_layer
        self.train_steps = train_step
        self.seq_len=seq_len
        self.n_seq=n_seq
        self.results_path = Path(save_directory)
        if not self.results_path.exists():
            self.results_path.mkdir()
        experiment = 0
        self.log_dir = self.results_path / f'experiment_{experiment:02}'
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        self.hdf_store = self.results_path / 'TimeSeriesGAN.h5'
        self.writer = tf.summary.create_file_writer(self.log_dir.as_posix())

    def create_model(self):
        ##create_model
        self.X = Input(shape=[self.seq_len, self.n_seq], name='RealData')
        self.Z = Input(shape=[self.seq_len, self.n_seq], name='RandomData')

        def make_rnn(n_layers, hidden_units, output_units, name):
            return Sequential([GRU(units=hidden_units,
                                   return_sequences=True,
                                   name=f'GRU_{i + 1}') for i in range(n_layers)] +
                              [Dense(units=output_units,
                                     activation='sigmoid',
                                     name='OUT')], name=name)

        self.embedder = make_rnn(n_layers=3,
                                 hidden_units=self.hidden_dim,
                                 output_units=self.hidden_dim,
                                 name='Embedder')

        self.recovery = make_rnn(n_layers=3,
                                 hidden_units=self.hidden_dim,
                                 output_units=self.n_seq,
                                 name='Recovery')

        self.generator = make_rnn(n_layers=3,
                                  hidden_units=self.hidden_dim,
                                  output_units=self.hidden_dim,
                                  name='Generator')

        self.discriminator = make_rnn(n_layers=3,
                                      hidden_units=self.hidden_dim,
                                      output_units=1,
                                      name='Discriminator')

        self.supervisor = make_rnn(n_layers=2,
                                   hidden_units=self.hidden_dim,
                                   output_units=self.hidden_dim,
                                   name='Supervisor')

        self.gamma = 1

        ##Generic loss functions
        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()

        ##Fase 1 Entrenamiento de los autoencoder
        self.H = self.embedder(self.X)
        X_tilde = self.recovery(self.H)

        self.autoencoder = Model(inputs=self.X,
                                 outputs=X_tilde,
                                 name='Autoencoder')

        ##Autoencoder Optimization
        steps_decay = self.train_steps/3
        lr_fn_autoencoder = tf.optimizers.schedules.PolynomialDecay(1e-3, steps_decay, 1e-5, 2)
        self.autoencoder_optimizer = Adam(lr_fn_autoencoder)
        ##Optimizadores
        lr_fn_generator = tf.optimizers.schedules.PolynomialDecay(1e-3, steps_decay, 1e-5, 2)
        self.generator_optimizer = Adam(lr_fn_generator)
        lr_fn_discriminator = tf.optimizers.schedules.PolynomialDecay(1e-3, steps_decay, 1e-5, 2)
        self.discriminator_optimizer = Adam(lr_fn_discriminator)
        lr_fn_embedding = tf.optimizers.schedules.PolynomialDecay(1e-3, steps_decay, 1e-5, 2)
        self.embedding_optimizer = Adam(lr_fn_embedding)
        lr_fn_supervisor = tf.optimizers.schedules.PolynomialDecay(1e-3, steps_decay, 1e-5, 2)
        self.supervisor_optimizer = Adam(lr_fn_supervisor)


        #opt = tf.optimizers.Adam(lr_fn)

    def training_loop(self):
        ##Prepare data
        self.prepare_data()
        ##Create_model
        self.create_model()
        ##Training Encoder
        self.training_autoencoder()
        ###Fase 2 Entrenamiento supervisado
        self.training_supervisor()
        ###joint training
        self.joint_training()
        ##Plot data
        #self.plot_data()

    def prepare_data(self,plot_data=False,batch_size=512):
        # PREPARE DATA
        self.batch_size=batch_size
        import yfinance as yf

        ###obtengo valores de un stock  por minuto
        def get_yfinance_m(stock, period):
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            val = yf.Ticker(stock)
            val_historical = val.history(period=period, interval="1m")
            return val_historical[features]

        def get_database_data(SYMBOL, DIR_PATH='C:/Users/56979/PycharmProjects/TimeGAN/Data/Prices', n_values=None):
            search_path = DIR_PATH + '/{}.csv'.format(SYMBOL)
            index_column = 1
            dataframe = pd.read_csv(search_path, index_col=index_column, parse_dates=True)
            if n_values != None:
                dataframe = dataframe.tail(n_values)
            dataframe.index = pd.to_datetime(dataframe.index, utc=True)
            dataframe['datetime'] = dataframe.index
            dataframe['datetime'] = dataframe['datetime'].apply(lambda x: set_datetime(today=x, type='price'))
            time_index = dataframe['datetime'].values
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            data = dataframe[features]

            return data, time_index



        #self.df=get_database_data(SYMBOL='AAPL')
        #self.df =get_yfinance_m(stock='AAPL',period='7d')
        saved_minute_path = 'C:/Users/56979/PycharmProjects/TimeGAN/Data/DOWJONES/1_minute'
        self.df,time_array=get_database_data(SYMBOL='AAPL',DIR_PATH=saved_minute_path)
        #print(self.df)
        if plot_data:
            #candle_data=candle_format(data=self.df)
            mpf.plot(self.df, type='candle')


        ##Normalizan los datos
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(self.df).astype(np.float32)
        ##Dividen los datos en secuencias
        seq_len=self.seq_len
        n_seq=self.n_seq
        time_index = self.df.index.values
        #print(time_index)
        time_data = []
        ##Dividen los datos en secuencias
        seq_len = seq_len
        data = []
        for i in range(len(self.df) - 2 * seq_len):
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
        prev_series = (tf.data.Dataset.from_tensor_slices(data_Z).batch(batch_size))
        prev_series_iter = iter(prev_series.repeat())

        def batch_noise_structure(n_batch, prev, seq_len):
            final_total = []
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
                #print(random_noise)
                random_noise2 = np.append(random_noise_3, random_noise_4, axis=1)
                # print(random_noise2)
                random_noise3 = np.append(random_noise, random_noise2, axis=1)
                # print(random_noise3)
                random_final = np.append(random_noise3, random_noise_5, axis=1)
                final_total.append(random_final)
            return np.array(final_total)

        def make_noise_data():
            while True:
                prev_data = next(prev_series_iter)
                prev = np.array(prev_data)
                random_final = batch_noise_structure(n_batch=len(prev), prev=prev, seq_len=seq_len)
                #print(random_final)
                noised_data = prev_data + random_final
                yield noised_data
        self.individual_series=iter(tf.data.Dataset.from_tensor_slices(data_X).batch(batch_size).repeat())

        self.random_series = iter(tf.data.Dataset
                             .from_generator(make_noise_data, output_types=tf.float32)
                             .repeat())

        ##Generador de data
        #z_series = (tf.data.Dataset.from_tensor_slices(data_Z).batch(batch_size))
        #self.random_series = iter(z_series.repeat())
        x_series = (tf.data.Dataset.from_tensor_slices(data_X).batch(batch_size))
        self.real_series_iter = iter(x_series.repeat())



    def training_autoencoder(self):
        autoencoder=self.autoencoder
        autoencoder_optimizer=self.autoencoder_optimizer
        loss_function=self.mse
        recovery=self.recovery
        embedder=self.embedder

        ###Autoencoder training steps
        @tf.function
        def train_autoencoder_init(x):
            with tf.GradientTape() as tape:
                x_tilde = autoencoder(x)
                embedding_loss_t0 = loss_function(x, x_tilde)
                e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

            var_list = embedder.trainable_variables + recovery.trainable_variables
            gradients = tape.gradient(e_loss_0, var_list)
            autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
            return tf.sqrt(embedding_loss_t0)

        ##Training Autoencoder
        for step in tqdm(range(self.train_steps)):
            X_ = next(self.individual_series)
            step_e_loss_t0 = train_autoencoder_init(X_)
            with self.writer.as_default():
                tf.summary.scalar('Loss Autoencoder Init', step_e_loss_t0, step=step)
        self.autoencoder.save(self.log_dir / 'autoencoder')




    def training_supervisor(self):
        supervisor=self.supervisor
        loss_function=self.mse
        # Optimizador del supervisor
        supervisor_optimizer = self.supervisor_optimizer
        embedder=self.embedder
        # funcion de entrenamiento del supervisor
        @tf.function
        def train_supervisor(x):
            with tf.GradientTape() as tape:
                h = embedder(x)
                h_hat_supervised = supervisor(h)
                g_loss_s = loss_function(h[:, 1:, :], h_hat_supervised[:, 1:, :])

            var_list = supervisor.trainable_variables
            gradients = tape.gradient(g_loss_s, var_list)
            supervisor_optimizer.apply_gradients(zip(gradients, var_list))
            return g_loss_s

        ##Loop de entrenamiento del supervisor
        for step in tqdm(range(self.train_steps)):
            X_ = next(self.individual_series)
            step_g_loss_s = train_supervisor(X_)
            with self.writer.as_default():
                tf.summary.scalar('Loss Generator Supervised Init', step_g_loss_s, step=step)
        self.supervisor.save(self.log_dir / 'supervisor')


    def joint_training(self):
        # Entramiento conjunto
        E_hat = self.generator(self.Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)

        adversarial_supervised = Model(inputs=self.Z,
                                       outputs=Y_fake,
                                       name='AdversarialNetSupervised')

        adversarial_supervised.summary()

        # Arquitectura adversaria en el espacio latente
        Y_fake_e = self.discriminator(E_hat)

        adversarial_emb = Model(inputs=self.Z,
                                outputs=Y_fake_e,
                                name='AdversarialNet')

        ##promedio y varianza de coste
        X_hat = self.recovery(H_hat)
        self.synthetic_data = Model(inputs=self.Z,
                               outputs=X_hat,
                               name='SyntheticData')

        def get_generator_moment_loss(y_true, y_pred):
            y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
            y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
            g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
            g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1e-6) - tf.sqrt(y_pred_var + 1e-6)))
            return g_loss_mean + g_loss_var

        # Discriminador
        # Arquitectura: data real
        Y_real = self.discriminator(self.H)
        discriminator_model = Model(inputs=self.X,
                                    outputs=Y_real,
                                    name='DiscriminatorReal')

        ##Optimizadores
        generator_optimizer = self.generator_optimizer
        discriminator_optimizer = self.discriminator_optimizer
        embedding_optimizer = self.embedding_optimizer

        # Generator training steps
        bce=self.bce
        embedder=self.embedder
        supervisor=self.supervisor
        mse=self.mse
        generator=self.generator
        autoencoder=self.autoencoder
        recovery=self.recovery
        discriminator=self.discriminator
        gamma=self.gamma
        @tf.function
        def train_generator(x, z):
            with tf.GradientTape() as tape:
                y_fake = adversarial_supervised(z)
                generator_loss_unsupervised = bce(y_true=tf.ones_like(y_fake),
                                                  y_pred=y_fake)

                y_fake_e = adversarial_emb(z)
                generator_loss_unsupervised_e = bce(y_true=tf.ones_like(y_fake_e),
                                                    y_pred=y_fake_e)
                h = embedder(x)
                h_hat_supervised = supervisor(h)
                generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

                x_hat = self.synthetic_data(z)
                generator_moment_loss = get_generator_moment_loss(x, x_hat)

                generator_loss = (generator_loss_unsupervised +
                                  generator_loss_unsupervised_e +
                                  100 * tf.sqrt(generator_loss_supervised) +
                                  100 * generator_moment_loss)

            var_list = generator.trainable_variables + supervisor.trainable_variables
            gradients = tape.gradient(generator_loss, var_list)
            generator_optimizer.apply_gradients(zip(gradients, var_list))
            return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

        @tf.function
        def train_embedder(x):
            with tf.GradientTape() as tape:
                h = embedder(x)
                h_hat_supervised = supervisor(h)
                generator_loss_supervised = mse(h[:, 1:, :], h_hat_supervised[:, 1:, :])

                x_tilde = autoencoder(x)
                embedding_loss_t0 = mse(x, x_tilde)
                e_loss = 10 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

            var_list = embedder.trainable_variables + recovery.trainable_variables
            gradients = tape.gradient(e_loss, var_list)
            embedding_optimizer.apply_gradients(zip(gradients, var_list))
            return tf.sqrt(embedding_loss_t0)

        @tf.function
        def get_discriminator_loss(x, z):
            y_real = discriminator_model(x)
            discriminator_loss_real = bce(y_true=tf.ones_like(y_real),
                                          y_pred=y_real)

            y_fake = adversarial_supervised(z)
            discriminator_loss_fake = bce(y_true=tf.zeros_like(y_fake),
                                          y_pred=y_fake)

            y_fake_e = adversarial_emb(z)
            discriminator_loss_fake_e = bce(y_true=tf.zeros_like(y_fake_e),
                                            y_pred=y_fake_e)
            return (discriminator_loss_real +
                    discriminator_loss_fake +
                    gamma * discriminator_loss_fake_e)

        @tf.function
        def train_discriminator(x, z):
            with tf.GradientTape() as tape:
                discriminator_loss = get_discriminator_loss(x, z)

            var_list = discriminator.trainable_variables
            gradients = tape.gradient(discriminator_loss, var_list)
            discriminator_optimizer.apply_gradients(zip(gradients, var_list))
            return discriminator_loss

        # Training LOOP
        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        for step in range(self.train_steps):
            # Train generator (twice as often as discriminator)
            for kk in range(9):
                Z_ = next(self.random_series)
                X_ = next(self.real_series_iter)

                # Train generator
                step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
                # Train embedder
                step_e_loss_t0 = train_embedder(X_)

            Z_ = next(self.random_series)
            X_ = next(self.real_series_iter)

            step_d_loss = get_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                step_d_loss = train_discriminator(X_, Z_)

            if step % 5 == 0:
                print(f'{step:6,.0f} | d_loss: {step_d_loss:6.4f} | g_loss_u: {step_g_loss_u:6.4f} | '
                      f'g_loss_s: {step_g_loss_s:6.4f} | g_loss_v: {step_g_loss_v:6.4f} | e_loss_t0: {step_e_loss_t0:6.4f}')

            with self.writer.as_default():
                tf.summary.scalar('G Loss S', step_g_loss_s, step=step)
                tf.summary.scalar('G Loss U', step_g_loss_u, step=step)
                tf.summary.scalar('G Loss V', step_g_loss_v, step=step)
                tf.summary.scalar('E Loss T0', step_e_loss_t0, step=step)
                tf.summary.scalar('D Loss', step_d_loss, step=step)

        self.synthetic_data.save(self.log_dir / 'synthetic_data')

    def plot_data(self):

        #generated_data = []
        #for i in range(int(self.n_windows / self.batch_size)):
        Z_ = next(self.random_series)
        d = self.synthetic_data(Z_)
        R_ = next(self.real_series_iter)
        print('Este es el real')
        print(R_)
        real_data = np.array(R_)
        print(real_data.shape)
        generated_data = np.array(d)
        print(generated_data.shape)
        print(len(generated_data))
        generated_data = (self.scaler.inverse_transform(generated_data.reshape(-1, self.n_seq)).reshape(-1, self.seq_len, self.n_seq))
        real_data = (self.scaler.inverse_transform(real_data.reshape(-1, self.n_seq)).reshape(-1, self.seq_len, self.n_seq))

        for i in range(3):
            # print(' sequence:{}'.format(i))
            generated = generated_data[i, :, :]
            real = real_data[i, :, :]
            # print(generated)
            # numpy_data = np.array([[1, 2], [3, 4]])
            base = datetime.datetime(2020, 11, 20)
            arr = np.array([base + datetime.timedelta(minutes=j) for j in range(20)])
            didx = pd.DatetimeIndex(data=arr, tz='America/New_York')
            gen_df = pd.DataFrame(data=generated, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
            print(gen_df)
            real_df = pd.DataFrame(data=real, index=didx, columns=["Open", "High", "Low", "Close", "Volume"])
            # mpf.plot(real_dat, type='candle',mav=(3,6,9),volume=True)
            fig = mpf.figure(style='yahoo', figsize=(14, 16))

            ax1 = fig.add_subplot(2, 2, 1)
            ax2 = fig.add_subplot(2, 2, 2)

            av1 = fig.add_subplot(3, 2, 5, sharex=ax1)
            av2 = fig.add_subplot(3, 2, 6, sharex=ax2)

            mpf.plot(gen_df, type='candle', ax=ax1, volume=av1, mav=(10, 20), axtitle='generated data')
            mpf.plot(real_df, type='candle', ax=ax2, volume=av2, mav=(10, 20), axtitle='real data')

            mpf.show()


training_steps=[13000,16000,20000,25000,30000]
for i in training_steps:
     model = TGAN(train_step=i,save_directory='TimeGAN_future2_noise_past{}'.format(i))
     model.training_loop()