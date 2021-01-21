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

class TGAN():

    def __init__(self,hidden_dim=60,num_layer=3,seq_len=60,n_seq=5,train_step=1,save_directory='time_gan'):
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
        self.autoencoder_optimizer = Adam()

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
        self.plot_data()

    def prepare_data(self,plot_data=False,batch_size =5):
        # PREPARE DATA
        self.batch_size=batch_size
        import yfinance as yf

        ###obtengo valores de un stock  por minuto
        def get_yfinance_m(stock, period):
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            val = yf.Ticker(stock)
            val_historical = val.history(period=period, interval="1m")
            return val_historical[features]

        def get_database_data(SYMBOL,DIR_PATH = 'C:/Users/56979/PycharmProjects/TimeGAN/Data/Prices'):
            search_path=DIR_PATH+'/{}.csv'.format(SYMBOL)
            index_column=1
            dataframe=pd.read_csv(search_path,index_col=index_column,parse_dates=True)
            features=['Open','High','Low','Close','Volume']
            data=dataframe[features]
            data.index = pd.to_datetime(data.index,utc=True)
            return data



        #self.df=get_database_data(SYMBOL='AAPL')
        self.df =get_yfinance_m(stock='AAPL',period='1d')
        print(self.df)
        if plot_data:
            #candle_data=candle_format(data=self.df)
            mpf.plot(self.df, type='candle')


        ##Normalizan los datos
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(self.df).astype(np.float32)
        ##Dividen los datos en secuencias
        seq_len=self.seq_len
        n_seq=self.n_seq
        data = []
        for i in range(len(self.df) - seq_len):
            data.append(scaled_data[i:i + seq_len])

        ##Numero de secuencias
        self.n_windows = len(data)

        ##Generador de data
        real_series = (tf.data.Dataset
                       .from_tensor_slices(data)
                       .shuffle(buffer_size=self.n_windows)
                       .batch(self.batch_size))
        self.real_series_iter = iter(real_series.repeat())

        def make_random_data():
            while True:
                yield np.random.uniform(low=0, high=1, size=(seq_len, n_seq))

        self.random_series = iter(tf.data.Dataset
                             .from_generator(make_random_data, output_types=tf.float32)
                             .batch(batch_size)
                             .repeat())




    def training_autoencoder(self):
        autoencoder=self.autoencoder
        loss_function=self.mse

        ###Autoencoder training steps
        @tf.function
        def train_autoencoder_init(x):
            with tf.GradientTape() as tape:
                x_tilde = autoencoder(x)
                embedding_loss_t0 = loss_function(x, x_tilde)
                e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

            var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
            gradients = tape.gradient(e_loss_0, var_list)
            self.autoencoder_optimizer.apply_gradients(zip(gradients, var_list))
            return tf.sqrt(embedding_loss_t0)

        ##Training Autoencoder
        for step in tqdm(range(self.train_steps)):
            X_ = next(self.real_series_iter)
            step_e_loss_t0 = train_autoencoder_init(X_)
            with self.writer.as_default():
                tf.summary.scalar('Loss Autoencoder Init', step_e_loss_t0, step=step)
        self.autoencoder.save(self.log_dir / 'autoencoder')




    def training_supervisor(self):
        supervisor=self.supervisor
        loss_function=self.mse
        # Optimizador del supervisor
        supervisor_optimizer = Adam()
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
            X_ = next(self.real_series_iter)
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
        generator_optimizer = Adam()
        discriminator_optimizer = Adam()
        embedding_optimizer = Adam()

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
            for kk in range(2):
                X_ = next(self.real_series_iter)
                Z_ = next(self.random_series)

                # Train generator
                step_g_loss_u, step_g_loss_s, step_g_loss_v = train_generator(X_, Z_)
                # Train embedder
                step_e_loss_t0 = train_embedder(X_)

            X_ = next(self.real_series_iter)
            Z_ = next(self.random_series)
            step_d_loss = get_discriminator_loss(X_, Z_)
            if step_d_loss > 0.15:
                step_d_loss = train_discriminator(X_, Z_)

            if step % 10 == 0:
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
        #generated_data.append(d)

        #len(generated_data)

        generated_data = np.array(d)
        print('generated data')
        print(generated_data)
        print('shape generated data')
        print(generated_data.shape)

        generated_data = (self.scaler.inverse_transform(generated_data
                                                   .reshape(-1, self.n_seq))
                          .reshape(-1, self.seq_len, self.n_seq))
        print(generated_data.shape)
        print('generated data inverse transoform')
        print(generated_data)
        print('last sequence:')
        generated_data=generated_data[-1,:,:]
        print(generated_data)
        #numpy_data = np.array([[1, 2], [3, 4]])
        base = datetime.datetime(2020,11,20)
        arr = np.array([base + datetime.timedelta(minutes=i) for i in range(60)])
        didx = pd.DatetimeIndex( data=arr, tz='America/New_York')
        df = pd.DataFrame(data=generated_data, index=didx, columns=["Open", "High","Low","Close","Volume"])
        print(df)
        #dir_path='C:/Users/56979/PycharmProjects/TimeGAN/tensorflow2_implementation/time_gan'
        file_name=str(self.results_path)+'/'+'{}.csv'.format('AAPL')
        df.to_csv(file_name)
        mpf.plot(df, type='candle')

        # index = list(range(1, 25))
        # synthetic = generated_data[np.random.randint(self.n_windows)]
        #
        # idx = np.random.randint(len(self.df) - self.seq_len)
        # real = self.df.iloc[idx: idx + self.seq_len]




#example=TGAN(save_directory='tgan_prueba')
#example.training_loop()

training_steps=[1000,15000,20000]
for i in training_steps:
     model = TGAN(train_step=i,save_directory='TimeGAN{}'.format(i))
     model.training_loop()


