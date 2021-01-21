import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import copy


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # print(names)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # print(agg.columns)
    return agg


def data_preprocessing(step_size, df_final, n_train_days):
    '''load data and transform data into trainset and testset
        features: O,H,L,C ,情态变量*2
        '''
    df_final.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Pct_change_raw', 'Compound_multiplied_raw']
    # avoid the data leakage problem
    df_final['Pct_change'] = df_final['Pct_change_raw'].shift(1)
    df_final.drop(['Pct_change_raw'], axis=1, inplace=True)
    df_final['Compound_multiplied'] = df_final['Compound_multiplied_raw'].shift(1)
    df_final.drop(['Compound_multiplied_raw'], axis=1, inplace=True)
    df_final.dropna(axis=0, how='any', inplace=True)
    dataset = df_final
    dataset['Date'] = pd.to_datetime(dataset.Date, format='%Y-%m-%d')
    dataset.index = dataset['Date']
    dataset = dataset.sort_index(ascending=True, axis=0)
    dataset.drop(['Date'], axis=1, inplace=True)

    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    # frame as supervised learning
    reframed = series_to_supervised(scaled, step_size, 1)
    print(reframed.shape)

    # split into train and test sets
    values = reframed.values

    train = values[:n_train_days, :]
    test = values[n_train_days:, :]

    return train, test, scaler


class TimeGAN():

    def __init__(self,step_size,feature_num,df_final,load=True):
        self.step_size=step_size
        self.load=load
        self.data=df_final
        self.feature_num=feature_num
        self.generator_network=self.generator()
        self.discriminator_network=self.discriminator()
        self.gan_network=self.gan()

    def generator(self):
        generator_input = keras.Input(shape=(self.step_size, self.feature_num))
        x = layers.LSTM(75, return_sequences=True)(generator_input)
        # x = layers.Dropout(0.2)(x)
        x = layers.LSTM(25)(x)
        x = layers.Dense(self.feature_num)(x)
        x = layers.LeakyReLU()(x)
        generator = keras.models.Model(generator_input, x)
        print('generator model')
        generator.summary()
        if self.load:
            generator.load_weights('C:/Users/56979/PycharmProjects/TimeGAN/models/generator.h5')
        return generator

    def discriminator(self):
        discriminator_input = layers.Input(shape=(self.feature_num))
        y = layers.Dense(72)(discriminator_input)
        y = layers.LeakyReLU(alpha=0.05)(y)
        y = layers.Dense(100)(y)
        y = layers.LeakyReLU(alpha=0.05)(y)
        y = layers.Dense(10)(y)
        y = layers.LeakyReLU(alpha=0.05)(y)
        y = layers.Dense(1, activation='sigmoid')(y)
        #y = layers.LeakyReLU()(y)
        discriminator = keras.models.Model(discriminator_input, y)
        print('discriminator model')
        discriminator.summary()
        discriminator_optimizer = keras.optimizers.RMSprop(lr=8e-4, clipvalue=1.0, decay=1e-8)
        discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')
        return discriminator
    def gan(self):
        self.discriminator_network.trainable = False
        gan_input = keras.Input(shape=(self.step_size, self.feature_num))
        gan_output = self.discriminator_network(self.generator_network(gan_input))
        gan = keras.models.Model(gan_input, gan_output)
        print('gan model')
        gan.summary()
        gan_optimizer = keras.optimizers.RMSprop(lr=4e-4, clipvalue=1.0, decay=1e-8)
        gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
        if self.load:
            gan.load_weights('C:/Users/56979/PycharmProjects/TimeGAN/models/gan.h5')
        return gan

    def train(self):
        import random
        n_train_days = len(self.data.values) - int(len(self.data.values) / 10)
        # PREPARATION OF TIME SERIES DATASET
        train, test, scaler = data_preprocessing(step_size=self.step_size, df_final=self.data,
                                                 n_train_days=n_train_days)
        # split into input and outputs
        n_obs = self.step_size * self.feature_num
        train_X, train_Y = train[:, :n_obs], train[:, n_obs:]
        print('train Y ')
        print(len(train_Y))
        print(len(train_X))
        test_X, test_Y = test[:, :n_obs], test[:,n_obs:]
        # reshape input to be 3D [samples, timesteps, features]
        trainX = train_X.reshape((train_X.shape[0], self.step_size, self.feature_num))
        testX = test_X.reshape((test_X.shape[0], self.step_size, self.feature_num))
        #Training
        iterations = n_train_days
        batch_size = 1
        start = 0
        final_predictions = []
        final_real=[]
        discriminator_losses=[]
        gan_losses=[]
        for step in range(iterations):
            print(step)
            temp_X = copy.deepcopy(trainX[step])
            temp_X = temp_X.reshape(batch_size, self.step_size, self.feature_num)
            temp_Y = copy.deepcopy(train_Y[step])
            temp_Y = temp_Y.reshape(batch_size, self.feature_num)
            num_drsciminator_train=random.randint(15,20)
            for i in range(num_drsciminator_train):
                #calcula la prediccion
                predictions = self.generator_network.predict(temp_X)
                #prepara data para entrenamiento del discriminador
                print('entrenamiento discriminador numero: {}'.format(i))
                input_f=np.array(predictions)
                input_r=np.array(temp_Y)
                input = np.concatenate((input_f, input_r),axis=0)
                label_created = [np.zeros(batch_size)]
                label_real=[np.ones(batch_size)]
                labels = np.concatenate((label_created,label_real),axis=0)
                d_loss = self.discriminator_network.train_on_batch(input, labels)
            num_gan_train=random.randint(2,10)
            for j in range(num_gan_train):
                print('entrenamiento gan numero: {}'.format(j))
                misleading_targets = np.zeros((batch_size, 1))
                g_loss = self.gan_network.train_on_batch(temp_X, [misleading_targets])
            print('d_loss: {} ,g_loss: {}'.format(d_loss,g_loss))
            final_predictions.append(predictions[0][0])
            final_real.append(temp_Y[0][0])
            discriminator_losses.append(d_loss)
            gan_losses.append(g_loss)

        final_dataframe=pd.DataFrame()
        final_dataframe['open_r']=final_real
        final_dataframe['open_f']=final_predictions
        final_dataframe['d_loss']=discriminator_losses
        final_dataframe['g_loss']=gan_losses
        save_path= '/Results/results.csv'
        final_dataframe.to_csv(save_path)
        self.gan_network.save_weights('C:/Users/56979/PycharmProjects/TimeGAN/models/gan.h5')
        self.generator_network.save_weights('C:/Users/56979/PycharmProjects/TimeGAN/models/generator.h5')

        return final_dataframe

