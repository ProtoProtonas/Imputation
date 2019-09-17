
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import tensorflow as tf
import warnings

from matplotlib import style
from pandas.plotting import register_matplotlib_converters
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding
from keras.optimizers import Adam

print(tf.__version__)

warnings.simplefilter(action = 'ignore', category = FutureWarning)
register_matplotlib_converters()

ACCURACY_PERCENTAGE = 0.1

def relatively_equal(val1, val2):
    percentage = ACCURACY_PERCENTAGE
    val1ten = percentage * val1
    val2ten = percentage * val2

    if val2 > (val1 - val1ten) and val2 < (val1 + val1ten):
        return True
    elif val1 > (val2 - val2ten) and val1 < (val2 + val2ten):
        return True
    return False

def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.legend()
    plt.show()

def main_copy():
    label = '2017-07-01'
    raw_df = pd.read_csv('csvs/predict2_updated.csv', encoding = 'utf-16', sep = '\t')
    df = raw_df.copy()
    df = df[np.isfinite(df[label])]

    df = df.fillna(0.0)

    #countrydf = df.loc[:, ['VLST_NR', 'VLST_KODAS2']]
    #vlst = df.pop('VLST_NR')
    _ = df.pop('VLST_KODAS2')
    _ = df.pop('COMPANY')

    #tuples = list(set([tuple(x) for x in countrydf.values]))

    #for nr, code in tuples:
    #    df[code] = (vlst == nr) * 1.0

    train_df = df.sample(frac = 0.8, random_state = 0)
    test_df = df.drop(train_df.index)

    train_stats = train_df.describe()
    train_stats.pop(label)

    train_labels = train_df.pop(label)
    test_labels = test_df.pop(label)

    normed_train_data, normed_test_data = pd.DataFrame(), pd.DataFrame()
    for col in train_df:
        normed_train_data[col] = (train_df[col] - train_stats[col]['mean']) / train_stats[col]['std']
        normed_test_data[col] = (test_df[col] - train_stats[col]['mean']) / train_stats[col]['std']

    def build_model():
        model = Sequential([Embedding(input_dim = len(train_df), input_length = len(train_df.iloc[0]), output_dim = len(train_df), trainable = False, mask_zero = True),
                        #Masking(mask_value = 0.0),
                        LSTM(64, return_sequences = False, dropout = 0.1, recurrent_dropout = 0.1),
                        Dense(64, activation = tf.nn.relu),
                        Dropout(0.5), 
                        Dense(1, activation = tf.nn.softmax)])

        optimizer = Adam(0.001)
        model.compile(loss = 'mean_squared_error',
                    optimizer = optimizer,
                    metrics = ['mean_absolute_error', 'mean_squared_error'])
        return model


    
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 50 == 0: 
                print('Epoch no. ', epoch)

    model = build_model()
    EPOCHS = 1000

    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10) # patience - kiek epochų išlaukia prieš nutraukdamas, kai validation error nustoja gerėti

    print(train_df)

    history = model.fit(train_df, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])
    plot_history(history)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print('Total number of epochs: ', list(hist['epoch'])[-1] + 1)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 0)
    print('Testing set Mean Absolute Error', mae)
    print('Testing set Mean Squared Error', mse)
    print('Testing set loss', loss)

    test_predictions = model.predict(normed_test_data).flatten()

    min_value = 10000
    #min_value = -9999999
    score, total = 0, 0
    for i in range(len(test_labels)):
        actual = float(test_labels.iloc[i])
        predicted = float(test_predictions[i])
        if actual > min_value and predicted > min_value:
            if relatively_equal(actual, predicted):
                score += 1
            total += 1

    print(score, total, score/total)

    test_labels = list(test_labels)
    test_predictions = list(test_predictions)

    for i in reversed(range(len(test_labels))):
        if test_labels[i] < min_value:
            del test_labels[i]
            del test_predictions[i]

    plt.scatter(test_labels, test_predictions, color = 'orange')
    plt.xlabel('Tikros TUI reikšmės')
    plt.ylabel('Nuspėtos TUI reikšmės')
    plt.axis('equal')
    plt.axis('square')
    _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000])
    _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000 * (1 - ACCURACY_PERCENTAGE)], color = 'green')
    _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000 / (1 - ACCURACY_PERCENTAGE)], color = 'green')
    plt.show()
    
def main():
    label = '2017-07-01'
    raw_df = pd.read_csv('csvs/predict2_updated.csv', encoding = 'utf-16', sep = '\t')
    df = raw_df.copy()
    df = df[np.isfinite(df[label])]

    df = df.fillna(0.0)

    countrydf = df.loc[:, ['VLST_NR', 'VLST_KODAS2']]
    vlst = df.pop('VLST_NR')
    _ = df.pop('VLST_KODAS2')
    _ = df.pop('COMPANY')

    tuples = list(set([tuple(x) for x in countrydf.values]))

    for nr, code in tuples:
        df[code] = (vlst == nr) * 1.0

    train_df = df.sample(frac = 0.8, random_state = 0)
    test_df = df.drop(train_df.index)

    train_df.reset_index(inplace = True)
    test_df.reset_index(inplace = True)

    train_stats = train_df.describe()
    train_stats.pop(label)

    train_labels = train_df.pop(label)
    test_labels = test_df.pop(label)

    normed_train_data, normed_test_data = pd.DataFrame(), pd.DataFrame()
    for col in train_df:
        normed_train_data[col] = (train_df[col] - train_stats[col]['mean']) / train_stats[col]['std']
        normed_test_data[col] = (test_df[col] - train_stats[col]['mean']) / train_stats[col]['std']

    train_dataset = np.array(normed_train_data.values)
    train_labels = np.array(train_labels.values)

    print('^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v^v')

    model = Sequential([Embedding(input_dim = len(train_df), input_length = len(train_df.iloc[0]), output_dim = len(train_df), trainable = False, mask_zero = True),
                        #Masking(mask_value = 0.0),
                        LSTM(64, return_sequences = False, dropout = 0.1, recurrent_dropout = 0.1),
                        Dense(64, activation = tf.nn.relu),
                        Dropout(0.5), 
                        Dense(1, activation = tf.nn.softmax)])

    optimizer = Adam(0.001)

    print('<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>', train_df.iloc[30, 28])

    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 50 == 0: 
                print('Epoch no. ', epoch)
            print('.')

    EPOCHS = 1000

    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10) # patience - kiek epochų išlaukia prieš nutraukdamas, kai validation error nustoja gerėti

    history = model.fit(train_dataset, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [PrintDot()])
    #history = model.fit(train_dataset, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])
    plot_history(history)

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print('Total number of epochs: ', list(hist['epoch'])[-1] + 1)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 0)
    print('Testing set Mean Absolute Error', mae)
    print('Testing set Mean Squared Error', mse)
    print('Testing set loss', loss)


    return 0



# su recurrent nn bus fiasko, nes jiems reikia tęstinumo (gerai kažkam nuspėti), 
# bet kadangi spėjam iš vidurio sekos tai šiuo atveju netgi neišeina paleisti apmokymų, jau nekalbant apie gerus rezultatus

