
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
import warnings

from matplotlib import style
from pandas.plotting import register_matplotlib_converters
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

warnings.simplefilter(action = 'ignore', category = FutureWarning)
register_matplotlib_converters()
REPLACE_NAN = -9999

# nupiešia grafiką, kruis parodo, kaip mažėja spėjimo paklaida
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    # vidutinė absoliučioji paklaida
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label = 'Val Error')
    plt.legend()

    # vidutinė kvadratinė paklaida
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label = 'Val Error')
    plt.legend()
    plt.show()

def atmesti_mazas_tui(df):
    min_value = 100
    rows, cols = df.shape
    for n in reversed(range(rows)):
        row = df.iloc[n]
        to_drop = ['VLST_KODAS2', 'DARB', 'PAJAMOS_EUR', 'veikla', 'COMPANY', 'ROD_KOD', ]
        for a in to_drop:
            row = row.drop(a)

        score = row.sum() / row.count()
        if score < min_value:
            df = df.drop(row.name)
    return df

def tidy_up_file(filename, encoding = 'utf-16'):
    with open(filename, 'r', encoding = encoding) as f:
        file = f.read()
        file = file.split('\n')
        file = [line.split('\t') for line in file]

    # isimti visus -9999 ir float paversti i int
    for i in range(len(file)):
        for j in range(len(file[i])):
            if '.' in file[i][j]:
                file[i][j] = file[i][j].split('.')[0]
            if ',' in file[i][j]:
                file[i][j] = file[i][j].split(',')[0]
            if '-999' in file[i][j]:
                file[i][j] = ''

    new_file = ''
    for line in file:
        new_line = ''
        for cell in line:
            new_line = new_line + '\t' + cell
        new_file = new_file + '\n' + new_line[1:]

    with open(filename, 'w', encoding = 'utf-16') as f:
        f.write(new_file[1:])

    return 0

def main_test(filename):
    # pasirenkamas kažkuris vienas laiko periodas
    label = '2017-07-01'

    raw_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    df = raw_df.copy()
    # atsirenkamos tik tos reikšmės, kurios jau yra žinomos
    df = df[np.isfinite(df[label])]

    # užpildomi tušti langeliai, kad pasileistų algoritmas (NaN reikšmių nepriima)
    df = df.fillna(REPLACE_NAN)

    countrydf = df.loc[:, ['VLST_NR', 'VLST_KODAS2']]
    vlst = df.pop('VLST_NR')
    _ = df.pop('VLST_KODAS2')
    _ = df.pop('COMPANY')

    # šalys, iš kurių atkeliavo TUI yra paverčiamos į one-hot masyvą ir tai yra priklijuojama prie pagrindinės lentelės (NN algoritmui taip yra lengviau interpretuoti informaciją)
    tuples = list(set([tuple(x) for x in countrydf.values]))
    for nr, code in tuples:
        df[code] = (vlst == nr) * 1.0

    # atskiriami duomenys testavimui ir apmokymams
    train_df = df.sample(frac = 0.8, random_state = 0)
    test_df = df.drop(train_df.index)

    # sudaroma statistika apie duomenis (standartinis nuokrypis, vidurkis ir t.t.)
    train_stats = train_df.describe()
    train_stats.pop(label)

    train_labels = train_df.pop(label)
    test_labels = test_df.pop(label)

    # aptvarkomi duomenys, kad NN neturėtų jokių nuokrypių ir nebūtų šališkas
    normed_train_data, normed_test_data = pd.DataFrame(), pd.DataFrame()
    for col in train_df:
        normed_train_data[col] = (train_df[col] - train_stats[col]['mean']) / train_stats[col]['std']
        normed_test_data[col] = (test_df[col] - train_stats[col]['mean']) / train_stats[col]['std']

    # NN modelis su sluoksniais ir jų dydžiais
    def build_model():
        model = keras.Sequential([
            layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_df.keys())]), 
            layers.Dense(128, activation = tf.nn.relu),
            layers.Flatten(),
            layers.Dense(4, activation = tf.nn.relu),
            layers.Dense(1)
            ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss = 'mean_squared_error',
                    optimizer = optimizer,
                    metrics = ['mean_absolute_error', 'mean_squared_error'])
        return model

    # kas epochą kviečiamas šitas objektas
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 50 == 0: 
                print('Epoch no. ', epoch)

    model = build_model()
    EPOCHS = 1000

    # nustatomas išankstinis sustojimas - kai paklaida nustoja mažėti tuomet yra sustabdomi apmokymai tam, kad pavyktų išventi overfitting
    early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10) # patience - kiek epochų išlaukia prieš nutraukdamas, kai validation error nustoja gerėti

    # apmokymai
    history = model.fit(normed_train_data, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])
    plot_history(history)

    # į ekraną išvedami keli parametrai apie algoritmą
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print('Total number of epochs: ', list(hist['epoch'])[-1] + 1)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 0)
    print('Testing set Mean Absolute Error', mae)
    print('Testing set Mean Squared Error', mse)
    print('Testing set loss', loss)

    test_predictions = model.predict(normed_test_data).flatten()

    # mažiausia reikšmė, kurią nuspėjus spėjimas dar įtraukiamas į galutinį rezultatą
    min_value = 10000
    # ištestuojamas algortimo tikslumas
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
    # atmetamos per mažos reikšmės
    for i in reversed(range(len(test_labels))):
        if test_labels[i] < min_value:
            del test_labels[i]
            del test_predictions[i]

    # nupiešiamas grafikas Spėta reikšmė / Tikroji reikšmė
    plt.scatter(test_labels, test_predictions, color = 'orange')
    plt.xlabel('Tikros TUI reikšmės')
    plt.ylabel('Nuspėtos TUI reikšmės')
    plt.axis('equal')
    plt.axis('square')
    _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000])
    _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000 * (1 - ACCURACY_PERCENTAGE)], color = 'green')
    _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000 / (1 - ACCURACY_PERCENTAGE)], color = 'green')
    plt.show()

def main_fill(filename):
    raw_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')

    # susirenka labelius čia
    labels = list(raw_df.columns.values)
    
    nancount = {}
    for col in labels:
        nancount[col] = raw_df[col].isna().sum()

    # išrikiuoja laiko eilutes pagal tai, kiek joms trūksta įrašų (pirma pilniausia, gale tuščiausia)
    labels = sorted(nancount.items(), key = lambda kv: kv[1])
    labels = [a[0] for a in labels]

    for a in ['COMPANY', 'ROD_KOD', 'VLST_KODAS2', 'veikla', 'DARB', 'PAJAMOS_EUR', 'VLST_NR']:
        try:
            labels.remove(a)
        except:
            pass

    # ar reikia atmesti mažas reikšmes
    new_filename = filename
    if False:
        raw_df = atmesti_mazas_tui(raw_df)
        new_filename = filename.split('.')
        new_filename = new_filename[0] + '_be_mazu_tui.' + new_filename[1]
        raw_df.to_csv(new_filename, sep = '\t', encoding = 'utf-16', index = False)

    for label in labels:
        print('\nWorking on label %s\n' % label)
        raw_df = pd.read_csv(new_filename, encoding = 'utf-16', sep = '\t')
        df = raw_df.copy()

        countrydf = df.loc[:, ['VLST_NR', 'VLST_KODAS2']]
        vlst = df.pop('VLST_NR')
        _ = df.pop('VLST_KODAS2')
        _ = df.pop('COMPANY')

        tuples = list(set([tuple(x) for x in countrydf.values]))

        # sudaromas one-hot masyvas šalims aprašyti
        for nr, code in tuples:
            df[code] = (vlst == nr) * 1.0

        # langelius su informacija atrenka apmokymui, be informacijos - užpildymui
        train_df = df[np.isfinite(df[label])]
        fill_df = df[np.isnan(df[label])]
        train_df = train_df.fillna(REPLACE_NAN)
        fill_df = fill_df.fillna(REPLACE_NAN)
        print('Nežinomų reikšmių yra %i' % len(fill_df))

        # statistika apie duomenis
        train_stats = train_df.describe()
        train_stats.pop(label)

        train_labels = train_df.pop(label)

        # normalizuojami duomenys - atimamas vidurkis ir padalinama iš standartinio nuokrypio
        normed_train_data, normed_fill_data = pd.DataFrame(), pd.DataFrame()
        for col in train_df:
            normed_train_data[col] = (train_df[col] - train_stats[col]['mean']) / train_stats[col]['std']
            normed_fill_data[col] = (fill_df[col] - train_stats[col]['mean']) / train_stats[col]['std']

        # sukuriamas NN modelis
        def build_model():
            model = keras.Sequential([
                layers.Dense(64, activation = tf.nn.relu, input_shape = [len(train_df.keys())]), 
                layers.Dense(128, activation = tf.nn.relu),
                layers.Flatten(),
                layers.Dense(4, activation = tf.nn.relu),
                layers.Dense(1)
                ])

            optimizer = tf.keras.optimizers.RMSprop(0.001)
            model.compile(loss = 'mean_squared_error',
                        optimizer = optimizer,
                        metrics = ['mean_absolute_error', 'mean_squared_error'])
            return model
            
        # rodo progresą (kelinta epocha)
        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 50 == 0: 
                    print('Epochos nr. ', epoch)

        model = build_model()
        EPOCHS = 1000

        # inicializuojamas objektas, kuris nutrauks apmokymus, kai rezultatai nustos gerėti
        early_stop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10) # patience - kiek epochų išlaukia prieš nutraukdamas, kai validation error nustoja gerėti

        history = model.fit(normed_train_data, train_labels, epochs = EPOCHS, validation_split = 0.2, verbose = 0, callbacks = [early_stop, PrintDot()])
        #plot_history(history)

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print('Viso epochų: ', list(hist['epoch'])[-1] + 1)

        max_epochs = 999

        # spėjimai apdorojami ir failas išsaugomas
        if list(hist['epoch'])[-1] + 1 < max_epochs:
            predictions = model.predict(normed_fill_data)
            predictions = np.array(predictions).flatten()
            fill_df[label] = predictions
            fill_df.replace(REPLACE_NAN, value = np.nan)

            raw_df.update(fill_df)
            raw_df.to_csv(new_filename, sep = '\t', encoding = 'utf-16', index = False)
            tidy_up_file(new_filename)
            print('Failas sėkmingai papildytas')

    return 0

main_fill('csvs/predict2_updated.csv')
