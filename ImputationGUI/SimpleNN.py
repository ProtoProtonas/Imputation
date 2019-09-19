
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



# <^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v
# <^>v<^>     REDAGUOTI ŠITUS     <^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v
# <^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v

ACCURACY = 0.1 # didžiausias nukrypimas nuo tikros reikšmės, kuris vis dar laikomas teisingu spėjimu (reikia tik testuojant)
REPLACE_NAN = -9999 # kokia reikšme pakeičiamos tuščios reikšmės (galima daryti outlier, pvz. -9999999, kad išsiskirtų, arba vidurkį, kad neiškreiptų duomenų)
MIN_AVG_VALUE = 10000 # mažiausia vidutinė reikšmė, kurią gali turėti eilutė ir vis dar nebūti atmesta, kaip per maža

TEST_LABEL = '2017-07-01' # kuris nors VIENAS stulpelis algoritmo testavimui
NEREIKIA_SPETI = ['VLST_KODAS2', 'DARB', 'PAJAMOS_EUR', 'veikla', 'COMPANY', 'ROD_KOD'] # stulpeliai, kurių nereikia spėti, tačiau jie vistiek gali turėti įtakos rezultatams, todėl yra paliekami
PAVERSTI_I_ONE_HOT = ['VLST_KODAS2'] # logistiniai duomenys (pvz. šalis arba industrija), kuriuos reikia paversti į one-hot masyvą
ATMESTI = ['VLST_NR', 'COMPANY'] # stulpeliai, kurie neturi jokios koreliacijos su spėjamais duomenimis ir tiesiog yra nenaudingi


# <^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v
# <^>v<^>     ČIA STENGTIS NEBERADAGUOTI      <^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v
# <^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v<^>v



# palygina dvi reikšmes, ar jos skiriasi per mažiau, nei nurodo ACCURACY kintamasis
def relatively_equal(val1, val2):
    val1ten = ACCURACY * val1
    val2ten = ACCURACY * val2

    if val2 > (val1 - val1ten) and val2 < (val1 + val1ten):
        return True
    elif val1 > (val2 - val2ten) and val1 < (val2 + val2ten):
        return True
    return False

# atmeta mažas reikšmes turinčias eilutes tam, kad jos neiškreptų rezultatų
def atmesti_mazas_tui(df):
    min_value = MIN_AVG_VALUE
    rows, cols = df.shape
    for n in reversed(range(rows)):
        row = df.iloc[n]
        to_drop = NEREIKIA_SPETI
        for a in to_drop:
            row = row.drop(a, errors = 'ignore')

        score = row.sum() / row.count()
        if score < min_value:
            df = df.drop(row.name, errors = 'ignore')
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

def process_df(df):
    # pridėti one-hot masyvą(-us) pagal pasirinktus stulpelius
    for col in PAVERSTI_I_ONE_HOT:
        all_values = list(set(x for x in df[col]))
        for code in all_values:
            df[code] = (code == df[col]) * 1.0
    
    # išmetami nereikalingi stulpeliai (taip part ir one-hot, kadangi jis jau panaudotas sukuriant kitus stulpelius)
    for label_to_remove in ATMESTI + PAVERSTI_I_ONE_HOT:
        _ = df.pop(label_to_remove)

    df = df.fillna(REPLACE_NAN)

    return df

def nn_test(filename):
    # pasirenkamas kažkuris vienas stulpelis, kuris bus spėjamas
    label = TEST_LABEL

    raw_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    df = raw_df.copy()
    # atsirenkamos tik tos reikšmės, kurios jau yra žinomos
    df = df[np.isfinite(df[label])]
    df = process_df(df)

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
            layers.Dense(64, activation = tf.nn.leaky_relu, input_shape = [len(train_df.keys())]), 
            layers.Dense(128, activation = tf.nn.leaky_relu),
            layers.Flatten(),
            layers.Dense(4, activation = tf.nn.leaky_relu),
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

    # į ekraną išvedami keli parametrai apie algoritmą
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    print('Total number of epochs: ', list(hist['epoch'])[-1] + 1)

    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 0)
    print('Testing set Mean Absolute Error', mae)
    print('Testing set Mean Squared Error', mse)
    print('Testing set loss', loss)

    test_predictions = model.predict(normed_test_data).flatten()

    # ištestuojamas algortimo tikslumas
    score, total = 0, 0
    for i in range(len(test_labels)):
        actual = float(test_labels.iloc[i])
        predicted = float(test_predictions[i])
        if actual > MIN_AVG_VALUE and predicted > MIN_AVG_VALUE:
            if relatively_equal(actual, predicted):
                score += 1
            total += 1

    print(score, total, score/total)

    test_labels = list(test_labels)
    test_predictions = list(test_predictions)
    # atmetamos per mažos reikšmės
    for i in reversed(range(len(test_labels))):
        if test_labels[i] < MIN_AVG_VALUE:
            del test_labels[i]
            del test_predictions[i]

    # nupiešiamas grafikas Spėta reikšmė / Tikroji reikšmė
    plt.scatter(test_labels, test_predictions, color = 'orange')
    plt.xlabel('Tikros TUI reikšmės')
    plt.ylabel('Nuspėtos TUI reikšmės')
    plt.axis('equal')
    plt.axis('square')
    _ = plt.plot([-10000000, 0, 10000000], [-10000000, 0, 10000000])
    _ = plt.plot([-10000000, 0, 10000000], [-10000000, 0, 10000000 * (1 - ACCURACY)], color = 'green')
    _ = plt.plot([-10000000, 0, 10000000], [-10000000, 0, 10000000 / (1 - ACCURACY)], color = 'green')
    plt.savefig('tikros_vs_nuspetos_nn.png', dpi = 300, bbox_inches = 'tight')
    plt.show()

def nn_fill(filename):
    raw_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')

    # susirenka labelius čia
    labels = list(raw_df.columns.values)
    
    nancount = {}
    for col in labels:
        nancount[col] = raw_df[col].isna().sum()

    # išrikiuoja laiko eilutes pagal tai, kiek joms trūksta įrašų (pirma pilniausia, gale tuščiausia)
    labels = sorted(nancount.items(), key = lambda kv: kv[1])
    labels = [a[0] for a in labels]

    for a in NEREIKIA_SPETI + PAVERSTI_I_ONE_HOT + ATMESTI:
        if a in labels:
            labels.remove(a)

    # ar reikia atmesti mažas reikšmes
    new_filename = filename
    if False:
        raw_df = atmesti_mazas_tui(raw_df)
        new_filename = filename.split('.')
        new_filename = new_filename[0] + '_be_mazu_tui.' + new_filename[1]
        raw_df.to_csv(new_filename, sep = '\t', encoding = 'utf-16', index = False)

    for label in labels:
        print('\nTiriamas %s stulpelis\n' % label)
        raw_df = pd.read_csv(new_filename, encoding = 'utf-16', sep = '\t')
        df = raw_df.copy()

        df = process_df(df)

        # langelius su informacija atrenka apmokymui, be informacijos - užpildymui
        train_df = df[np.isfinite(df[label])]
        fill_df = df[np.isnan(df[label])]

        if len(fill_df) > 0:
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
                    layers.Dense(64, activation = tf.nn.leaky_relu, input_shape = [len(train_df.keys())]), 
                    layers.Dense(128, activation = tf.nn.leaky_relu),
                    layers.Flatten(),
                    layers.Dense(4, activation = tf.nn.leaky_relu),
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

                new_filename = new_filename.split('.')[0] + '_updated.' + new_filename.split('.')[1]

                raw_df.update(fill_df)
                raw_df.to_csv(new_filename, sep = '\t', encoding = 'utf-16', index = False)
                tidy_up_file(new_filename)
                print('Failas sėkmingai papildytas')

    return 0

nn_test('csvs/predict2_updated.csv')

# nn_test('kelias/iki/failo.csv')
# nn_fill('kelias/iki/failo.csv')
