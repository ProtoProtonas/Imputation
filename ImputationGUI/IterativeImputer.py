
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import re
import tensorflow as tf
import time
import warnings

from matplotlib import style
from pandas.plotting import register_matplotlib_converters
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle
from tensorflow import keras

warnings.simplefilter(action = 'ignore', category = FutureWarning)
register_matplotlib_converters()

ACCURACY = 0.1
MIN_AVG_VALUE = 1000

def atmesti_mazas_tui(df):
    rows, cols = df.shape
    for n in reversed(range(rows)):
        row = df.iloc[n]
        to_drop = ['VLST_KODAS2', 'DARB', 'PAJAMOS_EUR', 'veikla', 'COMPANY', 'ROD_KOD', ]
        for a in to_drop:
            row = row.drop(a)

        score = row.sum() / row.count()
        if score < MIN_AVG_VALUE:
            df = df.drop(row.name)

    return df

def relatively_equal(val1, val2):
    val1ten = ACCURACY * val1
    val2ten = ACCURACY * val2

    if val2 > (val1 - val1ten) and val2 < (val1 + val1ten):
        return True
    elif val1 > (val2 - val2ten) and val1 < (val2 + val2ten):
        return True
    return False

def tidy_up_file(filename, encoding = 'utf-16'):
    with open(filename, 'r', encoding = encoding) as f:
        # susidalina failą į eilutes ir stulpelius
        file = f.read()
        file = file.split('\n')
        file = [line.split('\t') for line in file]

    # isimti visus -9999999 ir float paversti i int
    for i in range(len(file)):
        for j in range(len(file[i])):
            if '.' in file[i][j] or ',' in file[i][j]:
                # paverčia realiuosius skaičius į sveikuosius
                file[i][j] = file[i][j].split('.')[0]
            if '-999' in file[i][j]:
                file[i][j] = ''

    new_file = ''
    for line in file:
        new_line = ''
        for cell in line:
            new_line = new_line + '\t' + cell
        new_file = new_file + '\n' + new_line[1:]

    with open(filename, 'w', encoding = 'utf-16') as f:
        f.write(new_file[1:]) # pirma eilutė visada būna tuščia

    return 0

# funkcija, skirta duomenų užpildymui atlikti
def train_model_decision_trees_fill(filename):
    pd.options.mode.chained_assignment = None

    df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    rod_kods = list(set(df['ROD_KOD'].astype(int)))

    estimators = [ExtraTreesRegressor(), BayesianRidge(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()] # geriausiai veikia decision trees regressor
    # pasirenkamas algoritmas
    estimator = estimators[0]

    # ar reikia atmesti mažas reikšmes
    new_filename = filename
    atmesti_mazas_reiksmes = True
    if atmesti_mazas_reiksmes:
        df = atmesti_mazas_tui(df)
        new_filename = filename.split('.')
        new_filename = new_filename[0] + '_be_mazu_tui.' + new_filename[1]
        df.to_csv(new_filename, sep = '\t', encoding = 'utf-16', index = False)

    for rod_kod in rod_kods:

        print('Pildomas %s rodiklis' % rod_kod)
        maindf = pd.read_csv(new_filename, encoding = 'utf-16', sep = '\t')

        # atsirenkamos eilutės tik su tam tikra ROD_KOD reikšme
        df = maindf.loc[maindf['ROD_KOD'] == rod_kod]
        X = shuffle(df)

        # numetamos reikšmės, kurių neina konvertuoti į skaičius
        for name in ['COMPANY', 'VLST_KODAS2']:
            X = X.drop(name, axis = 1)

        # atsikratome tuščių stulpelių (neįmanoma teisingai nuspėti kai nėra jokio pavyzdžio)
        for col in X:
            if X[col].isnull().all():
                X = X.drop(col, axis = 1)
        
        # jei yra bent viena eilutė, kurią būtų galima užpildyti
        if len(X) > 0:
            index = list(X.index)
            columns = list(X.columns.values)
            # sukuriamas ir ištreniruojamas algoritmas
            imp = IterativeImputer(estimator = estimator, missing_values = np.nan)
            imp.fit(X)
            # užpildomos tuščios X reikšmės
            X = imp.transform(X) # čia X grąžinamas np.array pavidalu, todėl reikia jį atversti atgal į pandas.DataFrame
            X = pd.DataFrame(data = X, index = index, columns = columns)
            maindf.update(X)

            # išsaugomi spėjimai
            maindf.to_csv(new_filename, sep = '\t', encoding = 'utf-16', index = False)

    # sutvarko failą
    tidy_up_file(new_filename)
    return 0

# funkcija, skirta algoritmų tikslumui ištestuoti su tam tikrais duomenimis
def train_model_decision_trees_test(filename):
    save_plots = False
    pd.options.mode.chained_assignment = None

    raw_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    rows, cols = raw_df.shape

    # išmesti mažas investicijas lauk
    if True:
        raw_df = atmesti_mazas_tui(raw_df)

    rod_kods = list(set(raw_df['ROD_KOD'].astype(int)))

    sum, total = 0, 0
    estimators = [ExtraTreesRegressor(), BayesianRidge(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
    f = open('performance.txt', 'w')

    for estimator in estimators:
        print('Testuojamas', str(estimator).split('(')[0])
        algorithm_score, algorithm_total = 0, 0

        real, predicted = [], []
        
        for rod_kod in rod_kods:
            try:
                maindf = raw_df

                df = maindf.loc[maindf['ROD_KOD'] == rod_kod]
                X = shuffle(df)

                for name in ['VLST_KODAS2', 'DARB', 'PAJAMOS_EUR', 'veikla', 'COMPANY', 'ROD_KOD', 'VLST_NR']:
                    X = X.drop(name, axis = 1)

                split = 0.2 # kokiu santykiu padalinti duomenis treniravimui ir testavimui
                x_test = X.iloc[:int(len(X) * split)]
                x_test_to_plot = x_test # x_test_to_plot - testinis duomenų rinkinys, kurio reikšmės žinomos (galima vėliau palyginti su nuspėtom reikšmėm)
                x_test = x_test.fillna(-9999)
                x_train = X.iloc[int(len(X) * split):]
                # remove random cell values for testing
        
                sample = x_test.sample(int(len(X) * split * split))
                cols = list(sample.columns.values)

                # atsitiktiniai skaičiai yra ištrinami iš x_test masyvo
                for _, s in sample.iterrows():
                    m = random.randint(0, len(s)-1)
                    if x_test.at[s.name, cols[m]] != -9999:
                        x_test.at[s.name, cols[m]] = np.nan
                    n = random.randint(0, len(s)-1)
                    if x_test.at[s.name, cols[m]] != -9999:
                        x_test.at[s.name, cols[m]] = np.nan

                x_test_is_nan = x_test.isnull() # parodo, kuriose vietose nėra skaičių
                x_test = x_test.replace(-9999, np.nan) # ištrina skaičius, kurių nebuvo iš pradžių

                x_train = x_train.reindex(sorted(x_train.columns.values), axis = 1)
                x_train = np.array(x_train)
                x_test = np.array(x_test)

                # sukuriamas ir ištreniruojamas imputerio objektas
                imp = IterativeImputer(estimator = estimator, missing_values = np.nan)
                imp.fit(x_train)

                if len(x_test) > 0:
                    x_test = imp.transform(x_test)

                    rows, cols = x_test.shape
                    x_test = pd.DataFrame(x_test)

                    for row in range(rows):
                        for col in range(cols):
                            if list(x_test_is_nan.iloc[row])[col]:

                                # tikrina, ar atspėta ir tikroji reikšmės yra panašios
                                if relatively_equal(list(x_test.iloc[row])[col], list(x_test_to_plot.iloc[row])[col]):
                                    sum += 1
                                total += 1
                                # išsisaugo tas reikšmes, vėliau reikės vizualizacijai
                                real.append(list(x_test_to_plot.iloc[row])[col])
                                predicted.append(list(x_test.iloc[row])[col])


                    algorithm_score += sum
                    algorithm_total += total

            except Exception as e:
                print(e)

        # jei algoritmas atliko bent vieną spėjimą
        if algorithm_total > 0:
            # išveda rezultatus į ekraną
            print(str(estimator).split('(')[0])
            print(algorithm_score, algorithm_total)
            print(algorithm_score/algorithm_total)
            print('\n')

            # išsaugo rezultatus į failą performance.txt
            f.write(str(estimator).split('(')[0])
            f.write('\ntotal: %i / %i\n' % (algorithm_score, algorithm_total))
            f.write('score: %.1f%%\n\n' % (100 * algorithm_score/algorithm_total))
            
            # pavaizduoja rezultatus grafike
            plt.figure()
            plt.scatter(real, predicted, color = 'orange')
            plt.xlabel('Tikros TUI reikšmės')
            plt.ylabel('Nuspėtos TUI reikšmės')
            plt.axis('equal')
            plt.axis('square')
            plt.title(str(estimator).split('(')[0])
            plt.ylim(bottom = 0)
            plt.xlim(left = 0)

            plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000])
            plt.plot([-1000000 * (1 - ACCURACY), 0, 1000000], [-1000000, 0, 1000000 * (1 - ACCURACY)], color = 'green')
            plt.plot([-1000000 / (1 - ACCURACY), 0, 1000000], [-1000000, 0, 1000000 / (1 - ACCURACY)], color = 'green')
            plt.show()

    f.close()
    return 0

train_model_decision_trees_test('pngs/predict2_updated.csv')