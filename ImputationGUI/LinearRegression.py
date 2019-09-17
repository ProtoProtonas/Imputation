
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

ACCURACY_PERCENTAGE = 0.1


def relatively_equal(val1, val2):
    percentage = 0.1
    val1ten = percentage * val1
    val2ten = percentage * val2

    if val2 > (val1 - val1ten) and val2 < (val1 + val1ten):
        return True
    elif val1 > (val2 - val2ten) and val1 < (val2 + val2ten):
        return True
    return False

def train_model_linear_regression():
    save_plots = True
    pd.options.mode.chained_assignment = None

    df = pd.read_csv('csvs/predict2_updated.csv', encoding = 'utf-16', sep = '\t')
    rod_kods = list(set(df['ROD_KOD'].astype(int)))
    labels = list(df.columns.values)

    for a in ['COMPANY', 'ROD_KOD', 'VLST_KODAS2', 'veikla', 'DARB', 'PAJAMOS_EUR', 'VLST_NR']:
        try:
            labels.remove(a)
        except:
            pass


    nancount = {}
    for col in labels:
        nancount[col] = df[col].isna().sum()
    labels = sorted(nancount.items(), key = lambda kv: kv[1])
    labels = [a[0] for a in labels]
    
    if save_plots:
        fig = plt.figure()

    for label in ['2017-07-01']:#labels:
        for rod_kod in rod_kods:

            maindf = pd.read_csv('csvs/predict2_updated.csv', encoding = 'utf-16', sep = '\t')
            df = maindf.loc[maindf['ROD_KOD'] == rod_kod]
            X = shuffle(df)
            for name in ['COMPANY', 'VLST_KODAS2']:
                X = X.drop(name, axis = 1)

            for col in X:
                X[col] = pd.to_numeric(X[col].apply(lambda x: re.sub(',', '.', str(x))), downcast = 'float', errors = 'coerce')

            split = 0.3
            x_test = X.iloc[:int(len(X) * split)]
            x_test_to_plot = x_test
            x_test = x_test.fillna(-999)
            y_test = x_test[label]
            x_train = X.iloc[int(len(X) * split):]
            x_train = x_train.fillna(-999)
            y_train = x_train[label]

            print(x_train.shape, x_test.shape)

            #for _, s in x_test_to_plot.iterrows():
            #    for a in ['ROD_KOD', 'veikla', 'DARB', 'PAJAMOS_EUR', 'VLST_NR']:
            #        s = s.drop(a)
            #    s.index = pd.to_datetime(s.index)
            #    if save_plots:
            #        ax1.plot(s)

            x_train = x_train.drop(label, axis = 1)
            x_test = x_test.drop(label, axis = 1)

            x_train = x_train.reindex(sorted(x_train.columns.values), axis = 1)
            x_train = x_train.fillna(-999)
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_test = np.array(x_test)
            y_test = np.array(y_test)

            print('train, test', len(y_train), len(y_test))

            model = LinearRegression()
            model.fit(x_train, y_train)

            if len(x_test) > 0:
                predictions = model.predict(x_test)
                print(y_test.shape, predictions.shape)

                plt.scatter(y_test, predictions, color = 'orange')
                plt.xlabel('Tikros TUI reikšmės')
                plt.ylabel('Nuspėtos TUI reikšmės')
                plt.axis('equal')
                plt.axis('square')
                _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000])
                _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000 * (1 - ACCURACY_PERCENTAGE)])
                _ = plt.plot([-1000000, 0, 1000000], [-1000000, 0, 1000000 / (1 - ACCURACY_PERCENTAGE)])
                plt.show()


                sum, total = 0, 0
                for i in range(len(predictions)):
                    if relatively_equal(predictions[i], y_test[i]):
                        sum += 1
                    total += 1
                print(sum, total, sum/total)

                x_test_to_plot[label] = [int(a) for a in predictions]
                x_test_to_plot = x_test_to_plot.reindex(sorted(x_test_to_plot.columns.values), axis = 1)
    
                #for _, s in x_test_to_plot.iterrows():
                #    new_column = pd.DataFrame({label: [s[label]]}, index = [s.name])
                #    maindf.update(new_column, errors = 'ignore')

                #maindf.to_csv('csvs/predict2_updated.csv', sep = '\t', encoding = 'utf-16', index = False)


    return 0

train_model_linear_regression()