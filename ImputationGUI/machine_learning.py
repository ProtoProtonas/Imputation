
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


def relatively_equal(val1, val2):
    percentage = 0.1
    val1ten = percentage * val1
    val2ten = percentage * val2

    if val2 > (val1 - val1ten) and val2 < (val1 + val1ten):
        return True
    elif val1 > (val2 - val2ten) and val1 < (val2 + val2ten):
        return True
    return False

def get_data():
    path = 'tuiproc'
    filenames = os.listdir(path)
    filenames = [filename for filename in filenames if not filename.endswith('INT.csv')]
    filenames = shuffle(filenames)
    #filenames = filenames[:5]
    split_train_test = 0.2

    main_df = pd.DataFrame()
    for filename in filenames:
        print(filenames.index(filename))
        df = pd.read_csv(os.path.join(path, filename), sep = '\t', encoding = 'utf-16', low_memory = False)

        for _, s in df.iterrows():
            name = filename.split('.')[0]
            new_s = s.append(pd.Series(name, index = ['COMPANY']))
            main_df = main_df.append(new_s, ignore_index = True)

    main_df = shuffle(main_df)
    main_df = main_df.reindex(sorted(main_df.columns.values), axis = 1)

    train_size = int(main_df.shape[0] * (1 - split_train_test))
    train_df = main_df.iloc[:train_size]
    test_df = main_df.iloc[train_size:]

    train_df.to_csv('train.csv', sep = '\t', encoding = 'utf-16', index = False)
    test_df.to_csv('test.csv', sep = '\t', encoding = 'utf-16', index = False)
    main_df.to_csv('main.csv', sep = '\t', encoding = 'utf-16', index = False)

def tidy_up_train_test():
    filename = 'main.csv'
    with open(filename, 'r', encoding = 'utf-16') as f:
        file = f.read()
        file = file.split('\n')
        file = [line.split('\t') for line in file]

    # isimti visus -2147483648 ir float paversti i int
    for i in range(len(file)):
        for j in range(len(file[i])):
            if '.' in file[i][j]:
                file[i][j] = file[i][j].split('.')[0]

    new_file = ''
    for line in file:
        new_line = ''
        for cell in line:
            new_line = new_line + '\t' + cell
        full = [a for a in line if a != '']
        if (len(full) - 3) / (len(line) - 3) > 0.75:
            new_file = new_file + '\n' + new_line[1:]

    with open(filename, 'w', encoding = 'utf-16') as f:
        f.write(new_file[1:])

    return 0

def surikiuoti_pagal_rodikli(name):
    df = pd.read_csv(name, sep = '\t', encoding = 'utf-16', low_memory = False)
    print('Sorting now')
    df = df.sort_values('ROD_KOD')
    print('Sorted')
    df.to_csv(name, sep = '\t', encoding = 'utf-16', index = False)

def softmax(x):
    index = x.index
    x = np.array(list(x))
    softmaxed = softmax_raw(x)
    return pd.Series(softmaxed, index = index)

def softmax_raw(X):

    e_x = [x for x in X]# if x != np.nan else np.nan]
    sum = 0
    for a in e_x:
        if a == a:
            sum += abs(a)
    e_x = [a / sum for a in e_x]
    new_x = e_x

    e = 2.71828182845904523536
    e_x = [e**x for x in new_x]# if x != np.nan else np.nan]
    sum = 0
    for a in e_x:
        if a == a:
            sum += a
    e_x = [a / sum for a in e_x]
    return e_x

def plot_data():
    #'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
    #'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 
    #'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 
    #'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 
    #'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test'

    style.use('seaborn-poster')
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1), (0,0), rowspan = 2, colspan = 1)
    #ax2 = plt.subplot2grid((2,1), (1,0), rowspan = 1, colspan = 1)
    #ax1.xaxis_date()
    #ax2.xaxis_date()

    name = 'predict.csv'
    df = pd.read_csv(name, encoding = 'utf-16', sep = '\t')

    df_cols = list(df.columns.values)
    show_percentage = 1
    part = int(df.shape[0] * show_percentage)
    df = df.iloc[:part]

    for _, s in df.iterrows():
        if 'VLST_KODAS2' in list(s.index):
            s = s.drop('VLST_KODAS2')
        if 'COMPANY' in list(s.index):
            s = s.drop('COMPANY')
        if 'ROD_KOD' in list(s.index):
            s = s.drop('ROD_KOD')
        s.index = pd.to_datetime(s.index)
        s = s.astype(float)
        ax1.plot(s)
        ax1.axvline(x = '2017-07-01', alpha = 0.003)

    plt.show()

    #for col in df:
    #    plt.scatter(df[col], df['2017-07-01'], color = 'red')
    #    plt.xlabel(col)
    #    plt.ylabel('2017-07-01')
    #    plt.savefig('2017-07-01/%s.png' % col, dpi = 300)
    #    plt.clf()
    return 0
    
def train_model_linear_regression():
    save_plots = True
    pd.options.mode.chained_assignment = None


    df = pd.read_csv('main.csv', encoding = 'utf-16', sep = '\t')
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

    for label in labels:
        for rod_kod in rod_kods:
            #try:
            maindf = pd.read_csv('csvs/predict2_updated.csv', encoding = 'utf-16', sep = '\t')
            df = maindf.loc[maindf['ROD_KOD'] == rod_kod]
            X = shuffle(df)
            for name in ['COMPANY', 'VLST_KODAS2']:
                X = X.drop(name, axis = 1)

            print('\n')
            X = X.fillna(-999)

            for col in X:
                X[col] = pd.to_numeric(X[col].apply(lambda x: re.sub(',', '.', str(x))), downcast = 'float', errors = 'coerce')

            split = 0.15
            x_test = X.iloc[:int(len(X) * split)]
            x_test_to_plot = x_test
            #x_test = x_test.fillna(-999)
            y_test = x_test[label]
            x_train = X.iloc[int(len(X) * split):]
            y_train = x_train[label]

            print(x_train.shape, x_test.shape)

            #x_test = X[np.isnan(X[label])]
            #x_test_to_plot = x_test
            #x_test = x_test.fillna(-999)
            #y_test = x_test[label]
            #x_train = X[np.isfinite(X[label])]
            #y_train = x_train[label]

            if save_plots:
                plt.clf()
                fig.suptitle('Rodiklio kodas: ' + str(rod_kod), fontsize = 16)
                ax1 = plt.subplot2grid((2,1), (0,0), rowspan = 1, colspan = 1)
                ax2 = plt.subplot2grid((2,1), (1,0), rowspan = 1, colspan = 1, sharex = ax1, sharey = ax1)
                ax1.xaxis_date()
                ax2.xaxis_date()

                ax1.axvline(x = label, alpha = 0.3)
                ax2.axvline(x = label, alpha = 0.3)
                ax1.title.set_text('Tikri duomenys')
                ax2.title.set_text('Suskaičiuoti duomenys')

            for _, s in x_test_to_plot.iterrows():
                for a in ['ROD_KOD', 'veikla', 'DARB', 'PAJAMOS_EUR', 'VLST_NR']:
                    s = s.drop(a)
                s.index = pd.to_datetime(s.index)
                if save_plots:
                    ax1.plot(s)

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
            #print(model.coef_)

            if len(x_test) > 0:
                predictions = model.predict(x_test)

                sum, total = 0, 0
                for i in range(len(predictions)):
                    if relatively_equal(predictions[i], y_test[i]):
                        sum += 1
                    total += 1
                print(sum, total, sum/total)

                x_test_to_plot[label] = [int(a) for a in predictions]
                x_test_to_plot = x_test_to_plot.reindex(sorted(x_test_to_plot.columns.values), axis = 1)
    
                for _, s in x_test_to_plot.iterrows():
                    new_column = pd.DataFrame({label: [s[label]]}, index = [s.name])
                    maindf.update(new_column, errors = 'ignore')

                    if save_plots:
                        for a in ['COMPANY', 'ROD_KOD', 'VLST_KODAS2', 'veikla', 'DARB', 'PAJAMOS_EUR', 'VLST_NR']:
                            try:
                                s = s.drop(a)
                            except:
                                pass
                        s.index = pd.to_datetime(s.index)
                        ax2.plot(s)

                #maindf.to_csv('csvs/predict2_updated.csv', sep = '\t', encoding = 'utf-16', index = False)

                if save_plots:
                    fig = plt.gcf()
                    fig.set_size_inches((15, 18), forward = False)
                    plt.savefig('csvs/%s_%s.png' % (label, str(rod_kod)), dpi = 600, bbox_inches = 'tight')

            #except Exception as e:
            #    print(e)
    return 0

def train_model_linear_regression_barebones():
    label = '2017-10-01'
    rod_kod = 1030

    df = pd.read_csv('main.csv', encoding = 'utf-16', sep = '\t')

    df = df[np.isfinite(df[label])]
    df = df.loc[df['ROD_KOD'] == rod_kod]
    X = shuffle(df)
    for name in ['COMPANY', 'VLST_KODAS2', 'ROD_KOD']:
        X = X.drop(name, axis = 1)

    split = 0.15
    x_test = X.iloc[:int(len(X) * split)]
    x_test_to_plot = x_test
    x_test = x_test.fillna(-999)
    y_test = x_test[label]
    x_train = X.iloc[int(len(X) * split):]
    y_train = x_train[label]

    fig = plt.figure()
    fig.suptitle('Rodiklio kodas: ' + str(rod_kod), fontsize = 16)
    ax1 = plt.subplot2grid((2,1), (0,0), rowspan = 1, colspan = 1)
    ax2 = plt.subplot2grid((2,1), (1,0), rowspan = 1, colspan = 1, sharex = ax1, sharey = ax1)
    ax1.xaxis_date()
    ax2.xaxis_date()

    ax1.axvline(x = label, alpha = 0.3)
    ax2.axvline(x = label, alpha = 0.3)
    ax1.title.set_text('Tikri duomenys')
    ax2.title.set_text('Suskaičiuoti duomenys')

    for _, s in x_test_to_plot.iterrows():
        s.index = pd.to_datetime(s.index)
        ax1.plot(s)

    x_train = x_train.drop(label, axis = 1)
    x_test = x_test.drop(label, axis = 1)

    x_train = x_train.reindex(sorted(x_train.columns.values), axis = 1)
    x_train = x_train.fillna(-999)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.coef_)

    predictions = model.predict(x_test)

    sum, total = 0, 0
    for i in range(len(predictions)):
        if relatively_equal(predictions[i], y_test[i]):
            sum += 1
        total += 1

    print(sum, total, sum/total)

    x_test_to_plot[label] = predictions
    x_test_to_plot = x_test_to_plot.reindex(sorted(x_test_to_plot.columns.values), axis = 1)

    
    for _, s in x_test_to_plot.iterrows():
        s.index = pd.to_datetime(s.index)
        ax2.plot(s)

    x_test_to_plot.to_csv('predict.csv', sep = '\t', encoding = 'utf-16', index = False)

    plt.show()

def tidy_up_file(filename, encoding = 'utf-16'):
    with open(filename, 'r', encoding = encoding) as f:
        file = f.read()
        file = file.split('\n')
        file = [line.split('\t') for line in file]

    # isimti visus -9999999 ir float paversti i int
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
        full = [a for a in line if a != '']
        #if (len(full) - 3) / (len(line) - 3) > 0.75:
        new_file = new_file + '\n' + new_line[1:]

    with open(filename, 'w', encoding = 'utf-16') as f:
        f.write(new_file[1:])

    return 0

def train_model_decision_trees_test():
    save_plots = False
    pd.options.mode.chained_assignment = None

    df = pd.read_csv('main.csv', encoding = 'utf-16', sep = '\t')
    rod_kods = list(set(df['ROD_KOD'].astype(int)))
    
    if save_plots:
        fig = plt.figure()

    sum, total = 0, 0
    estimators = [ExtraTreesRegressor(), BayesianRidge(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
    f = open('performance.txt', 'w')

    for estimator in estimators:
        for rod_kod in rod_kods:
            try:
                maindf = pd.read_csv('csvs/predict.csv', encoding = 'utf-16', sep = '\t')

                df = maindf.loc[maindf['ROD_KOD'] == rod_kod]
                X = shuffle(df)

                for name in ['COMPANY', 'VLST_KODAS2', 'ROD_KOD']:
                    X = X.drop(name, axis = 1)

                split = 0.15
                x_test = X.iloc[:int(len(X) * split)]
                x_test_to_plot = x_test
                x_test = x_test.fillna(-9999999)
                x_train = X.iloc[int(len(X) * split):]
                # remove random cell values for testing
        
                total_values_changed = 0
                sample = x_test.sample(int(len(X) * split * split))
                cols = list(sample.columns.values)

                for _, s in sample.iterrows():
                    m = random.randint(0, len(s)-1)
                    if x_test.at[s.name, cols[m]] != -9999999:
                        x_test.at[s.name, cols[m]] = np.nan
                    n = random.randint(0, len(s)-1)
                    if x_test.at[s.name, cols[m]] != -9999999:
                        x_test.at[s.name, cols[m]] = np.nan
                    if m != n:
                        total_values_changed += 2
                    elif m == n:
                        total_values_changed += 1

                x_test_is_nan = x_test.isnull()
                x_test = x_test.replace(-9999999, np.nan)

                if save_plots:
                    plt.clf()
                    fig.suptitle('Rodiklio kodas: ' + str(rod_kod), fontsize = 16)
                    ax1 = plt.subplot2grid((2,1), (0,0), rowspan = 1, colspan = 1)
                    ax2 = plt.subplot2grid((2,1), (1,0), rowspan = 1, colspan = 1, sharex = ax1, sharey = ax1)
                    ax1.xaxis_date()
                    ax2.xaxis_date()

                    ax1.title.set_text('Tikri duomenys')
                    ax2.title.set_text('Suskaičiuoti duomenys')

                for _, s in x_test_to_plot.iterrows():
                    s.index = pd.to_datetime(s.index)
                    if save_plots:
                        ax1.plot(s)

                x_train = x_train.reindex(sorted(x_train.columns.values), axis = 1)
                x_train = np.array(x_train)
                x_test = np.array(x_test)

                imp = IterativeImputer(estimator = estimator, missing_values = np.nan)
                imp.fit(x_train)

                if len(x_test) > 0:
                    x_test = imp.transform(x_test)

                    rows, cols = x_test.shape
                    x_test = pd.DataFrame(x_test)
                    for row in range(rows):
                        for col in range(cols):
                            if list(x_test_is_nan.iloc[row])[col]:
                                if relatively_equal(list(x_test.iloc[row])[col], list(x_test_to_plot.iloc[row])[col]):
                                    sum += 1
                                total += 1

                    x_test_to_plot = x_test_to_plot.reindex(sorted(x_test_to_plot.columns.values), axis = 1)
    
                    for _, s in x_test_to_plot.iterrows():
                        maindf.update(x_test, errors = 'ignore')

                        if save_plots:
                            s.index = pd.to_datetime(s.index)
                            ax2.plot(s)

                    #maindf.to_csv('csvs/predict.csv', sep = '\t', encoding = 'utf-16', index = False)

                    if save_plots:
                        fig = plt.gcf()
                        fig.set_size_inches((15, 18), forward = False)
                        plt.savefig('csvs/%s_%s.png' % (random.randint(1, 200), str(rod_kod)), dpi = 600, bbox_inches = 'tight')

            except Exception as e:
                print(e)
            #break

        if total > 0:
            print(str(estimator).split('(')[0])
            print(sum, total)
            print(sum/total)
            print('\n')

            f.write(str(estimator).split('(')[0])
            f.write('\ntotal: %i / %i\n' % (sum, total))
            f.write('score: %.1f%%\n\n' % (100*sum/total))

    return 0

def train_model_decision_trees():
    save_plots = True
    pd.options.mode.chained_assignment = None

    df = pd.read_csv('main.csv', encoding = 'utf-16', sep = '\t')
    rod_kods = list(set(df['ROD_KOD'].astype(int)))
    
    if save_plots:
        fig = plt.figure()

    estimators = [ExtraTreesRegressor(), BayesianRidge(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]

    for estimator in estimators:
        for rod_kod in rod_kods:
            try:
                maindf = pd.read_csv('csvs/predict_%s.csv' % str(estimator).split('(')[0], encoding = 'utf-16', sep = '\t')

                df = maindf.loc[maindf['ROD_KOD'] == rod_kod]
                X = shuffle(df)

                for name in ['COMPANY', 'VLST_KODAS2']:
                    X = X.drop(name, axis = 1)

                # drop nan columns here
                for col in X:
                    if X[col].isnull().all():
                        X = X.drop(col, axis = 1)

                index = list(X.index)
                columns = list(X.columns.values)

                imp = IterativeImputer(estimator = estimator, missing_values = np.nan)
                imp.fit(X)

                if len(X) > 0:
                    X = imp.transform(X)
                    X = pd.DataFrame(data = X, index = index, columns = columns)
                    maindf.update(X, errors = 'ignore')

                    maindf.to_csv('csvs/predict_%s.csv' % str(estimator).split('(')[0], sep = '\t', encoding = 'utf-16', index = False)

            except Exception as e:
                print(e)
    return 0

def uzpildyti_pagal_rodiklius(filename):
    rod_kods = {'050': '000', '100': '060', '150': '110', '280': '260', '310': '290', '324': '320'}
    main_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    comps = list(set(main_df['COMPANY']))
    countries = list(set(main_df['VLST_KODAS2']))

    #comps = ['X00001', 'X00003', 'X00002', 'X00008']
    #countries = ['VG', 'FI', 'CA', 'DE']
    countries.sort()
    comps.sort()

    total = len(comps) * len(countries)
    count = 0

    main_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')

    for comp in comps:
        for country in countries:
            count += 1
            if count % 100 == 0:
                print(count, '\\', total)

            df = main_df

            df = df.loc[df['COMPANY'] == comp]
            df = df.loc[df['VLST_KODAS2'] == country]
            if not df.empty:
                df = df.fillna(np.nan)
                cols = list(df.columns.values)
                for name in ['COMPANY', 'VLST_KODAS2', 'ROD_KOD']:
                    cols.remove(name)

                df['ROD_KOD'] = df['ROD_KOD'].astype(int)
                for j in ['1', '2', '3', '4']:
                    for t in rod_kods:
                        t_df = df[df['ROD_KOD'].astype(str) == j + t]
                        if not t_df.empty:
                            tminus_df = df[df['ROD_KOD'].astype(str) == j + rod_kods[t]]
                            if not tminus_df.empty:
                                try:
                                    for i in range(len(cols) - 1):
                                        if not list(t_df[cols[i]]) == list(tminus_df[cols[i+1]]):
                                            if not float(list(t_df[cols[i]])[0]) == float(list(t_df[cols[i]])[0]):
                                                t_df.loc[:, cols[i]] = list(tminus_df[cols[i+1]])[0]
                                            if not float(list(tminus_df[cols[i+1]])[0]) == float(list(tminus_df[cols[i+1]])[0]):
                                                tminus_df.loc[:, cols[i+1]] = list(t_df[cols[i]])[0]

                                    main_df.update(t_df)
                                    main_df.update(tminus_df)
                                except Exception as e:
                                    print(comp, country, '\n', e)

    dot_position = filename.rfind('.')
    name = filename[:dot_position] + '_updated' + filename[dot_position:]
    main_df.to_csv(name, sep = '\t', encoding = 'utf-16', index = False)
    print('file save succesfully')

    return 0

def add_metadata(filename):
    df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    mdf = pd.read_csv('csvs/TUI_papildoma_info.csv', encoding = 'utf-16', sep = '\t')
    comps = list(set(df['COMPANY']))
    comps.sort()#reverse = True)
    mdf_cols = list(mdf.columns.values)
    mdf_cols.remove('kodas')

    print(df)

    for col in mdf_cols:
        df[col] = [np.nan] * len(df)

    for comp in comps:

        new_df = df[df['COMPANY'] == comp]
        print(comp, df.shape)
        if 'series' in str(type(new_df)).lower():
            if not new_df.empty:
                name = new_df.name
                arr = mdf.loc[mdf['kodas'] == comp]
                arr = arr.iloc[0]
                new_df.update(arr)
                new_df.name = name
                df.loc[name] = new_df
        else:
            for _, s in new_df.iterrows():
                name = s.name
                arr = mdf.loc[mdf['kodas'] == comp]
                if not arr.empty:
                    arr = arr.iloc[0]
                    s.update(arr)
                    s.name = name
                    df.loc[name] = s

    df = df.sort_values(by = ['COMPANY', 'ROD_KOD'])
    dot_position = filename.rfind('.')
    name = filename[:dot_position] + '_updated' + filename[dot_position:]
    df.to_csv(name, encoding = 'utf-16', sep = '\t', index = False)

def prideti_saliu_numerius(filename):
    df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    countries = list(set(df['VLST_KODAS2']))
    countries.sort()
    df['VLST_NR'] = df['VLST_KODAS2'].apply(lambda x: countries.index(x))
    df = df.sort_values(['COMPANY', 'VLST_NR', 'ROD_KOD'])
    df.to_csv(filename, encoding = 'utf-16', sep = '\t', index = False)
    return 0

