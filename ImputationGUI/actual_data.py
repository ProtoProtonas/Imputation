
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import time

from matplotlib import style
from pandas.plotting import register_matplotlib_converters
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import shuffle

register_matplotlib_converters()
KEYS = []

def vlst_nr(country_code):
    if country_code in KEYS:
        return int(KEYS.index(country_code) + 1)
    else:
        KEYS.append(country_code)
        return int(KEYS.index(country_code) + 1)
    return 0

def add_period(year, quarter):
    return str(year) + '-' + str(quarter)

def relatively_equal(val1, val2):
    percentage = 0.2
    val1ten = percentage * val1
    val2ten = percentage * val2

    if val2 > (val1 - val1ten) and val2 < (val1 + val1ten):
        return True
    elif val1 > (val2 - val2ten) and val1 < (val2 + val2ten):
        return True
    return False

def save_full_dfs():
    if not os.path.isdir('tui_data'):
        os.mkdir('tui_data')

    df = pd.read_csv('TUI_praleistos_2012_2018_pilnas.csv', sep = ';', low_memory = False)
    df = shuffle(df)
    df['Vlst_nr'] = df['VLST_KODAS2'].apply(vlst_nr)

    all_dfs = {}
    index = df.columns.values

    for i, s in enumerate(df.iterrows()):
        _, row = s
        if row['kodas'] in all_dfs:
            df1 = all_dfs[row['kodas']]
            df1 = df1.append(row, ignore_index = True)
            all_dfs[row['kodas']] = df1
        else:
            all_dfs[row['kodas']] = pd.DataFrame()
            df1 = all_dfs[row['kodas']]
            df1 = df1.append(row, ignore_index = True)
            all_dfs[row['kodas']] = df1
        if i % 500 == 0:
            print(i)
        #if i == 10000:
        #    break

    for key in all_dfs:
        df = all_dfs[key]
        df = df[['D1', 'D2', 'D3', 'ROD_KOD', 'VLST_KODAS2', 'Vlst_nr', 'PER_METAI', 'PER_PERIODAS_PAV', 'kodas']]
        for col in ['ROD_KOD', 'D1', 'D2', 'D3', 'PER_METAI', 'PER_PERIODAS_PAV', 'Vlst_nr']:
            df[col] = pd.to_numeric(df[col], errors = 'coerce', downcast = 'integer')

        df.reset_index(drop = True, inplace = True)
        df = df.sort_values(['PER_METAI', 'PER_PERIODAS_PAV'])

        df.to_csv('tui_data/%s.csv' % key, encoding = 'utf-16', sep = '\t', index = False)

    pickle_out = open('all_dfs.pickle', 'wb')
    pickle.dump(all_dfs, pickle_out)
    pickle_out.close()

def impute_tui():
    path = 'tui_data_d1/'
    filenames = os.listdir(path)
    filenames = [path + name for name in filenames if name.startswith('X')]

    main_df = pd.DataFrame()
    for n, filename in enumerate(filenames):
        df = pd.read_csv(filename, sep = '\t', low_memory = False, encoding = 'utf-16')
        main_df = main_df.append(df, ignore_index = True)
        if n % 100 == 0:
            print(n)
             
    main_df = main_df.drop(['VLST_KODAS2', 'kodas'], axis = 1)
    index = main_df.columns.values
    print(index)
    print(main_df)

    estimators = [ExtraTreesRegressor(), BayesianRidge(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor()]
    f = open('performance.txt', 'w')

    for estimator in estimators:

        imp = IterativeImputer(estimator = estimator, missing_values = np.nan)
        imp.fit(main_df)

        df = pd.read_csv(os.path.join(path, 'test.csv'), sep = '\t', low_memory = False, encoding = 'utf-16')
        df = df.drop(['VLST_KODAS2', 'kodas'], axis = 1)
        df = imp.transform(df)

        df = pd.DataFrame(df)
        df.columns = index
        df.to_csv(os.path.join(path, 'test_imp.csv'), sep = '\t', encoding = 'utf-16')

        df1 = pd.read_csv(os.path.join(path, 'X00411.csv'), sep = '\t', low_memory = False, encoding = 'utf-16')
        df1 = df1.drop(['VLST_KODAS2', 'kodas'], axis = 1)
        df2 = pd.read_csv(os.path.join(path, 'test_imp.csv'), sep = '\t', low_memory = False, encoding = 'utf-16')

        score, total = 0, 0
        for i in range(1, 400):
            val1 = df1.iloc[i]
            val2 = df2.iloc[i]
            val1 = int(val1['D1'])
            val2 = int(val2['D1'])
            if relatively_equal(val1, val2):
                print(val1, val2)
                score += 1
            total += 1

        print(str(estimator).split('(')[0])
        print('score: ', score / total)

        f.write(str(estimator).split('(')[0])
        f.write('\ntotal: %i / %i\n' % (score, total))
        f.write('score: %.1f%%\n\n' % (100*score/total))

def process_company_dfs():
    if not os.path.isdir('tuiproc'):
        os.mkdir('tuiproc')
    path = 'tui_data/'
    filenames = os.listdir(path)
    filenames = [path + name for name in filenames if name.startswith('X')]

    for name in filenames:
        df = pd.read_csv(name, sep = '\t', low_memory = False, encoding = 'utf-16')
        #df['PERIODAS'] = df['PER_METAI'].map(str) + ' Q' +  df['PER_PERIODAS_PAV'].map(str)
        df['PERIODAS'] = df['PER_METAI'].map(str) + '-' +  (3 * df['PER_PERIODAS_PAV'].astype(int) - 2).map(str) + '-01'
        df['PERIODAS'] = pd.to_datetime(df['PERIODAS'])
        df['PERIODAS'] = df['PERIODAS'].astype(str)
        df = df.drop(['kodas', 'D2', 'D3', 'PER_METAI', 'PER_PERIODAS_PAV'], axis = 1)

        df = df.sort_values('ROD_KOD')
        df = df.set_index('ROD_KOD')
        new_df = pd.DataFrame()
        
        if 10 in df.index and 20 in df.index:
            open = df.loc[10]['D1']
            close = df.loc[20]['D1']
            share_price = (open, close)
            df = df.drop([10, 20])
        elif 10 in df.index:
            df = df.drop(10)
        elif 20 in df.index:
            df = df.drop(20)
            
        rod_kods = list(set(df.index))
        rod_kods.sort()
        rod_kods = [str(a) for a in rod_kods]

        full_data = {}

        for rk in rod_kods:
            full_data[rk] = {}

        for _, line in df.iterrows():
            rod_kod = str(line.name)
            period = line['PERIODAS']
            country_code = line['VLST_KODAS2']
            country_number = line['Vlst_nr']
            if country_code in full_data[rod_kod]:
                full_data[rod_kod][country_code].append((period, line['D1']))
            else:
                full_data[rod_kod][country_code] = [(period, line['D1'])]
        
        new_df = pd.DataFrame()
        for rk in full_data:
            for country in full_data[rk]:
                index = ['ROD_KOD', 'VLST_KODAS2']
                data = [rk, country]
                for period, value in full_data[rk][country]:
                    if period not in index:
                        index.append(period)
                        data.append(value)

                s = pd.Series(data = data, index = index)
                new_df = new_df.append(s, ignore_index = True)

        new_df = new_df.reindex(sorted(new_df.columns.values), axis = 1)

        cols = list(new_df.columns.values)
        cols.remove('VLST_KODAS2')
        for col in cols:
            new_df[col] = pd.to_numeric(new_df[col], errors = 'coerce', downcast = 'integer')
            new_df[col] = new_df[col].astype(pd.Int64Dtype())

        new_df.to_csv('tuiproc/' + name.split('/')[1], sep = '\t', encoding = 'utf-16', index = False)
        print('Saved ' + 'tuiproc/' + name.split('/')[1])

def plot_data():
    path = 'tuiproc/'
    filenames = os.listdir(path)
    filenames = [path + name for name in filenames if name.startswith('X')]
    #filenames = filenames[:1000]

    #'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 
    #'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 
    #'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 
    #'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 
    #'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test'

    style.use('seaborn-poster')
    fig = plt.figure()
    ax1 = plt.subplot2grid((2,1), (0,0), rowspan = 1, colspan = 1)
    ax2 = plt.subplot2grid((2,1), (1,0), rowspan = 1, colspan = 1)
    ax1.xaxis_date()
    ax2.xaxis_date()

    for name in filenames:
        print('Opening', name)
        df = pd.read_csv(name, encoding = 'utf-16', sep = '\t')
        #df = df.fillna(np.nan)
        df = df.set_index('ROD_KOD')

        df_cols = list(df.columns.values)
        df_cols.remove('VLST_KODAS2')
        for col in df_cols:
            df[col] = df[col].astype(pd.Int64Dtype())

        #ax1.set_xticklabels(df_cols)
        try:
            # 'linear', 'time', 'index', 'values', 'pad', 'nearest', 'zero', 'slinear', 
            # 'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial', 'krogh', 
            # 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'from_derivatives',
            interpolation_method = 'polynomial'

            df = df.loc[1050]
            if 'series' in str(type(df)).lower():
                if 'VLST_KODAS2' in list(df.index):
                    s = df.drop('VLST_KODAS2')
                else:
                    s = df
                s.index = pd.to_datetime(s.index)
                s = s.astype(float)
                ax1.plot(s)
                s = s.interpolate(method = interpolation_method, order = 5, limit = 5, limit_area = 'inside')
                ax2.plot(s)
            else:
                for _, s in df.iterrows():
                    if 'VLST_KODAS2' in list(s.index):
                        s = s.drop('VLST_KODAS2')
                    s.index = pd.to_datetime(s.index)
                    s = s.astype(float)
                    ax1.plot(s)
                    s = s.interpolate(method = interpolation_method, order = 5, limit = 5, limit_area = 'inside')
                    ax2.plot(s)
        except Exception as e:
            print(e)
            pass

    plt.show()

def save_interpolated_data():
    path = 'tuiproc/'
    filenames = os.listdir(path)
    filenames = [path + name for name in filenames if name.startswith('X')]
    #filenames = filenames[:10]

    for name in filenames:
        print('Opening', name)
        df = pd.read_csv(name, encoding = 'utf-16', sep = '\t')
        df = df.set_index('ROD_KOD')

        df_cols = list(df.columns.values)
        df_cols.remove('VLST_KODAS2')
        for col in df_cols:
            df[col] = df[col].astype(pd.Int64Dtype())

        new_df = pd.DataFrame()

        # 'linear', 'time', 'index', 'values', 'pad', 'nearest', 'zero', 'slinear', 
        # 'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial', 'krogh', 
        # 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'from_derivatives',

        for _, s in df.iterrows():
            try:
                if 'VLST_KODAS2' in list(s.index):
                    int_s = s.drop('VLST_KODAS2')
                else:
                    int_s = s
                int_s.index = pd.to_datetime(int_s.index, errors = 'ignore', format = '%Y-%m-%d', yearfirst = True)
                int_s = int_s.astype(float)
                int_s = int_s.interpolate(method = 'linear', limit = 5)
                int_s = int_s.append(pd.Series(s['VLST_KODAS2'], index = ['VLST_KODAS2']))
                int_s = int_s.append(pd.Series(s.name, index = ['ROD_KOD']))
                new_df = new_df.append(int_s, ignore_index = True)
            except Exception as e:
                #print(e)
                int_s = int_s.astype(float)
                int_s = int_s.append(pd.Series(int(s.name), index = ['ROD_KOD']))
                int_s = int_s.append(pd.Series(s['VLST_KODAS2'], index = ['VLST_KODAS2']))
                new_df = new_df.append(int_s, ignore_index = True)

        name_to_save = name.split('/')[-1]
        name_to_save = name_to_save.split('.')[0] + 'INT.csv'

        new_df = new_df.fillna(-2147483648)

        for col in new_df:
            new_df[col] = pd.to_numeric(new_df[col], downcast = 'integer', errors = 'ignore')

        new_df.to_csv(os.path.join(path, name_to_save), sep = '\t', encoding = 'utf-16', index = False)

        with open(os.path.join(path, name_to_save), 'r', encoding = 'utf-16') as f:
            file = f.read()
            file = file.split('\n')
            file = [line.split('\t') for line in file]

        # nuo indekso nuimti 00:00:00
        for i in range(len(file[0])):
            if ' 00:00:00' in file[0][i]:
                file[0][i] = file[0][i].replace(' 00:00:00', '')

        # isimti visus -2147483648 ir float paversti i int
        for i in range(len(file)):
            for j in range(len(file[i])):
                if '.' in file[i][j]:
                    file[i][j] = file[i][j].split('.')[0]
                if '-2147483648' in file[i][j]:
                    file[i][j] = file[i][j].replace('-2147483648', '')


        new_file = ''
        for line in file:
            new_line = ''
            for cell in line:
                new_line = new_line + '\t' + cell
            new_file = new_file + '\n' + new_line[1:]

        with open(os.path.join(path, name_to_save), 'w', encoding = 'utf-16') as f:
            f.write(new_file[1:])

    return


#plot_data()
save_interpolated_data()


