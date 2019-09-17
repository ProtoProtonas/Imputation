import pandas as pd
import numpy as np


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
        new_file = new_file + '\n' + new_line[1:]

    with open(filename, 'w', encoding = 'utf-16') as f:
        f.write(new_file[1:])

    return 0

def uzpildyti_pagal_rodiklius(filename):
    rod_kods = {'050': '000', '100': '060', '150': '110', '280': '260', '310': '290', '324': '320'}
    main_df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    comps = list(set(main_df['COMPANY']))
    countries = list(set(main_df['VLST_KODAS2']))

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
    tidy_up_file(name)
    print('Failas išsaugotas sėkmingai')

    return 0

uzpildyti_pagal_rodiklius('csvs/predict2.csv')