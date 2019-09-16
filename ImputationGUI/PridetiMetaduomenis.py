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

def add_metadata(filename):
    df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    mdf = pd.read_csv('pngs/TUI_papildoma_info.csv', encoding = 'utf-16', sep = '\t')
    comps = list(set(df['COMPANY']))
    comps.sort()
    mdf_cols = list(mdf.columns.values)
    mdf_cols.remove('kodas')

    for col in mdf_cols:
        df[col] = [np.nan] * len(df)

    for comp in comps:

        new_df = df[df['COMPANY'] == comp]
        print(comp)
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
    tidy_up_file(name)


add_metadata('pngs/predict2_updated.csv')

