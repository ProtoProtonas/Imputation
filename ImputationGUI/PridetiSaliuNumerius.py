import pandas as pd

def prideti_saliu_numerius(filename):
    df = pd.read_csv(filename, encoding = 'utf-16', sep = '\t')
    countries = list(set(df['VLST_KODAS2']))
    countries.sort()
    df['VLST_NR'] = df['VLST_KODAS2'].apply(lambda x: countries.index(x))
    df = df.sort_values(['COMPANY', 'VLST_NR', 'ROD_KOD'])
    df.to_csv(filename, encoding = 'utf-16', sep = '\t', index = False)
    return 0

prideti_saliu_numerius('pngs/predict2.csv')
