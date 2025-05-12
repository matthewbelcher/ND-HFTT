import os

path = 'data/market_ticks/'
files = os.listdir(path)

for file in files:
    date = file.split('.csv')
    month, day, year = date[0].split('_')
    try:
        os.rename(path + file, f'{path}{year}_{month}_{day}.csv')
    except:
        print(f"did not rename {file}")