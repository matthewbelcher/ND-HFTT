import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np

def plot_bid_ask_vs_time(csv):
    df = pd.read_csv(csv)

    df['TIME'] = pd.to_datetime(df['TIME_M'])
    df['TIME_IN_SEC'] = df['TIME'].dt.nanosecond * 1e-9 + df['TIME'].dt.microsecond * 1e-6 + df['TIME'].dt.second + df['TIME'].dt.minute * 60 + df['TIME'].dt.hour * 3600
    df['TIME_FROM_RELEASE'] = df['TIME_IN_SEC'] - 14*3600.0000000

    df['ASK_CLEAN'] = df['ASK'][df['ASK'] > 5]
    df['BID_CLEAN'] = df['BID'][df['BID'] > 5]

    df[df['SYM_ROOT'] == 'SPY'].plot(kind='line', x='TIME_FROM_RELEASE', y = ['BID_CLEAN', 'ASK_CLEAN'])
    plt.title('SPY')
    plt.legend(["Bid", "Ask"])
    plt.show()

    df[df['SYM_ROOT'] == 'VOO'].plot(kind='line', x='TIME_FROM_RELEASE', y = ['BID_CLEAN', 'ASK_CLEAN'])
    plt.title('VOO')
    plt.legend(["Bid", "Ask"])
    plt.show()

def get_averages(csv):
    df = pd.read_csv(csv)
    df['TIME'] = pd.to_datetime(df['TIME_M'])
    df['TIME_IN_SEC'] = df['TIME'].dt.nanosecond * 1e-9 + df['TIME'].dt.microsecond * 1e-6 + df['TIME'].dt.second + df['TIME'].dt.minute * 60 + df['TIME'].dt.hour * 3600
    df['TIME_FROM_RELEASE'] = df['TIME_IN_SEC'] - 14*3600.0000000
    df['ASK_CLEAN'] = df['ASK'][df['ASK'] > 5]
    df['BID_CLEAN'] = df['BID'][df['BID'] > 5]

    print('\nSPY BEFORE RELEASE:')
    print((df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'SPY')]).apply([np.mean, np.max, np.min, np.var]))
    print((df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'SPY')]).apply([np.mean, np.max, np.min, np.var]))

    print('\nSPY AFTER RELEASE:')
    print((df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'SPY')]).apply([np.mean, np.max, np.min, np.var]))
    print((df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'SPY')]).apply([np.mean, np.max, np.min, np.var]))

    print('\nVOO BEFORE RELEASE:')
    print((df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'VOO')]).apply([np.mean, np.max, np.min, np.var]))
    print((df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'VOO')]).apply([np.mean, np.max, np.min, np.var]))

    print('\nVOO AFTER RELEASE:')
    print((df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'VOO')]).apply([np.mean, np.max, np.min, np.var]))
    print((df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'VOO')]).apply([np.mean, np.max, np.min, np.var]))



if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_bid_ask_vs_time(f'data/market_ticks/{sys.argv[1]}.csv')
        get_averages(f'data/market_ticks/{sys.argv[1]}.csv')
    else:
        plot_bid_ask_vs_time('data/market_ticks/24_01_31.csv')
        get_averages('data/market_ticks/24_01_31.csv')



# initial observations
# on 12/18/2024, rate was lowered
# bid prices dropped drastically, ask prices stayed the same

# on 05/01/2024, rate stayed the same
# bid prices decreases while ask prices increased



