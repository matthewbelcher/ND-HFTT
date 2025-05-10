import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

def get_means(path):
    means_spy_df = pd.DataFrame(columns = ['date', 'pre-release bid mean', 'pre-release ask mean', 'post-release bid mean', 'post-release ask mean'])
    means_voo_df = pd.DataFrame(columns = ['date', 'pre-release bid mean', 'pre-release ask mean', 'post-release bid mean', 'post-release ask mean'])


    files = sorted(os.listdir(path))

    for file in files:
        df = pd.read_csv(path + file)
        df['TIME'] = pd.to_datetime(df['TIME_M'])
        df['TIME_IN_SEC'] = df['TIME'].dt.nanosecond * 1e-9 + df['TIME'].dt.microsecond * 1e-6 + df['TIME'].dt.second + df['TIME'].dt.minute * 60 + df['TIME'].dt.hour * 3600
        df['TIME_FROM_RELEASE'] = df['TIME_IN_SEC'] - 14*3600.0000000
        df['ASK_CLEAN'] = df['ASK'][df['ASK'] > 5]
        df['BID_CLEAN'] = df['BID'][df['BID'] > 5]

        pre_ask_mean = df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'SPY')].mean()
        pre_bid_mean = df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'SPY')].mean()

        post_ask_mean = df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'SPY')].mean()
        post_bid_mean = df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'SPY')].mean()

        means_spy_df.loc[len(means_spy_df)] = [file, pre_bid_mean, pre_ask_mean, post_bid_mean, post_ask_mean]

        pre_ask_mean = df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'VOO')].mean()
        pre_bid_mean = df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] < 0) & (df['SYM_ROOT'] == 'VOO')].mean()

        post_ask_mean = df['ASK_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'VOO')].mean()
        post_bid_mean = df['BID_CLEAN'][(df['TIME_FROM_RELEASE'] > 0) & (df['SYM_ROOT'] == 'VOO')].mean()

        means_voo_df.loc[len(means_voo_df)] = [file, pre_bid_mean, pre_ask_mean, post_bid_mean, post_ask_mean]
    return files, means_spy_df, means_voo_df


def plot_means(path='data/market_ticks/'):
    files, means_spy_df, means_voo_df = get_means(path)
    means_spy_plot = means_spy_df.plot()
    means_spy_plot.set_xticks(range(10))
    means_spy_plot.set_xticklabels(files)
    plt.title('SPY')
    #plt.show()

    means_voo_plot = means_voo_df.plot()
    means_voo_plot.set_xticks(range(10))
    means_voo_plot.set_xticklabels(files)
    plt.title('VOO')
    #plt.show()

    return(means_spy_plot, means_voo_plot)



means_spy_plot, means_voo_plot = plot_means()
#plt.show()