import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from plot_means import plot_means

def plot_means_and_rates(path='data/market_ticks/'):
    effr = pd.read_csv('data/effr.csv')
    means_spy_plot, means_voo_plot = plot_means()
    files = sorted(os.listdir(path))

    plot_rates = []

    for file in files:
        date = file.split('.csv')
        year, month, day = date[0].split('_')
        rate = effr.loc[effr['date'] == f'20{year}-{month}-{day}', 'rate']
        plot_rates.append(rate.values)

    rates_df = pd.DataFrame(plot_rates, columns=['rate'])

    # rate_plot = means_spy_plot.twinx()

    # # this is Figure 3
    # rates_df.plot(ax=rate_plot)
    # lines, labels = means_spy_plot.get_legend_handles_labels()
    # lines2, labels2 = rate_plot.get_legend_handles_labels()
    # means_spy_plot.legend(lines + lines2, labels + labels2, loc="center left")
    # rate_plot.get_legend().remove()
    # plt.show()

    # comment out the other plot to show this one
    rate_plot2 = means_voo_plot.twinx()
    rates_df_2 = pd.DataFrame(plot_rates, columns=['rate'])

    rates_df_2.plot(ax=rate_plot2)

    lines, labels = means_voo_plot.get_legend_handles_labels()
    lines2, labels2 = rate_plot2.get_legend_handles_labels()
    means_voo_plot.legend(lines + lines2, labels + labels2, loc="center left")
    rate_plot2.get_legend().remove()
    plt.show()

plot_means_and_rates()