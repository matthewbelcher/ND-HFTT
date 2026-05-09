
import csv
from datetime import datetime
from math import comb
import os 
import pandas as pd
import matplotlib.pyplot as plt

futures = {file.split('_')[0] for file in os.listdir('data/federal_funds') if file.endswith('_trade.csv')}

for contract in futures:
    with open(f'data/federal_funds/{contract}_trade.csv', 'r') as fh:
        trades = sorted([(datetime.strptime(trade[0], '%Y-%m-%d %H:%M:%S'), float(trade[1])) for trade in list(csv.reader(fh, delimiter='\t'))[1:]], key = lambda x: x[0])
        

    with open(f'data/federal_funds/{contract}_bid.csv', 'r') as fh:
        bids = sorted([(datetime.strptime(bid[0], '%Y-%m-%d %H:%M:%S'), float(bid[1])) for bid in list(csv.reader(fh, delimiter='\t'))[1:]], key = lambda x: x[0])

    with open(f'data/federal_funds/{contract}_ask.csv', 'r') as fh:
        asks = sorted([(datetime.strptime(ask[0], '%Y-%m-%d %H:%M:%S'), float(ask[1])) for ask in list(csv.reader(fh, delimiter='\t'))[1:]], key = lambda x: x[0])

    trade_index = 0
    bid_index = 0
    ask_index = 0

    combined_data = []

    while trade_index < len(trades) or bid_index < len(bids) or ask_index < len(asks):
        trade_time = trades[trade_index][0] if trade_index < len(trades) else None
        bid_time = bids[bid_index][0] if bid_index < len(bids) else None
        ask_time = asks[ask_index][0] if ask_index < len(asks) else None

        next_time = min([t for t in [trade_time, bid_time, ask_time] if t is not None], default=None)

        if next_time is None:
            break
            
        next_row = [next_time]

        print(trade_index, bid_index , ask_index)
        print(next_time, trade_time, bid_time, ask_time)

        if trade_time == next_time:
            next_row.append(float(trades[trade_index][1]))
            trade_index += 1
        else:
            next_row.append(float(trades[trade_index-1][1]) if trade_index > 0 else None)
        
        if bid_time == next_time:
            next_row.append(float(bids[bid_index][1]))
            bid_index += 1
        else:
            next_row.append(float(bids[bid_index-1][1]) if bid_index > 0 else None)

        if ask_time == next_time:
            next_row.append(float(asks[ask_index][1]))
            ask_index += 1
        else:
            next_row.append(float(asks[ask_index-1][1]) if ask_index > 0 else None)

        if any(value is None for value in next_row[1:]):
            continue

        combined_data.append(next_row)

    combined_data = pd.DataFrame(combined_data, columns=['Time', 'Trade', 'Bid', 'Ask'])

    combined_data.to_csv(f'data/federal_funds/combined/{contract}.csv', index=False, header=True)