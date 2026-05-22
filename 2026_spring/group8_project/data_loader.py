import os
import pandas as pd
import databento as db
from dotenv import load_dotenv

load_dotenv()  #For reading the API key


def get_client():
    api_key = os.environ.get("DATABENTO_API_KEY")
    return db.Historical(api_key)


def fetch_book(client, start, end, limit=None):
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="mbp-10",
        symbols=["ES.c.0"],
        stype_in="continuous",
        start=start,
        end=end,
        **({"limit": limit} if limit else {}),
    )

    return data.to_df()

def fetch_trades(client, start, end):
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="trades",
        symbols=["ES.c.0"],
        stype_in="continuous",
        start=start,
        end=end,
    )
    return data.to_df()

def process_book(raw_df):
    df = raw_df.copy()

    #Best bid and ask at level 1
    df['bid_price_1'] = df['bid_px_00']
    df['ask_price_1'] = df['ask_px_00']
    df['bid_qty_1']   = df['bid_sz_00']
    df['ask_qty_1']   = df['ask_sz_00']

    #Mid price and spread
    df['mid_price'] = (df['bid_px_00'] + df['ask_px_00']) / 2
    df['spread']    = df['ask_px_00'] - df['bid_px_00']

    #L1 Order Book Imbalance
    df['imbalance_l1'] = (
        (df['bid_sz_00'] - df['ask_sz_00']) /
        (df['bid_sz_00'] + df['ask_sz_00'])
    )

    #L3 order book imbalance (sum of the top 3 levels)
    bid_depth = df['bid_sz_00'] + df['bid_sz_01'] + df['bid_sz_02']
    ask_depth = df['ask_sz_00'] + df['ask_sz_01'] + df['ask_sz_02']
    df['imbalance_l3'] = (bid_depth - ask_depth) / (bid_depth + ask_depth)

    #Filter the dataset
    cols = [
        'mid_price', 'spread',
        'bid_price_1', 'ask_price_1',
        'bid_qty_1', 'ask_qty_1',
        'imbalance_l1', 'imbalance_l3'
    ]

    df = df[cols]

    df = df[
        (df['mid_price'] > 0) &
        (df['ask_price_1'] > 0) &
        (df['spread'] > 0) &
        (df['ask_price_1'] > df['bid_price_1'])
    ].reset_index(drop=True)

    return df

def process_trades(raw_df):
    df = raw_df.copy()

    #Keep only what the VWAP baseline needs
    df = df[['ts_event', 'price', 'size']].copy()
    df = df.rename(columns={"price": "trade_price", "size": "trade_size"})

    #Drop any rows where the price or size is zero
    df = df[(df["trade_price"] > 0) & (df["trade_size"] > 0)].reset_index(drop=True)

    return df






    