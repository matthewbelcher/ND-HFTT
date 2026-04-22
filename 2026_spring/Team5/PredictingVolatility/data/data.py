import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class TAQ():
    def __init__(self, ticker_symbol, start_year, end_year, output_path):
        self.ticker_symbol = ticker_symbol
        self.start = start_year
        self.end = end_year
        self.conn = wrds.Connection()
        self.df = pd.DataFrame()
        self.output_path = output_path
    
    def close(self):
        self.conn.close()

    def ticker_to_id(self, ticker):
        """
        library = "optionm"
        table = "optionmnames"
        df = self.conn.raw_sql(f'''
        SELECT *
        FROM {library}.{table}
        WHERE ticker = '{ticker}';
        ''')
        print(df)
        """
        # This just returns the secid of SPY for now
        return 109820

    def plot_volatility_surface(self, entries):
        surface = entries.pivot_table(
            index="days",
            columns="delta",
            values="impl_volatility",
            aggfunc="mean"
        )

        X, Y = np.meshgrid(surface.columns.values, surface.index.values)
        Z = surface.values

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(X, Y, Z, cmap="viridis")

        ax.set_xlabel("Delta")
        ax.set_ylabel("Days to Maturity")
        ax.set_zlabel("Implied Volatility")
        ax.set_title("SPY Implied Volatility Surface")

        plt.show()

    def get_volatility_surface(self, secid, year):
        # volatility_surface_YYYY
        # tick_volatility_surface_YYYY
        # print(self.conn.describe_table("optionm", "vsurfd2025"))
        # date = The day of the volatility surface
        # days = Time to maturity of the option
        # delta = delta of the option (Moneyness)
        # impl_volatility = the interpolated implied volatility of the option
        # impl_strike = The Strike Price Corresponding to this Delta
        # impl_premium = The premium of a Theoretical Option with this implied volatility
        # dispersion = A measure of how well the implied volatility surface fits the underlying option data
        library = "optionm"
        table = f"vsurfd{year}"
        return self.conn.raw_sql(f"""
        SELECT *
        FROM {library}.{table}
        WHERE impl_volatility IS NOT NULL
        AND secid = {secid}
        """)

    def get_option_price_tables(self):
        # option_price_YYYY
        # option_price_view
        # std_option_price_*
        # tick_option_price_()
        pass

    def get_raw_option_data(self):
        # opprcdYYYY
        pass

    def get_underlying_price(self):
        # security_price
        # secprdYYYY
        pass

    def get_options_data(self):
        # print(self.conn.list_tables("optionm"))
        secid = self.ticker_to_id("SPY")
        dfs = []
        for year in range(int(self.start), int(self.end)+1):
            df = self.get_volatility_surface(secid, year)

            # Filter by call options only
            df = df[(df["cp_flag"] == "C")]

            if not df.empty:
                dfs.append(df)

        
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self.df.drop(columns=["secid", "cp_flag"])

        self.save_data("all_data.xlsx")

        # self.plot_volatility_surface(df[(df["date"] == "2025-01-02") & (df["cp_flag"] == "C")])
        # self.get_option_price_tables()
        # self.get_raw_option_data()
        # self.get_underlying_price()
        pass

    def collect_data(self):
        # self.get_underlying_data()
        self.get_options_data()
    
    def save_data(self, filename):
        self.df.to_excel(self.output_path / filename)

    """
    # Possible future expansion
    def get_trades(self):
        #   1) ctm_[YYYYMMDD] --> Trades (Core Data)
        #       Use for returns, realized volatility, trade intensity

        # Table Columns
        # date = Trade Date
        # time_m = Trade time (second resolution)
        # time_m_nano = Trade time (nanosecond resolution) 
        # sym_root = Ticker Symbol
        # ex = Exchange where trade occurred
        # price = Trade Price
        # size = Number of shares traded
        # tr_scond = Sale condition code (Filter by regular trades only)
        # tr_corr = Correction Indicator (Keep only valid, non-corrected, trades)
        # tr_stop_ind = Stop Trade Indicator (Marks special trade types, usually ignore or filter out)
        # tr_rf = Trade reporting facility flags (Used to filter out off-exchange trades)

        # tr_scond most common types:
        #       tr_scond   count
        # 0         I  295975 <-- Intermarket sweep orders
        # 1       F I  152889 
        # 2      <NA>  118200 <-- These are the standard orders
        # 3         F  110863 
        # 4        TI   14747
        # ...

        # tr_corr   count
        # 0      00  720933
        # 1      10       5
        # 2      08       5
        # 3      01       1
        # 4      12       1

        # No need to filter this one
        #   tr_stop_ind   count
        # 0           N  720945

        #   tr_rf   count
        # 0  <NA>  459512 <-- normal trades
        # 1     T  254411 <-- off-exchange trades (i.e. dark pools)
        # 2     N    4633
        # 3     B    2389

        library = "taqm_2025"
        table = "ctm_20250110"

        sql = f'''SELECT date, time_m, time_m_nano, sym_root, ex, price, size, tr_scond, tr_corr, tr_stop_ind, tr_rf
                  FROM {library}.{table}
                  WHERE sym_root = '{self.ticker_symbol}'
                  AND tr_scond IS NULL
                  AND tr_corr = '00'
                  AND (tr_rf IS NULL OR tr_rf = 'T'); 
        '''
        return self.conn.raw_sql(sql)

    
    def get_quotes(self):
        #   2) cqm_[YYYYMMDD] --> Quotes (Bid Price, Ask Price, Sizes)
        #           Use for spread (ask - bid), midprice (bid + ask) / 2

        # print(self.conn.describe_table("taqm_2025", table="cqm_20250110"))
        # TODO: Skip for now, can add this later to provide the model with extra context
        pass

    def get_nbbo(self):
        #   3) complete_nbbo_[YYYYMMDD] --> Best quotes
        #       Use for nbbo
        pass
    
    def get_metadata(self):
        #   4) mastm_[YYYYMMDD] --> Metadata
        #       Use for filter stocks and map symbols
        pass

    def get_luld(self):
        #   5) luld_[YYYYMMDD] --> Limit Up / Limit Down
        #       Trading halts, price bands
        pass

    def get_wct(self):
        #   6) wct_[YYYYMMDD] --> Weighted / Cleaned Trades
        #       Processed version of trades
        pass

    def get_underlying_data(self):
        df = self.get_trades()
        self.get_quotes()
        self.get_nbbo()
        self.get_metadata()
        self.get_luld()
        self.get_wct()
    """


if __name__ == "__main__":
    start_year  = "1996"
    end_year    = "2025"
    ticker_symbol = "SPY"

    BASE_DIR = Path(__file__).resolve().parent
    output_path = BASE_DIR / "raw"
    output_path.mkdir(parents=True, exist_ok=True)

    taq = TAQ(ticker_symbol, start_year, end_year, output_path)
    taq.collect_data()
    taq.close()



