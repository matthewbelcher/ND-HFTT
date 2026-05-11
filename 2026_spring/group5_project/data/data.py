import wrds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class OptionsData():
    def __init__(self, ticker_symbol, start_year, end_year, output_path):
        self.ticker_symbol = ticker_symbol
        self.start = start_year
        self.end = end_year
        self.conn = wrds.Connection()
        self.df = pd.DataFrame()
        self.output_path = output_path
    
    def close(self):
        self.conn.close()

    def get_SP500_daily(self, secid, year):
        library = "optionm"
        table = f"secprd{year}"
        # print(self.conn.describe_table(library, table))
        return self.conn.raw_sql(f"""
            SELECT 
                date, 
                close AS SP500_close, 
                return AS SP500_daily_simple_return, 
                open AS SP500_open, 
                high AS SP500_high, 
                low AS SP500_low
            FROM {library}.{table}
            WHERE secid = {secid}
        """)

    def get_underlying_data(self, secid, year):
        df = self.get_SP500_daily(secid, year)
        return df 

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
        
        # SPY ID: 109820 (S&500 Option (American))
        # SPX ID: 108105 (S&500 Option (European))
        return 108105

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

    def get_volatility_surface(self, secid, year, cp_flag):
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
            SELECT date, 
                   days AS days_to_exp, 
                   delta, 
                   impl_volatility, 
                   impl_strike, 
                   impl_premium
            FROM {library}.{table}
            WHERE impl_volatility IS NOT NULL
            AND cp_flag = '{cp_flag}'
            AND secid = {secid}
        """)

    def get_options_data(self, secid, year):
        df = self.get_volatility_surface(secid, year, "C")

        # self.plot_volatility_surface(df[(df["date"] == "2025-01-02") & (df["cp_flag"] == "C")])

        return df

    def get_daily_risk_free_rate(self):
        library = "optionm"
        table = f"zerocd"
        return self.conn.raw_sql(f"""
            SELECT date,
                   days AS rate_days,
                   rate / 100.00 AS risk_free_rate
            FROM {library}.{table}
            WHERE date BETWEEN '{self.start}-01-01' AND '{self.end}-12-31'
            ORDER BY date, rate_days
        """)

    def add_rates_to_df(self):
        """
            Attach interpolated zero-curve rates to each surface row.
            Risk-free rate days until maturity may not align with option days until maturity
            so the equivalent risk-free rate must be interpolated. If an entire
            zero-curve date is missing, carry forward the previous available
            curve by maturity before interpolating.
        """
        if self.df.empty:
            return

        rates_df = self.get_daily_risk_free_rate().copy()

        if rates_df.empty:
            self.df = self.df.copy()
            self.df["risk_free_rate"] = np.nan
            return

        self.df = self.df.copy()
        self.df["date"] = pd.to_datetime(self.df["date"])
        rates_df["date"] = pd.to_datetime(rates_df["date"])

        # Build a complete date x maturity grid for the dates we actually need,
        # then carry forward the prior day's zero curve within each maturity.
        all_dates = pd.DataFrame(
            {"date": sorted(self.df["date"].dropna().unique())}
        )
        all_rate_days = pd.DataFrame(
            {"rate_days": sorted(rates_df["rate_days"].dropna().unique())}
        )

        full_curve_index = all_dates.merge(all_rate_days, how="cross")
        rates_df = full_curve_index.merge(
            rates_df,
            on=["date", "rate_days"],
            how="left"
        )
        rates_df = rates_df.sort_values(["rate_days", "date"])
        rates_df["risk_free_rate"] = (
            rates_df.groupby("rate_days")["risk_free_rate"].ffill()
        )

        rates_by_date = {}
        for date, grp in rates_df.groupby("date"):
            curve = (
                grp[["rate_days", "risk_free_rate"]]
                .dropna()
                .sort_values("rate_days")
            )
            if curve.empty:
                continue

            x = curve["rate_days"].to_numpy(dtype=float)
            y = curve["risk_free_rate"].to_numpy(dtype=float)

            # Remove duplicate maturities so interpolation gets a clean curve.
            x_unique, idx = np.unique(x, return_index=True)
            y_unique = y[idx]
            rates_by_date[date] = (x_unique, y_unique)

        out = []
        for date, group in self.df.groupby("date"):
            group = group.copy()
            curve = rates_by_date.get(date)

            if curve is None:
                group["risk_free_rate"] = np.nan
            else:
                x, y = curve
                d = group["days_to_exp"].to_numpy(dtype=float)
                # np.interp linearly interpolates and clamps out-of-range maturities.
                group["risk_free_rate"] = np.interp(d, x, y)

            out.append(group)

        if out:
            self.df = (
                pd.concat(out, ignore_index=True)
                .sort_values(["date", "days_to_exp", "delta"])
                .reset_index(drop=True)
            )
        else:
            self.df["risk_free_rate"] = np.nan


    def get_daily_vix(self):
        library = "cboe_all"
        table = "cboe"

        return self.conn.raw_sql(f"""
            SELECT date, 
                vix AS vix_close,
                vixo AS vix_open,
                vixh AS vix_high,
                vixl AS vix_low
            FROM {library}.{table}
            WHERE date BETWEEN '{self.start}-01-01' AND '{self.end}-12-31';
        """)

    def collect_data(self):
        secid = self.ticker_to_id(self.ticker_symbol)

        dfs = []
        for year in range(int(self.start), int(self.end)+1):
            underlying_df = self.get_underlying_data(secid, year)
            options_df = self.get_options_data(secid, year)

            df = pd.merge(underlying_df, options_df, on="date")

            # self.get_daily_vix()

            if not df.empty:
                dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        self.add_rates_to_df()

        vix_df = self.get_daily_vix()
        self.df["date"] = pd.to_datetime(self.df["date"])
        vix_df["date"] = pd.to_datetime(vix_df["date"])
        vix_df = vix_df.sort_values("date").drop_duplicates(subset=["date"])

        # There are three entries missing, just ffill
        vix_df[["vix_close", "vix_open", "vix_high", "vix_low"]] = vix_df[["vix_close", "vix_open", "vix_high", "vix_low"]].ffill()

        self.df = pd.merge(
            self.df,
            vix_df,
            on="date",
            how="left",
            validate="many_to_one"
        )

        # There is one day, 2007-11-13

        self.print_stats()
        self.save_data("all_data.xlsx")
    
    def print_stats(self):
        print("\nDataFrame Stats")

        if self.df.empty:
            print("DataFrame is empty.")
            return

        print(f"Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns")
        print(f"Columns: {list(self.df.columns)}")

        if "date" in self.df.columns:
            date_series = pd.to_datetime(self.df["date"])
            print("\nDate Range")
            print(f"Min date: {date_series.min()}")
            print(f"Max date: {date_series.max()}")
            print(f"Unique dates: {date_series.nunique()}")

        print("\nNull Values")
        total_nulls = int(self.df.isna().sum().sum())
        print(f"Total null cells: {total_nulls}")
        null_counts = self.df.isna().sum()
        null_counts = null_counts[null_counts > 0]
        if null_counts.empty:
            print("No null values found.")
        else:
            print(null_counts)

        print("\nDuplicates")
        full_row_duplicates = int(self.df.duplicated().sum())
        print(f"Full row duplicates: {full_row_duplicates}")

        surface_cols = ["date", "days_to_exp", "delta"]
        if all(col in self.df.columns for col in surface_cols):
            surface_duplicates = int(self.df.duplicated(subset=surface_cols).sum())
            print(f"Duplicate surface nodes (date, days_to_exp, delta): {surface_duplicates}")
    
    def save_data(self, filename):
        print(f"Saving data...")
        self.df.to_excel(self.output_path / filename)
        print(f"Dataframe saved to {self.output_path / filename}")

if __name__ == "__main__":
    start_year  = "2005"
    end_year    = "2025"
    ticker_symbol = "SPX"

    BASE_DIR = Path(__file__).resolve().parent
    output_path = BASE_DIR / "raw"
    output_path.mkdir(parents=True, exist_ok=True)

    od = OptionsData(ticker_symbol, start_year, end_year, output_path)
    od.collect_data()
    od.close()
