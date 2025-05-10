import click
import multiprocessing
import os
import pandas as pd
import subprocess

CURRENCIES = {"USD", "EUR", "JPY", "GBP", "CNH", "AUD", "CAD", "CHF", "HKD", "SGD"}
ARB_BINARY = "./build/bin/arbitrage_possible"

def process(file):
    filtered_file = file.replace(".csv.gz", "_filtered.csv")
    if not os.path.exists(filtered_file):
        df = pd.read_csv(file, compression="gzip")
        # Clean, filter, and sort the raw dataframe.

        def is_symbol_of_interest(symbol: str):
            currency1, currency2 = symbol[2:].split("-")
            return currency1 in CURRENCIES and currency2 in CURRENCIES

        # Sort by the timestamp and convert the timestamp to seconds.
        # The timestamp is provided in nanoseconds but has 1 second resolution.
        df = df.sort_values(by="participant_timestamp", kind="stable")
        df["participant_timestamp"] //= int(1e9)

        # The exchange always seems to equal 48.
        df = df.drop(["ask_exchange", "bid_exchange"], axis=1)

        # Remove currencies in which we are not interested.
        df = df[df["ticker"].map(is_symbol_of_interest)]

        # OPTIONAL
        # Drop duplicate symbols within the same timestamp.
        # When this happens, we should keep the last (most up-to-date) prices.
        df = df.drop_duplicates(subset=["ticker", "participant_timestamp"], keep="last")
        df.to_csv(filtered_file)
        print(f"Successfully processed {file.split('/')[-1].split(".")[0]}")
    
    arb(filtered_file)
    
def arb(file):
    if not os.path.exists(ARB_BINARY):
        print("Arb binary does not exist")
        return
    arb_file = file.replace("filtered", "arbs")
    if not os.path.exists(arb_file):
        result = subprocess.run([ARB_BINARY, file, "-s", arb_file], capture_output=True)
        print(result.stdout.decode("utf-8")[:-1])
        print(f"Successfully wrote arb file for {file.split("/")[-1].split("_")[0]}")

@click.command()
@click.argument('src_dir')
def main(src_dir):
    files = []
    for file in os.listdir(src_dir):
        if ".csv.gz" not in file:
            continue
        files.append(f"{src_dir}{file}")
    with multiprocessing.Pool(processes=4) as pool:
        pool.map(process, files)
    return

if __name__ == "__main__":
    main()