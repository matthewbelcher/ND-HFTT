import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
from datetime import datetime

# analyze the correlation between several variables and the resulting market movement.
# Variables may inlcude the change in the fed funds rate,
# the difference between the expected and actual rate changes, etc.


def load_market_data(
    date, base_dir="./data/market_ticks", filename_format="%m_%d_%y.csv"
):
    # Convert string date to datetime if needed
    if isinstance(date, str):
        try:
            date = datetime.strptime(date, "%m_%d_%y")
        except ValueError:
            raise ValueError("Date must be in the format 'DD_MM_YY'")

    # Construct the filename
    filename = date.strftime(filename_format)
    filepath = os.path.join(base_dir, filename)

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"No data file found for {date.strftime('%m_%d_%y')} at {filepath}"
        )

    # Read the CSV into a pandas DataFrame
    try:
        data = pd.read_csv(filepath)
        # Create a new DataFrame with one series: the average of the third and fifth columns
        if data.shape[1] >= 5:  # Ensure there are at least 5 columns
            data = pd.DataFrame(
                {
                    "market_mid": (data.iloc[:, 2] + data.iloc[:, 4]) / 2,
                    # "time": pd.to_datetime(data.iloc[:, 1]),
                    "spread": data.iloc[:, 4] - data.iloc[:, 2],
                }
            )
        else:
            raise ValueError(
                "Data must have at least 5 columns to compute the average."
            )
        return data
    except Exception as e:
        raise Exception(f"Error reading CSV file: {e}")


def analyze_correlation(data, variables, target):
    """
    Analyze the correlation between several variables and the target market movement.

    Parameters:
    - data: pd.DataFrame containing the dataset
    - variables: list of column names to analyze
    - target: column name of the target variable (market movement)

    Returns:
    - correlation_results: dict containing correlation coefficients and p-values
    """
    correlation_results = {}
    for var in variables:
        if var in data.columns and target in data.columns:
            corr_coef, p_value = stats.pearsonr(data[var], data[target])
            correlation_results[var] = {"correlation": corr_coef, "p_value": p_value}
        else:
            print(f"Variable {var} or target {target} not found in data.")
    return correlation_results


def plot_correlations(data, variables, target):
    """
    Plot scatter plots for each variable against the target variable.

    Parameters:
    - data: pd.DataFrame containing the dataset
    - variables: list of column names to analyze
    - target: column name of the target variable (market movement)
    """
    for var in variables:
        if var in data.columns and target in data.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(data[var], data[target], alpha=0.7)
            plt.title(f"{var} vs {target}")
            plt.xlabel(var)
            plt.ylabel(target)
            plt.grid(True)
            plt.show()
        else:
            print(f"Variable {var} or target {target} not found in data.")


# Example usage
if __name__ == "__main__":
    market_data = load_market_data("01_29_25")
    print(market_data.head())

    variables = ["time", "spread"]
    target = "market_mid"

    # Analyze correlations
    results = analyze_correlation(market_data, variables, target)
    for var, stats in results.items():
        print(
            f"{var}: Correlation = {stats['correlation']:.2f}, p-value = {stats['p_value']:.4f}"
        )

    # Plot correlations
    plot_correlations(data, variables, target)
