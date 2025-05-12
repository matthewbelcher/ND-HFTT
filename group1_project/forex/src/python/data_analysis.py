import pandas as pd

# Read the CSV file (replace 'data.csv' with your actual file path)
df = pd.read_csv('../../2025-03-31.csv')

def extract_currency(ticker):
    # Remove prefix if present (e.g., "C:")
    if ':' in ticker:
        ticker = ticker.split(':', 1)[1]
    # Split the ticker on the dash and take the first part
    return ticker.split('-')[0]

# Create a new column 'currency' that contains the first currency code from the ticker
df['currency'] = df['ticker'].apply(extract_currency)

# Count the number of unique currencies in the new column
unique_currencies = df['currency'].nunique()
print("Number of unique currencies:", unique_currencies)