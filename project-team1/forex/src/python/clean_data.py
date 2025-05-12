import os
import pandas as pd
import subprocess

input_base = "../../HFTT_Data"
output_base = "../../HFTT_Data_Sorted"

allowed_currencies = {"USD", "EUR", "JPY", "GBP", "CNH", "AUD", "CAD", "CHF", "HKD", "NZD", "SGD"}

def ticker_filter(ticker):
    # Remove prefix if present
    if ':' in ticker:
        pair = ticker.split(':')[1]
    else:
        pair = ticker
    # Split into individual currencies and check if both are allowed
    currencies = pair.split('-')
    return all(cur in allowed_currencies for cur in currencies)

# Traverse the input directory recursively
for root, dirs, files in os.walk(input_base):
    for file in files:
        if file.endswith('.csv'):
            input_file = os.path.join(root, file)
            
            # Create the corresponding output directory
            relative_path = os.path.relpath(root, input_base)
            output_dir = os.path.join(output_base, relative_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Define output file name with _sorted suffix
            file_name, ext = os.path.splitext(file)
            output_file = os.path.join(output_dir, f"{file_name}_sorted{ext}")
            
            # Read, filter, and sort the CSV file
            try:
                df = pd.read_csv(input_file)
                df_filtered = df[df["ticker"].apply(ticker_filter)]
                df_sorted = df_filtered.sort_values(by="participant_timestamp")
                
                # Write the sorted dataframe to the output CSV
                df_sorted.to_csv(output_file, index=False)
                subprocess.run(["gzip", "-f", output_file],check=True)
                print(f"Processed and Zipped: {input_file} -> {output_file}")
            except Exception as e:
                print(f"Error processing {input_file}: {e}")

