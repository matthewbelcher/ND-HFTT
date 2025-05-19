import os
import json
import csv
import glob
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_trades_file(file_path):
    """Extract trade data from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Handle different possible structures of the JSON file
        trades = []
        if isinstance(data, dict) and 'trades' in data:
            trades = data['trades']
        elif isinstance(data, list):
            trades = data
        
        return trades
    except json.JSONDecodeError:
        logger.error(f"Failed to parse JSON from {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return []

def main():
    # Define the raw_responses directory path
    raw_responses_dir = '/Users/juanporras/Desktop/Cash&Carry/raw_responses'
    
    # Define output CSV file
    output_csv = 'coinbase_trades.csv'
    
    # Define CSV headers based on the trade data structure
    fieldnames = [
        'trade_id', 'product_id', 'price', 'size', 'time', 
        'side', 'bid', 'ask', 'exchange', 'file_source'
    ]
    
    # Find all JSON files in the raw_responses directory
    json_files = []
    for root, dirs, files in os.walk(raw_responses_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    # If no json extension, try to find files with trade or coinbase in the name
    if not json_files:
        json_files = glob.glob(os.path.join(raw_responses_dir, "*trade*"))
        json_files.extend(glob.glob(os.path.join(raw_responses_dir, "*coinbase*")))
    
    logger.info(f"Found {len(json_files)} potential trade data files")
    
    # Initialize counters
    total_trades = 0
    processed_files = 0
    
    # Open CSV file for writing
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each file
        for file_path in sorted(json_files):
            logger.info(f"Processing {file_path}")
            
            trades = process_trades_file(file_path)
            if not trades:
                continue
            
            # Write trades to CSV
            for trade in trades:
                # Add source file for reference
                trade_data = {k: v for k, v in trade.items()}
                trade_data['file_source'] = os.path.basename(file_path)
                
                # Write only the fields that match our headers
                row_data = {field: trade_data.get(field, '') for field in fieldnames}
                writer.writerow(row_data)
                total_trades += 1
            
            processed_files += 1
            logger.info(f"Extracted {len(trades)} trades from {file_path}")
    
            # Re-open and sort the CSV by the 'time' column
        try:
            with open(output_csv, 'r', newline='') as infile:
                reader = csv.DictReader(infile)
                sorted_rows = sorted(reader, key=lambda row: row['time'])

            with open(output_csv, 'w', newline='') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_rows)

            logger.info(f"✅ Sorted {output_csv} by time column")
        except Exception as e:
            logger.error(f"Failed to sort CSV by time: {e}")

    logger.info(f"Processing complete. Processed {processed_files} files, extracted {total_trades} trades.")
    logger.info(f"Data saved to {output_csv}")

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting trade data extraction at {start_time}")
    main()
    end_time = datetime.now()
    logger.info(f"Extraction completed at {end_time}. Total time: {end_time - start_time}")