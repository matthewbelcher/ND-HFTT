#!/usr/bin/env python3
import subprocess

def main():
    try:
        # Run the clean_data.py script
        print("Running clean_data.py ...")
        subprocess.run(["python", "clean_data.py"], check=True)
        print("Finished processing data with clean_data.py.")

        # Run the gzip_script.py in gunzip mode on ../HFTT_Data_Sorted
        print("Running gzip_script.py in gzip mode on ../HFTT_Data_Sorted ...")
        subprocess.run(["python", "unzip.py", "../HFTT_Data_Sorted", "gzip"], check=True)
        print("Finished processing zipping processed files with unzip.py.")

        print("Running unzip.py in gzip mode on ../HFTT_Data ...")
        subprocess.run(["python", "unzip.py", "../HFTT_Data", "gzip"], check=True)
        print("Finished processing zipping original files with unzip.py.")
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
