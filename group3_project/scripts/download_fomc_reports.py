import requests
import os
import re
from datetime import datetime

OUTPUT_DIR = "data/fomc_statements"
FED_PR_BASE_URL = "https://www.federalreserve.gov/newsevents/pressreleases"

def download_file(output_dir: str, base_url: str, file_name: str) -> None:
    r = requests.get(os.path.join(base_url, file_name))
    if r.status_code == 200:
        title = re.search("<title>(.*)</title>", r.text)
        if not title:
            print(f"Downloaded: {file_name} but couldn't find title in content.")
            return
        if title.group(1) == "Federal Reserve Board - Federal Reserve issues FOMC statement":
            with open(os.path.join(output_dir, file_name), "wb") as f:
                f.write(r.content)
                print(f"Downloaded: {file_name}")
        else:
            print(f"Downloaded: {file_name} but title is not 'FOMC statement'. Found title: {title.group(1)}")
    else:
        print(f"Failed to download: {file_name}. Status code: {r.status_code}")

def main():

    for year in range(2011, 2026):
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    datetime(year, month, day)
                except ValueError:
                    break

                file_name = f"monetary{year}{month:02}{day:02}a.htm"
                
                download_file(OUTPUT_DIR, FED_PR_BASE_URL, file_name)

if __name__ == "__main__":
    main()
