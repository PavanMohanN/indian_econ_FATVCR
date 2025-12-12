"""
Helper functions to download data. The script contains placeholders and
URL examples. Edit functions to point to the exact MoSPI CSV/Excel URLs you want to use.
"""

import os
import requests
from pathlib import Path

def download_file(url, outpath):
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(outpath, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

def fetch_mospi_quarterly_gdp(out_dir='data'):
    """
    Example: download a CSV from MoSPI DataViz or your repository.
    Replace the URL with the correct MoSPI export link or upload manually.
    """
    os.makedirs(out_dir, exist_ok=True)
    url = 'https://example.com/mospi_quarterly_gdp.csv'  # <- REPLACE
    outpath = Path(out_dir) / 'mospi_quarterly_gdp.csv'
    print(f'Download placeholder (edit URL): {outpath}')
    # Uncomment the next line after setting a real URL:
    # download_file(url, outpath)
    return outpath

def fetch_monthly_indicators(out_dir='data'):
    """
    Downloads or expects CSVs for monthly IP, PMI, GST, electricity, credit.
    Replace URLs or place files in `data/` manually.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Placeholder names: user should replace with real files or APIs
    files = []
    for name in ['ip','pmi','gst','electricity','credit']:
        path = Path(out_dir) / f'{name}.csv'
        # download_file(url_for(name), path)  # implement real URLs
        files.append(path)
    return files

if __name__ == '__main__':
    print("Data fetcher is a placeholder. Edit the URLs in src/data_fetch.py or place CSVs into /data.")
