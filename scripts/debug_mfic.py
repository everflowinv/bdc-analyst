import sys
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
import re

# Need to set headers as in bdc_analyzer
def get_headers():
    return {'User-Agent': 'OpenClaw-BDC-Analyst <admin@openclaw.ai>'}

def get_cik(ticker):
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=get_headers())
    data = res.json()
    for _, v in data.items():
        if v['ticker'].upper() == ticker.upper():
            return str(v['cik_str']).zfill(10)
    return None

def fetch_latest_10k_url(cik):
    res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=get_headers())
    filings = res.json()['filings']['recent']
    for i, form in enumerate(filings['form']):
        if form == '10-K' and str(filings['filingDate'][i]).startswith('2026'):
            acc = filings['accessionNumber'][i].replace('-', '')
            doc = filings['primaryDocument'][i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    return None

cik = get_cik('MFIC')
url = fetch_latest_10k_url(cik)
print(f"10-K URL: {url}")

res = requests.get(url, headers=get_headers())
soup = BeautifulSoup(res.content, 'lxml')
tables = soup.find_all('table')

for idx, tbl in enumerate(tables):
    txt = tbl.get_text(' ', strip=True).lower()
    if 'lendingpoint' in txt:
        print(f"\nTable {idx} contains 'lendingpoint'")
        if 'schedule of investments' in txt:
             print("Table seems to be SoI.")
        try:
            df = pd.read_html(StringIO(str(tbl)))[0]
            # Print rows with lendingpoint
            mask = df.astype(str).apply(lambda row: row.str.contains('LENDINGPOINT', case=False).any(), axis=1)
            if mask.any():
                print(df[mask].to_string())
        except:
            pass
