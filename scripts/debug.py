import sys
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.sec.gov/Archives/edgar/data/1422183/000162828026011734/fsk-20251231.htm"
headers = {'User-Agent': 'OpenClaw-BDC-Analyst <admin@openclaw.ai>'}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.content, 'lxml')
tables = soup.find_all('table')

big_tables = []
for i, tbl in enumerate(tables):
    rows = tbl.find_all('tr')
    if len(rows) > 30:
        text = tbl.get_text().lower()
        print(f"Table {i}: {len(rows)} rows")
        if 'fair value' in text:
            print("  Contains 'fair value'")
        dfs = pd.read_html(str(tbl))
        if dfs:
            print(dfs[0].head(2))
        print("----")
