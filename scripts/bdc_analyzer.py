import os
import sys
import requests
import argparse
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tabulate import tabulate
import re

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

def get_headers():
    # Retrieve EDGAR identity from environment, or fallback to a default (SEC requires a User-Agent)
    user_agent = os.environ.get('EDGAR_IDENTITY', 'OpenClaw-BDC-Analyst <admin@openclaw.ai>')
    return {'User-Agent': user_agent}

def get_cik(ticker):
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=get_headers())
    if res.status_code != 200:
        print("Failed to fetch CIK list from SEC.")
        sys.exit(1)
        
    data = res.json()
    for k, v in data.items():
        if v['ticker'].upper() == ticker.upper():
            return str(v['cik_str']).zfill(10)
    print(f"Ticker {ticker} not found.")
    sys.exit(1)

def fetch_10k_html_url(cik, filing_year):
    res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=get_headers())
    if res.status_code != 200:
        return None
        
    filings = res.json()['filings']['recent']
    
    for i, form in enumerate(filings['form']):
        if form == '10-K':
            filing_date = filings['filingDate'][i]
            if filing_date.startswith(str(filing_year)):
                acc_no = filings['accessionNumber'][i]
                primary_doc = filings['primaryDocument'][i]
                acc_no_no_dash = acc_no.replace('-', '')
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_no_dash}/{primary_doc}"
    return None

def clean_value(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    if not val_str or val_str in ['-', '—', '$']:
        return np.nan
    
    # Remove non-numeric chars except minus, parens, dot
    val_str = re.sub(r'[^\d\.\(\)-]', '', val_str)
    
    if val_str == '':
        return np.nan
        
    # Handle accounting negatives
    if val_str.startswith('(') and val_str.endswith(')'):
        val_str = '-' + val_str[1:-1]
    
    try:
        # Some tables are in thousands, some in millions. SEC text usually raw digits if it's in the table.
        return float(val_str)
    except ValueError:
        return np.nan

def extract_schedule_of_investments(html_url):
    print(f"  Fetching HTML: {html_url}")
    res = requests.get(html_url, headers=get_headers())
    if res.status_code != 200:
        print("  Failed to download HTML.")
        return pd.DataFrame()
        
    print("  Parsing HTML (this may take a moment)...")
    soup = BeautifulSoup(res.content, 'lxml')
    tables = soup.find_all('table')
    
    print(f"  Found {len(tables)} tables. Identifying Schedule of Investments...")
    
    all_data = []
    
    for idx, tbl in enumerate(tables):
        # We need a table that's large and has characteristic headers
        text = tbl.get_text().lower()
        if 'portfolio company' not in text and 'investment' not in text:
            continue
        if 'fair value' not in text and 'amortized cost' not in text:
            continue
            
        rows = tbl.find_all('tr')
        if len(rows) < 30: # SoI tables are usually huge
            continue
            
        # Parse it with pandas to utilize its HTML table parsing capabilities for colspans etc.
        try:
            from io import StringIO
            dfs = pd.read_html(StringIO(str(tbl)))
            if not dfs: continue
            df = dfs[0]
        except Exception as e:
            continue
            
        # Drop columns that are mostly empty
        df = df.dropna(axis=1, how='all')
        
        # Determine the header row. We look for 'Fair Value'
        header_row_idx = -1
        for i in range(min(10, len(df))):
            row_str = ' '.join([str(x).lower() for x in df.iloc[i].values])
            if 'fair value' in row_str and ('cost' in row_str or 'principal' in row_str):
                header_row_idx = i
                break
                
        if header_row_idx == -1:
            continue
            
        # Flatten multi-index columns if present, otherwise just pick the best header
        cols = []
        for i, val in enumerate(df.iloc[header_row_idx].values):
            cols.append(str(val).lower().replace('\n', ' ').strip())
        df.columns = cols
        
        # Only keep data below header
        df = df.iloc[header_row_idx+1:].copy()
        
        # Identify columns
        comp_col = None
        cost_col = None
        fv_col = None
        
        # The first non-empty text column is usually the company name or investment description
        for col in df.columns:
            if 'company' in col or 'investment' in col or 'issuer' in col:
                comp_col = col
                break
        if not comp_col:
            comp_col = df.columns[0] # Fallback
            
        for col in df.columns:
            if 'cost' in col:
                cost_col = col
            if 'fair value' in col:
                fv_col = col
                
        if not (comp_col and cost_col and fv_col):
            continue
            
        # Extract subset carefully handling duplicate column names
        def safe_get_col(df, col_name):
            extracted = df[col_name]
            if isinstance(extracted, pd.DataFrame):
                return extracted.iloc[:, 0]
            return extracted

        comp_s = safe_get_col(df, comp_col)
        cost_s = safe_get_col(df, cost_col)
        fv_s = safe_get_col(df, fv_col)
        
        sub = pd.DataFrame({'Company': comp_s, 'Cost': cost_s, 'Fair_Value': fv_s})
        all_data.append(sub)

    if not all_data:
        return pd.DataFrame()
        
    master_df = pd.concat(all_data, ignore_index=True)
    
    # Cleaning
    master_df['Company'] = master_df['Company'].astype(str).str.strip().str.replace(r'\n', ' ', regex=True)
    master_df['Company'] = master_df['Company'].str.replace(r'\(.*?\)', '', regex=True) # remove footnotes (1)(2)
    master_df = master_df[master_df['Company'] != 'nan']
    
    # Ignore subtotals
    master_df = master_df[~master_df['Company'].str.lower().str.contains('total')]
    master_df = master_df[~master_df['Company'].str.lower().str.contains('subtotal')]
    master_df = master_df[master_df['Company'].str.len() > 2] # Ignore weird 1-char parsed strings
    
    master_df['Cost'] = master_df['Cost'].apply(clean_value)
    master_df['Fair_Value'] = master_df['Fair_Value'].apply(clean_value)
    
    # Drop where both are null
    master_df.dropna(subset=['Cost', 'Fair_Value'], how='all', inplace=True)
    
    # Group by company name. Because BDCs break loans into Tranches (e.g. "First Lien Term Loan"),
    # grouping by the first part of the company name usually aggregates the whole exposure.
    # We will grab the first 2-3 words of the company string to group tranches.
    
    def get_base_name(name):
        # Extremely rough heuristic: take first two words and strip non-alpha
        words = name.split()
        if len(words) >= 2:
            base = words[0] + " " + words[1]
        else:
            base = words[0]
        return re.sub(r'[^A-Za-z0-9 ]', '', base).upper().strip()

    master_df['Base_Company'] = master_df['Company'].apply(get_base_name)
    
    agg = master_df.groupby('Base_Company', as_index=False).agg({'Cost': 'sum', 'Fair_Value': 'sum'})
    agg = agg[agg['Cost'] > 0] # Filter valid rows
    return agg

def analyze(ticker):
    print(f"[{ticker}] Looking up CIK...")
    cik = get_cik(ticker)
    
    print(f"[{ticker}] Processing 2025 Year-End Data (Filed in early 2026)...")
    url_2026 = fetch_10k_html_url(cik, 2026)
    if not url_2026:
        print("  Could not find 2026 filing.")
        sys.exit(1)
    df_2025 = extract_schedule_of_investments(url_2026)
    df_2025.rename(columns={'Cost': 'Cost_2025', 'Fair_Value': 'FV_2025'}, inplace=True)
    
    print(f"\n[{ticker}] Processing 2024 Year-End Data (Filed in early 2025)...")
    url_2025 = fetch_10k_html_url(cik, 2025)
    if not url_2025:
        print("  Could not find 2025 filing.")
        sys.exit(1)
    df_2024 = extract_schedule_of_investments(url_2025)
    df_2024.rename(columns={'Cost': 'Cost_2024', 'Fair_Value': 'FV_2024'}, inplace=True)
    
    if df_2025.empty or df_2024.empty:
        print("Failed to extract data for one or both years.")
        sys.exit(1)
        
    print(f"\n[{ticker}] Merging and comparing the two years...")
    
    merged = pd.merge(df_2025, df_2024, on='Base_Company', how='inner')
    
    if merged.empty:
        print("No matching companies found between the two years. Parsing might have failed to align.")
        sys.exit(1)
        
    merged['Unrealized_Depr_2024'] = merged['FV_2024'] - merged['Cost_2024']
    merged['Unrealized_Depr_2025'] = merged['FV_2025'] - merged['Cost_2025']
    
    # We want to find the ones where Unrealized Depreciation WORSENED the most in absolute dollars
    # A negative number means Fair Value < Cost. Worsening means 2025 is MORE negative than 2024.
    merged['Depr_Change'] = merged['Unrealized_Depr_2025'] - merged['Unrealized_Depr_2024']
    
    # Filter only assets that are actually underwater in 2025
    underwater = merged[merged['Unrealized_Depr_2025'] < 0]
    
    # Sort by the most negative change (deterioration)
    worst = underwater.sort_values(by='Depr_Change', ascending=True).head(15)
    
    res_table = worst[['Base_Company', 'Cost_2024', 'FV_2024', 'Unrealized_Depr_2024', 
                       'Cost_2025', 'FV_2025', 'Unrealized_Depr_2025', 'Depr_Change']]
                       
    print(f"\n==========================================================================")
    print(f"Top 15 Companies by Deterioration of Unrealized Depreciation ({ticker})")
    print(f"Comparing 2024 Year-End to 2025 Year-End")
    print(f"==========================================================================\n")
    print(tabulate(res_table, headers='keys', tablefmt='psql', showindex=False, floatfmt=".2f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDC Asset Depreciation Analyzer (HTML Parser)')
    parser.add_argument("--ticker", required=True, help='Stock ticker of the BDC')
    args = parser.parse_args()
    analyze(args.ticker)