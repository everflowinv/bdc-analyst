import os
import sys
import requests
import argparse
import pandas as pd
import io
import numpy as np
from tabulate import tabulate

def get_headers():
    return {'User-Agent': 'OpenClaw-BDC-Analyst <admin@openclaw.ai>'}

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

def clean_num(x):
    if pd.isna(x): return np.nan
    s = str(x).replace('$', '').replace(',', '').replace('%', '').strip()
    # Handle accounting negatives
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return np.nan

def analyze(ticker):
    print(f"[{ticker}] Looking up CIK...")
    cik = get_cik(ticker)
    print(f"[{ticker}] CIK: {cik}")

    print(f"[{ticker}] Fetching latest filings...")
    res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=get_headers())
    filings = res.json()['filings']['recent']
    
    # Find the latest 10-K
    acc_no = None
    for i, form in enumerate(filings['form']):
        if form == '10-K':
            acc_no = filings['accessionNumber'][i]
            break
            
    if not acc_no:
        print(f"[{ticker}] No 10-K found in recent filings.")
        sys.exit(1)
        
    print(f"[{ticker}] Latest 10-K Accession Number: {acc_no}")
    acc_no_no_dash = acc_no.replace('-', '')
    excel_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_no_dash}/Financial_Report.xlsx"
    
    print(f"[{ticker}] Fetching Interactive Data Excel: {excel_url}")
    xl_res = requests.get(excel_url, headers=get_headers())
    if xl_res.status_code != 200:
        print(f"[{ticker}] Excel file not found. Falling back to the previous year's 10-K to see if it has the Excel file...")
        # Fallback to older 10-K
        fallback_acc = None
        for i, form in enumerate(filings['form']):
            if form == '10-K' and filings['accessionNumber'][i] != acc_no:
                fallback_acc = filings['accessionNumber'][i]
                break
        if fallback_acc:
            print(f"[{ticker}] Fallback 10-K Accession Number: {fallback_acc}")
            acc_no_no_dash = fallback_acc.replace('-', '')
            excel_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_no_dash}/Financial_Report.xlsx"
            print(f"[{ticker}] Fetching fallback Interactive Data Excel: {excel_url}")
            xl_res = requests.get(excel_url, headers=get_headers())
            
    if xl_res.status_code != 200:
        print(f"[{ticker}] Excel file not found for the fallback either. SEC is not packaging it here.")
        sys.exit(1)
        
    print(f"[{ticker}] Parsing Excel worksheets...")
    try:
        xls = pd.ExcelFile(io.BytesIO(xl_res.content))
    except Exception as e:
        print(f"[{ticker}] Failed to parse Excel: {e}")
        sys.exit(1)

    # Find sheets related to Schedule of Investments
    target_sheets = [s for s in xls.sheet_names if 'invest' in s.lower() or 'schedule' in s.lower() or 'portfolio' in s.lower()]
    
    if not target_sheets:
        print(f"[{ticker}] No sheets resembling a Schedule of Investments found.")
        print(f"Available sheets: {xls.sheet_names[:10]}")
        sys.exit(1)

    print(f"[{ticker}] Extracting tabular data from {len(target_sheets)} potential sheets...")
    all_assets = []

    for sheet in target_sheets:
        df = pd.read_excel(xls, sheet)
        
        # Locate the header row by searching for "Fair Value" and "Cost" or "Principal"
        header_idx = None
        for idx, row in df.head(15).iterrows():
            row_str = " ".join([str(x).lower() for x in row.values])
            if 'fair value' in row_str and ('cost' in row_str or 'principal' in row_str):
                header_idx = idx
                break
        
        if header_idx is None:
            continue
            
        # Set new header
        df.columns = df.iloc[header_idx]
        df = df.iloc[header_idx+1:].reset_index(drop=True)
        
        # Clean columns: strip, lowercase
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Identify columns dynamically
        company_col = None
        cost_col = None
        fv_col = None
        
        for c in df.columns:
            if 'company' in c or 'issuer' in c or 'portfolio' in c or 'investment' in c:
                if not company_col: company_col = c
            if ('cost' in c and 'amortized' in c) or ('cost' in c and not cost_col):
                cost_col = c
            if 'fair value' in c:
                fv_col = c
                
        if not (company_col and cost_col and fv_col):
            continue
            
        # Extract the necessary columns
        sub_df = df[[company_col, cost_col, fv_col]].copy()
        sub_df.columns = ['Company', 'Cost', 'Fair_Value']
        all_assets.append(sub_df)

    if not all_assets:
        print(f"[{ticker}] Could not extract standardized (Company, Cost, Fair Value) table from the Excel sheets.")
        sys.exit(1)
        
    master_df = pd.concat(all_assets, ignore_index=True)
    master_df.dropna(subset=['Company'], inplace=True)
    master_df['Company'] = master_df['Company'].astype(str).str.strip()
    
    # Filter out empty or header-like rows
    master_df = master_df[master_df['Company'] != 'nan']
    master_df = master_df[~master_df['Company'].str.lower().isin(['total', 'portfolio company'])]
    
    master_df['Cost'] = master_df['Cost'].apply(clean_num)
    master_df['Fair_Value'] = master_df['Fair_Value'].apply(clean_num)
    
    # Drop rows where both numeric fields are NaN
    master_df.dropna(subset=['Cost', 'Fair_Value'], how='all', inplace=True)
    master_df = master_df[master_df['Cost'] > 0]
    
    print(f"[{ticker}] Aggregating {len(master_df)} asset tranches...")
    
    # Aggregate by company (combine different tranches of the same loan)
    agg_df = master_df.groupby('Company', as_index=False).agg({
        'Cost': 'sum',
        'Fair_Value': 'sum'
    })
    
    agg_df['Unrealized_Depreciation'] = agg_df['Fair_Value'] - agg_df['Cost']
    # Filter for companies that are actually depreciated
    agg_df = agg_df[agg_df['Unrealized_Depreciation'] < 0]
    agg_df['Depreciation_Pct'] = (agg_df['Unrealized_Depreciation'] / agg_df['Cost']) * 100
    
    # Sort by the largest absolute depreciation (most negative difference)
    worst_assets = agg_df.sort_values(by='Unrealized_Depreciation', ascending=True).head(15)
    
    print(f"\n==========================================================================")
    print(f"Top 15 Assets by Unrealized Depreciation (Fair Value < Cost) for {ticker}")
    print(f"Source: Latest 10-K Schedule of Investments (Interactive Data)")
    print(f"==========================================================================\n")
    print(tabulate(worst_assets, headers=['Portfolio Company', 'Amortized Cost', 'Fair Value', 'Unrealized Depr', 'Depr %'], tablefmt='psql', showindex=False, floatfmt=".2f"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BDC Asset Depreciation Analyzer')
    parser.add_argument("--ticker", required=True, help='Stock ticker of the BDC')
    args = parser.parse_args()
    analyze(args.ticker)