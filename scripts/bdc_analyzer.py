import os
import re
import sys
from io import StringIO

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate


def get_headers():
    user_agent = os.environ.get('EDGAR_IDENTITY', 'OpenClaw-BDC-Analyst <admin@openclaw.ai>')
    return {'User-Agent': user_agent}


def get_cik(ticker):
    res = requests.get("https://www.sec.gov/files/company_tickers.json", headers=get_headers(), timeout=60)
    res.raise_for_status()
    data = res.json()
    for _, v in data.items():
        if v['ticker'].upper() == ticker.upper():
            return str(v['cik_str']).zfill(10)
    raise ValueError(f"Ticker {ticker} not found")


def fetch_latest_10k_url(cik, filing_year=2026):
    res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=get_headers(), timeout=60)
    res.raise_for_status()
    filings = res.json()['filings']['recent']
    for i, form in enumerate(filings['form']):
        if form == '10-K' and str(filings['filingDate'][i]).startswith(str(filing_year)):
            acc = filings['accessionNumber'][i].replace('-', '')
            doc = filings['primaryDocument'][i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    raise ValueError(f"No {filing_year} 10-K found")


def clean_num(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s or s in ['-', '—', '$']:
        return np.nan
    s = re.sub(r'[^\d\.\(\)-]', '', s)
    if not s:
        return np.nan
    if s.startswith('(') and s.endswith(')'):
        s = '-' + s[1:-1]
    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_company(name):
    s = str(name)
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.upper()


def parse_candidate_table(tbl, context_year=None):
    text = tbl.get_text(' ', strip=True).lower()
    if 'fair value' not in text:
        return None, None
    if 'amortized cost' not in text and 'cost' not in text and 'principal' not in text:
        return None, None

    # detect table year from caption/body text; fallback to section context
    year = None
    if 'december 31, 2025' in text or 'as of december 31, 2025' in text:
        year = 2025
    elif 'december 31, 2024' in text or 'as of december 31, 2024' in text:
        year = 2024
    elif context_year in (2025, 2024):
        year = context_year

    try:
        df = pd.read_html(StringIO(str(tbl)))[0]
    except Exception:
        return None, year

    if len(df) < 20:
        return None, year

    df = df.dropna(axis=1, how='all')

    header_idx = None
    for i in range(min(12, len(df))):
        row = ' '.join([str(v).lower() for v in df.iloc[i].values])
        if 'fair value' in row and ('cost' in row or 'principal' in row):
            header_idx = i
            break
    if header_idx is None:
        return None, year

    cols = [str(v).strip().lower().replace('\n', ' ') for v in df.iloc[header_idx].values]
    df.columns = cols
    df = df.iloc[header_idx + 1:].copy()

    # pick columns
    company_col = None
    cost_col = None
    fv_col = None
    amortized_candidates = []
    generic_cost_candidates = []
    principal_candidates = []

    for c in df.columns:
        lc = str(c).lower()
        if company_col is None and ('portfolio company' in lc or 'issuer' in lc or 'investment' in lc or lc == cols[0]):
            company_col = c
        if fv_col is None and 'fair value' in lc:
            fv_col = c
        if 'amortized cost' in lc:
            amortized_candidates.append(c)
        elif 'cost' in lc:
            generic_cost_candidates.append(c)
        elif 'principal' in lc:
            principal_candidates.append(c)

    if amortized_candidates:
        cost_col = amortized_candidates[0]
    elif generic_cost_candidates:
        cost_col = generic_cost_candidates[0]
    elif principal_candidates:
        cost_col = principal_candidates[0]

    if company_col is None or cost_col is None or fv_col is None:
        return None, year

    def pick_series(frame, col):
        out = frame[col]
        if isinstance(out, pd.DataFrame):
            return out.iloc[:, 0]
        return out

    out = pd.DataFrame({
        'Company': pick_series(df, company_col),
        'Face': pick_series(df, cost_col),
        'Fair': pick_series(df, fv_col),
    })

    out['Company'] = out['Company'].astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
    out = out[out['Company'].str.len() > 1]
    out = out[~out['Company'].str.lower().str.contains('total|subtotal|schedule of investments|interest rate')]

    out['Face'] = out['Face'].apply(clean_num)
    out['Fair'] = out['Fair'].apply(clean_num)
    out = out.dropna(subset=['Face', 'Fair'], how='all')
    out = out[out['Face'] > 0]
    out['CompanyKey'] = out['Company'].apply(normalize_company)

    # exact-name aggregation only (avoid aggressive base-name compression)
    out = out.groupby('CompanyKey', as_index=False).agg({'Face': 'sum', 'Fair': 'sum'})
    return out, year


def extract_two_year_tables(url):
    res = requests.get(url, headers=get_headers(), timeout=120)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, 'lxml')
    tables = soup.find_all('table')

    context_year = None
    parsed_records = []  # (idx, parsed_df, year_or_none)

    for idx, tbl in enumerate(tables):
        txt = tbl.get_text(' ', strip=True).lower()
        # track section context for split SoI tables
        if 'consolidated schedule of investments' in txt or 'schedule of investments' in txt:
            if 'december 31, 2025' in txt:
                context_year = 2025
            elif 'december 31, 2024' in txt:
                context_year = 2024

        parsed, year = parse_candidate_table(tbl, context_year=context_year)
        if parsed is None:
            continue
        parsed_records.append((idx, parsed, year))

    year_frames = {2025: [], 2024: []}

    # 1) use explicit year-tagged tables first
    unknown = []
    for idx, df, year in parsed_records:
        if year in (2025, 2024):
            year_frames[year].append(df)
        else:
            unknown.append((idx, df))

    # 2) if many unknown tables exist (FSK case), split them into two index clusters:
    #    first cluster -> 2025 table block, second cluster -> 2024 table block.
    if unknown:
        unknown = sorted(unknown, key=lambda x: x[0])
        clusters = []
        cur = [unknown[0]]
        for rec in unknown[1:]:
            if rec[0] - cur[-1][0] <= 10:
                cur.append(rec)
            else:
                clusters.append(cur)
                cur = [rec]
        clusters.append(cur)

        # take two largest clusters by length, then order by index
        clusters = sorted(clusters, key=lambda c: len(c), reverse=True)[:2]
        clusters = sorted(clusters, key=lambda c: c[0][0])

        if len(clusters) >= 1:
            for _, df in clusters[0]:
                year_frames[2025].append(df)
        if len(clusters) >= 2:
            for _, df in clusters[1]:
                year_frames[2024].append(df)

    final = {}
    for y in [2025, 2024]:
        if not year_frames[y]:
            final[y] = pd.DataFrame(columns=['CompanyKey', 'Face', 'Fair'])
        else:
            tmp = pd.concat(year_frames[y], ignore_index=True)
            tmp = tmp.groupby('CompanyKey', as_index=False).agg({'Face': 'sum', 'Fair': 'sum'})
            final[y] = tmp
    return final[2025], final[2024]


def add_simple_business_intro(df):
    # lightweight rule-based one-liners by name keywords (no hallucinated deep specifics)
    def intro(name):
        n = name.lower()
        if 'medallia' in n:
            return '客户体验管理（CXM）软件平台。'
        if 'peraton' in n:
            return '国防与政府IT服务承包商。'
        if 'global jet' in n:
            return '公务航空相关融资与服务。'
        if 'lionbridge' in n:
            return '本地化与语言技术服务提供商。'
        if 'dental' in n:
            return '牙科医疗服务或相关诊疗网络。'
        if 'networks' in n:
            return '通信网络设备与基础设施相关业务。'
        return '年报持仓项对应的企业借款主体。'

    df['业务简介'] = df['CompanyKey'].apply(intro)
    return df


def analyze(ticker):
    print(f"[{ticker}] Resolving CIK...")
    cik = get_cik(ticker)
    print(f"[{ticker}] Fetching 2026-filed 10-K (contains 2025 & 2024 tables)...")
    url = fetch_latest_10k_url(cik, filing_year=2026)
    print(f"[{ticker}] 10-K URL: {url}")

    df25, df24 = extract_two_year_tables(url)
    if df25.empty or df24.empty:
        print("Failed to extract one of the two year tables (2025/2024) from the same 10-K.")
        sys.exit(1)

    merged = pd.merge(df25, df24, on='CompanyKey', how='inner', suffixes=('_2025', '_2024'))
    merged = merged[(merged['Face_2025'] > 0) & (merged['Face_2024'] > 0)]

    merged['ratio_2025'] = merged['Fair_2025'] / merged['Face_2025']
    merged['ratio_2024'] = merged['Fair_2024'] / merged['Face_2024']
    merged['ratio_change'] = merged['ratio_2025'] - merged['ratio_2024']

    # sort by worsening in fair/face ratio
    out = merged.sort_values('ratio_change', ascending=True).head(20).copy()
    out = add_simple_business_intro(out)

    show = out[[
        'CompanyKey', 'Face_2025', 'Fair_2025', 'ratio_2025',
        'Face_2024', 'Fair_2024', 'ratio_2024', 'ratio_change', '业务简介'
    ]]

    print("\n====================================================================")
    print(f"{ticker} | 2025 vs 2024 (same 2025 annual report) | Sorted by ratio change")
    print("ratio_change = (fair/face)_2025 - (fair/face)_2024")
    print("====================================================================\n")
    print(tabulate(show, headers='keys', tablefmt='psql', showindex=False, floatfmt='.4f'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', required=True)
    args = p.parse_args()
    analyze(args.ticker)
