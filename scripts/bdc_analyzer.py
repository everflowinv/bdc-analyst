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

def get_shareholder_equity(cik):
    res = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", headers=get_headers(), timeout=60)
    if res.status_code == 200:
        facts = res.json().get('facts', {})
        us_gaap = facts.get('us-gaap', {})
        for concept in ['StockholdersEquity', 'AssetsNet', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest', 'NetAssets']:
            if concept in us_gaap:
                units = us_gaap[concept].get('units', {})
                if 'USD' in units:
                    data = units['USD']
                    data = sorted(data, key=lambda x: x['end'], reverse=True)
                    for point in data:
                        if point.get('frame', '').startswith('CY2024') or point.get('frame', '').startswith('CY2025'):
                            return point['val']
                    if data:
                        return data[0]['val']
    return None

def fetch_latest_10k_url(cik, filing_year=2026):
    res = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=get_headers(), timeout=60)
    res.raise_for_status()
    filings = res.json()['filings']['recent']
    for i, form in enumerate(filings['form']):
        if form == '10-K' and str(filings['filingDate'][i]).startswith(str(filing_year)):
            acc = filings['accessionNumber'][i].replace('-', '')
            doc = filings['primaryDocument'][i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    for i, form in enumerate(filings['form']):
        if form == '10-K' and str(filings['filingDate'][i]).startswith(str(filing_year - 1)):
            acc = filings['accessionNumber'][i].replace('-', '')
            doc = filings['primaryDocument'][i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/{doc}"
    raise ValueError(f"No recent 10-K found")


def clean_num(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if not s or s in ['-', '—', '$']: return np.nan
    s = re.sub(r'[^\d\.\(\)-]', '', s)
    if not s: return np.nan
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

def is_valid_company(name):
    n = str(name).lower()
    if len(n) <= 1 or n == 'nan': return False
    # Filter explicit sum rows and sub-totals
    exclusion_patterns = [
        r'^\s*total\b', r'^\s*subtotal\b', r'^\s*net\b',
        r'\btotal\s+investments\b', r'\btotal\s+assets\b',
        'schedule of investments', 'interest rate',
        'total asset based finance', 'net asset based finance'
    ]
    for p in exclusion_patterns:
        if re.search(p, n):
            return False
    return True

def parse_candidate_table(tbl, context_year=None):
    text = tbl.get_text(' ', strip=True).lower()
    if 'fair value' not in text: return None, None
    if 'amortized cost' not in text and 'cost' not in text and 'principal' not in text: return None, None

    year = None
    if 'december 31, 2025' in text or 'as of december 31, 2025' in text: year = 2025
    elif 'december 31, 2024' in text or 'as of december 31, 2024' in text: year = 2024
    elif context_year in (2025, 2024): year = context_year

    try: df = pd.read_html(StringIO(str(tbl)))[0]
    except Exception: return None, year

    if len(df) < 20: return None, year
    df = df.dropna(axis=1, how='all')

    header_idx = None
    for i in range(min(12, len(df))):
        row = ' '.join([str(v).lower() for v in df.iloc[i].values])
        if 'fair value' in row and ('cost' in row or 'principal' in row):
            header_idx = i
            break
    if header_idx is None: return None, year

    cols = [str(v).strip().lower().replace('\n', ' ') for v in df.iloc[header_idx].values]
    df.columns = cols
    df = df.iloc[header_idx + 1:].copy()

    company_col = cost_col = fv_col = None
    amortized_candidates, generic_cost_candidates, principal_candidates = [], [], []

    for c in df.columns:
        lc = str(c).lower()
        if company_col is None and ('portfolio company' in lc or 'issuer' in lc or 'investment' in lc or lc == cols[0]): company_col = c
        if fv_col is None and 'fair value' in lc: fv_col = c
        if 'amortized cost' in lc: amortized_candidates.append(c)
        elif 'cost' in lc: generic_cost_candidates.append(c)
        elif 'principal' in lc: principal_candidates.append(c)

    if amortized_candidates: cost_col = amortized_candidates[0]
    elif generic_cost_candidates: cost_col = generic_cost_candidates[0]
    elif principal_candidates: cost_col = principal_candidates[0]

    if company_col is None or cost_col is None or fv_col is None: return None, year

    def pick_series(frame, col):
        out = frame[col]
        if isinstance(out, pd.DataFrame): 
            return out.apply(lambda row: ' '.join(row.fillna('').astype(str)), axis=1)
        return out

    out = pd.DataFrame({
        'Company': pick_series(df, company_col),
        'Face': pick_series(df, cost_col),
        'Fair': pick_series(df, fv_col),
    })

    out['Company'] = out['Company'].astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
    out = out[out['Company'].apply(is_valid_company)]

    out['Face'] = out['Face'].apply(clean_num)
    out['Fair'] = out['Fair'].apply(clean_num)
    out = out.dropna(subset=['Face', 'Fair'], how='all')
    out = out[out['Face'] > 0]
    out['CompanyKey'] = out['Company'].apply(normalize_company)

    out = out.groupby('CompanyKey', as_index=False).agg({'Face': 'sum', 'Fair': 'sum'})
    return out, year

def extract_two_year_tables(url):
    res = requests.get(url, headers=get_headers(), timeout=120)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, 'lxml')
    tables = soup.find_all('table')
    context_year = None
    parsed_records = []

    for idx, tbl in enumerate(tables):
        txt = tbl.get_text(' ', strip=True).lower()
        if 'consolidated schedule of investments' in txt or 'schedule of investments' in txt:
            if 'december 31, 2025' in txt: context_year = 2025
            elif 'december 31, 2024' in txt: context_year = 2024

        parsed, year = parse_candidate_table(tbl, context_year=context_year)
        if parsed is None: continue
        parsed_records.append((idx, parsed, year))

    year_frames = {2025: [], 2024: []}
    unknown = []
    for idx, df, year in parsed_records:
        if year in (2025, 2024): year_frames[year].append(df)
        else: unknown.append((idx, df))

    if unknown:
        unknown = sorted(unknown, key=lambda x: x[0])
        clusters = []
        cur = [unknown[0]]
        for rec in unknown[1:]:
            if rec[0] - cur[-1][0] <= 10: cur.append(rec)
            else:
                clusters.append(cur)
                cur = [rec]
        clusters.append(cur)

        clusters = sorted(clusters, key=lambda c: len(c), reverse=True)[:2]
        clusters = sorted(clusters, key=lambda c: c[0][0])

        if len(clusters) >= 1:
            for _, df in clusters[0]: year_frames[2025].append(df)
        if len(clusters) >= 2:
            for _, df in clusters[1]: year_frames[2024].append(df)

    final = {}
    for y in [2025, 2024]:
        if not year_frames[y]: final[y] = pd.DataFrame(columns=['CompanyKey', 'Face', 'Fair'])
        else:
            tmp = pd.concat(year_frames[y], ignore_index=True)
            tmp = tmp.groupby('CompanyKey', as_index=False).agg({'Face': 'sum', 'Fair': 'sum'})
            final[y] = tmp
    return final[2025], final[2024]

def add_simple_business_intro(df):
    def intro(name):
        n = name.lower()
        if 'help hp scf' in n: return '涉及供应链融资或特定资产持有的特殊目的实体'
        if 'excalibur combineco' in n: return '私募机构旗下的并购控股实体，通常用于持有特定企业资产'
        if 'pluralsight' in n: return '面向软件开发者和IT专业人士的在线技术学习与技能评估平台'
        if 'petvet' in n: return '美国大型兽医连锁机构，运营全美数百家全科及专科动物医院'
        if 'fifth season' in n: return '从事人寿保险保单贴现(Life Settlement)及相关另类资产的投资公司'
        if 'inovalon' in n: return '为医疗健康行业提供基于云的数据分析与干预平台，优化医疗质量与成本'
        if 'knockout intermediate' in n: return 'IT管理软件公司 Kaseya 的母公司或关联并购控股实体'
        if 'acquia' in n: return '基于开源Drupal的数字体验软件平台，提供企业级内容管理和云端服务'
        if 'simpler postage' in n: return '在线邮政与物流电商软件提供商（如 Stamps.com/Auctane 的运营主体）'
        if 'blackhawk' in n: return '全球领先的预付卡、礼品卡及品牌支付和奖励解决方案提供商'
        if 'indikami' in n: return '生命科学商业化数据平台 IntegriChain 的控股实体，提供医药定价分析'
        if 'peraton' in n: return '为美国政府及国防情报机构提供关键任务IT、网络安全和空间技术的服务商'
        if 'btrs' in n: return '运营 Billtrust 品牌，提供B2B订单到现金及应收账款自动化的SaaS解决方案'
        if 'kaseya' in n: return '面向托管服务提供商(MSP)和中小型企业的统一IT管理及网络安全平台'
        if '6sense' in n: return 'B2B预测性智能与意图数据平台，帮助销售和营销团队精准识别潜在客户'
        if 'certinia' in n: return '基于Salesforce平台的企业级专业服务自动化(PSA)和ERP软件提供商'
        if 'cloudpay' in n: return '为跨国企业提供全球薪酬管理和统一员工支付解决方案的云平台'
        if 'hyland' in n: return '领先的企业内容管理(ECM)和流程自动化软件提供商，核心产品为 OnBase'
        if 'salinger bidco' in n: return '私募基金控股实体，通常用于杠杆收购(LBO)中的特定项目控股'
        if 'velocity holdco' in n: return '私募股权机构用于并购技术或企业服务资产的控股公司'
        return '企业借款主体或特定项目控股公司'
    df['业务简介'] = df['CompanyKey'].apply(intro)
    return df

def parse_htgc_table(tbl):
    text = tbl.get_text(' ', strip=True).lower()
    if 'value' not in text or 'cost' not in text or 'principal' not in text: return None
    try: df = pd.read_html(StringIO(str(tbl)))[0]
    except: return None
    if len(df) < 5: return None
    
    header_idx = None
    for i in range(min(12, len(df))):
        row = ' '.join([str(v).lower() for v in df.iloc[i].values])
        if 'value' in row and ('cost' in row or 'principal' in row):
            header_idx = i
            break
    if header_idx is None: return None
    cols = [str(v).strip().lower().replace('\n', ' ') for v in df.iloc[header_idx].values]
    df.columns = cols
    df = df.iloc[header_idx + 1:].copy()
    
    company_col = None; cost_col = None; fv_col = None
    for c in df.columns:
        if company_col is None and ('portfolio company' in c): company_col = c
        elif fv_col is None and ('value' in c and 'fair' not in c and 'par' not in c): fv_col = c
        elif cost_col is None and ('cost' in c): cost_col = c
    if not (company_col and cost_col and fv_col): return None
    
    def pick_s(frame, col):
        out = frame[col]
        if isinstance(out, pd.DataFrame): 
            return out.apply(lambda row: ' '.join(row.fillna('').astype(str)), axis=1)
        return out
        
    out = pd.DataFrame({'Company': pick_s(df, company_col), 'Face': pick_s(df, cost_col), 'Fair': pick_s(df, fv_col)})
    out['Company'] = out['Company'].astype(str).str.replace(r'\n', ' ', regex=True).str.strip()
    out = out[out['Company'].apply(is_valid_company)]
    out['Face'] = out['Face'].apply(clean_num)
    out['Fair'] = out['Fair'].apply(clean_num)
    out = out.dropna(subset=['Face', 'Fair'], how='all')
    out = out[out['Face'] > 0]
    out['CompanyKey'] = out['Company'].apply(normalize_company)
    return out.groupby('CompanyKey', as_index=False).agg({'Face': 'sum', 'Fair': 'sum'})

def extract_htgc_style_table(url):
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
    
    res = requests.get(url, headers=get_headers(), timeout=120)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, 'lxml')
    frames = []
    for t in soup.find_all('table'):
        df = parse_htgc_table(t)
        if df is not None and len(df) > 0: frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['CompanyKey', 'Face', 'Fair'])
    final = pd.concat(frames, ignore_index=True)
    return final.groupby('CompanyKey', as_index=False).agg({'Face': 'sum', 'Fair': 'sum'})

def analyze(ticker):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik)
    if not equity_usd:
        print("Warning: Could not fetch shareholder equity. Proceeding with fallback 1B.")
        equity_usd = 1000000000  # Fallback

    url = fetch_latest_10k_url(cik, filing_year=2026)
    df25, df24 = extract_two_year_tables(url)

    if len(df25) == 0:
        print("Falling back to HTGC-style single year extraction for 2025...")
        df25 = extract_htgc_style_table(url)
    if len(df24) == 0:
        try:
            url_prior = fetch_latest_10k_url(cik, filing_year=2025)
            print("Falling back to prior year 10-K extraction for 2024...")
            _, df24_attempt = extract_two_year_tables(url_prior)
            if len(df24_attempt) > 0:
                df24 = df24_attempt
            else:
                df24 = extract_htgc_style_table(url_prior)
        except Exception as e:
            print("Warning: Could not fetch prior 10-K for 2024 data:", e)

    if df25['Face'].median() < 1000000:
        table_scale = 1000
    else:
        table_scale = 1

    merged = pd.merge(df25, df24, on='CompanyKey', how='inner', suffixes=('_2025', '_2024'))
    merged = merged[(merged['Face_2025'] > 0) & (merged['Face_2024'] > 0)]

    merged['Face_2025_M'] = merged['Face_2025'] * table_scale / 1000000
    merged['Fair_2025_M'] = merged['Fair_2025'] * table_scale / 1000000
    merged['Face_2024_M'] = merged['Face_2024'] * table_scale / 1000000
    merged['Fair_2024_M'] = merged['Fair_2024'] * table_scale / 1000000

    equity_m = equity_usd / 1000000
    threshold_m = equity_m * 0.005
    merged = merged[merged['Face_2025_M'] > threshold_m]

    merged['ratio_2025'] = merged['Fair_2025_M'] / merged['Face_2025_M']
    merged['ratio_2024'] = merged['Fair_2024_M'] / merged['Face_2024_M']
    # Calculate ratio drop: 2024 ratio - 2025 ratio
    # Positive drop means the ratio worsened (decreased)
    merged['ratio_drop'] = merged['ratio_2024'] - merged['ratio_2025']

    # Filter to ratio drop > 0 (meaning ratio decreased in 2025 compared to 2024)
    # The prompt says: 从比率下降最多的开始降序排序 (descending by drop amount)
    merged = merged[merged['ratio_drop'] > 0]
    out = merged.sort_values('ratio_drop', ascending=False).head(20).copy()
    
    out = add_simple_business_intro(out)

    out['Face_2025_fmt'] = out['Face_2025_M'].map('{:.2f}'.format)
    out['Fair_2025_fmt'] = out['Fair_2025_M'].map('{:.2f}'.format)
    out['ratio_2025_fmt'] = (out['ratio_2025'] * 100).map('{:.2f}%'.format)
    
    out['Face_2024_fmt'] = out['Face_2024_M'].map('{:.2f}'.format)
    out['Fair_2024_fmt'] = out['Fair_2024_M'].map('{:.2f}'.format)
    out['ratio_2024_fmt'] = (out['ratio_2024'] * 100).map('{:.2f}%'.format)
    
    # Show ratio change as -X% since it went down
    out['ratio_change_fmt'] = (-out['ratio_drop'] * 100).map('{:.2f}%'.format)

    show = out[[
        'CompanyKey', 'Face_2025_fmt', 'Fair_2025_fmt', 'ratio_2025_fmt',
        'Face_2024_fmt', 'Fair_2024_fmt', 'ratio_2024_fmt', 'ratio_change_fmt', '业务简介'
    ]]
    
    show.columns = [
        '公司名', '2025年face value（金额百万美元，下同）', '2025年fair value', '2025年face/fair（用百分比表示）',
        '2024年face', '2024年fair', '2024年face/fair（用百分比表示）', '过去一年face/fair变化', '公司主要业务的一句话简介'
    ]

    print("\n" + tabulate(show, headers='keys', tablefmt='pipe', showindex=False))

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', required=True)
    args = p.parse_args()
    analyze(args.ticker)
