import re
import time
from io import StringIO

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

from bdc_analyzer import (
    get_headers,
    get_cik,
    get_shareholder_equity,
    fetch_latest_10k_url,
    clean_num,
    add_simple_business_intro,
)


def _is_blank(v):
    s = '' if pd.isna(v) else str(v).strip()
    return s == '' or s.lower() in ('nan', 'none', '—', '-')


def _collapse_repeated_name(s: str) -> str:
    s = re.sub(r'\s+', ' ', str(s)).strip().upper()
    s = re.split(r'\s[—–-]\s', s)[0]
    words = s.split()
    if len(words) >= 6:
        n = len(words)
        for k in range(n // 3, n // 2 + 1):
            if words[:k] == words[k:2 * k]:
                return ' '.join(words[:k])
    return s


def _canon_key(name: str) -> str:
    s = _collapse_repeated_name(name)
    s = re.sub(r'\([^\)]*\)', ' ', s)
    s = re.sub(r'[^A-Z0-9 ]', ' ', s)
    toks = [t for t in s.split() if t not in {
        'LLC', 'INC', 'LTD', 'LP', 'L', 'P', 'CORP', 'CORPORATION', 'CO', 'AND', 'THE',
        'HOLDINGS', 'HOLDING', 'PARENT', 'TOPCO', 'MIDCO', 'BIDCO', 'BLOCKER', 'SERIES'
    }]
    return ' '.join(toks[:4])


def _retry_get_content(url: str, retries: int = 6, sleep_s: int = 3):
    last = None
    for _ in range(retries):
        try:
            r = requests.get(url, headers=get_headers(), timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last = e
            time.sleep(sleep_s)
    raise RuntimeError(f'fetch failed after retries: {last}')


def _pick_num(vals, idx):
    if idx >= len(vals):
        return None
    v = clean_num(vals[idx])
    if pd.isna(v):
        return None
    return float(v)


def _detect_colmap(df: pd.DataFrame):
    """Detect key columns and candidate amount blocks for OBDC SoI tables."""
    if df.shape[1] < 20:
        return None

    tops = []
    for c in range(df.shape[1]):
        t0 = str(df.iloc[0, c]) if len(df) > 0 else ''
        t1 = str(df.iloc[1, c]) if len(df) > 1 else ''
        t2 = str(df.iloc[2, c]) if len(df) > 2 else ''
        txt = f"{t0} {t1} {t2}".lower()
        tops.append(txt)

    def find_cols(keyword):
        return [i for i, t in enumerate(tops) if keyword in t]

    company_cols = find_cols('company')
    invest_cols = find_cols('investment')
    amort_cols = find_cols('amortized cost')
    fair_cols = find_cols('fair value')

    if not company_cols or not amort_cols or not fair_cols:
        return None

    company_col = company_cols[0]
    invest_col = invest_cols[0] if invest_cols else max(0, company_col + 6)

    return {
        'company': company_col,
        'invest': invest_col,
        'amort_cols': sorted(set(amort_cols)),
        'fair_cols': sorted(set(fair_cols)),
    }


def _parse_obdc_year(url: str, target_year: int) -> pd.DataFrame:
    soup = BeautifulSoup(_retry_get_content(url), 'lxml')

    all_rows = []

    for tbl in soup.find_all('table'):
        txt = tbl.get_text(' ', strip=True).lower()
        if 'company' not in txt:
            continue
        if 'amortized cost' not in txt or 'fair value' not in txt:
            continue
        if '% of net assets' not in txt and 'percentage of net assets' not in txt:
            continue

        try:
            df = pd.read_html(StringIO(str(tbl)))[0]
        except Exception:
            continue

        colmap = _detect_colmap(df)
        if not colmap:
            continue

        company_col = colmap['company']
        invest_col = colmap['invest']

        amort_cols = colmap['amort_cols']
        fair_cols = colmap['fair_cols']

        # map by paired order: left block tends older, right block tends current in OBDC 2025 filing
        pairs = list(zip(amort_cols[:len(fair_cols)], fair_cols[:len(amort_cols)]))
        if not pairs:
            continue
        if target_year == 2025:
            amort_candidates = [p[0] for p in pairs[::-1]]
            fair_candidates = [p[1] for p in pairs[::-1]]
        else:
            amort_candidates = [p[0] for p in pairs]
            fair_candidates = [p[1] for p in pairs]

        detail = {}
        subtotal = {}

        current = None
        closed = False

        for _, row in df.iterrows():
            vals = list(row.values)

            company_raw = str(vals[company_col]).strip() if company_col < len(vals) else ''
            inv_raw = str(vals[invest_col]).strip() if invest_col < len(vals) else ''

            comp_blank = _is_blank(company_raw)
            inv_blank = _is_blank(inv_raw)

            if not comp_blank:
                low = company_raw.lower()
                if any(k in low for k in ['company', 'schedule of investments', 'industry', 'total']):
                    continue
                current = _collapse_repeated_name(company_raw)
                closed = False

            if current is None:
                continue

            face = None
            fair = None
            for c in amort_candidates:
                face = _pick_num(vals, c)
                if face is not None:
                    break
            for c in fair_candidates:
                fair = _pick_num(vals, c)
                if fair is not None:
                    break
            if face is None and fair is None:
                continue

            f = 0.0 if face is None else face
            r = 0.0 if fair is None else fair

            if comp_blank and inv_blank:
                if not closed:
                    subtotal[current] = (f, r)
                    closed = True
                continue

            if closed:
                continue

            # detail rows should carry investment text
            if inv_blank:
                continue

            a, b = detail.get(current, (0.0, 0.0))
            detail[current] = (a + f, b + r)

        # Prefer explicit subtotal rows (blank company/investment line) when present,
        # because detail lines can be partial tranches while subtotal reflects the
        # intended company-level amount for that table block.
        chosen = {}
        for name, (f, r) in detail.items():
            chosen[name] = (f, r, 0)
        for name, (f, r) in subtotal.items():
            chosen[name] = (f, r, 1)

        for name, (f, r, is_subtotal) in chosen.items():
            if f > 0:
                all_rows.append({'CanonKey': _canon_key(name), 'CompanyKey': name, 'Face': f, 'Fair': r, 'IsSubtotal': is_subtotal})

    if not all_rows:
        return pd.DataFrame(columns=['CanonKey', 'CompanyKey', 'Face', 'Fair'])

    out = pd.DataFrame(all_rows)
    out = out[out['CanonKey'].str.len() > 0]
    # Dedupe repeated analytical tables:
    # 1) prefer subtotal-derived company rows,
    # 2) then prefer larger face (more complete row),
    # 3) then lower fair as stable tie-breaker.
    out = out.sort_values(['CanonKey', 'IsSubtotal', 'Face', 'Fair'], ascending=[True, False, False, True])
    out = out.groupby('CanonKey', as_index=False).first()[['CanonKey', 'CompanyKey', 'Face', 'Fair']]
    return out


def analyze(ticker):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik) or 1000000000

    # OBDC filing tables can be single-year layouts with unstable multi-header structure.
    # To avoid accidentally reading the same columns for both years, pull each year from
    # its own annual filing: 2025 data from 2026-filed 10-K, 2024 data from 2025-filed 10-K.
    url_2025 = fetch_latest_10k_url(cik, filing_year=2026)
    url_2024 = fetch_latest_10k_url(cik, filing_year=2025)

    df25 = _parse_obdc_year(url_2025, 2025).rename(columns={'Face': 'Face_2025', 'Fair': 'Fair_2025', 'CompanyKey': 'CompanyKey_2025'})
    # Parse prior filing with current-year selector to capture that filing's active SoI block.
    # The resulting series is used as the 2024 baseline in YoY comparison.
    df24 = _parse_obdc_year(url_2024, 2025).rename(columns={'Face': 'Face_2024', 'Fair': 'Fair_2024', 'CompanyKey': 'CompanyKey_2024'})

    merged = pd.merge(df25, df24, on='CanonKey', how='inner')
    if len(merged) == 0:
        print('\n| 公司名 | 2025年face value（金额百万美元，下同） | 2025年fair value | 2025年face/fair（用百分比表示） | 2024年face | 2024年fair | 2024年face/fair（用百分比表示） | 过去一年face/fair变化 | 公司主要业务的一句话简介 |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        return

    merged['CompanyKey'] = merged['CompanyKey_2025']
    merged = merged[(merged['Face_2025'] > 0) & (merged['Face_2024'] > 0)]

    # OBDC SoI values are in $ thousands => convert to $ millions
    merged['Face_2025_M'] = merged['Face_2025'] / 1000
    merged['Fair_2025_M'] = merged['Fair_2025'] / 1000
    merged['Face_2024_M'] = merged['Face_2024'] / 1000
    merged['Fair_2024_M'] = merged['Fair_2024'] / 1000

    threshold_m = (equity_usd / 1000000) * 0.005
    merged = merged[merged['Face_2025_M'] > threshold_m]

    merged['ratio_2025'] = merged['Fair_2025_M'] / merged['Face_2025_M']
    merged['ratio_2024'] = merged['Fair_2024_M'] / merged['Face_2024_M']
    merged['ratio_drop'] = merged['ratio_2024'] - merged['ratio_2025']

    out = merged[merged['ratio_drop'] > 0].sort_values('ratio_drop', ascending=False).head(20).copy()
    out = add_simple_business_intro(out)

    out['Face_2025_fmt'] = out['Face_2025_M'].map('{:.2f}'.format)
    out['Fair_2025_fmt'] = out['Fair_2025_M'].map('{:.2f}'.format)
    out['ratio_2025_fmt'] = (out['ratio_2025'] * 100).map('{:.2f}%'.format)
    out['Face_2024_fmt'] = out['Face_2024_M'].map('{:.2f}'.format)
    out['Fair_2024_fmt'] = out['Fair_2024_M'].map('{:.2f}'.format)
    out['ratio_2024_fmt'] = (out['ratio_2024'] * 100).map('{:.2f}%'.format)
    out['ratio_change_fmt'] = (-out['ratio_drop'] * 100).map('{:.2f}%'.format)

    show = out[[
        'CompanyKey', 'Face_2025_fmt', 'Fair_2025_fmt', 'ratio_2025_fmt',
        'Face_2024_fmt', 'Fair_2024_fmt', 'ratio_2024_fmt', 'ratio_change_fmt', '业务简介'
    ]]
    show.columns = [
        '公司名', '2025年face value（金额百万美元，下同）', '2025年fair value', '2025年face/fair（用百分比表示）',
        '2024年face', '2024年fair', '2024年face/fair（用百分比表示）', '过去一年face/fair变化', '公司主要业务的一句话简介'
    ]

    print('\n' + tabulate(show, headers='keys', tablefmt='pipe', showindex=False))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', required=True)
    args = p.parse_args()
    analyze(args.ticker)
