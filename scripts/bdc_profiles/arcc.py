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
    normalize_company,
    add_simple_business_intro,
)


def _is_blank(v):
    s = '' if pd.isna(v) else str(v).strip()
    return s == '' or s.lower() in ('nan', 'none', '—', '-')


def _pick_num_from_cols(row, cols):
    for c in cols:
        if c < len(row):
            v = clean_num(row[c])
            if pd.notna(v) and v > 0:
                return float(v)
    return None


def _extract_arcc_company_rows(url):
    """Company-level parser for ARCC SoI tables (no hardcoded company names).

    Handles:
    - continuation rows with blank company name
    - subtotal rows (blank company, numeric face/fair) overriding detail sum
    """
    res = requests.get(url, headers=get_headers(), timeout=120)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, 'lxml')

    rows = []

    for tbl in soup.find_all('table'):
        txt = tbl.get_text(' ', strip=True)
        low = txt.lower()
        if 'company (1)' not in low:
            continue
        if 'amortized cost' not in low or 'fair value' not in low:
            continue
        # keep Schedule-of-Investments style tables; exclude activity tables ("for the year ended")
        if 'for the year ended' in low:
            continue
        if '% of net assets' not in low:
            continue

        try:
            df = pd.read_html(StringIO(str(tbl)))[0]
        except Exception:
            continue

        # SoI detail tables are wide in ARCC filing
        if df.shape[1] < 60:
            continue

        c25 = low.count('2025')
        c24 = low.count('2024')
        year = 2025 if c25 >= c24 else 2024

        # per-table accumulation with continuation support
        detail_sum = {}
        subtotal_override = {}
        current_key = None
        company_closed = False

        for _, r in df.iterrows():
            vals = list(r.values)

            company = str(vals[0]).strip() if len(vals) > 0 else ''
            invest = str(vals[12]).strip() if len(vals) > 12 else ''
            desc = str(vals[6]).strip() if len(vals) > 6 else ''

            face = _pick_num_from_cols(vals, [51, 52, 53])
            fair = _pick_num_from_cols(vals, [57, 58, 59])

            has_num = (face is not None) or (fair is not None)
            comp_blank = _is_blank(company)
            invest_blank = _is_blank(invest)
            desc_blank = _is_blank(desc)

            # skip obvious header/section rows
            if not comp_blank:
                low_name = company.lower()
                if any(k in low_name for k in ['company (1)', 'assets category', 'schedule of investments', 'industry']):
                    continue

            # new company anchor row
            if (not comp_blank) and (not company.lower().startswith('total')):
                current_key = normalize_company(company)
                company_closed = False

            if not has_num:
                continue

            # normalize missing side
            if face is None:
                face = fair
            if fair is None:
                fair = face

            # detail / continuation line: require company row or investment-type row
            if current_key is None:
                continue

            if comp_blank and invest_blank:
                # first blank subtotal-like row after detail lines is usually company subtotal; close the block.
                if (not company_closed) and (current_key in detail_sum):
                    subtotal_override[current_key] = (float(face), float(fair))
                    company_closed = True
                continue

            if company_closed:
                continue

            a, b = detail_sum.get(current_key, (0.0, 0.0))
            detail_sum[current_key] = (a + float(face), b + float(fair))

        for k, (f, r) in detail_sum.items():
            if k in subtotal_override:
                f, r = subtotal_override[k]
            if f > 0 and r > 0:
                rows.append({'year': year, 'CompanyKey': k, 'Face': f, 'Fair': r})

    if not rows:
        return pd.DataFrame(columns=['year', 'CompanyKey', 'Face', 'Fair'])

    out = pd.DataFrame(rows)
    # ARCC filing has repeated analytical tables; keep largest Face, and for ties pick smaller Fair (more conservative).
    out = out.sort_values(['year', 'CompanyKey', 'Face', 'Fair'], ascending=[True, True, False, True])
    out = out.groupby(['year', 'CompanyKey'], as_index=False).first()[['year', 'CompanyKey', 'Face', 'Fair']]
    return out


def analyze(ticker):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik) or 1000000000
    url = fetch_latest_10k_url(cik, filing_year=2026)

    raw = _extract_arcc_company_rows(url)
    if len(raw) == 0:
        print('\n| 公司名 | 2025年face value（金额百万美元，下同） | 2025年fair value | 2025年face/fair（用百分比表示） | 2024年face | 2024年fair | 2024年face/fair（用百分比表示） | 过去一年face/fair变化 | 公司主要业务的一句话简介 |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        return

    df25 = raw[raw['year'] == 2025][['CompanyKey', 'Face', 'Fair']].rename(columns={'Face': 'Face_2025', 'Fair': 'Fair_2025'})
    df24 = raw[raw['year'] == 2024][['CompanyKey', 'Face', 'Fair']].rename(columns={'Face': 'Face_2024', 'Fair': 'Fair_2024'})

    merged = pd.merge(df25, df24, on='CompanyKey', how='inner')
    merged = merged[(merged['Face_2025'] > 0) & (merged['Face_2024'] > 0)]

    merged['Face_2025_M'] = merged['Face_2025']
    merged['Fair_2025_M'] = merged['Fair_2025']
    merged['Face_2024_M'] = merged['Face_2024']
    merged['Fair_2024_M'] = merged['Fair_2024']

    threshold_m = (equity_usd / 1000000) * 0.002
    merged = merged[merged['Face_2025_M'] > threshold_m]

    merged['ratio_2025'] = merged['Fair_2025_M'] / merged['Face_2025_M']
    merged['ratio_2024'] = merged['Fair_2024_M'] / merged['Face_2024_M']
    merged['ratio_drop'] = merged['ratio_2024'] - merged['ratio_2025']

    out = merged[(merged['ratio_drop'] > 0) & (merged['ratio_2025'] <= 1.0)].sort_values('ratio_drop', ascending=False).head(20).copy()
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
