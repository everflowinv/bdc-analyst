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
    fetch_filing_url_for_period,
    period_to_year,
    period_to_quarter,
    clean_num,
    add_simple_business_intro,
)


def _is_blank(v):
    s = '' if pd.isna(v) else str(v).strip()
    return s == '' or s.lower() in ('nan', 'none', '—', '-')


def _clean_display_name(name: str) -> str:
    s = str(name).upper().strip()
    s = re.sub(r'\(\d+\)', '', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,;')
    return s


def _display_name_key(name: str) -> str:
    s = _clean_display_name(name)
    s = re.sub(r'[^A-Z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _is_summary_like_name(name: str) -> bool:
    n = _clean_display_name(name)
    if not n:
        return True
    patterns = [
        r'^TOTAL\b', r'^SUBTOTAL\b', r'^NET\b',
        r'\bDEBT INVESTMENTS\b', r'\bEQUITY INVESTMENTS\b',
        r'\bFIRST LIEN\b', r'\bSECOND LIEN\b',
        r'\bSENIOR SECURED\b', r'\bPREFERRED STOCK\b',
        r'\bLP INTEREST\b', r'\bREVOLVING LINE OF CREDIT\b',
        r'\bASSET BASED\b', r'\bALL ASSETS\b',
    ]
    for p in patterns:
        if re.search(p, n):
            return True
    return False


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


def _infer_table_context_year(tbl):
    year_dist = None
    year_val = None
    soi_dist = None

    for i, s in enumerate(tbl.find_all_previous(string=True, limit=400)):
        t = ' '.join(str(s).split())
        if not t:
            continue
        low = t.lower()

        if soi_dist is None and ('schedule of investments' in low):
            soi_dist = i

        if year_dist is None:
            m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*(20\d{2})', low)
            if m:
                year_dist = i
                year_val = int(m.group(2))

        if soi_dist is not None and year_dist is not None:
            break

    if soi_dist is None or year_val is None:
        return None
    return year_val


def _period_label(target_year: int, target_quarter: int):
    md = {1: 'march 31', 2: 'june 30', 3: 'september 30', 4: 'december 31'}.get(target_quarter)
    if not md:
        return None
    return f"{md}, {target_year}"


def _table_matches_period(tbl, target_year: int, target_quarter: int | None):
    if target_quarter is None:
        return True
    label = _period_label(target_year, target_quarter)
    if not label:
        return True
    for s in tbl.find_all_previous(string=True, limit=250):
        t = ' '.join(str(s).split()).lower()
        if label in t:
            return True
    return False


def _pick_num_from_cols(row_vals, cols):
    for c in cols:
        if c < len(row_vals):
            v = clean_num(row_vals[c])
            if pd.notna(v) and v > 0:
                return float(v)
    return None


def _detect_arcc_colmap(df: pd.DataFrame):
    if df.shape[1] < 20:
        return None

    tops = []
    for c in range(df.shape[1]):
        t0 = str(df.iloc[0, c]) if len(df) > 0 else ''
        t1 = str(df.iloc[1, c]) if len(df) > 1 else ''
        t2 = str(df.iloc[2, c]) if len(df) > 2 else ''
        tops.append(f"{t0} {t1} {t2}".lower())

    def find_cols(keyword):
        return [i for i, t in enumerate(tops) if keyword in t]

    company_cols = find_cols('company') + find_cols('portfolio company')
    invest_cols = find_cols('investment') + find_cols('investments')
    amort_cols = find_cols('amortized cost')
    fair_cols = find_cols('fair value')

    if company_cols and amort_cols and fair_cols:
        return {
            'company': company_cols[0],
            'invest': invest_cols[0] if invest_cols else min(company_cols[0] + 12, df.shape[1] - 1),
            'amort_cols': sorted(set(amort_cols)),
            'fair_cols': sorted(set(fair_cols)),
        }

    # ARCC common wide-table fallback layout
    if df.shape[1] >= 60:
        return {
            'company': 0,
            'invest': 12,
            'amort_cols': [51, 52, 53],
            'fair_cols': [57, 58, 59],
        }

    return None


def _parse_arcc_period(url: str, target_year: int, target_quarter: int | None = None) -> pd.DataFrame:
    soup = BeautifulSoup(_retry_get_content(url), 'lxml')
    tables = soup.find_all('table')

    candidates = []
    for idx, tbl in enumerate(tables):
        txt = tbl.get_text(' ', strip=True).lower()
        if 'schedule of investments' not in txt and 'company' not in txt:
            continue
        if 'fair value' not in txt:
            continue
        if 'amortized cost' not in txt and 'cost' not in txt and 'principal' not in txt:
            continue
        if 'for the year ended' in txt:
            continue
        year = _infer_table_context_year(tbl)
        candidates.append((idx, tbl, txt, year))

    known = [(i, y) for i, _, _, y in candidates if y is not None]
    resolved = []
    for i, tbl, txt, y in candidates:
        if y is not None:
            resolved.append((i, tbl, txt, y))
            continue
        if not known:
            continue
        prev = [k for k in known if k[0] <= i]
        nxt = [k for k in known if k[0] > i]
        picked = None
        if prev:
            picked = prev[-1]
            if i - picked[0] > 20:
                picked = None
        if picked is None and nxt:
            picked = nxt[0]
            if picked[0] - i > 12:
                picked = None
        if picked is not None:
            resolved.append((i, tbl, txt, picked[1]))

    rows = []
    for tbl_idx, tbl, _, ctx_year in resolved:
        if ctx_year != target_year:
            continue
        if not _table_matches_period(tbl, target_year, target_quarter):
            continue

        try:
            df = pd.read_html(StringIO(str(tbl)))[0]
        except Exception:
            continue

        colmap = _detect_arcc_colmap(df)
        if not colmap:
            continue

        company_col = colmap['company']
        invest_col = colmap['invest']

        detail_sum = {}
        subtotal_override = {}
        current = None
        closed = False

        for _, r in df.iterrows():
            vals = list(r.values)

            company = str(vals[company_col]).strip() if company_col < len(vals) else ''
            invest = str(vals[invest_col]).strip() if invest_col < len(vals) else ''

            face = _pick_num_from_cols(vals, colmap['amort_cols'])
            fair = _pick_num_from_cols(vals, colmap['fair_cols'])
            has_num = (face is not None) or (fair is not None)

            comp_blank = _is_blank(company)
            invest_blank = _is_blank(invest)

            if not comp_blank:
                low = company.lower()
                if any(k in low for k in ['company', 'assets category', 'schedule of investments', 'industry']):
                    continue
                if _is_summary_like_name(company):
                    current = None
                    closed = False
                    continue
                current = _clean_display_name(company)
                closed = False

            if current is None or not has_num:
                continue

            if face is None:
                face = fair
            if fair is None:
                fair = face

            if comp_blank and invest_blank:
                if (not closed) and (current in detail_sum):
                    subtotal_override[current] = (float(face), float(fair))
                    closed = True
                continue

            if closed:
                continue

            a, b = detail_sum.get(current, (0.0, 0.0))
            detail_sum[current] = (a + float(face), b + float(fair))

        for k, (f, r) in detail_sum.items():
            if k in subtotal_override:
                f, r = subtotal_override[k]
            if f > 0 and r > 0 and not _is_summary_like_name(k):
                rows.append({'DisplayKey': _display_name_key(k), 'CompanyKey': k, 'Face': f, 'Fair': r, 'TableIdx': tbl_idx})

    if not rows:
        return pd.DataFrame(columns=['DisplayKey', 'CompanyKey', 'Face', 'Fair'])

    out = pd.DataFrame(rows)
    out = out[out['DisplayKey'].str.len() > 0]
    # same spirit as PFLT: dedupe repeated table blocks by earliest table and larger face
    out = out.sort_values(['DisplayKey', 'TableIdx', 'Face'], ascending=[True, True, False])
    out = out.groupby('DisplayKey', as_index=False).first()[['DisplayKey', 'CompanyKey', 'Face', 'Fair']]
    return out


def analyze(ticker, periodA=None, periodB=None):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik) or 1000000000

    fallback_notes = []
    if periodA and periodB:
        year_a = period_to_year(periodA)
        year_b = period_to_year(periodB)
        q_a = period_to_quarter(periodA)
        q_b = period_to_quarter(periodB)
        url_a, resolved_a, fb_a = fetch_filing_url_for_period(cik, periodA, allow_fallback=True, return_meta=True)
        url_b, resolved_b, fb_b = fetch_filing_url_for_period(cik, periodB, allow_fallback=True, return_meta=True)
        if fb_a:
            fallback_notes.append(f"periodA 请求 {periodA} 不可用，已回退到最近可用期 {resolved_a}")
        if fb_b:
            fallback_notes.append(f"periodB 请求 {periodB} 不可用，已回退到最近可用期 {resolved_b}")
        dispA, dispB = periodA, periodB
    else:
        url_a = fetch_latest_10k_url(cik)
        url_b = url_a
        m = re.search(r'(20\d{2})', url_a)
        year_a = int(m.group(1)) if m else 2025
        year_b = year_a - 1
        q_a, q_b = None, None
        dispA, dispB = str(year_a), str(year_b)

    df_a = _parse_arcc_period(url_a, year_a, q_a).rename(columns={'Face': 'Face_A', 'Fair': 'Fair_A', 'CompanyKey': 'CompanyKey_A'})
    df_b = _parse_arcc_period(url_b, year_b, q_b).rename(columns={'Face': 'Face_B', 'Fair': 'Fair_B', 'CompanyKey': 'CompanyKey_B'})

    merged = pd.merge(df_a, df_b, on='DisplayKey', how='inner')
    if len(merged) == 0:
        print('\n| 公司名 | periodA face value（金额百万美元，下同） | periodA fair value | periodA face/fair（用百分比表示） | periodB face | periodB fair | periodB face/fair（用百分比表示） | 期间face/fair变化 | 公司主要业务的一句话简介 |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        return

    merged = merged[(merged['Face_A'] > 0) & (merged['Face_B'] > 0)]
    merged['DisplayName'] = merged['CompanyKey_A'].apply(_clean_display_name)
    merged['DisplayKey2'] = merged['DisplayName'].apply(_display_name_key)

    merged = merged.groupby('DisplayKey2', as_index=False).agg({
        'DisplayName': 'first',
        'Face_A': 'sum',
        'Fair_A': 'sum',
        'Face_B': 'sum',
        'Fair_B': 'sum',
    })
    merged['CompanyKey'] = merged['DisplayName']

    scale = 1000 if merged['Face_A'].median() > 2000 else 1
    merged['Face_A_M'] = merged['Face_A'] / scale
    merged['Fair_A_M'] = merged['Fair_A'] / scale
    merged['Face_B_M'] = merged['Face_B'] / scale
    merged['Fair_B_M'] = merged['Fair_B'] / scale

    threshold_m = (equity_usd / 1000000) * 0.002
    merged = merged[merged['Face_A_M'] > threshold_m]

    merged['ratio_A'] = merged['Fair_A_M'] / merged['Face_A_M']
    merged['ratio_B'] = merged['Fair_B_M'] / merged['Face_B_M']
    merged['ratio_drop'] = merged['ratio_B'] - merged['ratio_A']

    out = merged[(merged['ratio_drop'] > 0) & (merged['ratio_A'] <= 1.0) & (merged['ratio_B'] <= 1.2)].sort_values('ratio_drop', ascending=False).head(20).copy()
    out = add_simple_business_intro(out)

    out['Face_A_fmt'] = out['Face_A_M'].map('{:.2f}'.format)
    out['Fair_A_fmt'] = out['Fair_A_M'].map('{:.2f}'.format)
    out['ratio_A_fmt'] = (out['ratio_A'] * 100).map('{:.2f}%'.format)
    out['Face_B_fmt'] = out['Face_B_M'].map('{:.2f}'.format)
    out['Fair_B_fmt'] = out['Fair_B_M'].map('{:.2f}'.format)
    out['ratio_B_fmt'] = (out['ratio_B'] * 100).map('{:.2f}%'.format)
    out['ratio_change_fmt'] = (-out['ratio_drop'] * 100).map('{:.2f}%'.format)

    show = out[[
        'CompanyKey', 'Face_A_fmt', 'Fair_A_fmt', 'ratio_A_fmt',
        'Face_B_fmt', 'Fair_B_fmt', 'ratio_B_fmt', 'ratio_change_fmt', '业务简介'
    ]]
    show.columns = [
        '公司名', f'{dispA} face value（金额百万美元，下同）', f'{dispA} fair value', f'{dispA} face/fair（用百分比表示）',
        f'{dispB} face', f'{dispB} fair', f'{dispB} face/fair（用百分比表示）', '期间face/fair变化', '公司主要业务的一句话简介'
    ]

    if fallback_notes:
        print('\n' + '；'.join(fallback_notes))
    print('\n' + tabulate(show, headers='keys', tablefmt='pipe', showindex=False))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--ticker', required=True)
    args = p.parse_args()
    analyze(args.ticker)
