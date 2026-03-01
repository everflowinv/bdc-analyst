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
    extract_two_year_tables,
)


def _is_blank(v):
    s = '' if pd.isna(v) else str(v).strip()
    return s == '' or s.lower() in ('nan', 'none', '—', '-')


def _pick_series(df, col):
    out = df[col]
    if isinstance(out, pd.DataFrame):
        return out.apply(lambda row: ' '.join(row.fillna('').astype(str)), axis=1)
    return out


def _find_header_idx(df0):
    for i in range(min(16, len(df0))):
        row = ' '.join([str(v).lower() for v in df0.iloc[i].values])
        if 'fair value' in row and ('amortized cost' in row or 'cost' in row or 'principal' in row):
            return i
    return None


def _parse_group_subtotals(tbl):
    """Return (subtotals, members) from one SoI table.

    subtotals: {GROUP_KEY: (face, fair)}
    members: {GROUP_KEY: set(member_company_keys)}
    """
    try:
        df0 = pd.read_html(StringIO(str(tbl)))[0]
    except Exception:
        return {}, {}

    if len(df0) < 20:
        return {}, {}

    df0 = df0.dropna(axis=1, how='all')
    header_idx = _find_header_idx(df0)
    if header_idx is None:
        return {}, {}

    cols = [str(v).strip().lower().replace('\n', ' ') for v in df0.iloc[header_idx].values]
    df = df0.iloc[header_idx + 1:].copy()
    df.columns = cols

    company_col = None
    desc_col = None
    fair_candidates = []
    cost_candidates = []

    for c in df.columns:
        lc = str(c).lower()
        if company_col is None and ('portfolio company' in lc or 'issuer' in lc or lc == cols[0]):
            company_col = c
        if desc_col is None and ('investment' in lc or 'debt' in lc or 'equity' in lc or 'type' in lc):
            desc_col = c
        if 'fair value' in lc:
            fair_candidates.append(c)
        if 'amortized cost' in lc or ('cost' in lc and 'fair' not in lc) or 'principal' in lc:
            cost_candidates.append(c)

    if company_col is None or not fair_candidates or not cost_candidates:
        return {}, {}

    # use rightmost duplicated columns in this table block (MFIC layout)
    fair_col = fair_candidates[-1]
    cost_col = cost_candidates[-1]

    tmp = pd.DataFrame({
        'company': _pick_series(df, company_col).astype(str).str.replace(r'\n', ' ', regex=True).str.strip(),
        'desc': _pick_series(df, desc_col).astype(str).str.replace(r'\n', ' ', regex=True).str.strip() if desc_col is not None else '',
        'face': _pick_series(df, cost_col).apply(clean_num),
        'fair': _pick_series(df, fair_col).apply(clean_num),
    })

    subtotals = {}
    members = {}
    active_group = None

    for _, r in tmp.iterrows():
        comp = r['company']
        desc = r['desc']
        face = r['face']
        fair = r['fair']

        has_num = pd.notna(face) or pd.notna(fair)
        comp_blank = _is_blank(comp)
        desc_blank = _is_blank(desc)

        # group header row: text only
        if (not comp_blank) and (not has_num):
            low = comp.lower()
            if 'total' not in low and 'industry' not in low and 'schedule' not in low and 'portfolio' not in low:
                active_group = normalize_company(comp)
                members.setdefault(active_group, set())
            continue

        if not has_num:
            continue

        fv = 0.0 if pd.isna(face) else float(face)
        fr = 0.0 if pd.isna(fair) else float(fair)

        # subtotal row for active group
        if active_group and comp_blank and desc_blank:
            subtotals[active_group] = (fv, fr)
            active_group = None
            continue

        # detail rows under active group: collect member company keys
        if active_group and (not comp_blank):
            ckey = normalize_company(comp)
            members.setdefault(active_group, set()).add(ckey)

    # only keep groups that actually have subtotals
    members = {k: v for k, v in members.items() if k in subtotals}
    return subtotals, members


def _collect_group_candidates(soup):
    """Collect subtotal candidates per group from all parseable SoI-like tables.

    Returns {group: [{'face':..,'fair':..,'members':set(),'idx':table_idx}, ...]}
    """
    out = {}
    for idx, tbl in enumerate(soup.find_all('table')):
        txt = tbl.get_text(' ', strip=True).lower()
        if 'fair value' not in txt:
            continue

        subs, mems = _parse_group_subtotals(tbl)
        if not subs:
            continue

        for g, (f, r) in subs.items():
            out.setdefault(g, []).append({
                'face': float(f),
                'fair': float(r),
                'members': set(mems.get(g, set())),
                'idx': idx,
            })
    return out


def _split_candidates_to_year_maps(candidates):
    """Split subtotal candidates into 2025/2024 maps using descending face heuristic.

    In MFIC 2025 filing, the current-year subtotal is typically the larger face amount,
    while prior-year subtotal is the next candidate for the same group.
    """
    subs25, mem25 = {}, {}
    subs24, mem24 = {}, {}

    for g, rows in candidates.items():
        rows = sorted(rows, key=lambda x: x['face'], reverse=True)
        if not rows:
            continue

        subs25[g] = (rows[0]['face'], rows[0]['fair'])
        mem25[g] = set(rows[0]['members'])

        if len(rows) > 1:
            subs24[g] = (rows[1]['face'], rows[1]['fair'])
            mem24[g] = set(rows[1]['members'])

    return (subs25, mem25), (subs24, mem24)


def _apply_group_subtotals(df, subtotals, members):
    df = df.copy()

    # remove member rows + existing group row, then insert group subtotal row
    for g, (f, r) in subtotals.items():
        member_keys = set(members.get(g, set()))
        drop_mask = df['CompanyKey'].eq(g)
        if member_keys:
            drop_mask = drop_mask | df['CompanyKey'].isin(member_keys)
        df = df.loc[~drop_mask]
        df = pd.concat([df, pd.DataFrame([{'CompanyKey': g, 'Face': f, 'Fair': r}])], ignore_index=True)

    return df


def extract_two_year_tables_mfic(url):
    # base extraction from generic parser
    df25, df24 = extract_two_year_tables(url)

    res = requests.get(url, headers=get_headers(), timeout=120)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, 'lxml')

    candidates = _collect_group_candidates(soup)
    (subs25, mem25), (subs24, mem24) = _split_candidates_to_year_maps(candidates)

    if subs25:
        df25 = _apply_group_subtotals(df25, subs25, mem25)
    if subs24:
        df24 = _apply_group_subtotals(df24, subs24, mem24)

    return df25, df24


def analyze(ticker):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik) or 1000000000

    url = fetch_latest_10k_url(cik, filing_year=2026)
    df25, df24 = extract_two_year_tables_mfic(url)

    merged = pd.merge(df25, df24, on='CompanyKey', how='inner', suffixes=('_2025', '_2024'))
    merged = merged[(merged['Face_2025'] > 0) & (merged['Face_2024'] > 0)]

    table_scale = 1000 if (len(df25) and df25['Face'].median() < 1000000) else 1

    merged['Face_2025_M'] = merged['Face_2025'] * table_scale / 1000000
    merged['Fair_2025_M'] = merged['Fair_2025'] * table_scale / 1000000
    merged['Face_2024_M'] = merged['Face_2024'] * table_scale / 1000000
    merged['Fair_2024_M'] = merged['Fair_2024'] * table_scale / 1000000

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
