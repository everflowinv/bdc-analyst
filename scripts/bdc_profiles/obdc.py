import re
from lxml import etree
import pandas as pd
import requests
from tabulate import tabulate

from bdc_analyzer import (
    get_headers,
    get_cik,
    get_shareholder_equity,
    fetch_filing_url_for_period,
    period_to_year,
    period_to_quarter,
    fetch_latest_10k_url,
    add_simple_business_intro,
)


def _resolve_main_xml_url(filing_url: str) -> str:
    m = re.match(r'^(https://www\.sec\.gov/Archives/edgar/data/\d+/\d+/)', filing_url)
    if not m:
        return filing_url
    base = m.group(1)
    idx = requests.get(base + 'index.json', headers=get_headers(), timeout=120)
    idx.raise_for_status()
    items = idx.json().get('directory', {}).get('item', [])

    # Prefer main inline-XBRL xml doc (obdc-YYYYMMDD_htm.xml)
    cands = []
    for it in items:
        n = it.get('name', '')
        low = n.lower()
        if low.endswith('_htm.xml'):
            cands.append(n)
    cands = sorted(cands, key=lambda n: (0 if n.lower().startswith('obdc-') else 1, len(n)))
    if cands:
        return base + cands[0]
    return filing_url


def _company_only(name: str) -> str:
    s = str(name)
    # Newer filings: "Company | Specialty finance ... | Affiliated"
    if '|' in s:
        s = s.split('|')[0].strip()

    # Strip loan/instrument suffix after comma
    s = re.split(r',\s*(First lien|Second lien|Unitranche|Unsecured|Common stock|Preferred|Warrants|Class\b|LLC Interest|Notes?|Term loan|Revolver)\b', s, flags=re.I)[0]

    # remove footnotes like (d), (14)
    s = re.sub(r'\(\w+\)', '', s)
    s = re.sub(r'\(\d+\)', '', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,;')
    return s


def _extract_company_rows_from_main_xml(xml_url: str, target_instant: str) -> pd.DataFrame:
    r = requests.get(xml_url, headers=get_headers(), timeout=120)
    r.raise_for_status()
    root = etree.fromstring(r.content)

    # context id -> investment identifier member text (same instant only)
    ctx_member = {}
    for c in root.findall('.//{http://www.xbrl.org/2003/instance}context'):
        cid = c.get('id')
        inst_node = c.find('.//{http://www.xbrl.org/2003/instance}instant')
        inst = inst_node.text.strip() if inst_node is not None and inst_node.text else ''
        if inst != target_instant:
            continue

        member = ''
        for tm in c.findall('.//{http://xbrl.org/2006/xbrldi}typedMember'):
            if 'InvestmentIdentifierAxis' in (tm.get('dimension') or ''):
                member = ''.join(tm.itertext()).strip()
                break
        if member:
            ctx_member[cid] = member

    rec = {}
    for el in root.iter():
        cref = el.get('contextRef')
        if cref not in ctx_member:
            continue

        tag = el.tag.split('}')[-1]
        if tag not in ('InvestmentOwnedAtCost', 'InvestmentOwnedAtFairValue'):
            continue

        txt = (el.text or '').strip()
        if not re.match(r'^-?\d+(\.\d+)?$', txt):
            continue
        val = float(txt)
        if val <= 0:
            continue

        key = ctx_member[cref]
        d = rec.setdefault(key, {'Face': None, 'Fair': None})
        if tag == 'InvestmentOwnedAtCost':
            d['Face'] = val
        elif tag == 'InvestmentOwnedAtFairValue':
            d['Fair'] = val

    rows = []
    for raw_name, vals in rec.items():
        if vals['Face'] and vals['Fair']:
            rows.append((raw_name, vals['Face'], vals['Fair']))

    if not rows:
        return pd.DataFrame(columns=['Company', 'Face', 'Fair'])

    df = pd.DataFrame(rows, columns=['RawName', 'Face', 'Fair'])
    df['Company'] = df['RawName'].map(_company_only)
    df = df[df['Company'].str.len() > 0]
    df = df.groupby('Company', as_index=False).agg({'Face': 'sum', 'Fair': 'sum'})
    return df


def _instant_from_period(period: str) -> str:
    y = period_to_year(period)
    q = period_to_quarter(period)
    md = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}[q]
    return f'{y}-{md}'


def analyze(ticker, periodA=None, periodB=None):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik) or 1_000_000_000

    fallback_notes = []
    if periodA and periodB:
        url_a, resolved_a, fb_a = fetch_filing_url_for_period(cik, periodA, allow_fallback=True, return_meta=True)
        url_b, resolved_b, fb_b = fetch_filing_url_for_period(cik, periodB, allow_fallback=True, return_meta=True)
        dispA, dispB = periodA, periodB
        instant_a = _instant_from_period(resolved_a)
        instant_b = _instant_from_period(resolved_b)
        if fb_a:
            fallback_notes.append(f"periodA 请求 {periodA} 不可用，已回退到最近可用期 {resolved_a}")
        if fb_b:
            fallback_notes.append(f"periodB 请求 {periodB} 不可用，已回退到最近可用期 {resolved_b}")
    else:
        url_a = fetch_latest_10k_url(cik)
        url_b = url_a
        dispA, dispB = '2025', '2024'
        instant_a, instant_b = '2025-12-31', '2024-12-31'

    xml_a = _resolve_main_xml_url(url_a)
    xml_b = _resolve_main_xml_url(url_b)

    a = _extract_company_rows_from_main_xml(xml_a, instant_a).rename(columns={'Face': 'Face_A', 'Fair': 'Fair_A', 'Company': 'Company_A'})
    b = _extract_company_rows_from_main_xml(xml_b, instant_b).rename(columns={'Face': 'Face_B', 'Fair': 'Fair_B', 'Company': 'Company_B'})

    merged = a.merge(b, left_on='Company_A', right_on='Company_B', how='inner')
    if len(merged) == 0:
        if fallback_notes:
            print('\n' + '；'.join(fallback_notes))
        print('\n| 公司名 | periodA face value（金额百万美元，下同） | periodA fair value | periodA face/fair（用百分比表示） | periodB face | periodB fair | periodB face/fair（用百分比表示） | 期间face/fair变化 | 公司主要业务的一句话简介 |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        return

    merged['Face_A_M'] = merged['Face_A'] / 1e6
    merged['Fair_A_M'] = merged['Fair_A'] / 1e6
    merged['Face_B_M'] = merged['Face_B'] / 1e6
    merged['Fair_B_M'] = merged['Fair_B'] / 1e6

    # 固定筛选口径
    threshold_m = (equity_usd / 1e6) * 0.002
    merged = merged[merged['Face_A_M'] > threshold_m]

    merged['ratio_A'] = merged['Fair_A_M'] / merged['Face_A_M']
    merged['ratio_B'] = merged['Fair_B_M'] / merged['Face_B_M']
    merged['ratio_drop'] = merged['ratio_B'] - merged['ratio_A']

    out = merged[(merged['ratio_drop'] > 0) & (merged['ratio_A'] <= 1.0)].sort_values('ratio_drop', ascending=False).head(20).copy()
    out['CompanyKey'] = out['Company_A']
    out = add_simple_business_intro(out)

    out['Face_A_fmt'] = out['Face_A_M'].map('{:.3f}'.format)
    out['Fair_A_fmt'] = out['Fair_A_M'].map('{:.3f}'.format)
    out['ratio_A_fmt'] = (out['ratio_A'] * 100).map('{:.2f}%'.format)
    out['Face_B_fmt'] = out['Face_B_M'].map('{:.3f}'.format)
    out['Fair_B_fmt'] = out['Fair_B_M'].map('{:.3f}'.format)
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
    p.add_argument('--periodA', required=False)
    p.add_argument('--periodB', required=False)
    args = p.parse_args()
    analyze(args.ticker, periodA=args.periodA, periodB=args.periodB)
