import re
from collections import defaultdict

import pandas as pd
import requests
from lxml import etree
from tabulate import tabulate

from bdc_analyzer import (
    get_headers,
    get_cik,
    get_shareholder_equity,
    fetch_latest_10k_url,
    fetch_filing_url_for_period,
    period_to_year,
    period_to_quarter,
)


def _resolve_main_xml_url(filing_url: str) -> str:
    m = re.match(r'^(https://www\.sec\.gov/Archives/edgar/data/\d+/\d+/)', filing_url)
    if not m:
        return filing_url
    base = m.group(1)
    r = requests.get(base + 'index.json', headers=get_headers(), timeout=120)
    r.raise_for_status()
    items = r.json().get('directory', {}).get('item', [])
    cands = [it.get('name', '') for it in items if it.get('name', '').lower().endswith('_htm.xml')]
    cands = sorted(cands, key=lambda n: (0 if n.lower().startswith('cswc-') else 1, len(n)))
    return base + cands[0] if cands else filing_url


def _company_only(name: str) -> str:
    s = str(name)
    s = s.split('|')[0].strip()
    s = re.sub(r'\(\d+\)', '', s)
    s = re.split(r',\s*(First lien|Second lien|Unitranche|Revolving|Delayed Draw|Common|Preferred|Warrants|Subordinated|LLC Interest|Equity|Debt|Shares?)\b', s, flags=re.I)[0]
    s = re.sub(r',\s*CLASS\s+[A-Z0-9\-]+\s+(UNITS?|STOCK|SHARES?)\b.*$', '', s, flags=re.I)
    s = re.sub(r',\s*(UNITS?|STOCK|SHARES?)\b.*$', '', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip(' ,;')
    return s


def _company_key(name: str) -> str:
    s = _company_only(name).upper()
    s = re.sub(r'[^A-Z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _instrument_key(name: str) -> str:
    s = str(name).upper()
    s = re.sub(r'\(\d+\)', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _instant_from_period(period: str) -> str:
    y = period_to_year(period)
    q = period_to_quarter(period)
    md = {1: '03-31', 2: '06-30', 3: '09-30', 4: '12-31'}[q]
    return f'{y}-{md}'


def _extract_from_xml(xml_url: str, target_instant: str) -> pd.DataFrame:
    root = etree.fromstring(requests.get(xml_url, headers=get_headers(), timeout=120).content)

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

    # context-level dedup: keep max cost/fair in each context (avoid duplicate facts)
    by_ctx = defaultdict(lambda: {'member': '', 'face': 0.0, 'fair': 0.0})
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
        v = float(txt)
        if v < 0:
            continue

        by_ctx[cref]['member'] = ctx_member[cref]
        if tag == 'InvestmentOwnedAtCost':
            by_ctx[cref]['face'] = max(by_ctx[cref]['face'], v)
        else:
            by_ctx[cref]['fair'] = max(by_ctx[cref]['fair'], v)

    # instrument-level dedup: same instrument can appear in multiple equivalent contexts
    by_instr = defaultdict(lambda: {'raw': '', 'face': 0.0, 'fair': 0.0})
    for rec in by_ctx.values():
        raw = rec['member']
        if not raw:
            continue
        ik = _instrument_key(raw)
        by_instr[ik]['raw'] = raw
        by_instr[ik]['face'] = max(by_instr[ik]['face'], rec['face'])
        by_instr[ik]['fair'] = max(by_instr[ik]['fair'], rec['fair'])

    # company-level aggregation across debt/equity instruments
    comp = defaultdict(lambda: {'Name': '', 'Face': 0.0, 'Fair': 0.0})
    for rec in by_instr.values():
        company = _company_only(rec['raw'])
        key = _company_key(rec['raw'])
        if not key:
            continue
        if not comp[key]['Name']:
            comp[key]['Name'] = company.upper()
        comp[key]['Face'] += rec['face']
        comp[key]['Fair'] += rec['fair']

    rows = []
    for k, v in comp.items():
        if v['Face'] > 0 or v['Fair'] > 0:
            rows.append({'CompanyKey': v['Name'] or k, 'Face': float(v['Face']), 'Fair': float(v['Fair'])})

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['CompanyKey', 'Face', 'Fair'])


def analyze(ticker, periodA=None, periodB=None):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik) or 1_000_000_000

    fallback_notes = []
    if periodA and periodB:
        url_a, resolved_a, fb_a = fetch_filing_url_for_period(cik, periodA, allow_fallback=True, return_meta=True)
        url_b, resolved_b, fb_b = fetch_filing_url_for_period(cik, periodB, allow_fallback=True, return_meta=True)
        inst_a = _instant_from_period(resolved_a)
        inst_b = _instant_from_period(resolved_b)
        dispA, dispB = periodA, periodB
        if fb_a:
            fallback_notes.append(f"periodA 请求 {periodA} 不可用，已回退到最近可用期 {resolved_a}")
        if fb_b:
            fallback_notes.append(f"periodB 请求 {periodB} 不可用，已回退到最近可用期 {resolved_b}")
    else:
        url_a = fetch_latest_10k_url(cik, filing_year=2026)
        url_b = url_a
        inst_a, inst_b = '2025-12-31', '2024-12-31'
        dispA, dispB = '2025', '2024'

    xml_a = _resolve_main_xml_url(url_a)
    xml_b = _resolve_main_xml_url(url_b)

    a = _extract_from_xml(xml_a, inst_a).rename(columns={'Face': 'Face_A', 'Fair': 'Fair_A'})
    b = _extract_from_xml(xml_b, inst_b).rename(columns={'Face': 'Face_B', 'Fair': 'Fair_B'})

    merged = a.merge(b, on='CompanyKey', how='inner')
    if len(merged) == 0:
        print('\n| 公司名 | periodA face value（金额百万美元，下同） | periodA fair value | periodA face/fair（用百分比表示） | periodB face | periodB fair | periodB face/fair（用百分比表示） | 期间face/fair变化 | 公司主要业务的一句话简介 |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        return

    merged = merged[(merged['Face_A'] > 0) & (merged['Face_B'] > 0)]

    # CSWC XBRL facts are in USD -> convert to $ millions
    merged['Face_A_M'] = merged['Face_A'] / 1_000_000
    merged['Fair_A_M'] = merged['Fair_A'] / 1_000_000
    merged['Face_B_M'] = merged['Face_B'] / 1_000_000
    merged['Fair_B_M'] = merged['Fair_B'] / 1_000_000

    threshold_m = (equity_usd / 1_000_000) * 0.002
    merged = merged[merged['Face_A_M'] > threshold_m]

    merged['ratio_A'] = merged['Fair_A_M'] / merged['Face_A_M']
    merged['ratio_B'] = merged['Fair_B_M'] / merged['Face_B_M']
    merged['ratio_drop'] = merged['ratio_B'] - merged['ratio_A']

    out = merged[(merged['ratio_drop'] > 0) & (merged['ratio_A'] <= 1.0) & (merged['ratio_B'] <= 1.2)].sort_values('ratio_drop', ascending=False).head(20).copy()
    out['业务简介'] = '企业借款主体或特定项目控股公司。'

    out['Face_A_fmt'] = out['Face_A_M'].map('{:.2f}'.format)
    out['Fair_A_fmt'] = out['Fair_A_M'].map('{:.2f}'.format)
    out['ratio_A_fmt'] = (out['ratio_A'] * 100).map('{:.2f}%'.format)
    out['Face_B_fmt'] = out['Face_B_M'].map('{:.2f}'.format)
    out['Fair_B_fmt'] = out['Fair_B_M'].map('{:.2f}'.format)
    out['ratio_B_fmt'] = (out['ratio_B'] * 100).map('{:.2f}%'.format)
    out['ratio_change_fmt'] = (-out['ratio_drop'] * 100).map('{:.2f}%'.format)

    show = out[['CompanyKey', 'Face_A_fmt', 'Fair_A_fmt', 'ratio_A_fmt', 'Face_B_fmt', 'Fair_B_fmt', 'ratio_B_fmt', 'ratio_change_fmt', '业务简介']]
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
