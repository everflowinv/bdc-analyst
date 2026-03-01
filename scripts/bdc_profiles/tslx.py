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


def _clean_display_name(name: str) -> str:
    s = _collapse_repeated_name(name)
    # remove footnote markers like (3)(4)(9)
    s = re.sub(r'\(\d+\)', '', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,;')
    return s


def _display_name_key(name: str) -> str:
    s = _clean_display_name(name).upper()
    s = re.sub(r'[^A-Z0-9 ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _enrich_business_intro(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based business intro enrichment for common BXSL names."""
    def intro(name: str) -> str:
        n = (name or '').upper()
        if 'MEDALLIA' in n:
            return '客户体验管理（CXM）软件平台，为企业提供客户反馈与体验分析SaaS。'
        if 'ACI GROUP' in n:
            return '以企业IT与数字化转型为主的技术服务平台，提供咨询与外包交付。'
        if 'DCA INVESTMENT' in n:
            return '投资控股主体，底层大概率为专业服务或工业细分资产。'
        if 'LD LOWER HOLDINGS' in n:
            return '控股型借款主体，底层业务需结合附注进一步确认。'
        if 'RED FOX CD' in n:
            return '控股型并购主体，通常对应特定细分行业资产整合平台。'
        if 'MODE PURCHASER' in n:
            return '并购载体公司，底层经营实体需结合披露附注识别。'
        if 'TRIPLE LIFT' in n:
            return '广告技术/数字营销相关平台概率较高，面向企业客户提供增长解决方案。'
        if 'NINTEX' in n:
            return '低代码流程自动化平台，帮助企业进行工作流与应用开发自动化。'
        if 'ZEUS, LLC' in n or n.startswith('ZEUS '):
            return '医疗器械与特种材料领域主体概率较高，具体以附注为准。'
        if 'SEKO GLOBAL LOGISTICS' in n:
            return '国际供应链与货代物流服务商，提供跨境运输与仓配管理。'
        if 'WINDOWS ACQUISITION' in n:
            return '门窗相关制造/分销资产的并购控股主体。'
        if 'CAMBIUM LEARNING' in n:
            return 'K-12 教育科技平台，提供课程内容、评测与教学数字化产品。'
        if 'GATEKEEPER SYSTEMS' in n:
            return '零售防损与资产追踪技术服务商，面向商超等场景提供系统方案。'
        if 'RED RIVER TECHNOLOGY' in n:
            return '企业IT基础设施集成与采购服务商，覆盖云、网络与安全领域。'
        if 'ZORRO BIDCO' in n:
            return '并购SPV主体，底层通常为规模化运营企业。'
        if 'WHCG PURCHASER' in n:
            return '医疗健康相关并购主体，底层资产多为医疗服务或技术平台。'
        if 'FENCING SUPPLY GROUP' in n:
            return '围栏与周边建材分销平台，服务承包商与工程客户。'
        if 'AMERILIFE' in n:
            return '保险分销与金融产品销售平台，聚焦寿险、年金及退休规划。'
        if 'DISCOVERY EDUCATION' in n:
            return '教育内容与数字课程平台，为学校提供教学资源与互动工具。'
        if 'NAVIGATOR ACQUIROR' in n:
            return '并购控股主体，底层经营资产需结合附注确认。'
        return '企业借款主体或特定项目控股公司。'

    out = df.copy()
    out['业务简介'] = out['CompanyKey'].apply(intro)
    return out


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
    """Detect key columns and candidate amount blocks for BXSL SoI tables."""
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

    company_cols = find_cols('company') + find_cols('investment') + find_cols('investments')
    invest_cols = find_cols('investment') + find_cols('investments')
    amort_cols = find_cols('amortized cost')
    # TSLX variants may label amount as Cost or Principal.
    cost_cols = [i for i, t in enumerate(tops) if (' cost' in t or t.strip().startswith('cost')) and 'amortized cost' not in t]
    principal_cols = find_cols('principal')
    fair_cols = find_cols('fair value') + find_cols('fairvalue')

    amount_cols = amort_cols if amort_cols else (cost_cols if cost_cols else principal_cols)

    if not company_cols or not amount_cols or not fair_cols:
        # TSLX continuation tables may lose header labels after pagination.
        # Fallback fixed layout seen in 27-column SoI fragments.
        if df.shape[1] >= 22:
            return {
                'company': 0,
                'invest': 2 if df.shape[1] > 2 else 0,
                'amort_cols': [16],
                'fair_cols': [21],
            }
        return None

    company_col = company_cols[0]
    invest_col = invest_cols[0] if invest_cols else max(0, company_col + 6)

    return {
        'company': company_col,
        'invest': invest_col,
        'amort_cols': sorted(set(amount_cols)),
        'fair_cols': sorted(set(fair_cols)),
    }


def _infer_table_context_year(tbl):
    """Infer whether a table belongs to 2025 or 2024 SoI section by nearby heading text."""
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
            m = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},\s*(202[45])', low)
            if m:
                year_dist = i
                year_val = int(m.group(2))

        if soi_dist is not None and year_dist is not None:
            break

    if soi_dist is None or year_val is None:
        return None
    return year_val


def _parse_obdc_year(url: str, target_year: int) -> pd.DataFrame:
    soup = BeautifulSoup(_retry_get_content(url), 'lxml')

    all_rows = []
    carry_company = None  # handle page-break continuation tables with blank company cells

    tables = soup.find_all('table')

    # Build candidate list first, then fill year=None by nearest known-year table.
    candidates = []
    for idx, tbl in enumerate(tables):
        txt = tbl.get_text(' ', strip=True).lower()
        if 'company' not in txt and 'investments' not in txt and 'investment' not in txt and 'loan' not in txt:
            continue
        if ('amortized cost' not in txt and ' cost ' not in f' {txt} ' and 'principal' not in txt and ' par ' not in f' {txt} ' and 'loan' not in txt):
            continue
        if 'fair value' not in txt and 'fairvalue' not in txt and ' value ' not in f' {txt} ' and 'loan' not in txt:
            continue
        candidates.append((idx, tbl, txt, _infer_table_context_year(tbl)))

    # Fill missing years by nearest known-year neighbor within SoI run.
    known = [(i, y) for i, _, _, y in candidates if y in (2025, 2024)]
    resolved = []
    for i, tbl, txt, y in candidates:
        if y in (2025, 2024):
            resolved.append((i, tbl, txt, y))
            continue
        if not known:
            continue
        nearest = min(known, key=lambda p: abs(p[0] - i))
        if abs(nearest[0] - i) <= 12:
            y = nearest[1]
            resolved.append((i, tbl, txt, y))

    for _, tbl, txt, ctx_year in resolved:
        if ctx_year != target_year:
            continue

        # TSLX layouts may omit explicit % of net assets in some split tables.
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

        current = carry_company
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

        # Preserve last seen company for continuation tables of the same year section.
        if current is not None:
            carry_company = current

    if not all_rows:
        return pd.DataFrame(columns=['CanonKey', 'CompanyKey', 'Face', 'Fair'])

    out = pd.DataFrame(all_rows)
    out = out[out['CanonKey'].str.len() > 0]
    # Dedupe repeated analytical tables for TSLX:
    # prioritize larger face first (more likely true instrument amount),
    # then lower fair as tie-breaker. Do not force subtotal priority because
    # continuation tables may surface section subtotal lines under current company.
    out = out.sort_values(['CanonKey', 'Face', 'Fair', 'IsSubtotal'], ascending=[True, False, True, False])
    out = out.groupby('CanonKey', as_index=False).first()[['CanonKey', 'CompanyKey', 'Face', 'Fair']]
    return out


def analyze(ticker):
    cik = get_cik(ticker)
    equity_usd = get_shareholder_equity(cik) or 1000000000

    # Single-file policy: use latest 10-K and extract both 2025 and 2024 from that filing.
    url_2025 = fetch_latest_10k_url(cik, filing_year=2026)

    df25 = _parse_obdc_year(url_2025, 2025).rename(columns={'Face': 'Face_2025', 'Fair': 'Fair_2025', 'CompanyKey': 'CompanyKey_2025'})
    df24 = _parse_obdc_year(url_2025, 2024).rename(columns={'Face': 'Face_2024', 'Fair': 'Fair_2024', 'CompanyKey': 'CompanyKey_2024'})

    # First align by canonical key, then aggregate by cleaned display name
    # so multiple tranches (first/second lien/revolver) under same entity are merged.
    merged = pd.merge(df25, df24, on='CanonKey', how='inner')
    if len(merged) == 0:
        print('\n| 公司名 | 2025年face value（金额百万美元，下同） | 2025年fair value | 2025年face/fair（用百分比表示） | 2024年face | 2024年fair | 2024年face/fair（用百分比表示） | 过去一年face/fair变化 | 公司主要业务的一句话简介 |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        return

    merged = merged[(merged['Face_2025'] > 0) & (merged['Face_2024'] > 0)]
    merged['DisplayName'] = merged['CompanyKey_2025'].apply(_clean_display_name)
    merged['DisplayKey'] = merged['DisplayName'].apply(_display_name_key)

    merged = merged.groupby('DisplayKey', as_index=False).agg({
        'DisplayName': 'first',
        'Face_2025': 'sum',
        'Fair_2025': 'sum',
        'Face_2024': 'sum',
        'Fair_2024': 'sum',
    })

    merged['CompanyKey'] = merged['DisplayName']

    # Auto-detect unit scale: some filings are already in $ millions, others in $ thousands.
    # Heuristic: if median face is large, treat as thousands and convert to millions.
    scale = 1000 if merged['Face_2025'].median() > 2000 else 1
    merged['Face_2025_M'] = merged['Face_2025'] / scale
    merged['Fair_2025_M'] = merged['Fair_2025'] / scale
    merged['Face_2024_M'] = merged['Face_2024'] / scale
    merged['Fair_2024_M'] = merged['Fair_2024'] / scale

    threshold_m = (equity_usd / 1000000) * 0.002
    merged = merged[merged['Face_2025_M'] > threshold_m]

    merged['ratio_2025'] = merged['Fair_2025_M'] / merged['Face_2025_M']
    merged['ratio_2024'] = merged['Fair_2024_M'] / merged['Face_2024_M']
    merged['ratio_drop'] = merged['ratio_2024'] - merged['ratio_2025']

    out = merged[(merged['ratio_drop'] > 0) & (merged['ratio_2025'] <= 1.0)].sort_values('ratio_drop', ascending=False).head(20).copy()
    out = _enrich_business_intro(out)

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
