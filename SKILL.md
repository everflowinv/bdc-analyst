---
name: bdc-analyst
description: SEC Financial Data Analyst specialized in parsing BDC (Business Development Company) Schedule of Investments to compare Face Value vs Fair Value year-over-year.
---

# BDC Analyst Skill

This skill allows the agent to download and parse the raw SEC HTML 10-K filings of Business Development Companies (BDCs) to extract their Schedule of Investments. 

## When to use
Use when a user asks to see which loans or assets inside a BDC's portfolio (like FSK, ARCC, MAIN, OBDC, OTF) have deteriorated the most year-over-year based on Fair Value to Face Value (or Amortized Cost) ratios.

## Core Logic & Filtering
- **Significant Assets Only**: Automatically fetches the BDC's Shareholder Equity and filters out any loans where the Face Value is less than **0.5%** of the total Shareholder Equity to remove long-tail noise.
- **Clean Aggregation**: Excludes summary rows (e.g., "Total Asset Based Finance", "Net Asset Based Finance", Subtotals) and aggregates loan tranches by base company.
- **Deterioration Ranking**: Sorts the portfolio descending by the largest drop in the `Fair Value / Face Value` ratio from 2024 to 2025. Extracts the **top 20 worst deteriorating assets** (where the ratio drop > 0).

## Usage
Use the bash command:
```bash
bash ~/.openclaw/workspace/skills/bdc-analyst/run.sh --ticker [TICKER]
```
For example: `bash ~/.openclaw/workspace/skills/bdc-analyst/run.sh --ticker OTF`

## ⚠️ AI Output Requirements (CRITICAL)

When presenting the final results to the user, the AI must ensure the output is a **Markdown Table** with the following exact column headers:
1. 公司名 (Company Name)
2. 2025年face value（金额百万美元，下同）
3. 2025年fair value
4. 2025年face/fair（用百分比表示）
5. 2024年face
6. 2024年fair
7. 2024年face/fair（用百分比表示）
8. 过去一年face/fair变化
9. **公司主要业务的一句话简介** (One-sentence business description)

### How to handle "公司主要业务的一句话简介"
The Python script provides a basic baseline, but **you (the AI) MUST use your internal knowledge to rewrite/enrich this column before outputting the table to the user**.
- Do not just output generic labels like "企业借款主体" (Corporate Borrower) if you know the company.
- Many names are PE shell entities (e.g., "Bidco", "Holdco", "Combineco", "Intermediate Holdings"). You must identify the **actual underlying operating company** (e.g., "SIMPLER POSTAGE, INC." -> Stamps.com/Auctane; "KNOCKOUT INTERMEDIATE" -> Kaseya) and describe its core business.
- Provide a concrete, one-sentence description explaining **what products they provide and who their target customers are** (e.g., "面向托管服务提供商(MSP)和中小型企业的统一IT管理及网络安全平台", "全球领先的预付卡、礼品卡及品牌支付和奖励解决方案提供商").