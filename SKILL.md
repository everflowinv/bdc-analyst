---
name: bdc-analyst
description: SEC Financial Data Analyst specialized in parsing BDC (Business Development Company) Schedule of Investments to compare Face Value vs Fair Value year-over-year.
---

# BDC Analyst Skill

This skill allows the agent to download and parse raw SEC HTML filings (10-K/10-Q) of Business Development Companies (BDCs) to extract their Schedule of Investments. 

## When to use
Use when a user asks to see which loans or assets inside a BDC's portfolio (like FSK, ARCC, MAIN, OBDC, OTF) have deteriorated the most year-over-year based on Fair Value to Face Value (or Amortized Cost) ratios.

## Core Logic & Filtering
- **Two-Period Filing Resolution**: For requested periods (`periodA`, `periodB`), automatically resolves each period to the corresponding SEC filing (10-K or 10-Q).
- **Significant Assets Only**: Automatically fetches the BDC's Shareholder Equity and filters out any loans where the Face Value is less than **0.2%** of the total Shareholder Equity to remove long-tail noise.
- **Clean Aggregation**: Excludes summary rows (e.g., "Total Asset Based Finance", "Net Asset Based Finance", Subtotals) and aggregates loan tranches by base company.
- **Deterioration Ranking**: Sorts the portfolio descending by the largest drop in the `Fair Value / Face Value` ratio from periodB to periodA. Extracts the **top 20 worst deteriorating assets** (where the ratio drop > 0), and excludes names with `periodA Fair/Face > 100%` from the final ranking list.

## Usage
Use the bash command:
```bash
bash ~/.openclaw/workspace/skills/bdc-analyst/run.sh --ticker [TICKER]
```
For example: `bash ~/.openclaw/workspace/skills/bdc-analyst/run.sh --ticker OTF`

To compare specific periods (recommended):
```bash
bash ~/.openclaw/workspace/skills/bdc-analyst/run.sh --ticker [TICKER] --periodA 25Q4 --periodB 24Q4
```
- `periodA` / `periodB` format: `YYQn` (e.g., `25Q4`, `25Q3`, `24Q2`).
- The skill auto-selects the filing containing each target period (10-K or 10-Q, depending on issuer timing).
- From each selected filing, extraction uses the latest period block in that filing for the target year/quarter context.

## Mandatory Post-Processing Contract (Direct Output)
After running the script, the assistant MUST post-process the result and output the final table directly to the user with enriched `公司主要业务的一句话简介`.

This enrichment MUST use the assistant's own model knowledge in the current session (no external LLM API calls).
If confidence is low for a company, use cautious wording rather than generic filler.

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
The Python script only outputs a baseline placeholder for this column.

**The assistant must enrich this column using its own model knowledge in the final response** (not by calling another external AI API).

Rules:
- Do not keep generic labels when you can infer the underlying operating business.
- If the name is a PE shell (Bidco/Holdco/Intermediate), infer and describe the likely operating company/business where possible.
- Output one concise Chinese sentence focusing on products/services and target customers.
- If confidence is low, clearly use a cautious generic wording.