---
name: bdc-analyst
description: SEC Financial Data Analyst specialized in parsing BDC (Business Development Company) Schedule of Investments to compare Amortized Cost vs Fair Value year-over-year.
---

# BDC Analyst Skill

This skill allows the agent to download and parse the raw SEC HTML 10-K filings of Business Development Companies (BDCs) to extract their Schedule of Investments. 

## When to use
Use when a user asks to see which loans or assets inside a BDC's portfolio (like FSK, ARCC, MAIN, OBDC) have deteriorated the most (i.e., Face Value / Amortized Cost vs Fair Value differences widening) year-over-year.

## Usage
Use the bash command:
```bash
bash ~/.openclaw/workspace/skills/bdc-analyst/run.sh --ticker [TICKER]
```
For example: `bash ~/.openclaw/workspace/skills/bdc-analyst/run.sh --ticker FSK`

The script will:
1. Initialize an isolated virtual environment using Python 3.12 (if available via Homebrew).
2. Fetch the 2024 and 2025 Year-End 10-K filings from the SEC.
3. Parse the massive HTML tables comprising the Schedule of Investments.
4. Clean and aggregate loan tranches by base company name.
5. Compute the Unrealized Depreciation difference between the two years and output the Top 15 worst performers.