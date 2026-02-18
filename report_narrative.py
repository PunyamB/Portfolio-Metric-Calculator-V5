import re
import requests


def _call_gemini(prompt, api_key, max_retries=3):
    """Call Gemini API with retry logic for rate limits."""
    import time
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4000}
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else 0
            if status == 429 and attempt < max_retries - 1:
                wait = (attempt + 1) * 15
                time.sleep(wait)
                continue
            elif status == 429:
                return "[Narrative generation failed: Gemini rate limit exceeded. Wait a few minutes and try again.]"
            elif status == 403:
                return "[Narrative generation failed: API key invalid or disabled. Check your key in Streamlit secrets.]"
            elif status == 404:
                return "[Narrative generation failed: Model not found. The Gemini model may have been updated.]"
            return f"[Narrative generation failed: HTTP {status}. Check your API key in Streamlit secrets.]"
        except Exception as e:
            err_msg = str(e)
            if api_key and api_key in err_msg:
                err_msg = err_msg.replace(api_key, "***")
            return f"[Narrative generation failed: {err_msg}]"


def _parse_sections(response):
    """Parse Gemini response into 4 sections using [SECTION_N] markers."""
    sections = {
        "executive_summary": "",
        "performance_attribution": "",
        "risk_analysis": "",
        "outlook": "",
    }

    if response.startswith("[Narrative generation failed"):
        for key in sections:
            sections[key] = response
        return sections

    # primary: split on [SECTION_N] markers
    marker_pattern = r'\[SECTION_(\d)\]'
    parts = re.split(marker_pattern, response)

    section_map = {"1": "executive_summary", "2": "performance_attribution",
                   "3": "risk_analysis", "4": "outlook"}

    marker_found = False
    for i, part in enumerate(parts):
        if part in section_map and i + 1 < len(parts):
            key = section_map[part]
            text = parts[i + 1].strip()
            # clean markdown headers and collapse whitespace
            text = re.sub(r'^#+\s*.*$', '', text, flags=re.MULTILINE).strip()
            text = re.sub(r'\*\*', '', text)  # remove bold markers
            text = re.sub(r'\n\s*\n', ' ', text)
            text = re.sub(r'\n', ' ', text)
            text = re.sub(r'\s{2,}', ' ', text)
            sections[key] = text.strip()
            marker_found = True

    if marker_found and any(v for v in sections.values()):
        return sections

    # fallback: header-based parsing
    current_section = None
    for line in response.split("\n"):
        lower = line.lower().strip().lstrip("#").strip()

        if any(kw in lower for kw in ["executive summary", "1. executive", "1)"]):
            current_section = "executive_summary"
            continue
        elif any(kw in lower for kw in ["performance attribution", "2. performance", "2)", "attribution"]):
            current_section = "performance_attribution"
            continue
        elif any(kw in lower for kw in ["risk profile", "risk analysis", "3. risk", "3)"]):
            current_section = "risk_analysis"
            continue
        elif any(kw in lower for kw in ["outlook", "recommendation", "4. outlook", "4)", "forward"]):
            current_section = "outlook"
            continue

        if current_section and line.strip():
            cleaned = line.strip().lstrip("#").strip()
            cleaned = re.sub(r'\*\*', '', cleaned)
            if cleaned and not cleaned.startswith("[SECTION"):
                sections[current_section] += cleaned + " "

    for key in sections:
        sections[key] = sections[key].strip()

    # last resort: split paragraphs
    if all(v == "" for v in sections.values()):
        clean = re.sub(r'^#+\s*.*$', '', response, flags=re.MULTILINE).strip()
        clean = re.sub(r'\*\*', '', clean)
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', clean) if p.strip()]
        if len(paragraphs) >= 4:
            sections["executive_summary"] = paragraphs[0].replace('\n', ' ')
            sections["performance_attribution"] = paragraphs[1].replace('\n', ' ')
            sections["risk_analysis"] = paragraphs[2].replace('\n', ' ')
            sections["outlook"] = " ".join(p.replace('\n', ' ') for p in paragraphs[3:])
        else:
            sections["executive_summary"] = clean[:2000].replace('\n', ' ')

    return sections


def generate_report_narrative(report_data, api_key):
    """Generate narrative sections for any date range report using Gemini."""
    d = report_data

    gainers_text = "\n".join([
        f"- {g['ticker']} ({g['company']}): {g['return_pct']:+.2f}%, ${g['pnl']:+,.2f} ({g['note']})"
        for g in d.get("top_gainers", [])
    ]) or "None"

    losers_text = "\n".join([
        f"- {l['ticker']} ({l['company']}): {l['return_pct']:+.2f}%, ${l['pnl']:+,.2f} ({l['note']})"
        for l in d.get("top_losers", [])
    ]) or "None"

    metrics_text = ""
    for key in ["Sharpe Ratio", "Sortino Ratio", "Beta", "Alpha (Annual)", "VaR (95%)", "Max Drawdown", "Std Dev (Annual)"]:
        start_val = d.get("start_metrics", {}).get(key, "N/A")
        end_val = d.get("end_metrics", {}).get(key, "N/A")
        delta = d.get("metric_deltas", {}).get(key, "N/A")
        metrics_text += f"- {key}: {start_val} -> {end_val} (delta {delta})\n"

    factor_text = ""
    fe = d.get("factor_exposures", {})
    for f_name in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
        if f_name in fe:
            factor_text += f"- {f_name}: {fe[f_name].get('exposure', 'N/A')} (t={fe[f_name].get('t_stat', 'N/A')})\n"

    all_stocks = "\n".join([
        f"- {a['ticker']}: ${a['pnl']:+,.2f} ({a['return_pct']:+.2f}%, {a['note']})"
        for a in d.get("attribution", [])
    ])

    prompt = f"""You are a portfolio analyst writing a professional equity research report for a paper trading portfolio (UConn MSFERM FNCE 5322). Write in first person as the portfolio manager.

REPORT PERIOD: {d['period_start']} to {d['period_end']} ({d['period_days']} days, {d['report_type']} report)

PORTFOLIO SUMMARY:
- Start Value: ${d['start_value']:,.2f} ({d['start_holdings_count']} positions)
- End Value: ${d['end_value']:,.2f} ({d['end_holdings_count']} positions)
- Value Change: ${d['value_change']:+,.2f} ({d['value_change_pct']:+.2f}%)
- S&P 500 return over same period: {d['benchmark_return']:+.2f}%
- Trades during period: {d['mid_period_trades']}
- Total Realised P&L (all time): ${d.get('total_realised_pnl', 0):+,.2f}

TOP GAINERS:
{gainers_text}

TOP LOSERS:
{losers_text}

ALL POSITION ATTRIBUTION:
{all_stocks}

METRICS (Start -> End):
{metrics_text}

FACTOR EXPOSURES (end of period):
{factor_text}

CRITICAL: You MUST write all 4 sections below. Start each section with its exact marker tag on its own line. Write 3-5 sentences of flowing prose per section. Use specific dollar amounts, percentages, and metric values. No bullet points. No markdown formatting.

[SECTION_1]
Executive Summary — overall performance, benchmark comparison, primary return driver.

[SECTION_2]
Performance Attribution — which positions drove gains/losses, reference actual P&L dollars and percentages, mention market events or earnings that caused moves.

[SECTION_3]
Risk Profile Analysis — how risk metrics changed (Sharpe, Beta, Alpha, VaR, Max Drawdown, Std Dev), what the changes mean for portfolio positioning.

[SECTION_4]
Outlook & Recommendations — what to watch next, rebalancing signals, key risks to monitor, reference specific positions or sectors."""

    response = _call_gemini(prompt, api_key)
    return _parse_sections(response)
