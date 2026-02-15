import requests


def _call_gemini(prompt, api_key):
    """Call Gemini API and return text response."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4000}
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "unknown"
        if status == 429:
            return "[Narrative generation failed: Gemini rate limit exceeded. Wait a minute and try again.]"
        return f"[Narrative generation failed: HTTP {status}. Check your API key in Streamlit secrets.]"
    except Exception as e:
        # sanitize: never include API key in error messages
        err_msg = str(e)
        if api_key and api_key in err_msg:
            err_msg = err_msg.replace(api_key, "***")
        return f"[Narrative generation failed: {err_msg}]"


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

    # full attribution list
    all_stocks = "\n".join([
        f"- {a['ticker']}: ${a['pnl']:+,.2f} ({a['return_pct']:+.2f}%, {a['note']})"
        for a in d.get("attribution", [])
    ])

    prompt = f"""You are a portfolio analyst writing a professional equity research report for a paper trading portfolio. Write in first person as the portfolio manager.

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

Write exactly 4 sections using these exact headers. Each section should be 2-4 sentences. Be specific with numbers.

## Executive Summary
Overall performance, comparison to benchmark, key driver of returns.

## Performance Attribution
Which positions drove the portfolio. Reference actual P&L numbers and percentages. Explain what market events may have caused these moves.

## Risk Profile Analysis
How portfolio risk changed over the period. Reference specific metric changes.

## Outlook
What to watch going forward, any rebalancing considerations, key risks.

Keep it professional, concise, data-driven. Do not use bullet points. Use actual numbers from the data."""

    response = _call_gemini(prompt, api_key)

    sections = {
        "executive_summary": "",
        "performance_attribution": "",
        "risk_analysis": "",
        "outlook": "",
    }

    # check if API call failed
    if response.startswith("[Narrative generation failed"):
        for key in sections:
            sections[key] = response
        return sections

    current_section = None
    for line in response.split("\n"):
        lower = line.lower().strip()
        if "executive summary" in lower:
            current_section = "executive_summary"
            continue
        elif "performance attribution" in lower:
            current_section = "performance_attribution"
            continue
        elif "risk profile" in lower or "risk analysis" in lower:
            current_section = "risk_analysis"
            continue
        elif "outlook" in lower or "recommendation" in lower or "forward" in lower:
            current_section = "outlook"
            continue

        if current_section and line.strip():
            # skip markdown header markers
            cleaned = line.strip().lstrip("#").strip()
            if cleaned:
                sections[current_section] += cleaned + " "

    for key in sections:
        sections[key] = sections[key].strip()

    # fallback: if all sections empty, put raw response in executive summary
    if all(v == "" for v in sections.values()):
        sections["executive_summary"] = response[:2000]

    return sections
