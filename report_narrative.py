import json
import requests


def _call_gemini(prompt, api_key):
    """Call Gemini API and return text response."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 4000}
    }
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[Narrative generation failed: {str(e)}]"


def generate_weekly_narrative(report_data, api_key):
    """Generate narrative sections for weekly report using Gemini."""

    # build structured prompt
    gainers_text = "\n".join([
        f"- {g['ticker']} ({g['company']}): {g['weekly_return_pct']:+.2f}%, ${g['pnl']:+,.2f}"
        for g in report_data.get("top_gainers", [])
    ])
    losers_text = "\n".join([
        f"- {l['ticker']} ({l['company']}): {l['weekly_return_pct']:+.2f}%, ${l['pnl']:+,.2f}"
        for l in report_data.get("top_losers", [])
    ])

    metrics_text = ""
    for key in ["Sharpe Ratio", "Sortino Ratio", "Beta", "Alpha (Annual)", "VaR (95%)", "Max Drawdown", "Std Dev (Annual)"]:
        prior = report_data.get("prior_metrics", {}).get(key, "N/A")
        current = report_data.get("current_metrics", {}).get(key, "N/A")
        delta = report_data.get("metric_deltas", {}).get(key, "N/A")
        metrics_text += f"- {key}: {prior} → {current} (Δ {delta})\n"

    factor_text = ""
    fe = report_data.get("factor_exposures", {})
    for f_name in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
        if f_name in fe:
            factor_text += f"- {f_name}: {fe[f_name]['exposure']} (t={fe[f_name]['t_stat']})\n"

    prompt = f"""You are a portfolio analyst writing a professional weekly equity research report for a paper trading portfolio. Write in first person as the portfolio manager.

PORTFOLIO DATA FOR THE WEEK {report_data['period_start']} to {report_data['period_end']}:

Portfolio Value: ${report_data['prior_value']:,.2f} → ${report_data['current_value']:,.2f} (change: ${report_data['value_change']:+,.2f}, {report_data['value_change_pct']:+.2f}%)
S&P 500 weekly return: {report_data['benchmark_weekly_return']:+.2f}%
Mid-week trades: {report_data['mid_week_trades']}
Positions: {report_data['current_holdings_count']}

TOP GAINERS:
{gainers_text}

TOP LOSERS:
{losers_text}

METRICS (Prior → Current):
{metrics_text}

FACTOR EXPOSURES:
{factor_text}

Write exactly 4 sections in this format (use these exact headers):

## Executive Summary
2-3 sentences on overall performance, what drove it, and how it compared to benchmark.

## Top Movers Commentary
1 paragraph explaining what drove the top gainers and losers. Reference actual market events, earnings, sector trends. Be specific.

## Risk Profile Analysis
1 paragraph on how portfolio risk changed this week. Reference the specific metric changes.

## Outlook & Recommendations
1 paragraph on what to watch next week, any rebalancing signals, and key risks.

Keep it professional, concise, and data-driven. Do not use bullet points. Use actual numbers from the data provided."""

    response = _call_gemini(prompt, api_key)

    # parse sections
    sections = {
        "executive_summary": "",
        "movers_commentary": "",
        "risk_analysis": "",
        "outlook": "",
    }

    current_section = None
    lines = response.split("\n")
    for line in lines:
        lower = line.lower().strip()
        if "executive summary" in lower:
            current_section = "executive_summary"
            continue
        elif "top movers" in lower:
            current_section = "movers_commentary"
            continue
        elif "risk profile" in lower:
            current_section = "risk_analysis"
            continue
        elif "outlook" in lower:
            current_section = "outlook"
            continue

        if current_section and line.strip():
            sections[current_section] += line.strip() + " "

    # clean up
    for key in sections:
        sections[key] = sections[key].strip()

    return sections


def generate_inception_narrative(report_data, api_key):
    """Generate narrative sections for inception report using Gemini."""

    gainers_text = "\n".join([
        f"- {g['ticker']} ({g['company']}): {g['weekly_return_pct']:+.2f}%, ${g['pnl']:+,.2f}"
        for g in report_data.get("top_gainers", [])
    ])
    losers_text = "\n".join([
        f"- {l['ticker']} ({l['company']}): {l['weekly_return_pct']:+.2f}%, ${l['pnl']:+,.2f}"
        for l in report_data.get("top_losers", [])
    ])

    metrics_text = ""
    for key, val in report_data.get("current_metrics", {}).items():
        metrics_text += f"- {key}: {val}\n"

    factor_text = ""
    fe = report_data.get("factor_exposures", {})
    for f_name in ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]:
        if f_name in fe:
            factor_text += f"- {f_name}: {fe[f_name]['exposure']} (t={fe[f_name]['t_stat']})\n"

    prompt = f"""You are a portfolio analyst writing a professional inception-to-date equity research report for a paper trading portfolio. Write in first person as the portfolio manager.

PORTFOLIO DATA FROM INCEPTION ({report_data['period_start']}) TO {report_data['period_end']}:

Portfolio Value: ${report_data.get('inception_value', 0):,.2f} → ${report_data['current_value']:,.2f}
Total P&L: ${report_data.get('total_pnl', 0):+,.2f} (Unrealised: ${report_data.get('unrealised_pnl', 0):+,.2f}, Realised: ${report_data.get('realised_pnl', 0):+,.2f})
S&P 500 return over same period: {report_data.get('benchmark_return', 0):+.2f}%
Total trades: {report_data.get('total_trades', 0)}
Current positions: {report_data['current_holdings_count']}

TOP PERFORMERS:
{gainers_text}

BOTTOM PERFORMERS:
{losers_text}

CURRENT METRICS:
{metrics_text}

FACTOR EXPOSURES:
{factor_text}

Write exactly 4 sections in this format (use these exact headers):

## Executive Summary
2-3 sentences summarising portfolio performance since inception. Compare to benchmark. Highlight the overall strategy.

## Performance Attribution
1 paragraph on what drove the portfolio's returns. Which positions were the biggest contributors and detractors. Reference actual numbers.

## Risk Profile Assessment
1 paragraph on the current risk characteristics of the portfolio. What the metrics tell us about how the portfolio is positioned.

## Forward Strategy
1 paragraph on the strategy going forward. What's working, what needs adjustment, and key considerations.

Keep it professional, concise, and data-driven. Do not use bullet points. Use actual numbers from the data provided."""

    response = _call_gemini(prompt, api_key)

    sections = {
        "executive_summary": "",
        "performance_attribution": "",
        "risk_assessment": "",
        "forward_strategy": "",
    }

    current_section = None
    lines = response.split("\n")
    for line in lines:
        lower = line.lower().strip()
        if "executive summary" in lower:
            current_section = "executive_summary"
            continue
        elif "performance attribution" in lower:
            current_section = "performance_attribution"
            continue
        elif "risk profile" in lower:
            current_section = "risk_assessment"
            continue
        elif "forward strategy" in lower:
            current_section = "forward_strategy"
            continue

        if current_section and line.strip():
            sections[current_section] += line.strip() + " "

    for key in sections:
        sections[key] = sections[key].strip()

    return sections
