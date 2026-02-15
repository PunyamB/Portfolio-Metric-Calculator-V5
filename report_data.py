import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from portfolio import load_stocktrak_csv, compute_realised_pnl, STARTING_CAPITAL
from metrics import (
    fetch_price_history, get_risk_free_rate, compute_portfolio_returns,
    calculate_sharpe, calculate_sortino, calculate_beta_alpha, calculate_var,
    calculate_cvar, calculate_max_drawdown, calculate_std_dev_annual,
    calculate_cagr, calculate_factor_exposures, calculate_sharpe_daily,
    calculate_alpha_daily,
)

INCEPTION_DATE = "2026-01-26"


def _to_date(d):
    """Convert any date-like to datetime.date."""
    if d is None:
        return datetime.today().date()
    if isinstance(d, str):
        return pd.to_datetime(d).date()
    if hasattr(d, 'date'):
        return d.date()
    return d


def reconstruct_holdings_on_date(target_date):
    """Replay CSV trades up to target_date, return holdings dict."""
    trades = load_stocktrak_csv()
    if trades.empty:
        return {}

    target = pd.to_datetime(target_date)
    trades_before = trades[trades["CreateDate"] <= target].sort_values("CreateDate")

    positions = {}
    for _, row in trades_before.iterrows():
        symbol = str(row["Symbol"]).upper().strip()
        txn = row["TransactionType"]
        qty = float(row["Quantity"])
        price = float(row["Price"])
        company = row.get("CompanyName", symbol)

        if symbol not in positions:
            positions[symbol] = {"shares": 0.0, "cost_total": 0.0, "buy_shares": 0.0,
                                 "short_cost_total": 0.0, "short_shares": 0.0,
                                 "company_name": company}
        pos = positions[symbol]

        if txn == "Buy":
            if pos["shares"] < 0:
                cover_qty = min(qty, abs(pos["shares"]))
                pos["shares"] += cover_qty
                if pos["short_shares"] > 0:
                    pos["short_cost_total"] -= (pos["short_cost_total"] / pos["short_shares"]) * cover_qty
                    pos["short_shares"] -= cover_qty
                remaining = qty - cover_qty
                if remaining > 0:
                    pos["shares"] += remaining
                    pos["cost_total"] += remaining * price
                    pos["buy_shares"] += remaining
            else:
                pos["shares"] += qty
                pos["cost_total"] += qty * price
                pos["buy_shares"] += qty
        elif txn == "Sell":
            if pos["shares"] > 0 and pos["buy_shares"] > 0:
                avg = pos["cost_total"] / pos["buy_shares"]
                sell_qty = min(qty, pos["shares"])
                pos["cost_total"] -= avg * sell_qty
                pos["buy_shares"] -= sell_qty
            pos["shares"] -= qty
        elif txn == "Short":
            pos["shares"] -= qty
            pos["short_cost_total"] += qty * price
            pos["short_shares"] += qty

    return {k: v for k, v in positions.items() if abs(v["shares"]) > 0.001}


def _get_prices_on_date(tickers, target_date):
    """Get closing prices for tickers on a specific date."""
    if not tickers:
        return {}
    target = pd.to_datetime(target_date)
    start = target - timedelta(days=7)
    prices = {}
    try:
        tklist = list(set(tickers))
        data = yf.download(tklist, start=start, end=target + timedelta(days=1),
                           auto_adjust=True, progress=False, group_by="ticker")
        for t in tklist:
            try:
                if len(tklist) == 1:
                    col = data["Close"].dropna()
                elif isinstance(data.columns, pd.MultiIndex):
                    col = data[(t, "Close")].dropna()
                else:
                    col = data["Close"].dropna()
                if not col.empty:
                    prices[t] = float(col.iloc[-1])
            except:
                pass
    except:
        pass
    return prices


def _get_trades_in_range(start_date, end_date):
    """Get all trades between start_date (exclusive) and end_date (inclusive)."""
    trades = load_stocktrak_csv()
    if trades.empty:
        return pd.DataFrame()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (trades["CreateDate"] > start) & (trades["CreateDate"] <= end)
    return trades[mask].copy()


def compute_stock_attribution(start_holdings, end_holdings, start_prices,
                               end_prices, mid_trades):
    """Compute per-stock P&L attribution handling mid-period trades."""
    attribution = []
    all_tickers = set(list(start_holdings.keys()) + list(end_holdings.keys()))

    mid_trades_by_symbol = {}
    if not mid_trades.empty:
        for _, row in mid_trades.iterrows():
            sym = str(row["Symbol"]).upper().strip()
            if sym not in mid_trades_by_symbol:
                mid_trades_by_symbol[sym] = []
            mid_trades_by_symbol[sym].append(row)

    for ticker in all_tickers:
        start_h = start_holdings.get(ticker)
        end_h = end_holdings.get(ticker)
        sp = start_prices.get(ticker, 0)
        ep = end_prices.get(ticker, 0)
        trades = mid_trades_by_symbol.get(ticker, [])
        company = ""
        if end_h:
            company = end_h.get("company_name", ticker)
        elif start_h:
            company = start_h.get("company_name", ticker)

        pnl = 0.0
        note = ""

        if start_h and end_h and not trades:
            pnl = (ep - sp) * start_h["shares"]
            note = "Held"
        elif not start_h and end_h and trades:
            for t in trades:
                if t["TransactionType"] == "Buy":
                    pnl += (ep - float(t["Price"])) * float(t["Quantity"])
                elif t["TransactionType"] == "Short":
                    pnl += (float(t["Price"]) - ep) * float(t["Quantity"])
            note = "New position"
        elif start_h and not end_h and trades:
            for t in trades:
                if t["TransactionType"] == "Sell":
                    pnl += (float(t["Price"]) - sp) * float(t["Quantity"])
                elif t["TransactionType"] == "Buy" and start_h["shares"] < 0:
                    pnl += (sp - float(t["Price"])) * float(t["Quantity"])
            note = "Closed"
        elif start_h and end_h and trades:
            pnl = (ep - sp) * start_h["shares"]
            for t in trades:
                if t["TransactionType"] == "Buy":
                    pnl += (ep - float(t["Price"])) * float(t["Quantity"])
                elif t["TransactionType"] == "Sell":
                    pnl += (float(t["Price"]) - sp) * float(t["Quantity"])
                elif t["TransactionType"] == "Short":
                    pnl += (float(t["Price"]) - ep) * float(t["Quantity"])
            note = "Held + traded"
        else:
            note = "No data"

        if start_h and sp > 0:
            return_pct = ((ep - sp) / abs(sp) * 100) if ep else 0
        elif trades:
            buy_prices = [float(t["Price"]) for t in trades if t["TransactionType"] in ["Buy", "Short"]]
            avg_buy = np.mean(buy_prices) if buy_prices else 0
            return_pct = ((ep - avg_buy) / abs(avg_buy) * 100) if avg_buy > 0 and ep else 0
        else:
            return_pct = 0

        shares = end_h["shares"] if end_h else (start_h["shares"] if start_h else 0)

        attribution.append({
            "ticker": ticker,
            "company": company,
            "pnl": round(pnl, 2),
            "return_pct": round(return_pct, 2),
            "start_price": round(sp, 2) if sp else None,
            "end_price": round(ep, 2) if ep else None,
            "shares": shares,
            "note": note,
        })

    attribution.sort(key=lambda x: x["pnl"], reverse=True)
    return attribution


def _compute_metrics_for_holdings(holdings, prices_df, rf):
    """Run metrics on a holdings dict using a prices DataFrame."""
    tickers = list(holdings.keys())
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return {}

    rows = [{"ticker": t, "shares": holdings[t]["shares"],
             "avg_buy_price": holdings[t].get("cost_total", 0) / max(holdings[t].get("buy_shares", 1), 0.001)
             if holdings[t]["shares"] > 0
             else holdings[t].get("short_cost_total", 0) / max(holdings[t].get("short_shares", 1), 0.001),
             "date_added": ""} for t in available]
    hdf = pd.DataFrame(rows)
    if hdf.empty:
        return {}

    port_ret = compute_portfolio_returns(prices_df, hdf)
    if port_ret.empty or len(port_ret) < 10:
        return {}

    bench_ret = prices_df["^GSPC"].pct_change().dropna() if "^GSPC" in prices_df.columns else pd.Series()
    beta, alpha = calculate_beta_alpha(port_ret, bench_ret) if not bench_ret.empty else (0, 0)
    alpha_daily = calculate_alpha_daily(port_ret, bench_ret) if not bench_ret.empty else 0
    max_dd = calculate_max_drawdown(port_ret)

    return {
        "Sharpe Ratio": round(calculate_sharpe(port_ret, rf), 4),
        "Sharpe (Daily)": round(calculate_sharpe_daily(port_ret, rf), 4),
        "Sortino Ratio": round(calculate_sortino(port_ret, rf), 4),
        "Beta": round(beta, 4),
        "Alpha (Annual)": round(alpha, 4),
        "Alpha (Daily)": round(alpha_daily, 6),
        "VaR (95%)": round(calculate_var(port_ret), 6),
        "CVaR (95%)": round(calculate_cvar(port_ret), 6),
        "Max Drawdown": round(max_dd, 4),
        "Std Dev (Annual)": round(calculate_std_dev_annual(port_ret), 4),
    }


def compute_report_data(start_date, end_date):
    """Compute all data needed for a report between any two dates.

    Args:
        start_date: Start of the report period
        end_date: End of the report period

    Returns dict with: start/end holdings, prices, metrics, attribution, deltas.
    """
    start = _to_date(start_date)
    end = _to_date(end_date)

    # reconstruct holdings at both dates
    start_holdings = reconstruct_holdings_on_date(start)
    end_holdings = reconstruct_holdings_on_date(end)

    all_tickers = list(set(list(start_holdings.keys()) + list(end_holdings.keys())))
    if not all_tickers:
        return {"error": "No holdings found for these dates."}

    # get prices
    start_prices = _get_prices_on_date(all_tickers, start)
    end_prices = _get_prices_on_date(all_tickers, end)

    # portfolio values
    start_value = sum(start_prices.get(t, 0) * h["shares"] for t, h in start_holdings.items())
    end_value = sum(end_prices.get(t, 0) * h["shares"] for t, h in end_holdings.items())

    # mid-period trades
    mid_trades = _get_trades_in_range(start, end)

    # attribution
    attribution = compute_stock_attribution(
        start_holdings, end_holdings, start_prices, end_prices, mid_trades)

    top_gainers = [a for a in attribution if a["pnl"] > 0][:3]
    top_losers = [a for a in attribution if a["pnl"] < 0]
    top_losers = list(reversed(top_losers[:3]))

    # metrics
    rf = get_risk_free_rate()
    prices_df = fetch_price_history(all_tickers, years=3)

    start_metrics = _compute_metrics_for_holdings(start_holdings, prices_df, rf)
    end_metrics = _compute_metrics_for_holdings(end_holdings, prices_df, rf)

    metric_deltas = {}
    for key in end_metrics:
        cv = end_metrics.get(key, 0)
        pv = start_metrics.get(key, 0)
        metric_deltas[key] = round(cv - pv, 6)

    # factor exposures
    hdf_end = pd.DataFrame([
        {"ticker": t, "shares": h["shares"], "avg_buy_price": 0, "date_added": ""}
        for t, h in end_holdings.items() if t in prices_df.columns
    ])
    port_ret = compute_portfolio_returns(prices_df, hdf_end) if not hdf_end.empty else pd.Series()
    factor_exposures = calculate_factor_exposures(port_ret) if not port_ret.empty and len(port_ret) > 30 else {}

    # benchmark return
    bench_start = _get_prices_on_date(["^GSPC"], start)
    bench_end = _get_prices_on_date(["^GSPC"], end)
    bs = bench_start.get("^GSPC", 0)
    be = bench_end.get("^GSPC", 0)
    bench_return = ((be - bs) / bs * 100) if bs > 0 else 0

    # realised P&L for closed positions in the period
    pnl_df, total_realised = compute_realised_pnl()

    # determine report type label
    days = (end - start).days
    if days <= 10:
        period_label = "Weekly"
    elif days <= 35:
        period_label = "Monthly"
    else:
        period_label = "Period"

    return {
        "report_type": period_label,
        "period_start": str(start),
        "period_end": str(end),
        "period_days": days,
        "start_value": round(start_value, 2),
        "end_value": round(end_value, 2),
        "value_change": round(end_value - start_value, 2),
        "value_change_pct": round((end_value - start_value) / start_value * 100, 2) if start_value else 0,
        "start_holdings_count": len(start_holdings),
        "end_holdings_count": len(end_holdings),
        "mid_period_trades": len(mid_trades),
        "total_realised_pnl": round(total_realised, 2),
        "attribution": attribution,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "start_metrics": start_metrics,
        "end_metrics": end_metrics,
        "metric_deltas": metric_deltas,
        "factor_exposures": factor_exposures,
        "benchmark_return": round(bench_return, 2),
        "risk_free_rate": rf,
    }
