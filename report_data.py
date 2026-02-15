import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from portfolio import load_stocktrak_csv, STARTING_CAPITAL
from metrics import (
    fetch_price_history, get_risk_free_rate, compute_portfolio_returns,
    calculate_sharpe, calculate_sortino, calculate_beta_alpha, calculate_var,
    calculate_cvar, calculate_max_drawdown, calculate_std_dev_annual,
    calculate_cagr, calculate_factor_exposures, calculate_sharpe_daily,
    calculate_alpha_daily,
)

INCEPTION_DATE = "2026-01-26"


def _get_last_friday(from_date=None):
    """Get the most recent Friday on or before from_date."""
    if from_date is None:
        from_date = datetime.today()
    if isinstance(from_date, str):
        from_date = pd.to_datetime(from_date)
    weekday = from_date.weekday()
    if weekday >= 4:  # Friday=4, Sat=5, Sun=6
        offset = weekday - 4
    else:
        offset = weekday + 3
    return (from_date - timedelta(days=offset)).date()


def _get_prior_friday(friday_date):
    """Get the Friday before a given Friday."""
    return friday_date - timedelta(days=7)


def reconstruct_holdings_on_date(target_date):
    """Replay CSV trades up to target_date, return holdings dict.
    Returns: {ticker: {shares, avg_price, cost_total, company_name}}
    """
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

    # filter out zero positions
    return {k: v for k, v in positions.items() if abs(v["shares"]) > 0.001}


def _get_prices_on_date(tickers, target_date):
    """Get closing prices for tickers on a specific date. Falls back to nearest prior trading day."""
    target = pd.to_datetime(target_date)
    start = target - timedelta(days=5)
    prices = {}
    try:
        data = yf.download(tickers, start=start, end=target + timedelta(days=1),
                           auto_adjust=True, progress=False, group_by="ticker")
        for t in tickers:
            try:
                if isinstance(data.columns, pd.MultiIndex):
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
    """Get all trades that occurred between start_date (exclusive) and end_date (inclusive)."""
    trades = load_stocktrak_csv()
    if trades.empty:
        return pd.DataFrame()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    mask = (trades["CreateDate"] > start) & (trades["CreateDate"] <= end)
    return trades[mask].copy()


def compute_stock_attribution(prior_holdings, current_holdings, prior_prices,
                               current_prices, mid_week_trades):
    """Compute per-stock P&L attribution handling mid-week trades.

    Logic:
    - Held both weeks → (current_price - prior_price) × shares
    - Bought mid-week → (current_price - buy_price_from_csv) × shares_bought
    - Sold mid-week → (sell_price_from_csv - prior_price) × shares_sold
    - Bought & sold same week → (sell_price - buy_price) from CSV
    """
    attribution = []

    # all tickers involved
    all_tickers = set(list(prior_holdings.keys()) + list(current_holdings.keys()))

    # index mid-week trades by symbol
    mid_trades_by_symbol = {}
    if not mid_week_trades.empty:
        for _, row in mid_week_trades.iterrows():
            sym = str(row["Symbol"]).upper().strip()
            if sym not in mid_trades_by_symbol:
                mid_trades_by_symbol[sym] = []
            mid_trades_by_symbol[sym].append(row)

    for ticker in all_tickers:
        prior = prior_holdings.get(ticker)
        current = current_holdings.get(ticker)
        prior_price = prior_prices.get(ticker, 0)
        current_price = current_prices.get(ticker, 0)
        mid_trades = mid_trades_by_symbol.get(ticker, [])
        company = ""
        if current:
            company = current.get("company_name", ticker)
        elif prior:
            company = prior.get("company_name", ticker)

        pnl = 0.0
        note = ""

        if prior and current and not mid_trades:
            # held both weeks, no trades
            shares = prior["shares"]
            pnl = (current_price - prior_price) * shares
            note = "Held"
        elif not prior and current and mid_trades:
            # new position bought mid-week
            for t in mid_trades:
                if t["TransactionType"] == "Buy":
                    pnl += (current_price - float(t["Price"])) * float(t["Quantity"])
                elif t["TransactionType"] == "Short":
                    pnl += (float(t["Price"]) - current_price) * float(t["Quantity"])
            note = "New position"
        elif prior and not current and mid_trades:
            # fully closed mid-week
            for t in mid_trades:
                if t["TransactionType"] == "Sell":
                    pnl += (float(t["Price"]) - prior_price) * float(t["Quantity"])
                elif t["TransactionType"] == "Buy" and prior["shares"] < 0:
                    pnl += (prior_price - float(t["Price"])) * float(t["Quantity"])
            note = "Closed"
        elif prior and current and mid_trades:
            # held + additional trades mid-week
            # base P&L on shares held from before
            prior_shares = prior["shares"]
            pnl = (current_price - prior_price) * prior_shares
            # add P&L from mid-week trades
            for t in mid_trades:
                if t["TransactionType"] == "Buy":
                    pnl += (current_price - float(t["Price"])) * float(t["Quantity"])
                elif t["TransactionType"] == "Sell":
                    pnl += (float(t["Price"]) - prior_price) * float(t["Quantity"])
                elif t["TransactionType"] == "Short":
                    pnl += (float(t["Price"]) - current_price) * float(t["Quantity"])
            note = "Held + traded"
        else:
            # edge case: position exists but no price data
            pnl = 0
            note = "No data"

        # weekly return %
        if prior and prior_price > 0:
            weekly_return_pct = ((current_price - prior_price) / prior_price * 100) if current_price else 0
        elif mid_trades:
            buy_prices = [float(t["Price"]) for t in mid_trades if t["TransactionType"] in ["Buy", "Short"]]
            avg_buy = np.mean(buy_prices) if buy_prices else 0
            weekly_return_pct = ((current_price - avg_buy) / avg_buy * 100) if avg_buy > 0 and current_price else 0
        else:
            weekly_return_pct = 0

        attribution.append({
            "ticker": ticker,
            "company": company,
            "pnl": round(pnl, 2),
            "weekly_return_pct": round(weekly_return_pct, 2),
            "prior_price": round(prior_price, 2) if prior_price else None,
            "current_price": round(current_price, 2) if current_price else None,
            "shares": current["shares"] if current else (prior["shares"] if prior else 0),
            "note": note,
        })

    # sort by P&L
    attribution.sort(key=lambda x: x["pnl"], reverse=True)
    return attribution


def _compute_metrics_for_holdings(holdings, prices_df, rf):
    """Run metrics on a holdings dict using a prices DataFrame."""
    tickers = list(holdings.keys())
    available = [t for t in tickers if t in prices_df.columns]
    if not available:
        return {}

    # build holdings df for metrics functions
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


def compute_weekly_report_data(week_end_date=None):
    """Compute all data needed for a weekly report.

    Args:
        week_end_date: The Friday to report on. Defaults to most recent Friday.

    Returns dict with: prior/current holdings, prices, metrics, attribution, deltas.
    """
    current_friday = _get_last_friday(week_end_date)
    prior_friday = _get_prior_friday(current_friday)

    # reconstruct holdings
    prior_holdings = reconstruct_holdings_on_date(prior_friday)
    current_holdings = reconstruct_holdings_on_date(current_friday)

    # get all tickers
    all_tickers = list(set(list(prior_holdings.keys()) + list(current_holdings.keys())))
    if not all_tickers:
        return {"error": "No holdings found for these dates."}

    # get prices
    prior_prices = _get_prices_on_date(all_tickers, prior_friday)
    current_prices = _get_prices_on_date(all_tickers, current_friday)

    # portfolio values
    prior_value = sum(prior_prices.get(t, 0) * h["shares"] for t, h in prior_holdings.items())
    current_value = sum(current_prices.get(t, 0) * h["shares"] for t, h in current_holdings.items())

    # mid-week trades
    mid_week_trades = _get_trades_in_range(prior_friday, current_friday)

    # attribution
    attribution = compute_stock_attribution(
        prior_holdings, current_holdings, prior_prices, current_prices, mid_week_trades)

    # top movers
    top_gainers = [a for a in attribution if a["pnl"] > 0][:3]
    top_losers = [a for a in attribution if a["pnl"] < 0]
    top_losers = list(reversed(top_losers[:3]))

    # metrics (using 3yr lookback)
    rf = get_risk_free_rate()
    prices_df = fetch_price_history(all_tickers, years=3)

    prior_metrics = _compute_metrics_for_holdings(prior_holdings, prices_df, rf)
    current_metrics = _compute_metrics_for_holdings(current_holdings, prices_df, rf)

    # metric deltas
    metric_deltas = {}
    for key in current_metrics:
        cv = current_metrics.get(key, 0)
        pv = prior_metrics.get(key, 0)
        metric_deltas[key] = round(cv - pv, 6)

    # factor exposures
    hdf_current = pd.DataFrame([
        {"ticker": t, "shares": h["shares"], "avg_buy_price": 0, "date_added": ""}
        for t, h in current_holdings.items() if t in prices_df.columns
    ])
    port_ret = compute_portfolio_returns(prices_df, hdf_current) if not hdf_current.empty else pd.Series()
    factor_exposures = calculate_factor_exposures(port_ret) if not port_ret.empty and len(port_ret) > 30 else {}

    # benchmark return for the week
    bench_prices = _get_prices_on_date(["^GSPC"], prior_friday)
    bench_prices_cur = _get_prices_on_date(["^GSPC"], current_friday)
    bench_prior = bench_prices.get("^GSPC", 0)
    bench_current = bench_prices_cur.get("^GSPC", 0)
    bench_weekly_return = ((bench_current - bench_prior) / bench_prior * 100) if bench_prior > 0 else 0

    return {
        "report_type": "weekly",
        "period_start": str(prior_friday),
        "period_end": str(current_friday),
        "prior_value": round(prior_value, 2),
        "current_value": round(current_value, 2),
        "value_change": round(current_value - prior_value, 2),
        "value_change_pct": round((current_value - prior_value) / prior_value * 100, 2) if prior_value else 0,
        "prior_holdings_count": len(prior_holdings),
        "current_holdings_count": len(current_holdings),
        "mid_week_trades": len(mid_week_trades),
        "attribution": attribution,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "prior_metrics": prior_metrics,
        "current_metrics": current_metrics,
        "metric_deltas": metric_deltas,
        "factor_exposures": factor_exposures,
        "benchmark_weekly_return": round(bench_weekly_return, 2),
        "risk_free_rate": rf,
    }


def compute_inception_report_data():
    """Compute all data from inception (Jan 26, 2026) to current.
    Same structure as weekly but with inception as the start date.
    """
    inception = pd.to_datetime(INCEPTION_DATE).date()
    current_friday = _get_last_friday()

    # inception holdings = what was held after first day of trading
    inception_holdings = reconstruct_holdings_on_date(inception)
    current_holdings = reconstruct_holdings_on_date(current_friday)

    all_tickers = list(set(list(inception_holdings.keys()) + list(current_holdings.keys())))
    if not all_tickers:
        return {"error": "No holdings found."}

    inception_prices = _get_prices_on_date(all_tickers, inception)
    current_prices = _get_prices_on_date(all_tickers, current_friday)

    inception_value = sum(inception_prices.get(t, 0) * h["shares"] for t, h in inception_holdings.items())
    current_value = sum(current_prices.get(t, 0) * h["shares"] for t, h in current_holdings.items())

    # all trades since inception
    all_trades = _get_trades_in_range(inception - timedelta(days=1), current_friday)

    # attribution — for inception, compare buy price from CSV to current price
    attribution = []
    trades_df = load_stocktrak_csv()

    for ticker, holding in current_holdings.items():
        company = holding.get("company_name", ticker)
        current_price = current_prices.get(ticker, 0)
        shares = holding["shares"]

        # get avg cost from trade history
        if shares > 0 and holding["buy_shares"] > 0:
            avg_cost = holding["cost_total"] / holding["buy_shares"]
        elif shares < 0 and holding["short_shares"] > 0:
            avg_cost = holding["short_cost_total"] / holding["short_shares"]
        else:
            avg_cost = 0

        if shares > 0:
            pnl = (current_price - avg_cost) * shares
        else:
            pnl = (avg_cost - current_price) * abs(shares)

        return_pct = ((current_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0
        if shares < 0:
            return_pct = ((avg_cost - current_price) / avg_cost * 100) if avg_cost > 0 else 0

        attribution.append({
            "ticker": ticker,
            "company": company,
            "pnl": round(pnl, 2),
            "weekly_return_pct": round(return_pct, 2),  # reused field name for consistency
            "prior_price": round(avg_cost, 2),
            "current_price": round(current_price, 2),
            "shares": shares,
            "note": "Since inception",
        })

    # also add closed positions P&L
    from portfolio import compute_realised_pnl
    pnl_df, total_realised = compute_realised_pnl()
    if not pnl_df.empty:
        for _, row in pnl_df.iterrows():
            # check if already in attribution (still held)
            existing = [a for a in attribution if a["ticker"] == row["Symbol"]]
            if not existing:
                attribution.append({
                    "ticker": row["Symbol"],
                    "company": row["Symbol"],
                    "pnl": round(row["RealisedPnL"], 2),
                    "weekly_return_pct": 0,
                    "prior_price": round(row.get("AvgCostAtSell", 0), 2),
                    "current_price": round(row.get("SellPrice", 0), 2),
                    "shares": 0,
                    "note": "Closed",
                })

    attribution.sort(key=lambda x: x["pnl"], reverse=True)
    top_gainers = [a for a in attribution if a["pnl"] > 0][:3]
    top_losers = [a for a in attribution if a["pnl"] < 0]
    top_losers = list(reversed(top_losers[:3]))

    # metrics
    rf = get_risk_free_rate()
    prices_df = fetch_price_history(all_tickers, years=3)
    current_metrics = _compute_metrics_for_holdings(current_holdings, prices_df, rf)

    # factor exposures
    hdf = pd.DataFrame([
        {"ticker": t, "shares": h["shares"], "avg_buy_price": 0, "date_added": ""}
        for t, h in current_holdings.items() if t in prices_df.columns
    ])
    port_ret = compute_portfolio_returns(prices_df, hdf) if not hdf.empty else pd.Series()
    factor_exposures = calculate_factor_exposures(port_ret) if not port_ret.empty and len(port_ret) > 30 else {}

    # benchmark return since inception
    bench_inception = _get_prices_on_date(["^GSPC"], inception)
    bench_current = _get_prices_on_date(["^GSPC"], current_friday)
    bp = bench_inception.get("^GSPC", 0)
    bc = bench_current.get("^GSPC", 0)
    bench_return = ((bc - bp) / bp * 100) if bp > 0 else 0

    # total P&L (unrealised + realised)
    unrealised = sum(a["pnl"] for a in attribution if a["note"] != "Closed")
    total_pnl = unrealised + total_realised

    return {
        "report_type": "inception",
        "period_start": INCEPTION_DATE,
        "period_end": str(current_friday),
        "inception_value": round(inception_value, 2),
        "current_value": round(current_value, 2),
        "value_change": round(current_value - inception_value, 2),
        "value_change_pct": round((current_value - inception_value) / inception_value * 100, 2) if inception_value else 0,
        "total_pnl": round(total_pnl, 2),
        "unrealised_pnl": round(unrealised, 2),
        "realised_pnl": round(total_realised, 2),
        "current_holdings_count": len(current_holdings),
        "total_trades": len(trades_df),
        "attribution": attribution,
        "top_gainers": top_gainers,
        "top_losers": top_losers,
        "current_metrics": current_metrics,
        "factor_exposures": factor_exposures,
        "benchmark_return": round(bench_return, 2),
        "risk_free_rate": rf,
    }
