import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from portfolio import (
    get_portfolio, get_trade_history, compute_realised_pnl, calculate_current_capital,
)
from metrics import (
    run_all_metrics, run_all_metrics_with_prices, fetch_price_history,
    get_risk_free_rate, compute_portfolio_returns, calculate_factor_exposures,
    calculate_ctr, run_markowitz_optimization, calculate_inception_metrics,
)
import metrics as m

st.set_page_config(page_title="Paper Trading Dashboard", page_icon="📈", layout="wide")
st.markdown("""<style>
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
    [data-testid="stSidebarContent"] { padding-top: 0.5rem; }
</style>""", unsafe_allow_html=True)

st.title("Paper Trading Portfolio Dashboard")
st.caption("US Equities & ETFs | Benchmark: S&P 500 | Data: Yahoo Finance EOD | Source: stocktrak_history.csv")

portfolio = get_portfolio()

# sidebar: info only
with st.sidebar:
    st.header("Dashboard Info")
    st.caption("Trade data loaded from `stocktrak_history.csv`.")
    st.caption("To update trades, replace the CSV file and refresh.")
    if not portfolio.empty:
        st.metric("Positions", len(portfolio))
        long_count = len(portfolio[portfolio["shares"] > 0])
        short_count = len(portfolio[portfolio["shares"] < 0])
        st.caption(f"Long: {long_count} | Short: {short_count}")

# --- tabs ---
tab_overview, tab_portfolio, tab_history, tab_simulator = st.tabs(["Overview", "Portfolio", "Trade History", "What-If Simulator"])


# === TAB 0: OVERVIEW ===
with tab_overview:
    st.subheader("Project Overview")
    st.caption("Paper Trading Portfolio Dashboard — Architecture, Methodology & Feature Guide")

    st.markdown("---")

    st.markdown("### Data Pipeline")
    st.markdown("""
The system reads `stocktrak_history.csv` as its single source of truth. On every page load, the raw CSV is parsed, cleaned, and transformed into a live portfolio. Transaction types like "Market - Buy", "Limit - Sell", "Short Proceeds", and "Buy to Cover" are all normalised into three standard actions: Buy, Sell, and Short. Dividend rows are excluded since Yahoo Finance's adjusted prices already account for dividend returns in our metric calculations. Prices and amounts are stripped of accounting notation (dollar signs, commas, parentheses) and quantities are converted to absolute values with direction inferred from the transaction type. Multiple lots of the same security are merged into weighted average positions.
""")

    st.markdown("### Portfolio Tab")
    st.markdown("""
The holdings table displays each position with its current market price (fetched live from Yahoo Finance), cost basis, unrealised P&L, and portfolio weight. Positions are tagged as Long or Short based on share direction. Summary cards show Portfolio Value, Cash Remaining, Total Capital, and a three-way P&L breakdown — unrealised, realised, and total. Realised P&L is calculated using the average cost method: when shares are sold, the gain or loss is computed against the weighted average purchase price of all lots for that security.
""")

    st.markdown("### Risk Metrics")
    st.markdown("""
Fourteen metrics are computed across the portfolio using historical daily returns and organised into logical groupings.

**Return & Growth** — CAGR, Max Drawdown, Calmar Ratio, and Turnover Ratio. These tell you how the portfolio has grown, how bad the worst decline was, and how actively it's being traded.

**Volatility & Distribution** — Daily and annualised standard deviation, variance, and skewness. These describe how much the portfolio moves and whether those movements are symmetric.

**Risk Measures** — Kurtosis (tail risk), Value at Risk at 95% confidence (the worst daily loss you'd expect 19 out of 20 days), Conditional VaR (the average loss on those bad days), and Beta (sensitivity to the S&P 500).

**Performance Ratios** — Sharpe (excess return per unit of total risk), Sortino (excess return per unit of downside risk only), and annualised Alpha (return above what Beta alone would predict).

**Daily Values** — Non-annualised Sharpe and Alpha for direct comparison with StockTrak's reported figures.

**Since Inception** — Sharpe and Alpha computed from the portfolio's start date of January 26, 2026, regardless of the lookback period selected. This provides a consistent performance anchor across all time horizons.

All metrics use the 13-week US Treasury Bill rate as the risk-free rate, fetched live. The benchmark is the S&P 500. Lookback periods of 1, 3, 5, 10, or 15 years determine how much historical price data is used.
""")

    st.markdown("### Fama-French 5-Factor Analysis")
    st.markdown("""
Beyond simple Beta, the dashboard runs a full regression against the Fama-French five-factor model. This decomposes portfolio returns into exposure to five systematic risk factors: Market (overall equity risk), Size (small vs large cap tilt), Value (cheap vs expensive stocks), Profitability (strong vs weak earnings), and Investment (conservative vs aggressive capital spending). Each factor shows its exposure coefficient, t-statistic, statistical significance, an optimal range for diversified portfolios, and a plain-English interpretation. The R-squared tells you what percentage of portfolio returns are explained by these five factors — anything left over is idiosyncratic risk specific to your stock picks.
""")

    st.markdown("### Contribution to Risk")
    st.markdown("""
This analysis answers the question "which positions are adding the most risk to my portfolio?" Using the covariance matrix of daily returns, it calculates each position's marginal contribution to total portfolio volatility. A position with high CTR relative to its weight is contributing disproportionate risk — it might be worth trimming. A position with low CTR relative to its weight is a diversifier.
""")

    st.markdown("### Trade History")
    st.markdown("""
A read-only view of all imported trades with filters by transaction type, symbol, and security type. Summary cards show total bought, total sold/shorted, net cash flow, and realised P&L. The tab exists for transparency and audit — since the CSV is the source of truth, edits happen in the file itself.
""")

    st.markdown("### What-If Simulator")
    st.markdown("""
The simulator lets you build hypothetical portfolios and compare their risk profiles against your current holdings without making any actual trades.

**Step 1** — Select which existing positions to keep by toggling checkboxes.

**Step 2** — Add hypothetical positions. Positive shares for new longs, negative shares for simulated sells or new shorts. When you simulate selling an existing position or opening a new short, the system calculates the cash proceeds at current market prices and displays them. These proceeds flow into Step 4's optimiser as additional available capital.

**Step 3** — Side-by-side metric comparison between your current portfolio and the simulated one. Every metric is shown with a delta and colour-coded — green for improvement, red for deterioration. The simulation uses the same price data and lookback period for both portfolios to ensure an apples-to-apples comparison. Simulated cash from hypothetical sells does not affect Step 3's metrics because in practice, uninvested cash earns the risk-free rate and has zero volatility — it would only dilute the metrics without adding information.
""")

    st.markdown("### Markowitz Portfolio Optimization")
    st.markdown("""
Step 4 uses mean-variance optimisation to find the mathematically optimal allocation across your selected tickers. The optimiser runs three scenarios:

- **Max Sharpe** finds the allocation with the highest risk-adjusted return.
- **Min Variance** finds the allocation with the lowest possible volatility.
- **Unconstrained** runs with no long/short caps to show what the optimiser would do if your constraints were removed.

Five parameters control the optimisation: **Total Capital** (auto-calculated from trade history plus any simulated proceeds, editable), **Max Long %** (the ceiling for total long exposure, default 73%), **Max Short %** (the ceiling for total short exposure, default 27%), **Max Per Position** (a dollar cap preventing any single position from dominating, default $90,000), and **Min Deploy %** (ensures at least this percentage of capital is allocated rather than sitting in cash, default 95%).

The long and short percentages are maximum caps, not targets — the optimiser can use less on either side if the math doesn't justify filling the allocation. This reflects how real portfolio managers operate: they set risk limits but deploy capital based on opportunity.

Each ticker is tagged as Long or Short with defaults inferred from current portfolio positions. The efficient frontier chart plots 10,000 randomly generated portfolios colour-coded by Sharpe ratio, with the three optimal points and your current portfolio marked.

A **Constraint Analysis** box appears after results — if the unconstrained optimal would prefer a different long/short split than your caps allow, it shows as a blue info box with the Sharpe gap. If your constraints aren't binding, it shows as a green success box. The metrics comparison table adds the unconstrained column so you can see exactly what relaxing your limits would gain or cost. Three allocation tables show the recommended share counts, dollar amounts, and weights for each scenario, with a CASH row when the optimiser doesn't fully deploy capital.
""")

    st.markdown("### Technical Architecture")
    st.markdown("""
The system is built with Python and Streamlit, using three modules: `portfolio.py` handles CSV parsing, position computation, P&L tracking, and capital calculation. `metrics.py` contains all financial calculations — individual metrics, factor regression, contribution to risk, and the Markowitz optimiser using SciPy's SLSQP solver. `app.py` is the Streamlit interface that ties everything together. Market data comes from Yahoo Finance via the yfinance library, factor data from Kenneth French's data library, and interactive charts from Plotly.
""")


# === TAB 1: PORTFOLIO ===
with tab_portfolio:
    st.subheader("Current Holdings")
    if portfolio.empty:
        st.info("No positions found. Make sure `stocktrak_history.csv` is in the app directory.")
    else:
        latest_prices = {}
        for t in portfolio["ticker"].tolist():
            try: latest_prices[t] = round(yf.Ticker(t).fast_info.last_price, 2)
            except: latest_prices[t] = None

        display_df = portfolio.copy()
        display_df["Current Price ($)"] = display_df["ticker"].map(latest_prices)
        display_df["Market Value ($)"] = (display_df["shares"] * display_df["Current Price ($)"]).round(2)
        display_df["Cost Basis ($)"] = (display_df["shares"].abs() * display_df["avg_buy_price"]).round(2)

        def calc_pnl(row):
            if row["shares"] >= 0: return row["Market Value ($)"] - row["Cost Basis ($)"]
            else: return row["Cost Basis ($)"] - abs(row["Market Value ($)"])

        display_df["Unrealised P&L ($)"] = display_df.apply(calc_pnl, axis=1).round(2)
        display_df["P&L %"] = (display_df["Unrealised P&L ($)"] / display_df["Cost Basis ($)"] * 100).round(2)
        total_market_value = display_df["Market Value ($)"].sum()
        abs_total = display_df["Market Value ($)"].abs().sum()
        display_df["Weight (%)"] = (display_df["Market Value ($)"] / abs_total * 100).round(2) if abs_total > 0 else 0
        display_df["Position"] = display_df["shares"].apply(lambda s: "Short" if s < 0 else "Long")
        display_df = display_df.rename(columns={"ticker": "Ticker", "shares": "Shares",
                                                 "avg_buy_price": "Avg Price ($)", "date_added": "First Trade"})

        # format prices with commas for display
        fmt_df = display_df[["Ticker", "Position", "Shares", "Avg Price ($)", "Current Price ($)",
                              "Market Value ($)", "Weight (%)", "Cost Basis ($)",
                              "Unrealised P&L ($)", "P&L %", "First Trade"]].copy()

        st.dataframe(fmt_df.style.format({
            "Avg Price ($)": "${:,.2f}",
            "Current Price ($)": "${:,.2f}",
            "Market Value ($)": "${:,.2f}",
            "Cost Basis ($)": "${:,.2f}",
            "Unrealised P&L ($)": "${:,.2f}",
            "Weight (%)": "{:.2f}",
            "P&L %": "{:.2f}",
        }), use_container_width=True, hide_index=True)

        total_cost = display_df["Cost Basis ($)"].sum()
        total_pnl = display_df["Unrealised P&L ($)"].sum()
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        _, total_realised = compute_realised_pnl()
        total_combined = total_pnl + total_realised

        cap_result = calculate_current_capital(total_market_value)
        if isinstance(cap_result, tuple):
            auto_capital, cash_remaining = cap_result
        else:
            auto_capital, cash_remaining = cap_result, 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${total_market_value:,.2f}")
        c2.metric("Cash Remaining", f"${cash_remaining:,.2f}")
        c3.metric("Total Capital", f"${auto_capital:,.2f}")
        c4.metric("Positions", len(portfolio))

        p1, p2, p3 = st.columns(3)
        p1.metric("Unrealised P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl_pct:.2f}%")
        realised_pct = (total_realised / total_cost * 100) if total_cost > 0 else 0
        p2.metric("Realised P&L", f"${total_realised:,.2f}", delta=f"{realised_pct:.2f}%")
        combined_pct = (total_combined / total_cost * 100) if total_cost > 0 else 0
        p3.metric("Total P&L", f"${total_combined:,.2f}", delta=f"{combined_pct:.2f}%")

        # --- sub-tabs ---
        st.divider()
        sub_metrics, sub_risk = st.tabs(["Metrics", "Risk Analysis"])

        with sub_metrics:
            col_lb, col_btn = st.columns([2, 1])
            with col_lb:
                lookback = st.selectbox("Lookback Period",
                    ["1 Year", "3 Years", "5 Years", "10 Years", "15 Years"], index=1, key="pf_lookback")
            with col_btn:
                st.write("")
                if st.button("Calculate Metrics", type="primary", use_container_width=True, key="calc_pf"):
                    with st.spinner("Fetching data and computing metrics..."):
                        m.LOOKBACK_YEARS = {"1 Year": 1, "3 Years": 3, "5 Years": 5,
                                            "10 Years": 10, "15 Years": 15}[lookback]
                        trade_hist = get_trade_history()
                        turnover_inputs = None
                        if not trade_hist.empty and abs(total_market_value) > 0:
                            buys_total = trade_hist.loc[trade_hist["TransactionType"] == "Buy", "Amount"].sum()
                            sells_total = trade_hist.loc[trade_hist["TransactionType"].isin(["Sell", "Short"]), "Amount"].sum()
                            turnover_inputs = {"buys": buys_total, "sells": sells_total,
                                               "avg_portfolio_value": abs(total_market_value)}
                        results = run_all_metrics(portfolio, turnover_inputs)
                        prices_ff = fetch_price_history(portfolio["ticker"].tolist())
                        port_ret = compute_portfolio_returns(prices_ff, portfolio)
                        factor_results = calculate_factor_exposures(port_ret)
                        rf = get_risk_free_rate()
                        inception_results = calculate_inception_metrics(portfolio, rf)
                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.session_state["portfolio_metrics"] = results
                        st.session_state["factor_exposures"] = factor_results
                        st.session_state["inception_metrics"] = inception_results

            if "portfolio_metrics" in st.session_state:
                r = st.session_state["portfolio_metrics"]
                inc = st.session_state.get("inception_metrics", {})

                # Row 1: Return & Growth
                mc1 = st.columns(4)
                for i, (label, value, tip) in enumerate([
                    ("CAGR", f"{r.get('CAGR', 0)*100:.2f}%", "Compound annual growth"),
                    ("Max Drawdown", f"{r.get('Max Drawdown', 0)*100:.2f}%", "Worst peak-to-trough"),
                    ("Calmar Ratio", r.get("Calmar Ratio", "N/A"), "CAGR / Max Drawdown"),
                    ("Turnover Ratio", r.get("Turnover Ratio", "N/A"), "From trade history"),
                ]):
                    with mc1[i]: st.metric(label=label, value=value, help=tip)

                # Row 2: Volatility & Distribution
                mc2 = st.columns(4)
                for i, (label, value, tip) in enumerate([
                    ("Variance (Daily)", r.get("Variance (Daily)", "N/A"), "Daily return variance"),
                    ("Std Dev (Daily)", f"{r.get('Std Dev (Daily)', 0)*100:.4f}%", "Daily volatility"),
                    ("Std Dev (Annual)", f"{r.get('Std Dev (Annual)', 0)*100:.2f}%", "Annualised volatility"),
                    ("Skewness", r.get("Skewness", "N/A"), "Negative = left tail, Positive = right tail"),
                ]):
                    with mc2[i]: st.metric(label=label, value=value, help=tip)

                # Row 3: Risk Measures
                mc3 = st.columns(4)
                for i, (label, value, tip) in enumerate([
                    ("Kurtosis", r.get("Kurtosis", "N/A"), ">0 = fat tails"),
                    ("VaR (95%)", f"{r.get('VaR (95%)', 0)*100:.2f}%", "Max daily loss at 95% confidence"),
                    ("CVaR (95%)", f"{r.get('CVaR (95%)', 0)*100:.2f}%", "Avg loss beyond VaR"),
                    ("Beta", r.get("Beta", "N/A"), "vs S&P 500"),
                ]):
                    with mc3[i]: st.metric(label=label, value=value, help=tip)

                # Row 4: Performance Ratios (Annualised)
                mc4 = st.columns(4)
                for i, (label, value, tip) in enumerate([
                    ("Sharpe Ratio", r.get("Sharpe Ratio", "N/A"), "Annualised excess return / volatility"),
                    ("Sortino Ratio", r.get("Sortino Ratio", "N/A"), "Annualised excess return / downside vol"),
                    ("Alpha (Annual)", f"{r.get('Alpha (Annual)', 0)*100:.2f}%", "Annualised excess return vs benchmark"),
                ]):
                    with mc4[i]: st.metric(label=label, value=value, help=tip)

                # Row 5: Daily Values
                mc5 = st.columns(4)
                for i, (label, value, tip) in enumerate([
                    ("Sharpe (Daily)", r.get("Sharpe (Daily)", "N/A"), "Non-annualised daily Sharpe"),
                    ("Alpha (Daily)", f"{r.get('Alpha (Daily)', 0)*100:.4f}%", "Non-annualised daily alpha"),
                ]):
                    with mc5[i]: st.metric(label=label, value=value, help=tip)

                # Row 6: Since Inception
                if inc:
                    st.caption("Since Inception (Jan 26, 2026)")
                    mc6 = st.columns(4)
                    for i, (label, value, tip) in enumerate([
                        ("Sharpe (Inception)", inc.get("Sharpe (Inception)", "N/A"), "Sharpe since Jan 26, 2026"),
                        ("Alpha (Inception)", f"{inc.get('Alpha (Inception)', 0)*100:.2f}%", "Alpha since Jan 26, 2026"),
                    ]):
                        with mc6[i]: st.metric(label=label, value=value, help=tip)

                st.caption(f"Risk-free rate: {r.get('Risk-Free Rate Used', 0)*100:.2f}% (13-week T-Bill) | Lookback: {lookback}")

                # factor exposures
                if "factor_exposures" in st.session_state:
                    fe = st.session_state["factor_exposures"]
                    with st.expander("Fama-French 5-Factor Exposures"):
                        if "error" in fe:
                            st.warning(fe["error"])
                        else:
                            st.caption(f"R-squared = {fe.get('R_squared', 'N/A')} | Alpha (annualised) = {fe.get('Alpha_annual', 'N/A')}")
                            factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
                            factor_labels = {"Mkt-RF": "Market", "SMB": "Size (Small-Big)",
                                             "HML": "Value (High-Low)", "RMW": "Profitability (Robust-Weak)",
                                             "CMA": "Investment (Conservative-Aggressive)"}
                            rows = []
                            for f in factor_names:
                                if f in fe:
                                    d = fe[f]
                                    rows.append({"Factor": factor_labels.get(f, f), "Exposure": d["exposure"],
                                                 "t-stat": d["t_stat"],
                                                 "Significant": "Yes" if d["significant"] else "No",
                                                 "Optimal Range": d.get("optimal_range", "-"),
                                                 "Interpretation": d.get("interpretation", "-")})
                            if rows:
                                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with sub_risk:
            if "portfolio_metrics" not in st.session_state:
                st.info("Click 'Calculate Metrics' in the Metrics tab first.")
            else:
                wr1, wr2 = st.columns(2)
                with wr1: st.markdown("**Weight Distribution**")
                with wr2:
                    rc_h1, rc_h2 = st.columns([2, 1])
                    with rc_h1: st.markdown("**Risk Contribution**")
                    with rc_h2: calc_ctr_btn = st.button("Calculate CTR", key="calc_ctr")

                if calc_ctr_btn:
                    with st.spinner("Computing..."):
                        prices_ctr = fetch_price_history(portfolio["ticker"].tolist())
                        ctr_df = calculate_ctr(prices_ctr, portfolio)
                    if ctr_df.empty: st.warning("Need at least 2 positions for CTR.")
                    else: st.session_state["ctr_data"] = ctr_df

                left_col, right_col = st.columns(2)
                with left_col:
                    st.bar_chart(display_df.set_index("Ticker")["Weight (%)"])
                with right_col:
                    if "ctr_data" in st.session_state:
                        ctr = st.session_state["ctr_data"]
                        st.bar_chart(ctr.set_index("Ticker")["CTR (%)"])
                        st.dataframe(ctr, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Click 'Calculate CTR' to see risk contributions.")


# === TAB 2: TRADE HISTORY (read-only) ===
with tab_history:
    st.subheader("Trade History")
    st.caption("Read-only. Data loaded from `stocktrak_history.csv`. To update, replace the file and refresh.")
    trade_history = get_trade_history()

    if trade_history.empty:
        st.info("No trades found. Make sure `stocktrak_history.csv` is in the app directory.")
    else:
        fc1, fc2, fc3 = st.columns(3)
        with fc1: filter_type = st.selectbox("Transaction Type", ["All", "Buy", "Sell", "Short"], key="th_filter_type")
        with fc2:
            all_symbols = sorted(trade_history["Symbol"].unique().tolist())
            filter_symbol = st.selectbox("Symbol", ["All"] + all_symbols, key="th_filter_symbol")
        with fc3: filter_sectype = st.selectbox("Security Type", ["All"] + sorted(trade_history["SecurityType"].dropna().unique().tolist()), key="th_filter_sectype")

        filtered = trade_history.copy()
        if filter_type != "All": filtered = filtered[filtered["TransactionType"] == filter_type]
        if filter_symbol != "All": filtered = filtered[filtered["Symbol"] == filter_symbol]
        if filter_sectype != "All": filtered = filtered[filtered["SecurityType"] == filter_sectype]

        total_buys = filtered.loc[filtered["TransactionType"] == "Buy", "Amount"].sum()
        total_sells = filtered.loc[filtered["TransactionType"].isin(["Sell", "Short"]), "Amount"].sum()
        pnl_df, total_realised = compute_realised_pnl()

        sm1, sm2, sm3, sm4, sm5 = st.columns(5)
        sm1.metric("Total Trades", len(filtered))
        sm2.metric("Total Bought", f"${total_buys:,.2f}")
        sm3.metric("Total Sold/Shorted", f"${total_sells:,.2f}")
        sm4.metric("Net Flow", f"${total_buys - total_sells:,.2f}")
        sm5.metric("Realised P&L", f"${total_realised:,.2f}")

        show_cols = ["CreateDate", "Symbol", "CompanyName", "TransactionType",
                     "Quantity", "Price", "Amount", "Exchange", "Currency", "SecurityType"]
        available_cols = [c for c in show_cols if c in filtered.columns]
        show_df = filtered[available_cols].sort_values("CreateDate", ascending=False).copy()

        # format date for display
        if "CreateDate" in show_df.columns:
            show_df["CreateDate"] = pd.to_datetime(show_df["CreateDate"], errors="coerce").dt.strftime("%m/%d/%Y")

        st.dataframe(show_df.style.format({
            "Price": "${:,.2f}",
            "Amount": "${:,.2f}",
        }), use_container_width=True, hide_index=True)


# === TAB 3: WHAT-IF SIMULATOR ===
with tab_simulator:
    st.subheader("What-If Portfolio Simulator")
    st.caption("Build a hypothetical portfolio and compare metrics. Nothing here is saved.")

    if portfolio.empty:
        st.info("Add at least one position first.")
    else:
        st.markdown("**Step 1 - Choose base holdings**")
        all_tickers = portfolio["ticker"].tolist()
        cols = st.columns(5)
        selected_base = []
        for i, ticker in enumerate(all_tickers):
            with cols[i % 5]:
                shares = portfolio.loc[portfolio["ticker"] == ticker, "shares"].iloc[0]
                label = f"{ticker} (Short)" if shares < 0 else ticker
                if st.checkbox(label, value=True, key=f"sim_base_{ticker}"):
                    selected_base.append(ticker)
        if not selected_base: st.warning("Select at least one holding.")

        st.divider()
        st.markdown("**Step 2 - Add hypothetical positions**")
        st.caption("Use negative shares to simulate short positions.")
        if "sim_hypothetical" not in st.session_state:
            st.session_state["sim_hypothetical"] = []

        with st.form("add_hypo_form"):
            hc1, hc2, hc3, hc4 = st.columns([2, 1, 1, 1])
            with hc1: hypo_ticker = st.text_input("Ticker", placeholder="e.g. MSFT")
            with hc2: hypo_shares = st.number_input("Shares", step=1, value=1, help="Negative = short")
            with hc3: hypo_price = st.number_input("Price ($)", min_value=0.01, step=0.25, format="%.2f", value=100.0)
            with hc4:
                st.write(""); st.write("")
                add_hypo = st.form_submit_button("Add", use_container_width=True)
            if add_hypo and hypo_ticker:
                hypo_ticker = hypo_ticker.upper().strip()
                if hypo_shares == 0: st.error("Shares cannot be zero.")
                else:
                    try:
                        _ = yf.Ticker(hypo_ticker).fast_info.last_price
                        existing = [h for h in st.session_state["sim_hypothetical"] if h["ticker"] == hypo_ticker]
                        if existing:
                            idx = st.session_state["sim_hypothetical"].index(existing[0])
                            old = existing[0]
                            ns = old["shares"] + hypo_shares
                            if ns == 0: st.session_state["sim_hypothetical"].pop(idx)
                            else:
                                tc = abs(old["shares"]) * old["avg_buy_price"] + abs(hypo_shares) * hypo_price
                                tq = abs(old["shares"]) + abs(hypo_shares)
                                st.session_state["sim_hypothetical"][idx] = {
                                    "ticker": hypo_ticker, "shares": ns,
                                    "avg_buy_price": round(tc/tq, 4), "date_added": old["date_added"]}
                        else:
                            st.session_state["sim_hypothetical"].append({
                                "ticker": hypo_ticker, "shares": hypo_shares,
                                "avg_buy_price": hypo_price, "date_added": "hypothetical"})
                        st.rerun()
                    except: st.error(f"Could not validate '{hypo_ticker}'.")

        if st.session_state["sim_hypothetical"]:
            st.markdown("**Hypothetical positions:**")
            hypo_df = pd.DataFrame(st.session_state["sim_hypothetical"])
            hd = hypo_df[["ticker", "shares", "avg_buy_price"]].copy()
            hd["Type"] = hd["shares"].apply(lambda s: "Short" if s < 0 else "Long")
            st.dataframe(hd.rename(columns={"ticker": "Ticker", "shares": "Shares", "avg_buy_price": "Price ($)"}).style.format({
                "Price ($)": "${:,.2f}",
            }), use_container_width=True, hide_index=True)

            # calculate simulated proceeds from sells/shorts
            sim_proceeds = 0.0
            for h in st.session_state["sim_hypothetical"]:
                if h["shares"] < 0:
                    try:
                        cur_price = yf.Ticker(h["ticker"]).fast_info.last_price
                    except:
                        cur_price = h["avg_buy_price"]
                    sim_proceeds += abs(h["shares"]) * cur_price
            if sim_proceeds > 0:
                st.caption(f"Simulated Proceeds: **${sim_proceeds:,.2f}** (from hypothetical sells/shorts)")
            st.session_state["sim_proceeds"] = sim_proceeds

            rc1, rc2 = st.columns([3, 1])
            with rc1:
                remove_hypo = st.selectbox("Remove", options=[""] + [h["ticker"] for h in st.session_state["sim_hypothetical"]], key="rm_hypo")
            with rc2:
                st.write(""); st.write("")
                if st.button("Remove", key="rm_btn") and remove_hypo:
                    st.session_state["sim_hypothetical"] = [h for h in st.session_state["sim_hypothetical"] if h["ticker"] != remove_hypo]
                    st.rerun()

        st.divider()
        st.markdown("**Step 3 - Run simulation**")
        s1, s2 = st.columns([2, 1])
        with s1: sim_lookback = st.selectbox("Lookback", ["1 Year", "3 Years", "5 Years", "10 Years", "15 Years"], index=1, key="sim_lookback")
        with s2:
            st.write("")
            run_sim = st.button("Run Simulation", type="primary", use_container_width=True, key="run_sim_btn")

        if run_sim:
            if not selected_base and not st.session_state["sim_hypothetical"]:
                st.error("Need at least one position.")
            else:
                base_rows = portfolio[portfolio["ticker"].isin(selected_base)].copy()
                hypo_rows = pd.DataFrame(st.session_state["sim_hypothetical"]) if st.session_state["sim_hypothetical"] else pd.DataFrame()
                sim_portfolio = pd.concat([base_rows, hypo_rows], ignore_index=True) if not hypo_rows.empty else base_rows.copy()
                with st.spinner("Running simulation..."):
                    m.LOOKBACK_YEARS = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10, "15 Years": 15}[sim_lookback]
                    all_sim_tickers = list(set(portfolio["ticker"].tolist() + sim_portfolio["ticker"].tolist()))
                    shared_prices = fetch_price_history(all_sim_tickers)
                    shared_rf = get_risk_free_rate()
                    current_metrics = run_all_metrics_with_prices(portfolio, shared_prices, shared_rf)
                    sim_metrics = run_all_metrics_with_prices(sim_portfolio, shared_prices, shared_rf)
                    cur_ctr = calculate_ctr(shared_prices, portfolio)
                    sim_ctr = calculate_ctr(shared_prices, sim_portfolio)
                if "error" in sim_metrics: st.error(sim_metrics["error"])
                else:
                    st.session_state["sim_results"] = {
                        "current": current_metrics, "simulated": sim_metrics,
                        "selected_base": selected_base,
                        "hypothetical": [h["ticker"] for h in st.session_state["sim_hypothetical"]],
                        "sim_portfolio": sim_portfolio, "cur_ctr": cur_ctr, "sim_ctr": sim_ctr}

        if "sim_results" in st.session_state:
            res = st.session_state["sim_results"]
            cur, sim = res["current"], res["simulated"]
            st.divider()
            st.markdown("**Results**")
            ic1, ic2 = st.columns(2)
            ic1.caption(f"Base: {', '.join(res['selected_base']) or 'None'}")
            ic2.caption(f"Hypothetical: {', '.join(res['hypothetical']) or 'None'}")

            metric_defs = [
                ("Variance (Daily)", "Variance (Daily)", "{:.6f}", False),
                ("Std Dev (Daily)", "Std Dev (Daily)", "{:.6f}", False),
                ("Std Dev (Annual)", "Std Dev (Annual)", "{:.2%}", False),
                ("Skewness", "Skewness", "{:.4f}", False),
                ("Kurtosis", "Kurtosis", "{:.4f}", False),
                ("CAGR", "CAGR", "{:.2%}", True),
                ("Max Drawdown", "Max Drawdown", "{:.2%}", True),
                ("Sharpe Ratio", "Sharpe Ratio", "{:.4f}", True),
                ("Sortino Ratio", "Sortino Ratio", "{:.4f}", True),
                ("Calmar Ratio", "Calmar Ratio", "{:.4f}", True),
                ("VaR (95%)", "VaR (95%)", "{:.4%}", False),
                ("CVaR (95%)", "CVaR (95%)", "{:.4%}", False),
                ("Beta", "Beta", "{:.4f}", False),
                ("Alpha (Annual)", "Alpha (Annual)", "{:.2%}", True),
            ]
            h1, h2, h3, h4 = st.columns([2, 1.5, 1.5, 1.5])
            h1.markdown("**Metric**"); h2.markdown("**Current**"); h3.markdown("**Simulated**"); h4.markdown("**Delta**")
            st.markdown("---")
            for label, key, fmt, hb in metric_defs:
                cv, sv = cur.get(key), sim.get(key)
                r1, r2, r3, r4 = st.columns([2, 1.5, 1.5, 1.5])
                r1.write(label)
                r2.write(fmt.format(cv) if cv is not None else "N/A")
                r3.write(fmt.format(sv) if sv is not None else "N/A")
                if cv is not None and sv is not None:
                    d = sv - cv; p = "+" if d > 0 else ""
                    c = ("green" if d > 0 else "red") if hb else ("green" if d < 0 else "red")
                    r4.markdown(f":{c}[{p}{fmt.format(d)}]")
                else: r4.write("-")
            st.caption(f"Risk-free rate: {cur.get('Risk-Free Rate Used', 0)*100:.2f}% | Lookback: {sim_lookback}")

            st.divider()
            st.markdown("**Portfolio Comparison**")
            wl, wr = st.columns(2)
            with wl:
                st.markdown("Current - Weight")
                cw = portfolio[portfolio["ticker"].isin(res["selected_base"])].copy()
                if not cw.empty: st.bar_chart(cw.set_index("ticker")["shares"].rename("Weight"))
            with wr:
                st.markdown("Simulated - Weight")
                sp = res["sim_portfolio"]
                if not sp.empty: st.bar_chart(sp.set_index("ticker")["shares"].rename("Weight"))

            cl, cr = st.columns(2)
            with cl:
                st.markdown("Current - CTR")
                if not res["cur_ctr"].empty:
                    st.bar_chart(res["cur_ctr"].set_index("Ticker")["CTR (%)"])
                    st.dataframe(res["cur_ctr"], use_container_width=True, hide_index=True)
            with cr:
                st.markdown("Simulated - CTR")
                if not res["sim_ctr"].empty:
                    st.bar_chart(res["sim_ctr"].set_index("Ticker")["CTR (%)"])
                    st.dataframe(res["sim_ctr"], use_container_width=True, hide_index=True)

            st.divider()
            if st.button("Reset Simulator", key="reset_sim"):
                for k in ["sim_results", "sim_hypothetical", "markowitz_results"]:
                    st.session_state.pop(k, None)
                st.rerun()

        # --- step 4: markowitz ---
        st.divider()
        st.markdown("**Step 4 - Portfolio Optimization (Markowitz)**")
        st.caption("Mean-variance optimization. Long/Short % are maximum caps, not targets.")

        # auto-calculate capital (includes simulated proceeds from Step 2)
        sim_proceeds = st.session_state.get("sim_proceeds", 0.0)
        if not portfolio.empty:
            tmv = 0
            for t in portfolio["ticker"].tolist():
                try:
                    p = yf.Ticker(t).fast_info.last_price
                    s = portfolio.loc[portfolio["ticker"] == t, "shares"].iloc[0]
                    tmv += p * s
                except: pass
            cap_result = calculate_current_capital(tmv)
            default_capital = int(cap_result[0]) if isinstance(cap_result, tuple) else int(cap_result)
            default_capital += int(sim_proceeds)
        else:
            default_capital = 1_000_000

        opt_c1, opt_c2, opt_c3, opt_c4, opt_c5 = st.columns(5)
        with opt_c1: opt_capital = st.number_input("Total Capital ($)", value=default_capital, step=10000, key="opt_capital")
        with opt_c2: opt_long_pct = st.number_input("Max Long %", value=73, min_value=0, max_value=100, step=1, key="opt_long_pct")
        with opt_c3: opt_short_pct = st.number_input("Max Short %", value=27, min_value=0, max_value=100, step=1, key="opt_short_pct")
        with opt_c4: opt_max_pos = st.number_input("Max Per Position ($)", value=90000, step=5000, key="opt_max_pos")
        with opt_c5: opt_min_deploy = st.number_input("Min Deploy %", value=95, min_value=50, max_value=100, step=1, key="opt_min_deploy")

        if opt_long_pct + opt_short_pct > 100:
            st.warning(f"Max Long + Max Short = {opt_long_pct + opt_short_pct}%. Exceeds 100%.")

        all_opt_tickers = list(set(
            [t for t in selected_base] + [h["ticker"] for h in st.session_state.get("sim_hypothetical", [])]))

        if all_opt_tickers:
            st.markdown("**Tag tickers as Long or Short:**")
            st.caption("Defaults from current portfolio positions.")
            opt_cols = st.columns(min(len(all_opt_tickers), 5))
            opt_long_list, opt_short_list = [], []
            for i, t in enumerate(all_opt_tickers):
                with opt_cols[i % min(len(all_opt_tickers), 5)]:
                    default_side = "Long"
                    if not portfolio.empty and t in portfolio["ticker"].values:
                        if portfolio.loc[portfolio["ticker"] == t, "shares"].iloc[0] < 0:
                            default_side = "Short"
                    side = st.radio(t, ["Long", "Short"], index=0 if default_side == "Long" else 1,
                                    key=f"opt_side_{t}", horizontal=True)
                    if side == "Long": opt_long_list.append(t)
                    else: opt_short_list.append(t)

            if not opt_long_list: st.warning("Select at least one Long ticker.")

            if st.button("Run Optimization", type="primary", key="run_opt_btn"):
                if not opt_long_list:
                    st.error("Need at least one Long ticker.")
                elif len(opt_long_list) + len(opt_short_list) < 2:
                    st.error("Need at least 2 tickers total.")
                else:
                    with st.spinner("Running Markowitz optimization..."):
                        opt_yrs = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10, "15 Years": 15}.get(sim_lookback, 3)
                        opt_prices = fetch_price_history(opt_long_list + opt_short_list, years=opt_yrs)
                        opt_rf = get_risk_free_rate()
                        cur_port_ret = compute_portfolio_returns(opt_prices, portfolio)
                        cur_ret = float(cur_port_ret.mean() * 252) if len(cur_port_ret) > 0 else 0
                        cur_vol = float(cur_port_ret.std() * np.sqrt(252)) if len(cur_port_ret) > 0 else 0
                        cur_sh = (cur_ret - opt_rf) / cur_vol if cur_vol > 0 else 0
                        opt_result = run_markowitz_optimization(
                            long_tickers=opt_long_list, short_tickers=opt_short_list,
                            prices=opt_prices, rf=opt_rf, total_capital=opt_capital,
                            max_long_pct=opt_long_pct, max_short_pct=opt_short_pct,
                            max_per_position=opt_max_pos, min_deploy_pct=opt_min_deploy)
                    if "error" in opt_result: st.error(opt_result["error"])
                    else:
                        opt_result["current"] = {"return": round(cur_ret, 4), "volatility": round(cur_vol, 4), "sharpe": round(cur_sh, 4)}
                        st.session_state["markowitz_results"] = opt_result

            if "markowitz_results" in st.session_state:
                mk = st.session_state["markowitz_results"]
                ms = mk["max_sharpe"]
                mv = mk["min_variance"]
                uc = mk["unconstrained"]
                cur_m = mk["current"]
                ca = mk["constraint_analysis"]

                # efficient frontier
                st.markdown("**Efficient Frontier**")
                frontier = mk["frontier"]
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frontier["volatilities"], y=frontier["returns"], mode="markers",
                    marker=dict(size=3, color=frontier["sharpes"], colorscale="Viridis", showscale=True,
                                colorbar=dict(title="Sharpe")),
                    name="Simulated", hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}"))
                fig.add_trace(go.Scatter(x=[ms["volatility"]], y=[ms["return"]], mode="markers",
                    marker=dict(size=15, color="red", symbol="star"),
                    name=f"Max Sharpe ({ms['sharpe']:.2f})"))
                fig.add_trace(go.Scatter(x=[mv["volatility"]], y=[mv["return"]], mode="markers",
                    marker=dict(size=15, color="blue", symbol="diamond"),
                    name=f"Min Variance ({mv['sharpe']:.2f})"))
                fig.add_trace(go.Scatter(x=[uc["volatility"]], y=[uc["return"]], mode="markers",
                    marker=dict(size=15, color="green", symbol="triangle-up"),
                    name=f"Unconstrained ({uc['sharpe']:.2f})"))
                fig.add_trace(go.Scatter(x=[cur_m["volatility"]], y=[cur_m["return"]], mode="markers",
                    marker=dict(size=15, color="orange", symbol="x"),
                    name=f"Current ({cur_m['sharpe']:.2f})"))
                fig.update_layout(xaxis_title="Annualised Volatility", yaxis_title="Annualised Return",
                                  xaxis_tickformat=".1%", yaxis_tickformat=".1%",
                                  height=500, margin=dict(l=40, r=40, t=30, b=40))
                st.plotly_chart(fig, use_container_width=True)

                # constraint analysis box
                if ca["is_limiting"]:
                    st.info(
                        f"**Constraint Analysis**\n\n"
                        f"Unconstrained Optimal: **{ca['uc_long_pct']}% Long / {ca['uc_short_pct']}% Short**\n\n"
                        f"Your Limits: **{ca['constrained_long_pct']}% Long / {ca['constrained_short_pct']}% Short**\n\n"
                        f"Your constraints are limiting potential returns. "
                        f"Unconstrained Sharpe: **{ca['uc_sharpe']:.2f}** vs Constrained: **{ca['constrained_sharpe']:.2f}** "
                        f"(gap: {ca['sharpe_gap']:.4f})"
                    )
                else:
                    st.success(
                        f"**Constraint Analysis**\n\n"
                        f"Unconstrained Optimal: **{ca['uc_long_pct']}% Long / {ca['uc_short_pct']}% Short**\n\n"
                        f"Your Limits: **{ca['constrained_long_pct']}% Long / {ca['constrained_short_pct']}% Short**\n\n"
                        f"Your constraints are not limiting the optimizer."
                    )

                # metrics comparison — 5 columns
                st.markdown("**Metrics Comparison**")
                comp = [("Expected Return", "return", "{:.2%}"), ("Volatility", "volatility", "{:.2%}"),
                        ("Sharpe Ratio", "sharpe", "{:.4f}"), ("VaR (95%)", "var_95", "{:.4%}"),
                        ("CVaR (95%)", "cvar_95", "{:.4%}"), ("Cash Held", "cash", "${:,.0f}"),
                        ("Long %", "long_pct", "{:.1f}%"), ("Short %", "short_pct", "{:.1f}%")]
                mh1, mh2, mh3, mh4, mh5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
                mh1.markdown("**Metric**"); mh2.markdown("**Current**"); mh3.markdown("**Max Sharpe**")
                mh4.markdown("**Min Variance**"); mh5.markdown("**Unconstrained**")
                st.markdown("---")
                for label, key, fmt in comp:
                    mc1, mc2, mc3, mc4, mc5 = st.columns([2, 1.2, 1.2, 1.2, 1.2])
                    mc1.write(label)
                    cv = cur_m.get(key); msv = ms.get(key); mvv = mv.get(key); ucv = uc.get(key)
                    mc2.write(fmt.format(cv) if cv is not None else "N/A")
                    mc3.write(fmt.format(msv) if msv is not None else "N/A")
                    mc4.write(fmt.format(mvv) if mvv is not None else "N/A")
                    mc5.write(fmt.format(ucv) if ucv is not None else "N/A")

                # allocation tables — 3 columns
                st.divider()
                at1, at2, at3 = st.columns(3)
                tbl_fmt = {"Dollar Amount ($)": "${:,.2f}", "Weight (%)": "{:.2f}"}
                with at1:
                    st.markdown("**Max Sharpe Allocation**")
                    if not ms["table"].empty:
                        st.dataframe(ms["table"].style.format(tbl_fmt), use_container_width=True, hide_index=True)
                with at2:
                    st.markdown("**Min Variance Allocation**")
                    if not mv["table"].empty:
                        st.dataframe(mv["table"].style.format(tbl_fmt), use_container_width=True, hide_index=True)
                with at3:
                    st.markdown("**Unconstrained Allocation**")
                    if not uc["table"].empty:
                        st.dataframe(uc["table"].style.format(tbl_fmt), use_container_width=True, hide_index=True)
