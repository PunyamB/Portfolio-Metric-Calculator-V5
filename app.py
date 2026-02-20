import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from portfolio import (
    get_portfolio, get_trade_history, compute_realised_pnl, calculate_current_capital,
)
from metrics import (
    run_all_metrics, run_all_metrics_with_prices, fetch_price_history,
    get_risk_free_rate, compute_portfolio_returns, calculate_factor_exposures,
    calculate_ctr, run_markowitz_optimization, calculate_inception_metrics,
    _run_custom_target_optimization,
)
import metrics as m

st.set_page_config(page_title="Paper Trading Dashboard", page_icon="📈", layout="wide")

# === CUSTOM CSS THEME ===
st.markdown("""<style>
    /* Sidebar */
    section[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
    [data-testid="stSidebarContent"] { padding-top: 0.5rem; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #8892b0 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e6f1ff !important;
        font-weight: 600;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0a0a1a;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
        color: #8892b0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a2e !important;
        color: #64ffda !important;
    }

    /* Dataframes */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* Headers */
    h1, h2, h3 { color: #ccd6f6 !important; }

    /* KPI strip */
    .kpi-strip {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d; border-radius: 10px;
        padding: 10px 24px; margin-bottom: 16px;
    }
    .kpi-item { text-align: center; flex: 1; }
    .kpi-label { color: #8b949e; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #e6edf3; font-size: 1.1rem; font-weight: 600; }
    .kpi-value.positive { color: #3fb950; }
    .kpi-value.negative { color: #f85149; }
    .kpi-divider { width: 1px; height: 36px; background: #30363d; margin: 0 8px; }

    /* Badge styles */
    .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin-left: 6px; }
    .badge-good { background: #1a3a2a; color: #3fb950; border: 1px solid #238636; }
    .badge-watch { background: #3a2a1a; color: #d29922; border: 1px solid #9e6a03; }
    .badge-risk { background: #3a1a1a; color: #f85149; border: 1px solid #da3633; }

    /* Divider */
    hr { border-color: #21262d !important; }
</style>""", unsafe_allow_html=True)

st.title("Paper Trading Portfolio Dashboard")
st.caption("US Equities & ETFs | Benchmark: S&P 500 | Data: Yahoo Finance EOD | Source: stocktrak_history.csv")

portfolio = get_portfolio()


# === HELPER: metric badge ===
def metric_badge(value, good_fn, watch_fn=None):
    """Return HTML badge: good/watch/risk based on value."""
    if value is None or value == "N/A": return ""
    try:
        v = float(value) if not isinstance(value, (int, float)) else value
    except: return ""
    if good_fn(v): return '<span class="badge badge-good">Good</span>'
    if watch_fn and watch_fn(v): return '<span class="badge badge-watch">Watch</span>'
    return '<span class="badge badge-risk">Risk</span>'

# badge rules for key metrics
BADGE_RULES = {
    "Sharpe Ratio": (lambda v: v > 1.0, lambda v: 0.5 <= v <= 1.0),
    "Sortino Ratio": (lambda v: v > 1.5, lambda v: 0.75 <= v <= 1.5),
    "Beta": (lambda v: 0.8 <= v <= 1.2, lambda v: 0.5 <= v <= 1.5),
    "Max Drawdown": (lambda v: v > -0.10, lambda v: -0.20 <= v <= -0.10),
    "Sharpe (Daily)": (lambda v: v > 0.05, lambda v: 0.02 <= v <= 0.05),
    "Alpha (Annual)": (lambda v: v > 0.02, lambda v: 0.0 <= v <= 0.02),
    "VaR (95%)": (lambda v: v > -0.02, lambda v: -0.03 <= v <= -0.02),
    "CVaR (95%)": (lambda v: v > -0.03, lambda v: -0.04 <= v <= -0.03),
    "Calmar Ratio": (lambda v: v > 1.0, lambda v: 0.5 <= v <= 1.0),
    "CAGR": (lambda v: v > 0.10, lambda v: 0.0 <= v <= 0.10),
}


# === KPI STRIP (pre-compute values) ===
if not portfolio.empty:
    _kpi_prices = {}
    for t in portfolio["ticker"].tolist():
        try: _kpi_prices[t] = yf.Ticker(t).fast_info.last_price
        except: _kpi_prices[t] = None

    _kpi_mv = sum((s * (_kpi_prices.get(t) or 0)) for t, s in zip(portfolio["ticker"], portfolio["shares"]))
    _kpi_cap_result = calculate_current_capital(_kpi_mv)
    _kpi_total = _kpi_cap_result[0]
    _kpi_pnl = _kpi_total - 1_000_000
    _kpi_pnl_pct = (_kpi_pnl / 1_000_000) * 100
    _kpi_long_count = len(portfolio[portfolio["shares"] > 0])
    _kpi_short_count = len(portfolio[portfolio["shares"] < 0])

    pnl_class = "positive" if _kpi_pnl >= 0 else "negative"
    pnl_sign = "+" if _kpi_pnl >= 0 else ""

    st.markdown(f"""<div class="kpi-strip">
        <div class="kpi-item"><div class="kpi-label">Account Value</div><div class="kpi-value">${_kpi_total:,.0f}</div></div>
        <div class="kpi-divider"></div>
        <div class="kpi-item"><div class="kpi-label">Total P&L</div><div class="kpi-value {pnl_class}">{pnl_sign}${_kpi_pnl:,.0f} ({pnl_sign}{_kpi_pnl_pct:.2f}%)</div></div>
        <div class="kpi-divider"></div>
        <div class="kpi-item"><div class="kpi-label">Positions</div><div class="kpi-value">{len(portfolio)}</div></div>
        <div class="kpi-divider"></div>
        <div class="kpi-item"><div class="kpi-label">Long / Short</div><div class="kpi-value">{_kpi_long_count}L / {_kpi_short_count}S</div></div>
    </div>""", unsafe_allow_html=True)


# sidebar
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
tab_overview, tab_reports, tab_portfolio, tab_history, tab_simulator = st.tabs(["Overview", "Reports", "Portfolio", "Trade History", "What-If Simulator"])


# === TAB: REPORTS ===
with tab_reports:
    st.subheader("Portfolio Reports")
    st.caption("Select any two dates to generate a report comparing portfolio performance between them.")

    gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    if not gemini_key:
        st.warning("Gemini API key not found. Add `GEMINI_API_KEY` to `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets.")
    else:
        st.success("Gemini API connected.")

    rc1, rc2, rc3 = st.columns([2, 2, 1])
    with rc1: report_start = st.date_input("Start Date", value=None, key="report_start_date")
    with rc2: report_end = st.date_input("End Date", value=None, key="report_end_date")
    with rc3:
        st.write(""); st.write("")
        gen_disabled = not gemini_key or report_start is None or report_end is None
        gen_clicked = st.button("Generate Report", type="primary", use_container_width=True, key="gen_report_btn", disabled=gen_disabled)

    if gen_clicked and report_start and report_end:
        if report_start >= report_end:
            st.error("Start date must be before end date.")
        else:
            with st.spinner("Reconstructing portfolios, computing metrics, generating narrative..."):
                try:
                    from report_data import compute_report_data
                    from report_narrative import generate_report_narrative
                    from report_gen import generate_report
                    import os
                    report_data = compute_report_data(report_start, report_end)
                    if "error" in report_data:
                        st.error(report_data["error"])
                    else:
                        narrative = generate_report_narrative(report_data, gemini_key)
                        output_path = os.path.join(os.getcwd(), "portfolio_report.docx")
                        generate_report(report_data, narrative, output_path)
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Report", data=f.read(),
                                file_name=f"portfolio_report_{report_data['period_start']}_to_{report_data['period_end']}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                key="dl_report")
                        st.session_state["report_data"] = report_data
                        st.session_state["report_narrative"] = narrative
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")

    if "report_narrative" in st.session_state:
        with st.expander("Report Preview", expanded=True):
            n = st.session_state["report_narrative"]
            d = st.session_state["report_data"]
            st.caption(f"{d['period_start']} to {d['period_end']}")
            st.markdown(f"**Portfolio:** ${d['start_value']:,.2f} → ${d['end_value']:,.2f} ({d['value_change_pct']:+.2f}%)")
            st.markdown(f"**Benchmark (S&P 500):** {d['benchmark_return']:+.2f}%")
            st.markdown("---")
            st.markdown(f"**Executive Summary:** {n.get('executive_summary', 'N/A')}")
            st.markdown(f"**Performance Attribution:** {n.get('performance_attribution', 'N/A')}")
            st.markdown(f"**Risk Profile:** {n.get('risk_analysis', 'N/A')}")
            st.markdown(f"**Outlook:** {n.get('outlook', 'N/A')}")


# === TAB 0: OVERVIEW ===
with tab_overview:
    st.subheader("Project Overview")
    st.caption("Paper Trading Portfolio Dashboard — Architecture, Methodology & Feature Guide")

    st.markdown("---")

    st.markdown("### Data Pipeline")
    st.markdown("""
The system reads `stocktrak_history.csv` as its single source of truth. On every page load, the raw CSV is parsed, cleaned, and transformed into a live portfolio. Transaction types like "Market - Buy", "Limit - Sell", "Short Proceeds", and "Buy to Cover" are all normalised into three standard actions: Buy, Sell, and Short. Dividend rows are excluded since Yahoo Finance's adjusted prices already account for dividend returns in metric calculations. Prices and amounts are stripped of accounting notation (dollar signs, commas, parentheses) and quantities are converted to absolute values with direction inferred from the transaction type. Multiple lots of the same security are merged into weighted average positions.
""")

    st.markdown("### Portfolio Tab")
    st.markdown("""
The holdings table displays each position with its current market price (fetched live from Yahoo Finance), cost basis, unrealised P&L, and portfolio weight. Positions are tagged as Long or Short based on share direction. Summary cards show Portfolio Value (net market value of all positions), Cash Balance (actual spendable cash excluding short collateral), Margin Held (cost basis of open short positions held as collateral), Total Account Value (sum of all three), and a three-way P&L breakdown — unrealised, realised, and total. Realised P&L is calculated using the average cost method: when shares are sold, the gain or loss is computed against the weighted average purchase price of all lots for that security.
""")

    st.markdown("### Risk Metrics")
    st.markdown("""
Metrics are computed across the portfolio using historical daily returns, organised into logical groupings.

**Return & Growth** — CAGR, Max Drawdown, Calmar Ratio, and Turnover Ratio. These show how the portfolio has grown, how bad the worst decline was, and how actively it is being traded.

**Volatility & Distribution** — Daily and annualised standard deviation, variance, and skewness. These describe how much the portfolio moves and whether those movements are symmetric.

**Risk Measures** — Kurtosis (tail risk), Value at Risk at 95% confidence (the worst daily loss expected 19 out of 20 days), Conditional VaR (the average loss on those bad days), and Beta (sensitivity to the S&P 500).

**Performance Ratios** — Sharpe (excess return per unit of total risk), Sortino (excess return per unit of downside risk only), and annualised Alpha (return above what Beta alone would predict).

**Daily Values** — Non-annualised Sharpe and Alpha for direct comparison with StockTrak's reported figures.

**Since Inception** — Sharpe and Alpha computed from the portfolio's start date of January 26, 2026, regardless of the lookback period selected. This provides a consistent performance anchor across all time horizons.

All metrics use the 13-week US Treasury Bill rate as the risk-free rate, fetched live. The S&P 500 serves as the benchmark. Lookback periods of 1, 3, 5, 10, or 15 years determine how much historical price data is used.
""")

    st.markdown("### Fama-French 5-Factor Analysis")
    st.markdown("""
Beyond simple Beta, a full regression is run against the Fama-French five-factor model. This decomposes portfolio returns into exposure to five systematic risk factors: Market (overall equity risk), Size (small vs large cap tilt), Value (cheap vs expensive stocks), Profitability (strong vs weak earnings), and Investment (conservative vs aggressive capital spending). Each factor shows its exposure coefficient, t-statistic, statistical significance, an optimal range for diversified portfolios, and a plain-English interpretation. The R-squared indicates what percentage of portfolio returns are explained by these five factors — anything left over is idiosyncratic risk specific to the stock picks.
""")

    st.markdown("### Contribution to Risk")
    st.markdown("""
This analysis answers the question "which positions are adding the most risk to the portfolio?" Using the covariance matrix of daily returns, each position's marginal contribution to total portfolio volatility is calculated. A position with high CTR relative to its weight is contributing disproportionate risk — it might be worth trimming. A position with low CTR relative to its weight is a diversifier.
""")

    st.markdown("### Reports")
    st.markdown("""
The report generator allows selection of any two dates and produces a professional portfolio analysis comparing performance between them. The system reconstructs holdings on each date by replaying the CSV trade history, fetches historical prices from Yahoo Finance, and computes all metric deltas. Per-stock P&L attribution handles mid-period trades correctly — if a stock was bought on Wednesday, the return is calculated from the buy price to the end date, not from the start date. Narrative commentary is generated by Google's Gemini AI using the structured data as input. The output is a downloadable DOCX report with performance snapshot, top movers, risk profile changes, factor exposures, and forward outlook.
""")

    st.markdown("### Trade History")

    st.markdown("### What-If Simulator")
    st.markdown("""
The simulator allows construction of hypothetical portfolios and comparison of their risk profiles against current holdings without making any actual trades.

**Step 1** — Existing positions are selected for inclusion by toggling checkboxes.

**Step 2** — Hypothetical positions are added. Positive shares for new longs, negative shares for simulated sells or new shorts. When a simulated sell or new short is entered, the system calculates the cash proceeds at current market prices and displays them. These proceeds flow into Step 4's optimiser as additional available capital.

**Step 3** — A side-by-side metric comparison between the current portfolio and the simulated one. Every metric is shown with a delta and colour-coded — green for improvement, red for deterioration. The simulation uses the same price data and lookback period for both portfolios to ensure an apples-to-apples comparison. Simulated cash from hypothetical sells does not affect Step 3's metrics because in practice, uninvested cash earns the risk-free rate and has zero volatility — it would only dilute the metrics without adding information.
""")

    st.markdown("### Markowitz Portfolio Optimization")
    st.markdown("""
Step 4 uses mean-variance optimisation to find the mathematically optimal allocation across the selected tickers. Three scenarios are run:

- **Max Sharpe** finds the allocation with the highest risk-adjusted return.
- **Min Variance** finds the allocation with the lowest possible volatility.
- **Unconstrained** runs with no long/short caps to show what the optimiser would do if constraints were removed.

Five parameters control the optimisation: **Total Capital** (auto-calculated from the trade history plus any simulated proceeds, editable), **Max Long %** (the ceiling for total long exposure, auto-filled from current portfolio allocation), **Max Short %** (the ceiling for total short exposure, auto-filled from current portfolio allocation), **Max Per Position** (a dollar cap preventing any single position from dominating, default $90,000), and **Min Deploy %** (ensures at least this percentage of capital is allocated rather than sitting in cash, default 95%).

The long and short percentages are maximum caps, not targets — the optimiser can use less on either side if the math doesn't justify filling the allocation. This reflects how real portfolio managers operate: they set risk limits but deploy capital based on opportunity.

Each ticker is tagged as Long or Short with defaults inferred from current portfolio positions. The efficient frontier chart plots 10,000 randomly generated portfolios colour-coded by Sharpe ratio, with the three optimal points and the current portfolio marked.

A **Constraint Analysis** box appears after results — if the unconstrained optimal would prefer a different long/short split than the caps allow, it shows as a blue info box with the Sharpe gap. If constraints aren't binding, it shows as a green success box. The metrics comparison table adds the unconstrained column for visibility into what relaxing limits would gain or cost. Three allocation tables show the recommended share counts, dollar amounts, and weights for each scenario, with a CASH row when the optimiser doesn't fully deploy capital.
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
        # reuse KPI prices if available, otherwise fetch
        latest_prices = _kpi_prices if '_kpi_prices' in dir() else {}
        if not latest_prices:
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

        # --- Treemap + Donut ---
        viz1, viz2 = st.columns([3, 2])
        with viz1:
            st.markdown("**Position Map**")
            tree_df = display_df[display_df["Current Price ($)"].notna()].copy()
            tree_df["Abs MV"] = tree_df["Market Value ($)"].abs()
            tree_df["Color"] = tree_df["P&L %"]
            fig_tree = px.treemap(tree_df, path=["Ticker"], values="Abs MV", color="Color",
                color_continuous_scale=["#f85149", "#21262d", "#3fb950"],
                color_continuous_midpoint=0,
                hover_data={"Market Value ($)": ":$,.2f", "P&L %": ":.2f", "Weight (%)": ":.1f"})
            fig_tree.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=320,
                                   coloraxis_colorbar=dict(title="P&L %"))
            st.plotly_chart(fig_tree, use_container_width=True)

        with viz2:
            st.markdown("**Weight Allocation**")
            donut_df = display_df[display_df["Current Price ($)"].notna()].copy()
            donut_df["Abs Weight"] = donut_df["Weight (%)"].abs()
            fig_donut = px.pie(donut_df, values="Abs Weight", names="Ticker", hole=0.55)
            fig_donut.update_layout(margin=dict(t=10, l=10, r=10, b=10), height=320,
                                    showlegend=True, legend=dict(font=dict(size=10)))
            fig_donut.update_traces(textposition="inside", textinfo="label+percent")
            st.plotly_chart(fig_donut, use_container_width=True)

        # Holdings table
        fmt_df = display_df[["Ticker", "Position", "Shares", "Avg Price ($)", "Current Price ($)",
                              "Market Value ($)", "Weight (%)", "Cost Basis ($)",
                              "Unrealised P&L ($)", "P&L %", "First Trade"]].copy()

        st.dataframe(fmt_df.style.format({
            "Avg Price ($)": "${:,.2f}", "Current Price ($)": "${:,.2f}",
            "Market Value ($)": "${:,.2f}", "Cost Basis ($)": "${:,.2f}",
            "Unrealised P&L ($)": "${:,.2f}", "Weight (%)": "{:.2f}", "P&L %": "{:.2f}",
        }), use_container_width=True, hide_index=True)

        total_cost = display_df["Cost Basis ($)"].sum()
        total_pnl = display_df["Unrealised P&L ($)"].sum()
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        _, total_realised = compute_realised_pnl()
        total_combined = total_pnl + total_realised

        cap_result = calculate_current_capital(total_market_value)
        auto_capital, cash_balance, margin_held = cap_result

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Portfolio Value", f"${total_market_value:,.2f}")
        c2.metric("Cash Balance", f"${cash_balance:,.2f}")
        c3.metric("Margin Held", f"${margin_held:,.2f}")
        c4.metric("Total Account Value", f"${auto_capital:,.2f}")
        c5.metric("Positions", len(portfolio))

        p1, p2, p3 = st.columns(3)
        p1.metric("Unrealised P&L", f"${total_pnl:,.2f}", delta=f"{total_pnl_pct:.2f}%")
        realised_pct = (total_realised / total_cost * 100) if total_cost > 0 else 0
        p2.metric("Realised P&L", f"${total_realised:,.2f}", delta=f"{realised_pct:.2f}%")
        combined_pct = (total_combined / total_cost * 100) if total_cost > 0 else 0
        p3.metric("Total P&L", f"${total_combined:,.2f}", delta=f"{combined_pct:.2f}%")

        # --- Portfolio Value Time Series ---
        st.divider()
        st.markdown("**Portfolio Value vs S&P 500**")
        pv_tickers = portfolio["ticker"].tolist()
        pv_prices = fetch_price_history(pv_tickers + ["^GSPC"], years=1)
        if not pv_prices.empty:
            pv_shares = portfolio.set_index("ticker")["shares"]
            # handle potential duplicate tickers
            pv_shares = pv_shares.groupby(level=0).sum()
            avail_t = [t for t in pv_shares.index if t in pv_prices.columns]
            if avail_t:
                pv_values = pv_prices[avail_t].multiply(pv_shares[avail_t], axis=1).sum(axis=1)
                pv_norm = pv_values / pv_values.iloc[0] * 100
                if "^GSPC" in pv_prices.columns:
                    sp_norm = pv_prices["^GSPC"] / pv_prices["^GSPC"].iloc[0] * 100
                    fig_pv = go.Figure()
                    fig_pv.add_trace(go.Scatter(x=pv_norm.index, y=pv_norm.values,
                        name="Portfolio", line=dict(color="#64ffda", width=2)))
                    fig_pv.add_trace(go.Scatter(x=sp_norm.index, y=sp_norm.values,
                        name="S&P 500", line=dict(color="#8892b0", width=1.5, dash="dot")))
                    fig_pv.update_layout(yaxis_title="Indexed (100 = Start)", height=350,
                        margin=dict(l=40, r=40, t=20, b=40), legend=dict(orientation="h", y=1.02),
                        hovermode="x unified")
                    st.plotly_chart(fig_pv, use_container_width=True)

        # --- sub-tabs ---
        st.divider()
        sub_metrics, sub_risk = st.tabs(["Metrics", "Risk Analysis"])

        with sub_metrics:
            col_lb, col_btn = st.columns([2, 1])
            with col_lb:
                lookback = st.selectbox("Lookback Period",
                    ["1 Year", "3 Years", "5 Years", "10 Years", "15 Years"], index=1, key="pf_lookback")
            with col_btn:
                lookback = lookback
            calc_clicked = col_btn.button("Calculate Metrics", type="primary", use_container_width=True, key="calc_pf")
            if calc_clicked:
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

                def _metric_with_badge(label, value, tip, raw_val=None):
                    """Display metric with optional badge."""
                    badge_html = ""
                    if label in BADGE_RULES and raw_val is not None:
                        good_fn, watch_fn = BADGE_RULES[label]
                        badge_html = metric_badge(raw_val, good_fn, watch_fn)
                    if badge_html:
                        st.metric(label=label, value=value, help=tip)
                        st.markdown(badge_html, unsafe_allow_html=True)
                    else:
                        st.metric(label=label, value=value, help=tip)

                # Row 1: Return & Growth
                mc1 = st.columns(4)
                raw_cagr = r.get('CAGR', 0)
                raw_dd = r.get('Max Drawdown', 0)
                raw_calmar = r.get('Calmar Ratio', 0)
                with mc1[0]: _metric_with_badge("CAGR", f"{raw_cagr*100:.2f}%", "Compound annual growth", raw_cagr)
                with mc1[1]: _metric_with_badge("Max Drawdown", f"{raw_dd*100:.2f}%", "Worst peak-to-trough", raw_dd)
                with mc1[2]: _metric_with_badge("Calmar Ratio", r.get("Calmar Ratio", "N/A"), "CAGR / Max Drawdown", raw_calmar)
                with mc1[3]: st.metric("Turnover Ratio", r.get("Turnover Ratio", "N/A"), help="From trade history")

                # Row 2: Volatility & Distribution
                mc2 = st.columns(4)
                with mc2[0]: st.metric("Variance (Daily)", r.get("Variance (Daily)", "N/A"), help="Daily return variance")
                with mc2[1]: st.metric("Std Dev (Daily)", f"{r.get('Std Dev (Daily)', 0)*100:.4f}%", help="Daily volatility")
                with mc2[2]: st.metric("Std Dev (Annual)", f"{r.get('Std Dev (Annual)', 0)*100:.2f}%", help="Annualised volatility")
                with mc2[3]: st.metric("Skewness", r.get("Skewness", "N/A"), help="Negative = left tail, Positive = right tail")

                # Row 3: Risk Measures
                mc3 = st.columns(4)
                raw_var = r.get('VaR (95%)', 0)
                raw_cvar = r.get('CVaR (95%)', 0)
                raw_beta = r.get('Beta', 0)
                with mc3[0]: st.metric("Kurtosis", r.get("Kurtosis", "N/A"), help=">0 = fat tails")
                with mc3[1]: _metric_with_badge("VaR (95%)", f"{raw_var*100:.2f}%", "Max daily loss at 95% confidence", raw_var)
                with mc3[2]: _metric_with_badge("CVaR (95%)", f"{raw_cvar*100:.2f}%", "Avg loss beyond VaR", raw_cvar)
                with mc3[3]: _metric_with_badge("Beta", r.get("Beta", "N/A"), "vs S&P 500", raw_beta)

                # Row 4: Performance Ratios
                mc4 = st.columns(4)
                raw_sharpe = r.get('Sharpe Ratio', 0)
                raw_sortino = r.get('Sortino Ratio', 0)
                raw_alpha = r.get('Alpha (Annual)', 0)
                with mc4[0]: _metric_with_badge("Sharpe Ratio", r.get("Sharpe Ratio", "N/A"), "Annualised excess return / volatility", raw_sharpe)
                with mc4[1]: _metric_with_badge("Sortino Ratio", r.get("Sortino Ratio", "N/A"), "Annualised excess return / downside vol", raw_sortino)
                with mc4[2]: _metric_with_badge("Alpha (Annual)", f"{raw_alpha*100:.2f}%", "Annualised excess return vs benchmark", raw_alpha)

                # Row 5: Daily Values
                mc5 = st.columns(4)
                raw_sharpe_d = r.get('Sharpe (Daily)', 0)
                with mc5[0]: _metric_with_badge("Sharpe (Daily)", r.get("Sharpe (Daily)", "N/A"), "Non-annualised daily Sharpe", raw_sharpe_d)
                with mc5[1]: st.metric("Alpha (Daily)", f"{r.get('Alpha (Daily)', 0)*100:.4f}%", help="Non-annualised daily alpha")

                # Row 6: Since Inception
                if inc:
                    st.caption("Since Inception (Jan 26, 2026)")
                    mc6 = st.columns(4)
                    with mc6[0]: st.metric("Sharpe (Inception)", inc.get("Sharpe (Inception)", "N/A"), help="Sharpe since Jan 26, 2026")
                    with mc6[1]: st.metric("Alpha (Inception)", f"{inc.get('Alpha (Inception)', 0)*100:.2f}%", help="Alpha since Jan 26, 2026")

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
                        # also compute correlation matrix
                        avail_corr = [t for t in portfolio["ticker"].tolist() if t in prices_ctr.columns]
                        if len(avail_corr) >= 2:
                            corr_matrix = prices_ctr[avail_corr].pct_change().dropna().corr()
                            st.session_state["corr_matrix"] = corr_matrix
                    if ctr_df.empty: st.warning("Need at least 2 positions for CTR.")
                    else: st.session_state["ctr_data"] = ctr_df

                left_col, right_col = st.columns(2)
                with left_col:
                    st.bar_chart(display_df.set_index("Ticker")["Weight (%)"])
                with right_col:
                    if "ctr_data" in st.session_state:
                        ctr = st.session_state["ctr_data"]
                        st.bar_chart(ctr.set_index("Ticker")["CTR (%)"])
                    else:
                        st.caption("Click 'Calculate CTR' to see risk contributions.")

                # correlation heatmap
                if "corr_matrix" in st.session_state:
                    st.divider()
                    st.markdown("**Correlation Matrix**")
                    corr = st.session_state["corr_matrix"]
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                        colorscale=[[0, "#f85149"], [0.5, "#0d1117"], [1, "#3fb950"]],
                        zmid=0, zmin=-1, zmax=1,
                        text=corr.round(2).values, texttemplate="%{text}",
                        textfont=dict(size=10),
                        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
                        colorbar=dict(title="ρ"),
                    ))
                    fig_corr.update_layout(
                        height=max(400, len(corr) * 35),
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis=dict(side="bottom"),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)


# === TAB 2: TRADE HISTORY ===
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
        sm2.metric("Total Bought", f"${total_buys:,.0f}")
        sm3.metric("Total Sold/Shorted", f"${total_sells:,.0f}")
        sm4.metric("Net Flow", f"${total_buys - total_sells:,.0f}")
        sm5.metric("Realised P&L", f"${total_realised:,.2f}")

        show_cols = ["CreateDate", "Symbol", "CompanyName", "TransactionType",
                     "Quantity", "Price", "Amount", "Exchange", "Currency", "SecurityType"]
        available_cols = [c for c in show_cols if c in filtered.columns]
        show_df = filtered[available_cols].sort_values("CreateDate", ascending=False).copy()

        if "CreateDate" in show_df.columns:
            show_df["CreateDate"] = pd.to_datetime(show_df["CreateDate"], errors="coerce").dt.strftime("%m/%d/%Y")

        st.dataframe(show_df.style.format({
            "Price": "${:,.2f}", "Amount": "${:,.2f}", "Quantity": "{:,.0f}",
        }), use_container_width=True, hide_index=True,
        column_config={
            "CreateDate": st.column_config.TextColumn("CreateDate", width="small"),
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "CompanyName": st.column_config.TextColumn("CompanyName", width="large"),
            "TransactionType": st.column_config.TextColumn("TransactionType", width="small"),
            "Quantity": st.column_config.NumberColumn("Quantity", width="medium", format="%d"),
            "Price": st.column_config.NumberColumn("Price", width="medium", format="$%.2f"),
            "Amount": st.column_config.NumberColumn("Amount", width="medium", format="$%.2f"),
            "Exchange": st.column_config.TextColumn("Exchange", width="small"),
            "Currency": st.column_config.TextColumn("Currency", width="small"),
            "SecurityType": st.column_config.TextColumn("SecurityType", width="small"),
        })

        # --- Trade Journal ---
        st.divider()
        st.markdown("**Trade Journal**")
        st.caption("AI-generated notes for each trade. Cached locally — only new trades hit the API.")

        import json, os
        NOTES_FILE = "trade_notes.json"

        def _load_notes():
            if os.path.exists(NOTES_FILE):
                try:
                    with open(NOTES_FILE, "r") as f: return json.load(f)
                except: pass
            return {}

        def _save_notes(notes):
            with open(NOTES_FILE, "w") as f: json.dump(notes, f, indent=2)

        def _trade_key(row):
            return f"{row.get('CreateDate', '')}_{row.get('Symbol', '')}_{row.get('TransactionType', '')}_{row.get('Quantity', '')}"

        existing_notes = _load_notes()

        # find trades without notes
        all_trade_keys = []
        missing_trades = []
        for _, row in trade_history.iterrows():
            key = _trade_key(row)
            all_trade_keys.append(key)
            if key not in existing_notes:
                missing_trades.append(row)

        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
        jn_c1, jn_c2 = st.columns([3, 1])
        with jn_c1:
            if missing_trades:
                st.caption(f"{len(missing_trades)} trades without notes. Click to generate.")
            else:
                st.caption(f"All {len(all_trade_keys)} trades have notes.")
        with jn_c2:
            gen_notes_btn = st.button("Generate Notes", key="gen_notes_btn",
                type="primary", use_container_width=True,
                disabled=not gemini_key or len(missing_trades) == 0)

        if gen_notes_btn and missing_trades:
            with st.spinner(f"Generating notes for {len(missing_trades)} trades..."):
                try:
                    import requests as req
                    import time

                    all_notes = []
                    # batch in chunks of 15 to avoid token limits
                    chunk_size = 15
                    chunks = [missing_trades[i:i+chunk_size] for i in range(0, len(missing_trades), chunk_size)]

                    for chunk_idx, chunk in enumerate(chunks):
                        trade_lines = []
                        for row in chunk:
                            trade_lines.append(
                                f"- {row.get('TransactionType', 'Buy')} {abs(row.get('Quantity', 0)):.0f} "
                                f"{row.get('Symbol', '?')} ({row.get('CompanyName', '')}) "
                                f"@ ${row.get('Price', 0):,.2f} on {row.get('CreateDate', '?')}")

                        prompt = (
                            "You are a portfolio analyst writing concise trade journal notes. "
                            "For each trade below, write ONE line (15-25 words) explaining the likely rationale — "
                            "what sector/theme exposure it adds, whether it's a hedge, value play, momentum bet, "
                            "profit-taking, or rebalancing. Be specific about the company's business.\n\n"
                            "Format: return ONLY a JSON array of strings, one note per trade, in the same order. "
                            "No markdown, no backticks, just raw JSON.\n\n"
                            "Trades:\n" + "\n".join(trade_lines)
                        )

                        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
                        payload = {
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}
                        }

                        resp = req.post(url, json=payload, timeout=90)
                        resp.raise_for_status()
                        data = resp.json()
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()

                        if raw_text.startswith("```"): raw_text = raw_text.split("\n", 1)[1]
                        if raw_text.endswith("```"): raw_text = raw_text.rsplit("```", 1)[0]
                        raw_text = raw_text.strip()

                        chunk_notes = json.loads(raw_text)
                        all_notes.extend(chunk_notes)

                        # rate limit pause between chunks
                        if chunk_idx < len(chunks) - 1:
                            time.sleep(2)

                    for i, row in enumerate(missing_trades):
                        key = _trade_key(row)
                        if i < len(all_notes):
                            existing_notes[key] = all_notes[i]
                        else:
                            existing_notes[key] = "Note generation incomplete."

                    _save_notes(existing_notes)
                    st.success(f"Generated {len(missing_trades)} notes.")
                    st.rerun()
                except Exception as e:
                    err_msg = str(e)
                    if gemini_key and gemini_key in err_msg:
                        err_msg = err_msg.replace(gemini_key, "***")
                    st.error(f"Note generation failed: {err_msg}")

        # display notes
        if existing_notes:
            notes_display = []
            for _, row in filtered.sort_values("CreateDate", ascending=False).iterrows():
                key = _trade_key(row)
                date_str = pd.to_datetime(row.get("CreateDate"), errors="coerce")
                date_fmt = date_str.strftime("%m/%d/%Y") if pd.notna(date_str) else str(row.get("CreateDate", ""))
                note = existing_notes.get(key, "")
                if note:
                    notes_display.append({
                        "Date": date_fmt,
                        "Ticker": row.get("Symbol", ""),
                        "Action": row.get("TransactionType", ""),
                        "Qty": abs(row.get("Quantity", 0)),
                        "Note": note,
                    })
            if notes_display:
                notes_df = pd.DataFrame(notes_display)
                st.dataframe(notes_df, use_container_width=True, hide_index=True,
                    column_config={
                        "Date": st.column_config.TextColumn("Date", width="small"),
                        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                        "Action": st.column_config.TextColumn("Action", width="small"),
                        "Qty": st.column_config.NumberColumn("Qty", width="small", format="%d"),
                        "Note": st.column_config.TextColumn("Note", width="large"),
                    })
                csv_data = notes_df.to_csv(index=False)
                st.download_button("Download Notes CSV", data=csv_data,
                    file_name="trade_journal.csv", mime="text/csv", key="dl_notes_csv")


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
                    live_price = None
                    try:
                        lp = yf.Ticker(hypo_ticker).fast_info.last_price
                        if lp and lp > 0: live_price = round(lp, 2)
                    except: pass
                    use_price = live_price if live_price else hypo_price
                    if not live_price:
                        st.warning(f"Could not fetch live price for '{hypo_ticker}'. Using manual price ${hypo_price:,.2f}.")
                    existing = [h for h in st.session_state["sim_hypothetical"] if h["ticker"] == hypo_ticker]
                    if existing:
                        idx = st.session_state["sim_hypothetical"].index(existing[0])
                        old = existing[0]
                        ns = old["shares"] + hypo_shares
                        if ns == 0: st.session_state["sim_hypothetical"].pop(idx)
                        else:
                            tc = abs(old["shares"]) * old["avg_buy_price"] + abs(hypo_shares) * use_price
                            tq = abs(old["shares"]) + abs(hypo_shares)
                            st.session_state["sim_hypothetical"][idx] = {
                                "ticker": hypo_ticker, "shares": ns,
                                "avg_buy_price": round(tc/tq, 4), "date_added": old["date_added"]}
                    else:
                        st.session_state["sim_hypothetical"].append({
                            "ticker": hypo_ticker, "shares": hypo_shares,
                            "avg_buy_price": use_price, "date_added": "hypothetical"})
                    st.rerun()

        if st.session_state["sim_hypothetical"]:
            st.markdown("**Hypothetical positions:**")
            hypo_df = pd.DataFrame(st.session_state["sim_hypothetical"])
            hd = hypo_df[["ticker", "shares", "avg_buy_price"]].copy()
            hd["Type"] = hd["shares"].apply(lambda s: "Short" if s < 0 else "Long")
            st.dataframe(hd.rename(columns={"ticker": "Ticker", "shares": "Shares", "avg_buy_price": "Price ($)"}).style.format({
                "Price ($)": "${:,.2f}",
            }), use_container_width=True, hide_index=True)

            sim_proceeds = 0.0
            for h in st.session_state["sim_hypothetical"]:
                if h["shares"] < 0:
                    try: cur_price = yf.Ticker(h["ticker"]).fast_info.last_price
                    except: cur_price = h["avg_buy_price"]
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
                hypo_rows = st.session_state["sim_hypothetical"]

                sim_dict = {}
                for _, row in base_rows.iterrows():
                    sim_dict[row["ticker"]] = {
                        "ticker": row["ticker"], "shares": row["shares"],
                        "avg_buy_price": row["avg_buy_price"],
                        "date_added": row.get("date_added", "")}
                for h in hypo_rows:
                    t = h["ticker"]
                    if t in sim_dict:
                        sim_dict[t]["shares"] += h["shares"]
                        if abs(sim_dict[t]["shares"]) < 0.0001:
                            del sim_dict[t]
                    else:
                        sim_dict[t] = {"ticker": t, "shares": h["shares"],
                            "avg_buy_price": h["avg_buy_price"],
                            "date_added": h.get("date_added", "hypothetical")}

                sim_portfolio = pd.DataFrame(list(sim_dict.values())) if sim_dict else pd.DataFrame(columns=["ticker", "shares", "avg_buy_price", "date_added"])
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

        sim_proceeds = st.session_state.get("sim_proceeds", 0.0)
        default_long_pct = 73
        default_short_pct = 27
        if not portfolio.empty:
            tmv = 0; long_val = 0; short_val = 0
            for t in portfolio["ticker"].tolist():
                try:
                    p = yf.Ticker(t).fast_info.last_price
                    s = portfolio.loc[portfolio["ticker"] == t, "shares"].iloc[0]
                    mv = p * s; tmv += mv
                    if s > 0: long_val += mv
                    else: short_val += abs(mv)
                except: pass
            cap_result = calculate_current_capital(tmv)
            default_capital = int(cap_result[0])
            default_capital += int(sim_proceeds)
            total_cap = default_capital if default_capital > 0 else 1
            default_long_pct = min(100, round(long_val / total_cap * 100))
            default_short_pct = min(100, round(short_val / total_cap * 100))
        else:
            default_capital = 1_000_000

        opt_c1, opt_c2, opt_c3, opt_c4, opt_c5 = st.columns(5)
        with opt_c1: opt_capital = st.number_input("Total Capital ($)", value=default_capital, step=10000, key="opt_capital")
        with opt_c2: opt_long_pct = st.number_input("Max Long %", value=default_long_pct, min_value=0, max_value=100, step=1, key="opt_long_pct")
        with opt_c3: opt_short_pct = st.number_input("Max Short %", value=default_short_pct, min_value=0, max_value=100, step=1, key="opt_short_pct")
        with opt_c4: opt_max_pos = st.number_input("Max Per Position ($)", value=90000, step=5000, key="opt_max_pos")
        with opt_c5: opt_min_deploy = st.number_input("Min Deploy %", value=95, min_value=50, max_value=100, step=1, key="opt_min_deploy")

        if opt_long_pct + opt_short_pct > 100:
            st.info(f"Note: Long + Short exposure = {opt_long_pct + opt_short_pct}%. This is expected for long-short portfolios using leverage.")

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

                        # compute current portfolio VaR/CVaR/cash/long%/short%
                        cur_var95, cur_cvar95 = None, None
                        if len(cur_port_ret) > 30:
                            cur_sim_daily = np.random.normal(cur_ret/252, cur_vol/np.sqrt(252), 10000)
                            cur_var95 = round(float(np.percentile(cur_sim_daily, 5)), 6)
                            cur_cvar95 = round(float(np.mean(cur_sim_daily[cur_sim_daily <= cur_var95])), 6)

                        cur_cash = cash_balance if 'cash_balance' in dir() else None
                        cur_long_pct_val = round(long_val / total_cap * 100, 1) if total_cap > 0 else None
                        cur_short_pct_val = round(short_val / total_cap * 100, 1) if total_cap > 0 else None

                        opt_result = run_markowitz_optimization(
                            long_tickers=opt_long_list, short_tickers=opt_short_list,
                            prices=opt_prices, rf=opt_rf, total_capital=opt_capital,
                            max_long_pct=opt_long_pct, max_short_pct=opt_short_pct,
                            max_per_position=opt_max_pos, min_deploy_pct=opt_min_deploy)
                    if "error" in opt_result: st.error(opt_result["error"])
                    else:
                        opt_result["current"] = {
                            "return": round(cur_ret, 4), "volatility": round(cur_vol, 4),
                            "sharpe": round(cur_sh, 4),
                            "var_95": cur_var95, "cvar_95": cur_cvar95,
                            "cash": cur_cash,
                            "long_pct": cur_long_pct_val, "short_pct": cur_short_pct_val,
                        }
                        # store optimizer internals for custom targets
                        opt_result["_internals"] = {
                            "prices": opt_prices, "rf": opt_rf,
                            "long_tickers": opt_long_list, "short_tickers": opt_short_list,
                            "total_capital": opt_capital, "max_long_pct": opt_long_pct,
                            "max_short_pct": opt_short_pct, "max_per_position": opt_max_pos,
                            "min_deploy_pct": opt_min_deploy,
                        }
                        st.session_state["markowitz_results"] = opt_result

            if "markowitz_results" in st.session_state:
                mk = st.session_state["markowitz_results"]
                ms = mk["max_sharpe"]
                mv = mk["min_variance"]
                uc = mk["unconstrained"]
                cur_m = mk["current"]
                ca = mk["constraint_analysis"]

                # efficient frontier chart
                st.markdown("**Efficient Frontier**")
                frontier = mk["frontier"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=frontier["volatilities"], y=frontier["returns"], mode="markers",
                    marker=dict(size=3, color=frontier["sharpes"], colorscale="Viridis", showscale=True,
                                colorbar=dict(title="Sharpe")),
                    name="Simulated", hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}"))

                # efficient frontier curve
                if "frontier_curve" in mk:
                    fc = mk["frontier_curve"]
                    fig.add_trace(go.Scatter(x=fc["volatilities"], y=fc["returns"],
                        mode="lines", line=dict(color="#64ffda", width=2.5, dash="solid"),
                        name="Efficient Frontier", hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}"))

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

                # --- Custom Target Allocation ---
                st.markdown("**Custom Target Allocation**")
                st.caption("Find the optimal allocation for a specific volatility or return target.")
                ct1, ct2 = st.columns(2)
                with ct1:
                    target_vol = st.number_input("Target Volatility (%)", min_value=1.0, max_value=50.0,
                        value=round(cur_m["volatility"] * 100, 1), step=0.5, key="target_vol")
                    find_vol = st.button("Find Max Return", key="find_vol_btn", use_container_width=True)
                with ct2:
                    target_ret = st.number_input("Target Return (%)", min_value=-20.0, max_value=200.0,
                        value=round(cur_m["return"] * 100, 1), step=1.0, key="target_ret")
                    find_ret = st.button("Find Min Volatility", key="find_ret_btn", use_container_width=True)

                # run custom optimization
                if (find_vol or find_ret) and "_internals" in mk:
                    internals = mk["_internals"]
                    with st.spinner("Finding optimal allocation..."):
                        if find_vol:
                            custom_result = _run_custom_target_optimization(
                                internals, mode="target_vol", target_value=target_vol / 100)
                        else:
                            custom_result = _run_custom_target_optimization(
                                internals, mode="target_ret", target_value=target_ret / 100)

                    if custom_result and "error" not in custom_result:
                        st.session_state["custom_target_result"] = custom_result
                    elif custom_result:
                        st.warning(custom_result["error"])

                if "custom_target_result" in st.session_state:
                    ctr_result = st.session_state["custom_target_result"]
                    ctr_c1, ctr_c2, ctr_c3, ctr_c4 = st.columns(4)
                    ctr_c1.metric("Expected Return", f"{ctr_result['return']*100:.2f}%")
                    ctr_c2.metric("Volatility", f"{ctr_result['volatility']*100:.2f}%")
                    ctr_c3.metric("Sharpe Ratio", f"{ctr_result['sharpe']:.4f}")
                    ctr_c4.metric("Cash Held", f"${ctr_result.get('cash', 0):,.0f}")
                    if not ctr_result["table"].empty:
                        tbl_fmt = {"Dollar Amount ($)": "${:,.2f}", "Weight (%)": "{:.2f}"}
                        st.dataframe(ctr_result["table"].style.format(tbl_fmt),
                            use_container_width=True, hide_index=True)

                # constraint analysis box
                if ca["is_limiting"]:
                    st.info(
                        f"**Constraint Analysis**\n\n"
                        f"Unconstrained Optimal: **{ca['uc_long_pct']}% Long / {ca['uc_short_pct']}% Short**\n\n"
                        f"Your Limits: **{ca['constrained_long_pct']}% Long / {ca['constrained_short_pct']}% Short**\n\n"
                        f"Your constraints are limiting potential returns. "
                        f"Unconstrained Sharpe: **{ca['uc_sharpe']:.2f}** vs Constrained: **{ca['constrained_sharpe']:.2f}** "
                        f"(gap: {ca['sharpe_gap']:.4f})")
                else:
                    st.success(
                        f"**Constraint Analysis**\n\n"
                        f"Unconstrained Optimal: **{ca['uc_long_pct']}% Long / {ca['uc_short_pct']}% Short**\n\n"
                        f"Your Limits: **{ca['constrained_long_pct']}% Long / {ca['constrained_short_pct']}% Short**\n\n"
                        f"Your constraints are not limiting the optimizer.")

                # metrics comparison
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

                # allocation tables
                st.divider()
                all_input_tickers = set(opt_long_list + opt_short_list)
                def _get_allocated(table):
                    if table.empty: return set()
                    return set(table[table["Ticker"] != "CASH"]["Ticker"].tolist())

                ms_allocated = _get_allocated(ms["table"])
                mv_allocated = _get_allocated(mv["table"])
                uc_allocated = _get_allocated(uc["table"])
                ms_zeroed = all_input_tickers - ms_allocated
                mv_zeroed = all_input_tickers - mv_allocated
                uc_zeroed = all_input_tickers - uc_allocated
                all_zeroed = ms_zeroed | mv_zeroed | uc_zeroed
                if all_zeroed:
                    zeroed_notes = []
                    for t in sorted(all_zeroed):
                        scenarios = []
                        if t in ms_zeroed: scenarios.append("Max Sharpe")
                        if t in mv_zeroed: scenarios.append("Min Variance")
                        if t in uc_zeroed: scenarios.append("Unconstrained")
                        zeroed_notes.append(f"**{t}** → 0% in {', '.join(scenarios)}")
                    st.warning("Some positions received zero allocation: " + " · ".join(zeroed_notes))

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
