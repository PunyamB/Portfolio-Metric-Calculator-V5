import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

BENCHMARK = "^GSPC"
RISK_FREE_TICKER = "^IRX"
LOOKBACK_YEARS = 3
INCEPTION_DATE = "2026-01-26"
FF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


# --- data fetching ---

def fetch_price_history(tickers, years=None):
    if years is None:
        years = LOOKBACK_YEARS
    end = datetime.today()
    start = end - timedelta(days=365 * years)
    all_tickers = list(set(tickers + [BENCHMARK]))
    raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame(
            {t: raw[(t, "Close")] for t in all_tickers if t in raw.columns.get_level_values(0)},
            index=raw.index)
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers
    return prices.dropna(how="all")


def fetch_price_history_from_date(tickers, start_date):
    all_tickers = list(set(tickers + [BENCHMARK]))
    raw = yf.download(all_tickers, start=start_date, end=datetime.today(),
                      auto_adjust=True, progress=False, group_by="ticker")
    if isinstance(raw.columns, pd.MultiIndex):
        prices = pd.DataFrame(
            {t: raw[(t, "Close")] for t in all_tickers if t in raw.columns.get_level_values(0)},
            index=raw.index)
    else:
        prices = raw[["Close"]].copy()
        prices.columns = all_tickers
    return prices.dropna(how="all")


def get_risk_free_rate():
    try:
        irx = yf.Ticker(RISK_FREE_TICKER)
        hist = irx.history(period="5d")
        return hist["Close"].iloc[-1] / 100
    except Exception:
        return 0.04


def fetch_fama_french_factors(years=None):
    if years is None:
        years = LOOKBACK_YEARS
    try:
        df = pd.read_csv(FF_URL, skiprows=3, index_col=0)
        df = df[df.index.astype(str).str.len() == 8]
        df.index = pd.to_datetime(df.index, format="%Y%m%d")
        df = df.apply(pd.to_numeric, errors="coerce") / 100
        end = datetime.today()
        start = end - timedelta(days=365 * years)
        df = df.loc[start:end]
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


# --- portfolio returns ---

def compute_portfolio_returns(prices, holdings):
    # aggregate duplicate tickers (e.g. base + hypothetical for same stock)
    agg = holdings.groupby("ticker")["shares"].sum().reset_index()
    agg = agg[agg["shares"].abs() > 0.0001]  # drop zeroed-out positions
    tickers = agg["ticker"].tolist()
    available = [t for t in tickers if t in prices.columns]
    if not available:
        return pd.Series(dtype=float)
    price_subset = prices[available].dropna()
    shares = agg.set_index("ticker")["shares"][available]
    market_values = price_subset.multiply(shares, axis=1)
    total_value = market_values.sum(axis=1)
    mask = total_value.abs() > 0
    total_value = total_value[mask]
    market_values = market_values.loc[mask]
    weights = market_values.divide(total_value, axis=0)
    daily_returns = price_subset.loc[mask].pct_change().dropna()
    return (daily_returns * weights.shift(1)).sum(axis=1).dropna()


# --- individual metrics ---

def calculate_variance(r): return float(r.var())
def calculate_std_dev_daily(r): return float(r.std())
def calculate_std_dev_annual(r): return float(r.std() * np.sqrt(252))
def calculate_skewness(r): return float(r.skew())
def calculate_kurtosis(r): return float(r.kurtosis())

def calculate_cagr(prices, holdings):
    tickers = [t for t in holdings["ticker"].tolist() if t in prices.columns]
    if not tickers: return 0.0
    shares = holdings.set_index("ticker")["shares"][tickers]
    pv = prices[tickers].dropna().multiply(shares, axis=1).sum(axis=1)
    start_val, end_val = pv.iloc[0], pv.iloc[-1]
    n_years = len(pv) / 252
    if start_val <= 0 or n_years <= 0: return 0.0
    return float((end_val / start_val) ** (1 / n_years) - 1)

def calculate_max_drawdown(r):
    cum = (1 + r).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())

def calculate_sharpe(r, rf):
    daily_rf = rf / 252
    excess = r - daily_rf
    if excess.std() == 0: return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))

def calculate_sharpe_daily(r, rf):
    # non-annualised sharpe: mean excess return / std of returns
    daily_rf = rf / 252
    excess = r - daily_rf
    if excess.std() == 0: return 0.0
    return float(excess.mean() / excess.std())

def calculate_sortino(r, rf):
    daily_rf = rf / 252
    excess = r - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0: return 0.0
    return float((excess.mean() / downside.std()) * np.sqrt(252))

def calculate_calmar(cagr, max_dd):
    if max_dd == 0: return 0.0
    return float(cagr / abs(max_dd))

def calculate_var(r, confidence=0.95):
    return float(r.quantile(1 - confidence))

def calculate_cvar(r, confidence=0.95):
    var = calculate_var(r, confidence)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) > 0 else var

def calculate_beta_alpha(port_r, bench_r):
    aligned = pd.concat([port_r, bench_r], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    if len(aligned) < 10: return 0.0, 0.0
    cov = np.cov(aligned["portfolio"], aligned["benchmark"])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0.0
    alpha_daily = aligned["portfolio"].mean() - beta * aligned["benchmark"].mean()
    return float(beta), float(alpha_daily * 252)

def calculate_alpha_daily(port_r, bench_r):
    # non-annualised alpha: daily excess return over beta * benchmark
    aligned = pd.concat([port_r, bench_r], axis=1).dropna()
    aligned.columns = ["portfolio", "benchmark"]
    if len(aligned) < 10: return 0.0
    cov = np.cov(aligned["portfolio"], aligned["benchmark"])
    beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0.0
    return float(aligned["portfolio"].mean() - beta * aligned["benchmark"].mean())

def calculate_turnover(buys, sells, avg_value):
    if avg_value <= 0: return 0.0
    return float((buys + sells) / 2 / avg_value)


# --- inception metrics ---

def calculate_inception_metrics(holdings, rf):
    tickers = holdings["ticker"].tolist()
    try:
        prices = fetch_price_history_from_date(tickers, INCEPTION_DATE)
    except Exception:
        return {}
    port_ret = compute_portfolio_returns(prices, holdings)
    if port_ret.empty or len(port_ret) < 5:
        return {}
    bench_ret = prices[BENCHMARK].pct_change().dropna()
    beta, alpha = calculate_beta_alpha(port_ret, bench_ret)
    sharpe = calculate_sharpe(port_ret, rf)
    return {
        "Sharpe (Inception)": round(sharpe, 4),
        "Beta (Inception)": round(beta, 4),
        "Alpha (Inception)": round(alpha, 4),
    }


# --- factor exposures (FF 5-factor) ---

def _interpret_factor(name, exposure, t_stat, significant):
    sig_label = "Strong" if abs(t_stat) > 3.0 else ("Moderate" if significant else "Weak (not significant)")
    interp = {
        "Mkt-RF": {"high_pos": f"{sig_label} — Aggressive market exposure",
                    "neutral": f"{sig_label} — Near market-neutral (~1.0)",
                    "low": f"{sig_label} — Defensive, less market sensitivity"},
        "SMB": {"pos": f"{sig_label} — Tilted toward small-caps",
                "neg": f"{sig_label} — Tilted toward large-caps",
                "zero": f"{sig_label} — No size bias"},
        "HML": {"pos": f"{sig_label} — Value tilt (cheap stocks)",
                "neg": f"{sig_label} — Growth tilt (expensive stocks)",
                "zero": f"{sig_label} — No value/growth bias"},
        "RMW": {"pos": f"{sig_label} — Favours profitable firms",
                "neg": f"{sig_label} — Exposed to weak-profitability firms",
                "zero": f"{sig_label} — No profitability bias"},
        "CMA": {"pos": f"{sig_label} — Conservative firms (low investment)",
                "neg": f"{sig_label} — Aggressive growth firms (high investment)",
                "zero": f"{sig_label} — No investment style bias"},
    }
    if name == "Mkt-RF":
        if exposure > 1.2: return interp[name]["high_pos"]
        elif exposure < 0.8: return interp[name]["low"]
        else: return interp[name]["neutral"]
    thresholds = {"SMB": 0.1, "HML": 0.1, "RMW": 0.08, "CMA": 0.08}
    thresh = thresholds.get(name, 0.1)
    if name in interp:
        if exposure > thresh: return interp[name]["pos"]
        elif exposure < -thresh: return interp[name]["neg"]
        else: return interp[name]["zero"]
    return f"Exposure: {exposure:.4f}"


def _get_optimal_range(name):
    return {"Mkt-RF": "0.8 to 1.2", "SMB": "-0.5 to +0.5", "HML": "-0.5 to +0.5",
            "RMW": "-0.3 to +0.3", "CMA": "-0.3 to +0.3"}.get(name, "-")


def calculate_factor_exposures(portfolio_returns):
    ff = fetch_fama_french_factors()
    if ff.empty:
        return {"error": "Could not fetch Fama-French data."}
    aligned = pd.concat([portfolio_returns, ff], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return {"error": "Insufficient overlapping data for factor regression."}
    y = aligned.iloc[:, 0] - aligned["RF"]
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    available_factors = [f for f in factors if f in aligned.columns]
    if not available_factors:
        return {"error": "Factor columns not found."}
    X = aligned[available_factors].values
    X_const = np.column_stack([np.ones(len(X)), X])
    y_arr = y.values
    try:
        betas = np.linalg.lstsq(X_const, y_arr, rcond=None)[0]
        y_pred = X_const @ betas
        residuals = y_arr - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n, k = len(y_arr), len(betas)
        mse = ss_res / (n - k)
        var_b = mse * np.linalg.inv(X_const.T @ X_const)
        se = np.sqrt(np.diag(var_b))
        t_stats = betas / se
        result = {"R_squared": round(r_sq, 4), "Alpha_annual": round(betas[0] * 252, 6)}
        for i, f in enumerate(available_factors):
            exp = round(betas[i + 1], 4)
            ts = round(t_stats[i + 1], 2)
            sig = abs(ts) > 2.0
            result[f] = {"exposure": exp, "t_stat": ts, "significant": sig,
                         "interpretation": _interpret_factor(f, exp, ts, sig),
                         "optimal_range": _get_optimal_range(f)}
        return result
    except Exception as e:
        return {"error": str(e)}


# --- contribution to risk ---

def calculate_ctr(prices, holdings):
    tickers = [t for t in holdings["ticker"].tolist() if t in prices.columns]
    if len(tickers) < 2: return pd.DataFrame()
    shares = holdings.set_index("ticker")["shares"][tickers]
    price_subset = prices[tickers].dropna()
    latest_prices = price_subset.iloc[-1]
    market_values = shares * latest_prices
    total_value = market_values.sum()
    if abs(total_value) < 0.01: return pd.DataFrame()
    weights = market_values / total_value
    daily_returns = price_subset.pct_change().dropna()
    cov_matrix = daily_returns.cov() * 252
    port_var = float(weights.values @ cov_matrix.values @ weights.values)
    port_std = np.sqrt(abs(port_var))
    if port_std == 0: return pd.DataFrame()
    marginal = (cov_matrix.values @ weights.values) / port_std
    ctr = weights.values * marginal
    ctr_pct = ctr / port_std * 100
    return pd.DataFrame({"Ticker": tickers, "Weight (%)": (weights.values * 100).round(2),
                          "Marginal CTR": np.round(marginal, 6), "CTR": np.round(ctr, 6),
                          "CTR (%)": np.round(ctr_pct, 2)})


# --- master metric runners ---

def _compute_metrics_from_prices(holdings, prices, rf, turnover_inputs=None):
    portfolio_returns = compute_portfolio_returns(prices, holdings)
    if portfolio_returns.empty or len(portfolio_returns) < 30:
        return {"error": "Insufficient price data. Need at least 30 trading days."}
    bench_returns = prices[BENCHMARK].pct_change().dropna()
    beta, alpha = calculate_beta_alpha(portfolio_returns, bench_returns)
    alpha_daily = calculate_alpha_daily(portfolio_returns, bench_returns)
    sharpe_daily = calculate_sharpe_daily(portfolio_returns, rf)
    cagr = calculate_cagr(prices, holdings)
    max_dd = calculate_max_drawdown(portfolio_returns)
    metrics = {
        "Variance (Daily)": round(calculate_variance(portfolio_returns), 6),
        "Std Dev (Daily)": round(calculate_std_dev_daily(portfolio_returns), 6),
        "Std Dev (Annual)": round(calculate_std_dev_annual(portfolio_returns), 4),
        "Skewness": round(calculate_skewness(portfolio_returns), 4),
        "Kurtosis": round(calculate_kurtosis(portfolio_returns), 4),
        "CAGR": round(cagr, 4),
        "Max Drawdown": round(max_dd, 4),
        "Sharpe Ratio": round(calculate_sharpe(portfolio_returns, rf), 4),
        "Sharpe (Daily)": round(sharpe_daily, 4),
        "Sortino Ratio": round(calculate_sortino(portfolio_returns, rf), 4),
        "Calmar Ratio": round(calculate_calmar(cagr, max_dd), 4),
        "VaR (95%)": round(calculate_var(portfolio_returns), 6),
        "CVaR (95%)": round(calculate_cvar(portfolio_returns), 6),
        "Beta": round(beta, 4),
        "Alpha (Annual)": round(alpha, 4),
        "Alpha (Daily)": round(alpha_daily, 6),
        "Risk-Free Rate Used": round(rf, 4),
    }
    if turnover_inputs:
        metrics["Turnover Ratio"] = round(calculate_turnover(
            turnover_inputs.get("buys", 0), turnover_inputs.get("sells", 0),
            turnover_inputs.get("avg_portfolio_value", 1)), 4)
    return metrics


def run_all_metrics(holdings, turnover_inputs=None):
    if holdings.empty: return {}
    prices = fetch_price_history(holdings["ticker"].tolist())
    rf = get_risk_free_rate()
    return _compute_metrics_from_prices(holdings, prices, rf, turnover_inputs)


def run_all_metrics_with_prices(holdings, prices, rf, turnover_inputs=None):
    if holdings.empty: return {}
    return _compute_metrics_from_prices(holdings, prices, rf, turnover_inputs)


# --- markowitz portfolio optimization ---

def _run_single_optimization(mean_returns, cov_matrix, rf, n, long_idx, short_idx,
                              bounds, constraints, w0):
    from scipy.optimize import minimize

    def portfolio_return(w): return float(w @ mean_returns.values)
    def portfolio_volatility(w): return float(np.sqrt(abs(w @ cov_matrix.values @ w)))
    def neg_sharpe(w):
        r, v = portfolio_return(w), portfolio_volatility(w)
        return -(r - rf) / v if v > 0 else 0

    ms = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                  constraints=constraints, options={"maxiter": 1000})
    mv = minimize(portfolio_volatility, w0, method="SLSQP", bounds=bounds,
                  constraints=constraints, options={"maxiter": 1000})
    return (ms.x if ms.success else w0), (mv.x if mv.success else w0)


def run_markowitz_optimization(
    long_tickers, short_tickers, prices, rf,
    total_capital=1_000_000, max_long_pct=73, max_short_pct=27,
    max_per_position=90_000, min_deploy_pct=95, n_simulations=10000,
):
    from scipy.optimize import minimize

    all_tickers = long_tickers + short_tickers
    available = [t for t in all_tickers if t in prices.columns]
    if len(available) < 2:
        return {"error": "Need at least 2 tickers for optimization."}

    daily_returns = prices[available].pct_change().dropna()
    if len(daily_returns) < 30:
        return {"error": "Insufficient price data for optimization."}

    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252

    n = len(available)
    long_idx = [i for i, t in enumerate(available) if t in long_tickers]
    short_idx = [i for i, t in enumerate(available) if t in short_tickers]

    max_long = max_long_pct / 100
    max_short = max_short_pct / 100
    max_w = max_per_position / total_capital if total_capital > 0 else 0.1
    min_deploy = min_deploy_pct / 100

    def portfolio_return(w): return float(w @ mean_returns.values)
    def portfolio_volatility(w): return float(np.sqrt(abs(w @ cov_matrix.values @ w)))

    # --- constrained optimization ---
    c_bounds = []
    for i in range(n):
        if i in long_idx:
            c_bounds.append((0.0, min(max_w, max_long)))
        else:
            c_bounds.append((-min(max_w, max_short), 0.0))
    c_bounds = tuple(c_bounds)

    # long sum <= max_long, long sum >= 0
    # short |sum| <= max_short, short sum <= 0
    # total deployment >= min_deploy
    c_constraints = [
        {"type": "ineq", "fun": lambda w: max_long - sum(w[i] for i in long_idx)},
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in long_idx)},
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in long_idx) + sum(-w[i] for i in short_idx) - min_deploy},
    ]
    if short_idx:
        c_constraints.extend([
            {"type": "ineq", "fun": lambda w: max_short + sum(w[i] for i in short_idx)},
            {"type": "ineq", "fun": lambda w: -sum(w[i] for i in short_idx)},
        ])

    # initial weights
    w0 = np.zeros(n)
    il = min(max_long / max(len(long_idx), 1), max_w)
    ish = min(max_short / max(len(short_idx), 1), max_w)
    for i in long_idx: w0[i] = il
    for i in short_idx: w0[i] = -ish

    ms_w, mv_w = _run_single_optimization(
        mean_returns, cov_matrix, rf, n, long_idx, short_idx, c_bounds, c_constraints, w0)

    # --- unconstrained optimization (no long/short caps, only per-position cap) ---
    u_bounds = []
    for i in range(n):
        if i in long_idx:
            u_bounds.append((0.0, max_w))
        else:
            u_bounds.append((-max_w, 0.0))
    u_bounds = tuple(u_bounds)

    u_constraints = [
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in long_idx)},
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in long_idx) + sum(-w[i] for i in short_idx) - min_deploy},
    ]
    if short_idx:
        u_constraints.append({"type": "ineq", "fun": lambda w: -sum(w[i] for i in short_idx)})

    # unconstrained initial: spread more freely
    u0 = np.zeros(n)
    for i in long_idx: u0[i] = min(0.5 / max(len(long_idx), 1), max_w)
    for i in short_idx: u0[i] = -min(0.3 / max(len(short_idx), 1), max_w)

    uc_w, _ = _run_single_optimization(
        mean_returns, cov_matrix, rf, n, long_idx, short_idx, u_bounds, u_constraints, u0)

    # compute unconstrained long/short split
    uc_long_pct = round(sum(uc_w[i] for i in long_idx) * 100, 1)
    uc_short_pct = round(sum(-uc_w[i] for i in short_idx) * 100, 1)

    # --- monte carlo frontier ---
    frontier_returns, frontier_vols, frontier_sharpes = [], [], []
    for _ in range(n_simulations):
        w = np.zeros(n)
        if long_idx:
            raw = np.random.dirichlet(np.ones(len(long_idx))) * np.random.uniform(0.3, max_long)
            raw = np.minimum(raw, max_w)
            for j, idx in enumerate(long_idx): w[idx] = raw[j]
        if short_idx:
            raw = np.random.dirichlet(np.ones(len(short_idx))) * np.random.uniform(0.05, max_short)
            raw = np.minimum(raw, max_w)
            for j, idx in enumerate(short_idx): w[idx] = -raw[j]
        ret = portfolio_return(w)
        vol = portfolio_volatility(w)
        sharpe = (ret - rf) / vol if vol > 0 else 0
        frontier_returns.append(ret)
        frontier_vols.append(vol)
        frontier_sharpes.append(sharpe)

    latest_prices = prices[available].iloc[-1]

    def build_alloc_table(weights):
        rows = []
        for i, t in enumerate(available):
            w = weights[i]
            if abs(w) < 0.0001: continue
            side = "Long" if t in long_tickers else "Short"
            dollar_amt = abs(w) * total_capital
            price = latest_prices[t]
            shares = int(dollar_amt / price) if price > 0 else 0
            rows.append({"Ticker": t, "Side": side, "Weight (%)": round(w * 100, 2),
                         "Shares": shares if side == "Long" else -shares,
                         "Dollar Amount ($)": round(dollar_amt, 2)})
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            allocated = df["Dollar Amount ($)"].sum()
            cash = total_capital - allocated
            if cash > 100:
                cash_row = pd.DataFrame([{"Ticker": "CASH", "Side": "-",
                    "Weight (%)": round(cash / total_capital * 100, 2),
                    "Shares": "-", "Dollar Amount ($)": round(cash, 2)}])
                df = pd.concat([df, cash_row], ignore_index=True)
        return df

    def calc_metrics(w):
        ret = portfolio_return(w)
        vol = portfolio_volatility(w)
        sharpe = (ret - rf) / vol if vol > 0 else 0
        sim_daily = np.random.normal(ret / 252, vol / np.sqrt(252), 10000)
        var95 = float(np.percentile(sim_daily, 5))
        cvar95 = float(np.mean(sim_daily[sim_daily <= var95]))
        long_sum = sum(w[i] for i in long_idx) * 100
        short_sum = sum(-w[i] for i in short_idx) * 100
        allocated = (sum(w[i] for i in long_idx) + sum(-w[i] for i in short_idx)) * total_capital
        cash = total_capital - allocated
        return {"return": round(ret, 4), "volatility": round(vol, 4),
                "sharpe": round(sharpe, 4), "var_95": round(var95, 6),
                "cvar_95": round(cvar95, 6), "cash": round(cash, 2),
                "long_pct": round(long_sum, 1), "short_pct": round(short_sum, 1)}

    # constraint analysis
    uc_metrics = calc_metrics(uc_w)
    ms_metrics = calc_metrics(ms_w)
    is_limiting = (uc_long_pct > max_long_pct + 1) or (uc_short_pct < max_short_pct - 1 and uc_long_pct > max_long_pct - 1)
    sharpe_gap = uc_metrics["sharpe"] - ms_metrics["sharpe"]

    constraint_analysis = {
        "uc_long_pct": uc_long_pct,
        "uc_short_pct": uc_short_pct,
        "constrained_long_pct": max_long_pct,
        "constrained_short_pct": max_short_pct,
        "uc_sharpe": uc_metrics["sharpe"],
        "constrained_sharpe": ms_metrics["sharpe"],
        "is_limiting": is_limiting or sharpe_gap > 0.05,
        "sharpe_gap": round(sharpe_gap, 4),
    }

    # --- efficient frontier curve (extract from Monte Carlo points) ---
    frontier_curve_rets, frontier_curve_vols = [], []
    if frontier_returns and frontier_vols:
        # bin by return, take min volatility in each bin
        paired = list(zip(frontier_returns, frontier_vols))
        min_ret, max_ret = min(frontier_returns), max(frontier_returns)
        n_bins = 60
        bin_width = (max_ret - min_ret) / n_bins if max_ret > min_ret else 1
        bins = {}
        for r_val, v_val in paired:
            b = int((r_val - min_ret) / bin_width)
            b = min(b, n_bins - 1)
            if b not in bins or v_val < bins[b][1]:
                bins[b] = (r_val, v_val)
        # sort by return
        sorted_pts = sorted(bins.values(), key=lambda x: x[0])
        # smooth: enforce non-decreasing vol as return increases (upper frontier only)
        for r_val, v_val in sorted_pts:
            frontier_curve_rets.append(r_val)
            frontier_curve_vols.append(v_val)

    return {
        "tickers": available,
        "max_sharpe": {**ms_metrics, "table": build_alloc_table(ms_w), "weights": ms_w},
        "min_variance": {**calc_metrics(mv_w), "table": build_alloc_table(mv_w), "weights": mv_w},
        "unconstrained": {**uc_metrics, "table": build_alloc_table(uc_w), "weights": uc_w},
        "constraint_analysis": constraint_analysis,
        "frontier": {"returns": frontier_returns, "volatilities": frontier_vols, "sharpes": frontier_sharpes},
        "frontier_curve": {"returns": frontier_curve_rets, "volatilities": frontier_curve_vols},
        "success": True,
    }


def _run_custom_target_optimization(internals, mode="target_vol", target_value=0.15):
    """Run optimizer for a custom volatility or return target.
    mode: 'target_vol' (maximize return at given vol) or 'target_ret' (minimize vol at given return)
    """
    from scipy.optimize import minimize

    prices = internals["prices"]
    rf = internals["rf"]
    long_tickers = internals["long_tickers"]
    short_tickers = internals["short_tickers"]
    total_capital = internals["total_capital"]
    max_long_pct = internals["max_long_pct"]
    max_short_pct = internals["max_short_pct"]
    max_per_position = internals["max_per_position"]
    min_deploy_pct = internals["min_deploy_pct"]

    all_tickers = long_tickers + short_tickers
    available = [t for t in all_tickers if t in prices.columns]
    if len(available) < 2:
        return {"error": "Need at least 2 tickers."}

    daily_returns = prices[available].pct_change().dropna()
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    n = len(available)
    long_idx = [i for i, t in enumerate(available) if t in long_tickers]
    short_idx = [i for i, t in enumerate(available) if t in short_tickers]

    max_long = max_long_pct / 100
    max_short = max_short_pct / 100
    max_w = max_per_position / total_capital if total_capital > 0 else 0.1
    min_deploy = min_deploy_pct / 100

    def portfolio_return(w): return float(w @ mean_returns.values)
    def portfolio_volatility(w): return float(np.sqrt(abs(w @ cov_matrix.values @ w)))

    bounds = []
    for i in range(n):
        if i in long_idx: bounds.append((0.0, min(max_w, max_long)))
        else: bounds.append((-min(max_w, max_short), 0.0))
    bounds = tuple(bounds)

    constraints = [
        {"type": "ineq", "fun": lambda w: max_long - sum(w[i] for i in long_idx)},
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in long_idx)},
        {"type": "ineq", "fun": lambda w: sum(w[i] for i in long_idx) + sum(-w[i] for i in short_idx) - min_deploy},
    ]
    if short_idx:
        constraints.extend([
            {"type": "ineq", "fun": lambda w: max_short + sum(w[i] for i in short_idx)},
            {"type": "ineq", "fun": lambda w: -sum(w[i] for i in short_idx)},
        ])

    w0 = np.zeros(n)
    il = min(max_long / max(len(long_idx), 1), max_w)
    ish = min(max_short / max(len(short_idx), 1), max_w)
    for i in long_idx: w0[i] = il
    for i in short_idx: w0[i] = -ish

    if mode == "target_vol":
        # maximize return subject to vol <= target
        constraints.append({"type": "ineq", "fun": lambda w: target_value - portfolio_volatility(w)})
        def neg_return(w): return -portfolio_return(w)
        result = minimize(neg_return, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 1000})
    else:
        # minimize vol subject to return >= target
        constraints.append({"type": "ineq", "fun": lambda w: portfolio_return(w) - target_value})
        result = minimize(portfolio_volatility, w0, method="SLSQP", bounds=bounds,
                          constraints=constraints, options={"maxiter": 1000})

    if not result.success:
        return {"error": f"Optimizer did not converge. Try a different target. ({result.message})"}

    w = result.x
    ret = portfolio_return(w)
    vol = portfolio_volatility(w)
    sharpe = (ret - rf) / vol if vol > 0 else 0

    # build allocation table
    latest_prices = prices[available].iloc[-1]
    rows = []
    for i, t in enumerate(available):
        wi = w[i]
        if abs(wi) < 0.0001: continue
        side = "Long" if t in long_tickers else "Short"
        dollar_amt = abs(wi) * total_capital
        price = latest_prices[t]
        shares = int(dollar_amt / price) if price > 0 else 0
        rows.append({"Ticker": t, "Side": side, "Weight (%)": round(wi * 100, 2),
                     "Shares": shares if side == "Long" else -shares,
                     "Dollar Amount ($)": round(dollar_amt, 2)})
    table = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not table.empty:
        allocated = table["Dollar Amount ($)"].sum()
        cash = total_capital - allocated
        if cash > 100:
            cash_row = pd.DataFrame([{"Ticker": "CASH", "Side": "-",
                "Weight (%)": round(cash / total_capital * 100, 2),
                "Shares": "-", "Dollar Amount ($)": round(cash, 2)}])
            table = pd.concat([table, cash_row], ignore_index=True)
    else:
        cash = total_capital

    return {"return": round(ret, 4), "volatility": round(vol, 4),
            "sharpe": round(sharpe, 4), "cash": round(cash, 2),
            "table": table}
