import pandas as pd
import numpy as np
import os
import re

STOCKTRAK_FILE = "stocktrak_history.csv"
STARTING_CAPITAL = 1_000_000

# --- load and parse stocktrak csv ---

def load_stocktrak_csv():
    if not os.path.exists(STOCKTRAK_FILE):
        return pd.DataFrame(columns=["Symbol", "TransactionType", "CompanyName", "Exchange",
                                      "Quantity", "Currency", "SecurityType", "Price", "Amount", "CreateDate"])

    df = pd.read_csv(STOCKTRAK_FILE, encoding="utf-8-sig")

    # strip whitespace from all column headers
    df.columns = df.columns.str.strip()

    # clean Price: strip $, commas, parens
    def clean_money(val):
        if pd.isna(val): return 0.0
        s = str(val).replace("$", "").replace(",", "").replace("(", "").replace(")", "").strip()
        try: return abs(float(s))
        except: return 0.0

    df["Price"] = df["Price"].apply(clean_money)

    # clean Quantity: always positive
    df["Quantity"] = df["Quantity"].apply(lambda x: abs(float(x)) if pd.notna(x) else 0)

    # parse TransactionType
    def parse_txn_type(val):
        if pd.isna(val): return None
        s = str(val).lower()
        if "dividend" in s: return None  # skip dividends
        if "short" in s: return "Short"
        if "cover" in s: return "Buy"  # buy to cover
        if "buy" in s: return "Buy"
        if "sell" in s: return "Sell"
        return None

    df["_parsed_type"] = df["TransactionType"].apply(parse_txn_type)
    df = df[df["_parsed_type"].notna()].copy()
    df["TransactionType"] = df["_parsed_type"]
    df.drop(columns=["_parsed_type"], inplace=True)

    # parse CreateDate: "02/11/2026 - 10:29" -> date only
    def parse_date(val):
        if pd.isna(val): return pd.NaT
        s = str(val).split(" - ")[0].strip()
        try: return pd.to_datetime(s, format="%m/%d/%Y")
        except:
            try: return pd.to_datetime(s)
            except: return pd.NaT

    df["CreateDate"] = df["CreateDate"].apply(parse_date)

    # recalculate Amount
    df["Amount"] = (df["Quantity"] * df["Price"]).round(2)

    # use SecurityType as-is from file
    # drop FXRate if present
    for col in ["FXRate", "_parsed_type"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")

    # ensure required columns exist
    for col in ["Symbol", "CompanyName", "Exchange", "Currency", "SecurityType"]:
        if col not in df.columns:
            df[col] = ""

    return df.reset_index(drop=True)


# --- portfolio from trades ---

def compute_portfolio_from_trades():
    trades = load_stocktrak_csv()
    if trades.empty:
        return pd.DataFrame(columns=["ticker", "shares", "avg_buy_price", "date_added"])

    trades_sorted = trades.sort_values("CreateDate").reset_index(drop=True)
    positions = {}

    for _, row in trades_sorted.iterrows():
        symbol = str(row["Symbol"]).upper().strip()
        txn_type = row["TransactionType"]
        qty = float(row["Quantity"])
        price = float(row["Price"])
        trade_date = row.get("CreateDate", None)

        if symbol not in positions:
            positions[symbol] = {"shares": 0.0, "cost_total": 0.0, "buy_shares": 0.0,
                                 "short_shares": 0.0, "short_cost_total": 0.0, "first_date": None}

        pos = positions[symbol]
        if trade_date is not None and pd.notna(trade_date) and pos["first_date"] is None:
            pos["first_date"] = trade_date

        if txn_type == "Buy":
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
        elif txn_type == "Sell":
            pos["shares"] -= qty
        elif txn_type == "Short":
            pos["shares"] -= qty
            pos["short_cost_total"] += qty * price
            pos["short_shares"] += qty

    rows = []
    for symbol, pos in positions.items():
        if abs(pos["shares"]) < 0.0001: continue
        if pos["shares"] > 0:
            avg = round(pos["cost_total"] / pos["buy_shares"], 4) if pos["buy_shares"] > 0 else 0.0
        else:
            avg = round(pos["short_cost_total"] / pos["short_shares"], 4) if pos["short_shares"] > 0 else 0.0
        first_dt = pos["first_date"]
        if first_dt is not None and pd.notna(first_dt):
            try: first_dt = pd.Timestamp(first_dt).strftime("%m/%d/%Y")
            except: first_dt = str(first_dt)
        else: first_dt = ""
        rows.append({"ticker": symbol, "shares": round(pos["shares"], 4),
                      "avg_buy_price": avg, "date_added": first_dt})

    if not rows:
        return pd.DataFrame(columns=["ticker", "shares", "avg_buy_price", "date_added"])
    return pd.DataFrame(rows)


# --- realised P&L ---

def compute_realised_pnl():
    trades = load_stocktrak_csv()
    if trades.empty: return pd.DataFrame(), 0.0

    trades_sorted = trades.sort_values("CreateDate").reset_index(drop=True)
    positions = {}
    pnl_rows = []

    for _, row in trades_sorted.iterrows():
        symbol = str(row["Symbol"]).upper().strip()
        txn_type = row["TransactionType"]
        qty = float(row["Quantity"])
        price = float(row["Price"])

        if symbol not in positions:
            positions[symbol] = {"shares": 0.0, "cost_total": 0.0}
        pos = positions[symbol]

        if txn_type == "Buy":
            if pos["shares"] < 0:
                avg_short = abs(pos["cost_total"] / pos["shares"]) if pos["shares"] != 0 else 0.0
                cover_qty = min(qty, abs(pos["shares"]))
                realised = (avg_short - price) * cover_qty
                pnl_rows.append({"Symbol": symbol, "SellDate": row.get("CreateDate", ""),
                                 "Quantity": cover_qty, "SellPrice": price,
                                 "AvgCostAtSell": round(avg_short, 4), "RealisedPnL": round(realised, 2),
                                 "Type": "Cover"})
                pos["cost_total"] -= avg_short * cover_qty * (-1)
                pos["shares"] += cover_qty
                remaining = qty - cover_qty
                if remaining > 0:
                    pos["cost_total"] += remaining * price
                    pos["shares"] += remaining
            else:
                pos["cost_total"] += qty * price
                pos["shares"] += qty
        elif txn_type == "Sell":
            avg_cost = pos["cost_total"] / pos["shares"] if pos["shares"] > 0 else 0.0
            realised = (price - avg_cost) * qty
            pnl_rows.append({"Symbol": symbol, "SellDate": row.get("CreateDate", ""),
                             "Quantity": qty, "SellPrice": price,
                             "AvgCostAtSell": round(avg_cost, 4), "RealisedPnL": round(realised, 2),
                             "Type": "Sell"})
            pos["cost_total"] -= avg_cost * qty
            pos["shares"] -= qty
        elif txn_type == "Short":
            pos["cost_total"] -= qty * price
            pos["shares"] -= qty

    pnl_df = pd.DataFrame(pnl_rows) if pnl_rows else pd.DataFrame()
    total = sum(r["RealisedPnL"] for r in pnl_rows)
    return pnl_df, round(total, 2)


# --- capital calculation ---

def _clean_money_signed(val):
    """Parse StockTrak Amount preserving sign: ($84,696.00) -> -84696.00"""
    if pd.isna(val): return 0.0
    s = str(val).replace("$", "").replace(",", "").strip()
    if "(" in s and ")" in s:
        s = "-" + s.replace("(", "").replace(")", "")
    try: return float(s)
    except: return 0.0


def calculate_current_capital(portfolio_market_value):
    if not os.path.exists(STOCKTRAK_FILE):
        return STARTING_CAPITAL, STARTING_CAPITAL

    df = pd.read_csv(STOCKTRAK_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    if "Amount" not in df.columns:
        return STARTING_CAPITAL, STARTING_CAPITAL

    # use StockTrak's Amount directly (already signed: negative=spent, positive=received)
    # skip dividend rows for trade-based capital calc
    if "TransactionType" in df.columns:
        trade_rows = df[~df["TransactionType"].str.lower().str.contains("dividend", na=False)]
    else:
        trade_rows = df

    total_cash_flow = trade_rows["Amount"].apply(_clean_money_signed).sum()
    cash_remaining = STARTING_CAPITAL + total_cash_flow
    total_capital = cash_remaining + portfolio_market_value
    return round(total_capital, 2), round(cash_remaining, 2)


# --- public API ---

def get_portfolio():
    return compute_portfolio_from_trades()

def get_trade_history():
    return load_stocktrak_csv()
