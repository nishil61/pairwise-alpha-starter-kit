import pandas as pd
import numpy as np

def get_coin_metadata():
    return {
        "targets": [
            {"symbol": "DOGE", "timeframe": "1H"},
            {"symbol": "AVAX", "timeframe": "1H"},
            {"symbol": "ADA", "timeframe": "1H"}
        ],
        "anchors": [
            {"symbol": "BTC", "timeframe": "4H"},
            {"symbol": "ETH", "timeframe": "4H"},
            {"symbol": "SOL", "timeframe": "1D"},
            {"symbol": "BNB", "timeframe": "4H"}
        ]
    }

def compute_anchor_signal(df):
    for sym, thr in [("BTC", 0.0125), ("ETH", 0.0125), ("BNB", 0.0125)]:
        col = f"close_{sym}_4H"
        if col in df.columns:
            df[f"{sym}_mom"] = df[col].pct_change(1).fillna(0) > thr
        else:
            df[f"{sym}_mom"] = False
    if "close_SOL_1D" in df.columns:
        df["SOL_mom"] = df["close_SOL_1D"].pct_change(1).fillna(0) > 0.02
    else:
        df["SOL_mom"] = False
    df["anchor_score"] = df[[c for c in df.columns if c.endswith("_mom")]].sum(axis=1)
    return df

def compute_target_features(df, sym):
    col = f"close_{sym}_1H"
    if col not in df.columns:
        return df
    df["price"] = df[col]
    df["sma20"] = df[col].rolling(20, min_periods=1).mean()
    df["zscore"] = (df["sma20"] - df["price"]) / df["sma20"]
    df["hr_vol"] = df["price"].pct_change().rolling(24, min_periods=1).std()
    df["rsi"] = compute_rsi(df["price"])
    return df

def compute_rsi(p, n=14):
    d = p.diff().fillna(0)
    u = d.clip(lower=0)
    d_ = (-d).clip(lower=0)
    rs = u.rolling(n, min_periods=1).mean() / d_.rolling(n, min_periods=1).mean().replace(0, 1e-8)
    return 100 - 100 / (1 + rs)

def generate_signals(anchor_df, target_df):
    ts = sorted(set(anchor_df.timestamp) | set(target_df.timestamp))
    df = pd.DataFrame({"timestamp": ts})
    df = df.merge(anchor_df, on="timestamp", how="left").ffill()
    df = df.merge(target_df, on="timestamp", how="left").ffill()
    targets = get_coin_metadata()["targets"]
    target_symbols = [t["symbol"] for t in targets]
    required_cols = [f"close_{s}_1H" for s in target_symbols if f"close_{s}_1H" in df.columns]
    if required_cols:
        df.dropna(subset=required_cols, how='all', inplace=True)
    df = df.reset_index(drop=True)
    if len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "symbol", "signal", "position_size"])
    df = compute_anchor_signal(df)
    out = []
    for sym in target_symbols:
        tmp = df[["timestamp"] + [f"close_{sym}_1H", "anchor_score"]].copy()
        tmp = compute_target_features(tmp, sym)
        out.append(signal_generation(tmp, sym))
    return pd.concat(out).reset_index(drop=True)

def signal_generation(df, sym):
    signals = []
    sizes = []
    in_pos = False
    entry = 0
    entry_i = 0
    trailing_stop = None

    # Tuned for high win rate, low drawdown, and stable profit
    take_profit = 0.012
    stop_loss = 0.006
    min_anchor_score = 2
    min_rsi = 25
    max_rsi = 75
    min_zscore = 0.01
    max_hr_vol = 0.025
    min_hold = 1
    max_hold = 6

    for i, row in df.iterrows():
        p = row["price"]
        sig = "HOLD"
        size = 0.0
        anchor_score = row["anchor_score"] if "anchor_score" in row else 0

        # Only allow BUY if not in position
        if not in_pos and anchor_score >= min_anchor_score and row.zscore > min_zscore and row.hr_vol < max_hr_vol and min_rsi < row.rsi < max_rsi:
            sig = "BUY"
            size = 1.0
            in_pos = True
            entry = p
            entry_i = i
            trailing_stop = p * (1 - stop_loss * 0.7)
        elif in_pos:
            profit = (p - entry) / entry
            age = i - entry_i
            if p > entry and (trailing_stop is None or p * (1 - stop_loss * 0.7) > trailing_stop):
                trailing_stop = p * (1 - stop_loss * 0.7)
            exit_cond = (
                (profit >= take_profit and age >= min_hold) or
                profit <= -stop_loss or
                p < trailing_stop or
                row.rsi > 80 or
                row.rsi < 20 or
                row.price < row.sma20 or
                age >= max_hold
            )
            if exit_cond:
                sig = "SELL"
                size = 0
                in_pos = False
                trailing_stop = None
        signals.append(sig)
        sizes.append(size)

    # Strict post-processing: enforce alternate BUY/SELL, never consecutive BUYs or SELLs
    filtered_signals = []
    filtered_sizes = []
    position = False
    for sig, size in zip(signals, sizes):
        if sig == "BUY":
            if not position:
                filtered_signals.append("BUY")
                filtered_sizes.append(1.0)
                position = True
            else:
                filtered_signals.append("HOLD")
                filtered_sizes.append(0.0)
        elif sig == "SELL":
            if position:
                filtered_signals.append("SELL")
                filtered_sizes.append(0.0)
                position = False
            else:
                filtered_signals.append("HOLD")
                filtered_sizes.append(0.0)
        else:
            filtered_signals.append("HOLD")
            filtered_sizes.append(0.0)

    return pd.DataFrame({
        "timestamp": df.timestamp.iloc[:len(filtered_signals)],
        "symbol": sym,
        "signal": filtered_signals,
        "position_size": filtered_sizes
    })
