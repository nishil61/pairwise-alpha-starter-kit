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
    for sym, thr in [("BTC", 0.010), ("ETH", 0.010), ("BNB", 0.010)]:
        col = f"close_{sym}_4H"
        if col in df.columns:
            df[f"{sym}_mom"] = df[col].pct_change(1).fillna(0) > thr
        else:
            df[f"{sym}_mom"] = False

    if "close_SOL_1D" in df.columns:
        df["SOL_mom"] = df["close_SOL_1D"].pct_change(1).fillna(0) > 0.018
    else:
        df["SOL_mom"] = False

    df["anchor_score"] = df[[c for c in df.columns if c.endswith("_mom")]].sum(axis=1)

    for sym in ["BTC", "ETH", "BNB"]:
        col = f"close_{sym}_4H"
        if col in df.columns:
            df[f"{sym}_trend"] = df[col].rolling(6).mean() > df[col].rolling(20).mean()
        else:
            df[f"{sym}_trend"] = False

    df["strong_trend"] = df[[c for c in df.columns if c.endswith("_trend")]].sum(axis=1)
    return df

def compute_target_features(df, sym):
    col = f"close_{sym}_1H"
    if col not in df.columns:
        return df

    df[f"price_{sym}"] = df[col]
    df[f"sma20_{sym}"] = df[col].rolling(20, min_periods=1).mean()
    df[f"sma50_{sym}"] = df[col].rolling(50, min_periods=1).mean()
    df[f"ema12_{sym}"] = df[col].ewm(span=12).mean()
    df[f"ema26_{sym}"] = df[col].ewm(span=26).mean()

    sma_std = df[col].rolling(20, min_periods=1).std()
    df[f"zscore_{sym}"] = (df[col] - df[f"sma20_{sym}"]) / sma_std.replace(0, 1e-8)

    df[f"hr_vol_{sym}"] = df[col].pct_change().rolling(24, min_periods=1).std()
    df[f"rsi_{sym}"] = compute_rsi(df[col])
    df[f"rsi_smooth_{sym}"] = df[f"rsi_{sym}"].rolling(2).mean()

    df[f"momentum_4h_{sym}"] = df[col].pct_change(4).fillna(0)
    df[f"momentum_8h_{sym}"] = df[col].pct_change(8).fillna(0)

    df[f"bb_upper_{sym}"] = df[f"sma20_{sym}"] + 2 * sma_std
    df[f"bb_lower_{sym}"] = df[f"sma20_{sym}"] - 2 * sma_std
    df[f"bb_position_{sym}"] = (df[col] - df[f"sma20_{sym}"]) / (2 * sma_std)

    macd = df[f"ema12_{sym}"] - df[f"ema26_{sym}"]
    df[f"macd_{sym}"] = macd
    df[f"macd_signal_{sym}"] = macd.ewm(span=9).mean()
    df[f"macd_hist_{sym}"] = macd - df[f"macd_signal_{sym}"]

    df[f"price_vs_sma20_{sym}"] = (df[col] / df[f"sma20_{sym}"]) - 1
    df[f"price_vs_sma50_{sym}"] = (df[col] / df[f"sma50_{sym}"]) - 1

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
    for sym in target_symbols:
        df = compute_target_features(df, sym)

    signals = []

    in_position = False
    held_symbol = None
    entry_price = 0
    entry_index = 0
    peak_price = 0
    trailing_stop = None
    last_trade_index = -100 

    take_profit = 0.035       
    stop_loss = 0.015         
    min_anchor_score = 2
    min_strong_trend = 1
    min_rsi = 30
    max_rsi = 70
    min_zscore = 0.6         
    max_hr_vol = 0.02         
    min_hold = 4
    max_hold = 48            
    trail_activation = 0.015
    trail_distance = 0.008
    cooldown_period = 4

    for i, row in df.iterrows():
        current_signal = {
            "timestamp": row["timestamp"],
            "symbol": held_symbol if in_position else target_symbols[0], # Default symbol
            "signal": "HOLD",
            "position_size": 1.0 if in_position else 0.0
        }

        if i < 50:
            signals.append(current_signal)
            continue

        if in_position and held_symbol:
            price_col = f"price_{held_symbol}"
            if price_col not in row or pd.isna(row[price_col]):
                signals.append(current_signal)
                continue

            p = row[price_col]
            profit = (p - entry_price) / entry_price
            age = i - entry_index

            if p > peak_price: peak_price = p
            if profit > trail_activation and (trailing_stop is None or peak_price * (1 - trail_distance) > trailing_stop):
                trailing_stop = peak_price * (1 - trail_distance)

            exit_cond = (
                profit >= take_profit or
                profit <= -stop_loss or
                (trailing_stop and p <= trailing_stop) or
                age >= max_hold
            )

            if exit_cond:
                current_signal = {
                    "timestamp": row["timestamp"], "symbol": held_symbol,
                    "signal": "SELL", "position_size": 0.0
                }
                in_position, held_symbol, trailing_stop, peak_price = False, None, None, 0
                last_trade_index = i

        elif not in_position and (i - last_trade_index) >= cooldown_period:
            candidates = []
            anchor_score = row.get("anchor_score", 0)
            strong_trend = row.get("strong_trend", 0)

            if anchor_score >= min_anchor_score and strong_trend >= min_strong_trend:
                for sym in target_symbols:
                    price_col, zscore_col, hr_vol_col, rsi_col, momentum_4h_col = f"price_{sym}", f"zscore_{sym}", f"hr_vol_{sym}", f"rsi_smooth_{sym}", f"momentum_4h_{sym}"

                    if all(c in row and not pd.isna(row[c]) for c in [price_col, zscore_col, hr_vol_col, rsi_col, momentum_4h_col]):
                        if (row[zscore_col] > min_zscore and row[hr_vol_col] < max_hr_vol and min_rsi < row[rsi_col] < max_rsi and row[momentum_4h_col] > 0.002):
                            quality_score = row[zscore_col] + (row[momentum_4h_col] * 50)
                            candidates.append({"symbol": sym, "score": quality_score})

            if candidates:
                best_candidate = max(candidates, key=lambda x: x["score"])
                held_symbol = best_candidate["symbol"]
                current_signal = {
                    "timestamp": row["timestamp"], "symbol": held_symbol,
                    "signal": "BUY", "position_size": 0.98
                }
                in_position = True
                entry_price = row[f"price_{held_symbol}"]
                entry_index = i
                peak_price = entry_price
                trailing_stop = None

        signals.append(current_signal)

    if not signals:
        return pd.DataFrame(columns=["timestamp", "symbol", "signal", "position_size"])

    result_df = pd.DataFrame(signals)
    result_df["timestamp"] = pd.to_datetime(result_df["timestamp"])
    result_df["position_size"] = result_df["position_size"].astype(float)

    result_df["symbol"].fillna(method='ffill', inplace=True)
    result_df["symbol"].fillna(method='bfill', inplace=True)

    return result_df.sort_values("timestamp").reset_index(drop=True)
