import pandas as pd
import numpy as np

def get_coin_metadata():
    """Defines the target and anchor coins for the strategy."""
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

def generate_signals(anchor_df, target_df):
    """
    Main function to generate trading signals for all target assets.
    """
    ts = sorted(set(anchor_df.timestamp) | set(target_df.timestamp))
    df = pd.DataFrame({"timestamp": ts})
    df = df.merge(anchor_df, on="timestamp", how="left").ffill()
    df = df.merge(target_df, on="timestamp", how="left")
    df.dropna(subset=[f"close_{t['symbol']}_1H" for t in get_coin_metadata()["targets"]], inplace=True)
    
    df = compute_anchor_signal(df)
    
    out = []
    targets = get_coin_metadata()["targets"]
    num_targets = len(targets) # Get the number of assets to trade

    for t in targets:
        sym = t["symbol"]
        # Create a clean dataframe for each symbol
        cols_to_keep = ['timestamp', f"close_{sym}_1H", "anchor_strong", "anchor_score"]
        tmp = df[cols_to_keep].copy()
        tmp = compute_target_features(tmp, sym)
        # Pass the number of targets to divide capital correctly
        out.append(signal_generation(tmp, sym, num_targets))
        
    return pd.concat(out).reset_index(drop=True)

def compute_anchor_signal(df):
    """Computes a market-wide momentum signal from anchor assets."""
    for sym, thr in [("BTC", 0.0125), ("ETH", 0.0125), ("BNB", 0.0125)]:
        col = f"close_{sym}_4H"
        df[f"{sym}_mom"] = df[col].pct_change(1).fillna(0) > thr
    df["SOL_mom"] = df["close_SOL_1D"].pct_change(1).fillna(0) > 0.02
    df["anchor_score"] = df[[c for c in df if c.endswith("_mom")]].sum(axis=1)
    df["anchor_strong"] = df.anchor_score >= 2
    return df

def compute_target_features(df, sym):
    """Computes technical indicators for a single target asset."""
    col = f"close_{sym}_1H"
    df["price"] = df[col]
    df["sma20"] = df.price.rolling(20, min_periods=1).mean()
    df["zscore"] = (df.sma20 - df.price) / df.sma20
    df["hr_vol"] = df.price.pct_change().rolling(24, min_periods=1).std()
    df["rsi"] = compute_rsi(df.price)
    return df

def signal_generation(df, sym, num_targets):
    """
    Generates BUY, SELL, and HOLD signals for a single asset with proper state management.
    """
    signals = []
    sizes = []
    
    # State variables
    in_position = False
    current_position_size = 0.0
    entry_price = 0
    entry_index = 0
    trailing_stop = None
    
    # Capital allocation per asset
    trade_size = 1.0 / num_targets

    # Strategy Parameters
    take_profit = 0.006
    stop_loss = 0.005
    min_anchor_score = 2
    min_rsi = 10
    max_rsi = 90
    min_zscore = 0.002
    max_hr_vol = 0.035
    min_hold = 1
    max_hold = 3

    for i, row in df.iterrows():
        signal = "HOLD" # Default signal is HOLD

        # --- ENTRY LOGIC ---
        # Only consider buying if not already in a position
        entry_condition = (
            not in_position and
            row["anchor_score"] >= min_anchor_score and
            row.zscore > min_zscore and
            row.hr_vol < max_hr_vol and
            min_rsi < row.rsi < max_rsi
        )
        
        if entry_condition:
            signal = "BUY"
            in_position = True
            current_position_size = trade_size
            entry_price = row["price"]
            entry_index = i
            trailing_stop = row["price"] * (1 - stop_loss * 0.7)

        # --- EXIT LOGIC ---
        # Only consider selling if currently in a position
        elif in_position:
            profit = (row["price"] - entry_price) / entry_price
            age = i - entry_index

            # Update trailing stop if price moves in our favor
            if row["price"] > entry_price and (trailing_stop is None or row["price"] * (1 - stop_loss * 0.7) > trailing_stop):
                trailing_stop = row["price"] * (1 - stop_loss * 0.7)

            exit_condition = (
                (profit >= take_profit and age >= min_hold) or
                profit <= -stop_loss or
                row["price"] < trailing_stop or
                row.rsi > 93 or
                row.rsi < 7 or
                (row["price"] < row.sma20 and age >= min_hold) or
                age >= max_hold
            )

            if exit_condition:
                signal = "SELL"
                in_position = False
                current_position_size = 0.0
                # Reset state variables
                entry_price = 0
                entry_index = 0
                trailing_stop = None
        
        signals.append(signal)
        sizes.append(current_position_size)

    return pd.DataFrame({
        "timestamp": df.timestamp,
        "symbol": sym,
        "signal": signals,
        "position_size": sizes
    })

def compute_rsi(p, n=14):
    """Calculates the Relative Strength Index (RSI)."""
    d = p.diff().fillna(0)
    u = d.clip(lower=0)
    d_ = (-d).clip(lower=0)
    rs = u.rolling(n).mean() / d_.rolling(n).mean().replace(0, 1e-8)
    return 100 - 100 / (1 + rs)
