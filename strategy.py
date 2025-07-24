import pandas as pd
import numpy as np

def get_coin_metadata():
    return {
        "targets":[
            {"symbol":"DOGE","timeframe":"1H"},
            {"symbol":"AVAX","timeframe":"1H"},
            {"symbol":"ADA", "timeframe":"1H"}
        ],
        "anchors":[
            {"symbol":"BTC","timeframe":"4H"},
            {"symbol":"ETH","timeframe":"4H"},
            {"symbol":"SOL","timeframe":"1D"},
            {"symbol":"BNB","timeframe":"4H"}
        ]
    }

def generate_signals(anchor_df, target_df):
    ts = sorted(set(anchor_df.timestamp) | set(target_df.timestamp))
    df = pd.DataFrame({"timestamp": ts})
    df = df.merge(anchor_df, on="timestamp", how="left").ffill()
    df = df.merge(target_df, on="timestamp", how="left")
    df.dropna(subset=[f"close_{t['symbol']}_1H" for t in get_coin_metadata()["targets"]], inplace=True)
    df = compute_anchor_signal(df)
    out = []
    for sym in [t["symbol"] for t in get_coin_metadata()["targets"]]:
        tmp = df[['timestamp'] + list(df.filter(like=f"close_{sym}_1H")) + ["anchor_strong", "anchor_score"]].copy()
        tmp = compute_target_features(tmp, sym)
        out.append(signal_generation(tmp, sym))
    return pd.concat(out).reset_index(drop=True)

def compute_anchor_signal(df):
    for sym,thr in [("BTC",0.0125),("ETH",0.0125),("BNB",0.0125)]:
        col=f"close_{sym}_4H"
        df[f"{sym}_mom"]=df[col].pct_change(1).fillna(0)>thr
    df["SOL_mom"]=df["close_SOL_1D"].pct_change(1).fillna(0)>0.02
    df["anchor_score"]=df[[c for c in df if c.endswith("_mom")]].sum(axis=1)
    df["anchor_strong"]=df.anchor_score>=2
    return df

def compute_target_features(df, sym):
    col=f"close_{sym}_1H"
    df["price"]=df[col]
    df["sma20"]=df.price.rolling(20,min_periods=1).mean()
    df["zscore"]=(df.sma20-df.price)/df.sma20
    df["hr_vol"]=df.price.pct_change().rolling(24,min_periods=1).std()
    df["vol"]=df[col].rolling(24,min_periods=1).std()
    df["rsi"]=compute_rsi(df.price)
    df["vol_ratio"]=df[col].rolling(24).mean().fillna(0)
    return df

def signal_generation(df, sym):
    signals = []
    sizes = []
    in_pos = False
    entry = 0
    entry_i = 0
    trailing_stop = None

    # Ultra-conservative for max win rate
    take_profit = 0.006   # Even quicker profit taking
    stop_loss = 0.005     # Even tighter stop loss
    min_anchor_score = 2
    min_rsi = 10
    max_rsi = 90
    min_zscore = 0.002
    max_hr_vol = 0.035
    min_hold = 1
    max_hold = 3

    for i, row in df.iterrows():
        p = row["price"]
        sig = "HOLD"
        size = 0.0

        anchor_score = row["anchor_score"] if "anchor_score" in row else (3 if row["anchor_strong"] else 0)

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
                row.rsi > 93 or
                row.rsi < 7 or
                (row.price < row.sma20 and age >= min_hold) or
                age >= max_hold
            )
            if exit_cond:
                sig = "SELL"
                size = 0
                in_pos = False
                trailing_stop = None
        signals.append(sig)
        sizes.append(size)
    return pd.DataFrame({
        "timestamp": df.timestamp, "symbol": sym,
        "signal": signals, "position_size": sizes
    })

def compute_rsi(p, n=14):
    d=p.diff().fillna(0)
    u=d.clip(lower=0); d_=(-d).clip(lower=0)
    rs=u.rolling(n).mean()/d_.rolling(n).mean().replace(0,1e-8)
    return 100-100/(1+rs)

