"""
This is a sample strategy that demonstrates how to implement a basic trading strategy
that passes all validation requirements. This strategy is for educational purposes only
and does not guarantee profitable trades. Users are encouraged to create their own
strategies based on their trading knowledge and risk management principles.
"""

import pandas as pd
import numpy as np




def get_coin_metadata() -> dict:
    """
    Specifies the target and anchor coins used in this strategy.
    
    Returns:
    {
        "targets": [{"symbol": "LDO", "timeframe": "1H"}],
        "anchors": [
            {"symbol": "BTC", "timeframe": "4H"},
            {"symbol": "ETH", "timeframe": "4H"}
        ]
    }
    """
    return {
        "targets": [{
            "symbol": "LDO",
            "timeframe": "1H"
        }],
        "anchors": [
            {"symbol": "BTC", "timeframe": "4H"},
            {"symbol": "ETH", "timeframe": "4H"}
        ]
    }

def generate_signals(anchor_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy: Buy LDO if BTC or ETH pumped >2% in the last 4H candle.
    Enhanced with sell conditions and position sizing for complete trading pairs.

    Inputs:
    - anchor_df: DataFrame with timestamp, close_BTC_4H, close_ETH_4H columns
    - target_df: DataFrame with timestamp, close_LDO_1H columns

    Output:
    - DataFrame with ['timestamp', 'symbol', 'signal', 'position_size']
    """
    try:
        # Merge anchor and target data on timestamp
        df = pd.merge(
            target_df[['timestamp', 'close_LDO_1H']],
            anchor_df[['timestamp', 'close_BTC_4H', 'close_ETH_4H']],
            on='timestamp',
            how='outer'  # Use outer join to get all timestamps
        ).sort_values('timestamp').reset_index(drop=True)
        
        # Calculate 4H returns for BTC and ETH
        df['btc_return_4h'] = df['close_BTC_4H'].pct_change(fill_method=None)
        df['eth_return_4h'] = df['close_ETH_4H'].pct_change(fill_method=None)
        
        # Calculate LDO price change for sell signals
        df['ldo_return_1h'] = df['close_LDO_1H'].pct_change(fill_method=None)
        
        # Initialize signal arrays
        signals = []
        position_sizes = []
        
        # Track position state for generating buy-sell pairs
        in_position = False
        entry_price = 0
        
        for i in range(len(df)):
            # Get current values (handle NaN)
            btc_pump = df['btc_return_4h'].iloc[i] > 0.02 if pd.notna(df['btc_return_4h'].iloc[i]) else False
            eth_pump = df['eth_return_4h'].iloc[i] > 0.02 if pd.notna(df['eth_return_4h'].iloc[i]) else False
            ldo_price = df['close_LDO_1H'].iloc[i]
            
            # Signal generation logic
            if not in_position:
                # Look for buy signals
                if (btc_pump or eth_pump) and pd.notna(ldo_price):
                    signals.append('BUY')
                    position_sizes.append(0.5)  # 50% position size
                    in_position = True
                    entry_price = ldo_price
                else:
                    signals.append('HOLD')
                    position_sizes.append(0.0)
            else:
                # Look for sell signals when in position
                if pd.notna(ldo_price) and entry_price > 0:
                    # Sell conditions: 5% profit or 3% loss
                    profit_pct = (ldo_price - entry_price) / entry_price
                    
                    if profit_pct >= 0.05 or profit_pct <= -0.03:
                        signals.append('SELL')
                        position_sizes.append(0.0)
                        in_position = False
                        entry_price = 0
                    else:
                        signals.append('HOLD')
                        position_sizes.append(0.5)  # Maintain position
                else:
                    signals.append('HOLD')
                    position_sizes.append(0.5 if in_position else 0.0)
        
        # Create result DataFrame with required columns
        result_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'symbol': 'LDO',  # All signals are for LDO (the target)
            'signal': signals,
            'position_size': position_sizes
        })
        
        return result_df
        
    except Exception as e:
        raise RuntimeError(f"Error in generate_signals: {e}")
