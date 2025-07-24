#!/usr/bin/env python3
"""
Strategy Parameter Optimizer
============================

Fine-tune strategy parameters to maximize performance metrics:
- Profitability (45 points max)
- Sharpe Ratio (35 points max)  
- Max Drawdown (20 points max)

This optimizer tests different parameter combinations to find the best settings.
"""

import pandas as pd
import numpy as np
from data_download_manager import CryptoDataManager
from strategy import get_coin_metadata, generate_signals
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class StrategyOptimizer:
    """Optimize strategy parameters for maximum performance."""
    
    def __init__(self):
        self.data_manager = CryptoDataManager()
        self.metadata = get_coin_metadata()
        
    def download_full_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Download the complete dataset for backtesting."""
        print("📊 Downloading full market data...")
        
        # Prepare symbol configs
        all_symbols = []
        
        # Add targets
        for target in self.metadata['targets']:
            all_symbols.append({
                "symbol": target['symbol'],
                "timeframe": target['timeframe']
            })
        
        # Add anchors
        for anchor in self.metadata['anchors']:
            all_symbols.append({
                "symbol": anchor['symbol'],
                "timeframe": anchor['timeframe']
            })
        
        # Download data
        full_df = self.data_manager.get_market_data(all_symbols)
        
        # Split into anchor and target dataframes
        anchor_cols = ['timestamp']
        target_cols = ['timestamp']
        
        for anchor in self.metadata['anchors']:
            symbol, timeframe = anchor['symbol'], anchor['timeframe']
            for col_type in ['open', 'high', 'low', 'close', 'volume']:
                col_name = f"{col_type}_{symbol}_{timeframe}"
                if col_name in full_df.columns:
                    anchor_cols.append(col_name)
        
        for target in self.metadata['targets']:
            symbol, timeframe = target['symbol'], target['timeframe']
            for col_type in ['open', 'high', 'low', 'close', 'volume']:
                col_name = f"{col_type}_{symbol}_{timeframe}"
                if col_name in full_df.columns:
                    target_cols.append(col_name)
        
        anchor_df = full_df[anchor_cols].copy()
        target_df = full_df[target_cols].copy()
        
        print(f"✅ Data downloaded - Anchor: {anchor_df.shape}, Target: {target_df.shape}")
        return anchor_df, target_df
    
    def calculate_performance_metrics(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate key performance metrics using actual price data."""
        metrics = {}

        portfolio_value = 1.0
        portfolio_history = [portfolio_value]
        daily_returns = []

        # Use actual price data for each trade
        for symbol in signals_df['symbol'].unique():
            symbol_signals = signals_df[signals_df['symbol'] == symbol].copy()
            symbol_signals = symbol_signals.sort_values('timestamp').reset_index(drop=True)

            position = 0
            entry_price = 0

            # Use the 'price' column from signals_df if available
            if 'price' not in symbol_signals.columns:
                # Try to reconstruct price from close columns
                price_col = [c for c in symbol_signals.columns if c.startswith('close_') and c.endswith('1H')]
                if price_col:
                    symbol_signals['price'] = symbol_signals[price_col[0]]
                else:
                    continue  # Skip if no price data

            for i in range(len(symbol_signals)):
                signal = symbol_signals['signal'].iloc[i]
                position_size = symbol_signals['position_size'].iloc[i]
                price = symbol_signals['price'].iloc[i]

                if signal == 'BUY' and position == 0:
                    position = position_size
                    entry_price = price
                elif signal == 'SELL' and position > 0:
                    if entry_price > 0:
                        return_pct = (price - entry_price) / entry_price
                        portfolio_value *= (1 + return_pct * position)
                        portfolio_history.append(portfolio_value)
                        daily_returns.append(return_pct * position)
                    position = 0
                    entry_price = 0

        # Calculate metrics
        if len(daily_returns) > 0:
            total_return = (portfolio_value - 1.0) * 100  # Percentage
            daily_returns = np.array(daily_returns)
            
            # Sharpe Ratio (assuming 252 trading days)
            if np.std(daily_returns) > 0:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Max Drawdown
            portfolio_history = np.array(portfolio_history)
            peak = np.maximum.accumulate(portfolio_history)
            drawdown = (portfolio_history - peak) / peak
            max_drawdown = np.min(drawdown) * 100  # Percentage
            
            metrics = {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': abs(max_drawdown),
                'num_trades': len(daily_returns),
                'win_rate': np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
            }
        else:
            metrics = {
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'num_trades': 0,
                'win_rate': 0
            }
        
        return metrics
    
    def score_performance(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Convert metrics to Lunor scoring system."""
        scores = {}
        
        # Profitability Score (45 points max)
        # Assuming linear scaling: 0% = 0 points, 20% = 45 points
        profit_score = min(45, max(0, metrics['total_return'] * 2.25))
        scores['profitability'] = profit_score
        
        # Sharpe Ratio Score (35 points max)
        # Assuming: Sharpe 0 = 0 points, Sharpe 2 = 35 points
        sharpe_score = min(35, max(0, metrics['sharpe_ratio'] * 17.5))
        scores['sharpe'] = sharpe_score
        
        # Max Drawdown Score (20 points max)
        # Lower drawdown = higher score: 0% = 20 points, 20% = 0 points
        drawdown_score = max(0, 20 - metrics['max_drawdown'])
        scores['drawdown'] = drawdown_score
        
        # Total Score
        scores['total'] = profit_score + sharpe_score + drawdown_score
        
        return scores
    
    def run_optimization(self):
        """Run the optimization process."""
        print("🎯 Starting Strategy Optimization...")
        print("="*50)
        
        try:
            # Download data
            anchor_df, target_df = self.download_full_dataset()
            
            # Generate signals with current strategy
            print("\n🔄 Generating signals...")
            signals_df = generate_signals(anchor_df, target_df)

            # --- Inject true close price from target_df for each symbol ---
            for target in self.metadata['targets']:
                sym = target['symbol']
                tf = target['timeframe']
                price_col = f"close_{sym}_{tf}"
                price_df = target_df[["timestamp", price_col]].rename(columns={price_col: "true_close"})
                mask = signals_df["symbol"] == sym
                if mask.any():
                    merged = signals_df.loc[mask].merge(
                        price_df, on="timestamp", how="left"
                    )
                    # Only assign if 'true_close' exists in merged columns
                    if "true_close" in merged.columns:
                        signals_df.loc[mask, "true_close"] = merged["true_close"].values
                    else:
                        signals_df.loc[mask, "true_close"] = np.nan

            # Use true_close for performance metrics if available
            def patched_performance_metrics(signals_df: pd.DataFrame) -> Dict[str, float]:
                metrics = {}
                portfolio_value = 1.0
                portfolio_history = [portfolio_value]
                daily_returns = []
                for symbol in signals_df['symbol'].unique():
                    symbol_signals = signals_df[signals_df['symbol'] == symbol].copy()
                    symbol_signals = symbol_signals.sort_values('timestamp').reset_index(drop=True)
                    position = 0
                    entry_price = 0
                    # Use 'true_close' if available, else fallback to 'price'
                    if 'true_close' in symbol_signals.columns:
                        symbol_signals['price'] = symbol_signals['true_close']
                    elif 'price' not in symbol_signals.columns:
                        price_col = [c for c in symbol_signals.columns if c.startswith('close_') and c.endswith('1H')]
                        if price_col:
                            symbol_signals['price'] = symbol_signals[price_col[0]]
                        else:
                            continue
                    for i in range(len(symbol_signals)):
                        signal = symbol_signals['signal'].iloc[i]
                        position_size = symbol_signals['position_size'].iloc[i]
                        price = symbol_signals['price'].iloc[i]
                        if signal == 'BUY' and position == 0:
                            position = position_size
                            entry_price = price
                        elif signal == 'SELL' and position > 0:
                            if entry_price > 0:
                                return_pct = (price - entry_price) / entry_price
                                portfolio_value *= (1 + return_pct * position)
                                portfolio_history.append(portfolio_value)
                                daily_returns.append(return_pct * position)
                            position = 0
                            entry_price = 0
                if len(daily_returns) > 0:
                    total_return = (portfolio_value - 1.0) * 100
                    daily_returns = np.array(daily_returns)
                    if np.std(daily_returns) > 0:
                        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
                    else:
                        sharpe_ratio = 0
                    portfolio_history = np.array(portfolio_history)
                    peak = np.maximum.accumulate(portfolio_history)
                    drawdown = (portfolio_history - peak) / peak
                    max_drawdown = np.min(drawdown) * 100
                    metrics = {
                        'total_return': total_return,
                        'sharpe_ratio': sharpe_ratio,
                        'max_drawdown': abs(max_drawdown),
                        'num_trades': len(daily_returns),
                        'win_rate': np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
                    }
                else:
                    metrics = {
                        'total_return': 0,
                        'sharpe_ratio': 0,
                        'max_drawdown': 0,
                        'num_trades': 0,
                        'win_rate': 0
                    }
                return metrics

            # Calculate performance using patched function
            print("📊 Calculating performance metrics...")
            metrics = patched_performance_metrics(signals_df)
            scores = self.score_performance(metrics)
            
            # Display results
            print("\n" + "="*50)
            print("📈 PERFORMANCE ANALYSIS")
            print("="*50)
            
            print(f"💰 Total Return: {metrics['total_return']:.2f}%")
            print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"📉 Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"🔄 Number of Trades: {metrics['num_trades']}")
            print(f"🎯 Win Rate: {metrics['win_rate']:.1%}")
            
            print("\n" + "="*50)
            print("🏆 LUNOR SCORING")
            print("="*50)
            
            print(f"📈 Profitability: {scores['profitability']:.1f}/45 {'✅' if scores['profitability'] >= 9 else '❌'}")
            print(f"📊 Sharpe Ratio: {scores['sharpe']:.1f}/35 {'✅' if scores['sharpe'] >= 10 else '❌'}")
            print(f"📉 Max Drawdown: {scores['drawdown']:.1f}/20 {'✅' if scores['drawdown'] >= 5 else '❌'}")
            print(f"💯 Total Score: {scores['total']:.1f}/100 {'✅' if scores['total'] >= 50 else '❌'}")
            
            # Qualification check
            qualifies = (
                scores['profitability'] >= 9 and
                scores['sharpe'] >= 10 and
                scores['drawdown'] >= 5 and
                scores['total'] >= 50
            )
            
            print("\n" + "="*50)
            if qualifies:
                print("🎉 STRATEGY QUALIFIES FOR COMPETITION!")
                print("✅ All cutoff criteria met")
            else:
                print("⚠️ STRATEGY NEEDS IMPROVEMENT")
                print("❌ Some cutoff criteria not met")
                
                # Provide improvement suggestions
                print("\n💡 Improvement Suggestions:")
                if scores['profitability'] < 9:
                    print("   • Increase profitability: Adjust take-profit levels or trade frequency")
                if scores['sharpe'] < 10:
                    print("   • Improve Sharpe ratio: Reduce position sizes or improve entry timing")
                if scores['drawdown'] < 5:
                    print("   • Reduce drawdown: Tighten stop-losses or add more exit conditions")
            
            print("="*50)
            
            return qualifies, metrics, scores
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return False, {}, {}
        

def main():
    """Main optimization runner."""
    optimizer = StrategyOptimizer()
    
    print("🚀 Lunor AI Strategy Optimizer")
    print("🎯 Target: $5000 Prize")
    
    qualifies, metrics, scores = optimizer.run_optimization()
    
    if qualifies:
        print("\n🎯 READY FOR SUBMISSION!")
        print("Next steps:")
        print("1. Submit to Lunor AI platform")
        print("2. Monitor leaderboard")
        print("3. Prepare for forward test")
    else:
        print("\n🔧 OPTIMIZATION NEEDED")
        print("Consider adjusting strategy parameters before submission")

if __name__ == "__main__":
    main()