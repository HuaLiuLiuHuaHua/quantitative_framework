"""
BB-ATR Strategy - Basic Backtest
BB-ATR策略 - 基礎回測測試

測試目的：
1. 驗證策略基本功能
2. 評估固定參數的績效
3. 生成權益曲線圖表
4. 保存回測結果

數據來源：
- Bybit 加密貨幣數據 (支持 BTC, SOL 等)
- 支持 1H / 1D 時間框架
- 自動抓取或從 data/ 載入
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import matplotlib
matplotlib.use('Agg')  # 禁用圖片自動顯示

from datetime import datetime

# 導入策略和工具
from strategies.bb_atr.strategy import BBATRStrategy
from shared.backtest import BacktestEngine
from shared.visualization import plot_backtest_results
from shared.data_loader import load_data_with_fallback

def load_data(symbol: str, timeframe: str, start_date: str, end_date: str, verbose: bool = True):
    """
    載入或抓取數據 (優先使用本地數據)
    """
    # 根據 symbol 和 timeframe 選擇對應的 fetcher
    if symbol == "BTCUSDT" and timeframe == "1h":
        from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data
        fetcher_func = fetch_bybit_btc_1h_data
    elif symbol == "BTCUSDT" and timeframe == "1d":
        from data_fetchers.bybit_btc_1d_fetcher import fetch_bybit_btc_1d_data
        fetcher_func = fetch_bybit_btc_1d_data
    elif symbol == "SOLUSDT" and timeframe == "1h":
        from data_fetchers.bybit_sol_1h_fetcher import fetch_bybit_sol_1h_data
        fetcher_func = fetch_bybit_sol_1h_data
    else:
        raise ValueError(f"Unsupported symbol/timeframe combination: {symbol}/{timeframe}")

    return load_data_with_fallback(
        data_source=timeframe,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )

def run_backtest(
    symbol: str = "BTCUSDT",
    timeframe: str = "1d",
    bb_window: int = 20,
    bb_std: float = 2.0,
    atr_window: int = 14,
    atr_multiplier: float = 2.0,
    start_date: str = "2022-10-01",
    end_date: str = "2024-09-30",
    transaction_cost: float = 0.002,
    slippage: float = 0.0005,
    initial_capital: float = 100000,
    verbose: bool = True
):
    """
    執行BB-ATR策略回測
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BB-ATR策略回測")
        print("=" * 70)
        print(f"參數設置:")
        print(f"  交易對:           {symbol}")
        print(f"  時間框架:         {timeframe}")
        print(f"  bb_window:        {bb_window}")
        print(f"  bb_std:           {bb_std}")
        print(f"  atr_window:       {atr_window}")
        print(f"  atr_multiplier:   {atr_multiplier}")
        print(f"  交易成本:         {transaction_cost*100:.2f}%")
        print(f"  滑價:             {slippage*100:.3f}%")
        print(f"  初始資金:         ${initial_capital:,.2f}")
        print("=" * 70)

    # 步驟1: 載入數據
    df = load_data(symbol, timeframe, start_date, end_date, verbose)

    # 步驟2: 生成策略信號
    if verbose:
        print("\n生成策略信號...")

    strategy = BBATRStrategy()
    signals = strategy.generate_signals(
        df,
        bb_window=bb_window,
        bb_std=bb_std,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier
    )

    if verbose:
        print(f"信號統計:")
        print(f"  做多信號: {(signals == 1).sum()}")
        print(f"  做空信號: {(signals == -1).sum()}")
        print(f"  空倉信號: {(signals == 0).sum()}")

    # 步驟3: 執行回測
    if verbose:
        print("\n執行回測...")

    periods_per_year = 365 if timeframe == "1d" else 365 * 24

    engine = BacktestEngine(
        data=df,
        signals=signals,
        transaction_cost=transaction_cost,
        slippage=slippage,
        initial_capital=initial_capital,
        periods_per_year=periods_per_year
    )

    results = engine.run()

    # 步驟4: 打印績效摘要
    if verbose:
        engine.print_summary()

    # 步驟5: 保存結果
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_name = f"Backtest_BB_ATR_{symbol}_{timeframe}_bb{bb_window}_std{bb_std}_atr{atr_window}_{date_str}"

    save_path = engine.save_results(data_source=test_name, output_dir=Path(__file__).parent / "results")

    # 步驟6: 繪製圖表
    if verbose:
        print("\n繪製權益曲線...")

    plot_save_path = save_path / "backtest_chart.png"
    plot_backtest_results(
        equity_curve=results['equity_curve'],
        data=df,
        signals=signals,
        title=f"BB-ATR策略回測 ({symbol} {timeframe} | bb={bb_window}, std={bb_std}, atr={atr_window})",
        save_path=plot_save_path,
        figsize=(15, 10)
    )

    if verbose:
        print(f"\n圖表已保存: {plot_save_path}")
        print("=" * 70)
        print("回測完成！")
        print("=" * 70)

    return {
        'results': results,
        'strategy': strategy,
        'signals': signals,
        'data': df,
        'save_path': save_path
    }


if __name__ == "__main__":
    # ========== 數據配置 ========== 
    SYMBOL = "SOLUSDT"
    TIMEFRAME = "1h"

    # ========== 策略參數 ========== 
    BB_WINDOW = 124
    BB_STD = 2.5
    ATR_WINDOW = 8
    ATR_MULTIPLIER = 1.5
    START_DATE = "2024-10-01"
    END_DATE = "2025-09-30"
    TRANSACTION_COST = 0.002
    SLIPPAGE = 0.0005
    INITIAL_CAPITAL = 100000

    # 執行回測
    backtest_results = run_backtest(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        bb_window=BB_WINDOW,
        bb_std=BB_STD,
        atr_window=ATR_WINDOW,
        atr_multiplier=ATR_MULTIPLIER,
        start_date=START_DATE,
        end_date=END_DATE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        initial_capital=INITIAL_CAPITAL,
        verbose=True
    )

    print(f"\n測試完成！結果已保存到: {backtest_results['save_path']}")