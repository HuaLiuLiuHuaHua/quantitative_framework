"""
MACD-ATR Strategy - Basic Backtest
MACD-ATR策略 - 基礎回測測試

測試目的：
1. 驗證策略基本功能
2. 評估固定參數的績效
3. 生成權益曲線圖表
4. 保存回測結果

數據來源：
- Bybit BTC 1H K線數據
- 自動抓取或從data/載入
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import matplotlib
matplotlib.use('Agg')  # 禁用圖片自動顯示

import pandas as pd
import numpy as np
from datetime import datetime

# 導入策略和工具
from strategies.macd_atr.strategy import MACDATRStrategy
from shared.backtest import BacktestEngine
from shared.visualization import plot_backtest_results
from data_fetchers.bybit_btc_1d_fetcher import fetch_bybit_btc_1d_data


def load_or_fetch_data(data_config: str = "BTCUSDT_1d", start_date: str = "2022-10-01", end_date: str = "2024-09-30", verbose: bool = True):
    """
    載入或抓取數據 (優先使用本地數據)

    使用統一的data_loader模組,優先從本地載入已下載的數據,
    大幅提升測試速度,避免重複下載。

    Args:
        data_config: 數據配置 (BTCUSDT_1d 或 BTCUSDT_1h)
        start_date: 開始日期
        end_date: 結束日期
        verbose: 是否打印詳細信息

    Returns:
        pd.DataFrame: OHLCV數據
    """
    from shared.data_loader import load_data_with_fallback

    # 根據配置選擇數據源和fetcher
    if data_config == "BTCUSDT_1h":
        from data_fetchers.bybit_btc_1h_fetcher import fetch_bybit_btc_1h_data
        data_source = "1h"
        fetcher_func = fetch_bybit_btc_1h_data
    else:  # 默認使用 BTCUSDT_1d
        data_source = "1d"
        fetcher_func = fetch_bybit_btc_1d_data

    return load_data_with_fallback(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )


def run_backtest(
    data_config: str = "BTCUSDT_1d",
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
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
    執行MACD-ATR策略回測

    Args:
        data_config: 數據配置 (BTCUSDT_1d 或 BTCUSDT_1h)
        macd_fast: MACD 快線週期
        macd_slow: MACD 慢線週期
        macd_signal: MACD 信號線週期
        atr_window: ATR 窗口期
        atr_multiplier: ATR 止損倍數
        start_date: 開始日期
        end_date: 結束日期
        transaction_cost: 交易成本（0.001 = 0.1%）
        slippage: 滑價（0.0005 = 0.05%）
        initial_capital: 初始資金
        verbose: 是否打印詳細信息

    Returns:
        Dict: 回測結果字典
    """
    if verbose:
        print("\n" + "=" * 70)
        print("MACD-ATR策略回測")
        print("=" * 70)
        print(f"參數設置:")
        print(f"  數據配置:         {data_config}")
        print(f"  macd_fast:        {macd_fast}")
        print(f"  macd_slow:        {macd_slow}")
        print(f"  macd_signal:      {macd_signal}")
        print(f"  atr_window:       {atr_window}")
        print(f"  atr_multiplier:   {atr_multiplier}")
        print(f"  交易成本:         {transaction_cost*100:.2f}%")
        print(f"  滑價:             {slippage*100:.3f}%")
        print(f"  初始資金:         ${initial_capital:,.2f}")
        print("=" * 70)

    # 步驟1: 載入數據
    df = load_or_fetch_data(data_config, start_date, end_date, verbose)

    # 步驟2: 生成策略信號
    if verbose:
        print("\n生成策略信號...")

    strategy = MACDATRStrategy()
    signals = strategy.generate_signals(
        df,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
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

    # 根據數據配置設定年化參數
    periods_per_year = 365 if data_config == "BTCUSDT_1d" else 365 * 24  # 日線365天，小時線365*24小時

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

    # 步驟5: 保存結果（改進命名方式以顯示測試名稱）
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 改進命名：包含測試名稱、策略、數據源、參數
    data_source = f"Backtest_MACD_ATR_{data_config}_macd{macd_fast}_{macd_slow}_{macd_signal}_atr{atr_window}_{date_str}"

    # 保存到strategies/macd_atr/results/
    save_path = engine.save_results(data_source=data_source, output_dir=Path(__file__).parent / "results")

    # 步驟6: 繪製圖表
    if verbose:
        print("\n繪製權益曲線...")

    plot_save_path = save_path / "backtest_chart.png"
    plot_backtest_results(
        equity_curve=results['equity_curve'],
        data=df,
        signals=signals,
        title=f"MACD-ATR策略回測 (macd={macd_fast}/{macd_slow}/{macd_signal}, atr={atr_window})",
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
    DATA_CONFIG = "SOLUSDT_1h"       # 數據配置: BTCUSDT_1d 或 BTCUSDT_1h

    # ========== 策略參數 ==========
    MACD_FAST = 1                    # MACD 快線週期
    MACD_SLOW = 4                    # MACD 慢線週期
    MACD_SIGNAL = 2                   # MACD 信號線週期
    ATR_WINDOW = 90                   # ATR 窗口期
    ATR_MULTIPLIER = 0.5              # ATR 止損倍數
    START_DATE = "2022-10-01"         # 開始日期
    END_DATE = "2024-09-30"           # 結束日期
    TRANSACTION_COST = 0.002          # 交易成本 0.2%
    SLIPPAGE = 0.0005                 # 滑價 0.05%
    INITIAL_CAPITAL = 100000          # 初始資金 $100,000

    # 執行回測
    print("\n" + "=" * 70)
    print("MACD-ATR策略 - 基礎回測測試")
    print("=" * 70)

    backtest_results = run_backtest(
        data_config=DATA_CONFIG,
        macd_fast=MACD_FAST,
        macd_slow=MACD_SLOW,
        macd_signal=MACD_SIGNAL,
        atr_window=ATR_WINDOW,
        atr_multiplier=ATR_MULTIPLIER,
        start_date=START_DATE,
        end_date=END_DATE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        initial_capital=INITIAL_CAPITAL,
        verbose=True
    )

    print("\n測試完成！")
    print(f"結果已保存到: {backtest_results['save_path']}")
