"""
BB-ATR Strategy - Walk-Forward Analysis
BB-ATR策略 - Walk-Forward分析

測試目的：
1. 模擬實盤的滾動優化過程
2. 評估樣本外(OOS)績效
3. 檢驗參數穩定性
4. 避免前視偏差

配置：
- train_window: 730根K棒
- test_window: 30根K棒
- step: 30根K棒
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
from datetime import datetime

# 導入策略和工具
from strategies.bb_atr.strategy import BBATRStrategy
from shared.walkforward import WalkForwardAnalyzer
from shared.visualization import plot_walkforward_performance

# 將策略函數定義在模組的最上層，以支持多進程
def bb_atr_strategy_func(data: pd.DataFrame, bb_window: int, bb_std: float,
                          atr_window: int, atr_multiplier: float):
    """BB-ATR策略包裝函數"""
    strategy = BBATRStrategy()
    return strategy.generate_signals(
        data,
        bb_window=bb_window,
        bb_std=bb_std,
        atr_window=atr_window,
        atr_multiplier=atr_multiplier
    )

def load_data(data_config: str = "BTCUSDT_1d", start_date: str = "2022-10-01", end_date: str = "2025-09-30", verbose: bool = True):
    """
    載入數據 (優先使用本地數據)

    使用統一的data_loader模組,優先從本地載入已下載的數據,
    大幅提升測試速度,避免重複下載。

    Args:
        data_config: 數據配置 (BTCUSDT_1d, BTCUSDT_1h, SOLUSDT_1h 等)
        start_date: 開始日期
        end_date: 結束日期
        verbose: 是否打印詳細信息

    Returns:
        pd.DataFrame: OHLCV數據

    Raises:
        ValueError: 如果日期格式錯誤或日期範圍無效
    """
    from shared.data_loader import load_data_with_fallback

    # 驗證日期範圍
    try:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start >= end:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")
    except Exception as e:
        raise ValueError(f"Invalid date format: {e}")

    # 從 data_config 解析 symbol 和 timeframe
    symbol = None
    timeframe = None
    if "_" in data_config:
        parts = data_config.split('_')
        symbol = parts[0]
        timeframe = parts[1]

    if not symbol or timeframe not in ["1h", "1d"]:
        raise ValueError(f"無法從 data_config '{data_config}' 解析出有效的 symbol 和 timeframe")

    # 根據配置選擇對應的 fetcher
    fetcher_func = None
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
        raise ValueError(f"Unsupported data_config: {data_config}")

    # 修正: 傳遞正確的 symbol 和 timeframe (作為 data_source)
    df = load_data_with_fallback(
        data_source=timeframe,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )

    if df is None or df.empty:
        raise ValueError(f"Failed to load data for {data_config}")

    return df


def run_walkforward(
    data_config: str = "BTCUSDT_1d",
    bb_window_range: tuple = (15, 25, 5),
    bb_std_range: tuple = (1.5, 2.5, 0.5),
    atr_window_range: tuple = (10, 20, 2),
    atr_multiplier_range: tuple = (1.5, 3.0, 0.5),
    start_date: str = "2022-10-01",
    end_date: str = "2025-09-30",
    train_window: int = 730,
    test_window: int = 30,
    step: int = 30,
    objective: str = 'profit_factor',
    transaction_cost: float = 0.002,
    slippage: float = 0.0005,
    n_jobs: int = 1,
    verbose: bool = True
):
    """
    執行Walk-Forward分析
    """
    if verbose:
        print("\n" + "=" * 70)
        print("BB-ATR策略Walk-Forward分析")
        print("=" * 70)
        print(f"參數設置:")
        print(f"  數據配置:         {data_config}")
        print(f"  bb_window範圍:    {bb_window_range}")
        print(f"  bb_std範圍:       {bb_std_range}")
        print(f"  atr_window範圍:   {atr_window_range}")
        print(f"  atr_multiplier範圍: {atr_multiplier_range}")
        print(f"  訓練窗口:         {train_window} K棒")
        print(f"  測試窗口:         {test_window} K棒")
        print(f"  滾動步長:         {step} K棒")
        print(f"  優化目標:         {objective}")
        print(f"  交易成本:         {transaction_cost*100:.2f}%")
        print(f"  滑價:             {slippage*100:.3f}%")
        print(f"  並行核心:         {n_jobs}")
        print("=" * 70)

    # 步驟1: 載入數據
    df = load_data(data_config, start_date, end_date, verbose)

    # 步驟2: 設置參數網格
    param_grid = {
        'bb_window': list(range(bb_window_range[0], bb_window_range[1] + 1, bb_window_range[2])),
        'bb_std': list(np.arange(bb_std_range[0], bb_std_range[1] + bb_std_range[2] * 0.001, bb_std_range[2])),
        'atr_window': list(range(atr_window_range[0], atr_window_range[1] + 1, atr_window_range[2])),
        'atr_multiplier': list(np.arange(atr_multiplier_range[0], atr_multiplier_range[1] + atr_multiplier_range[2] * 0.001, atr_multiplier_range[2]))
    }

    # 步驟3: 創建Walk-Forward分析器
    periods_per_year = 365 if "_1d" in data_config else 365 * 24

    analyzer = WalkForwardAnalyzer(
        strategy_func=bb_atr_strategy_func,
        data=df,
        param_grid=param_grid,
        train_window=train_window,
        test_window=test_window,
        step=step,
        objective=objective,
        transaction_cost=transaction_cost,
        slippage=slippage,
        n_jobs=n_jobs,
        periods_per_year=periods_per_year
    )

    # 步驟4: 執行Walk-Forward分析
    results_df = analyzer.run(verbose=verbose)

    # 步驟5: 保存結果
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_prefix = f"WalkForward_BB_ATR_{data_config}_{date_str}"

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = analyzer.save_results(data_source=result_prefix, output_dir=output_dir)

    # 步驟6: 繪製Walk-Forward分析圖表
    if verbose:
        print("\n繪製Walk-Forward分析圖表...")

    if analyzer.combined_equity_curve is not None and not analyzer.combined_equity_curve.empty:
        plot_save_path = save_path / "walkforward_chart.png"
        plot_walkforward_performance(
            results_df=results_df,
            title="BB-ATR策略Walk-Forward分析",
            save_path=plot_save_path,
            combined_equity_curve=analyzer.combined_equity_curve,
            buy_hold_equity_curve=analyzer.buy_hold_equity_curve
        )
    elif verbose:
        print("\n警告: 未能生成有效的合併權益曲線，跳過繪圖。" )

    # 步驟7: 打印樣本外績效摘要
    if verbose:
        print("\n" + "=" * 70)
        print("樣本外(OOS)績效摘要")
        print("=" * 70)
        print(f"總窗口數:         {len(results_df)}")
        print(f"平均OOS夏普:      {results_df['oos_sharpe_ratio'].mean():.3f}")
        print(f"平均OOS收益:      {results_df['oos_total_return'].mean()*100:.2f}%")
        print(f"平均OOS回撤:      {results_df['oos_max_drawdown'].mean()*100:.2f}%")
        print(f"平均OOS獲利因子:  {results_df['oos_profit_factor'].mean():.3f}")
        print(f"平均OOS勝率:      {results_df['oos_win_rate'].mean()*100:.2f}%")
        print(f"正收益窗口:       {(results_df['oos_total_return'] > 0).sum()} / {len(results_df)}")

        is_oos_ratio = (results_df['oos_sharpe_ratio'].mean() /
                       results_df['is_sharpe_ratio'].mean()
                       if results_df['is_sharpe_ratio'].mean() != 0 else 0)
        print(f"OOS/IS夏普比率:   {is_oos_ratio:.3f}")
        print("  (接近1表示沒有過度擬合)")
        print("=" * 70)

    return {
        'analyzer': analyzer,
        'results_df': results_df,
        'save_path': save_path
    }


if __name__ == "__main__":
    # ========== 數據配置 ========== 
    DATA_CONFIG = "SOLUSDT_1h"        # 數據配置: BTCUSDT_1d, BTCUSDT_1h, SOLUSDT_1h

    # ========== Walk-Forward參數 ========== 
    BB_WINDOW_RANGE = (120, 169, 1)
    BB_STD_RANGE = (2.0, 3.1, 0.5)
    ATR_WINDOW_RANGE = (8, 13, 1)
    ATR_MULTIPLIER_RANGE = (0.5, 1.6, 0.5)

    START_DATE = "2022-10-01"
    END_DATE = "2025-09-30"

    TRAIN_WINDOW = 17520
    TEST_WINDOW = 720
    STEP = 720

    OBJECTIVE = 'profit_factor'
    TRANSACTION_COST = 0.002
    SLIPPAGE = 0.0005
    N_JOBS = 3

    # 執行Walk-Forward分析
    print("\n" + "=" * 70)
    print("BB-ATR策略 - Walk-Forward分析")
    print("=" * 70)

    wf_results = run_walkforward(
        data_config=DATA_CONFIG,
        bb_window_range=BB_WINDOW_RANGE,
        bb_std_range=BB_STD_RANGE,
        atr_window_range=ATR_WINDOW_RANGE,
        atr_multiplier_range=ATR_MULTIPLIER_RANGE,
        start_date=START_DATE,
        end_date=END_DATE,
        train_window=TRAIN_WINDOW,
        test_window=TEST_WINDOW,
        step=STEP,
        objective=OBJECTIVE,
        transaction_cost=TRANSACTION_COST,
        slippage=SLIPPAGE,
        n_jobs=N_JOBS,
        verbose=True
    )

    print("\n測試完成！")
    if wf_results and 'save_path' in wf_results:
        print(f"結果已保存到: {wf_results['save_path']}")

        if 'results_df' in wf_results:
            results_df = wf_results['results_df']
            if 'oos_sharpe_ratio' in results_df.columns:
                print(f"平均樣本外夏普比率: {results_df['oos_sharpe_ratio'].mean():.3f}")
            else:
                print("警告：結果中缺少 oos_sharpe_ratio 列")
        else:
            print("警告：結果中缺少 results_df")
    else:
        print("錯誤：Walk-Forward分析失敗")