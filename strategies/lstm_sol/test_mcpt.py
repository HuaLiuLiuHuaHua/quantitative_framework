"""
LSTM SOL Strategy - Monte Carlo Permutation Test (MCPT)
LSTM SOL策略 - 蒙地卡羅排列測試

測試目的：
1. 驗證策略是否存在過度擬合
2. 計算統計顯著性(p-value)
3. 比較原始績效與隨機排列績效
4. 生成MCPT分佈圖表

測試模式：
- mode='normal': 使用固定參數
- n_permutations=1000: 執行1000次排列測試
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
from strategies.lstm_sol.strategy import LSTMSOLStrategy
from shared.mcpt import MonteCarloTester
from shared.visualization import plot_mcpt_distribution_medium_style


def lstm_sol_strategy_func(data: pd.DataFrame, confidence_threshold: float, look_back: int):
    """LSTM SOL策略包裝函數 (模組級別函數以支援多進程)"""
    strategy = LSTMSOLStrategy()
    return strategy.generate_signals(data, symbol="SOL/USD", confidence_threshold=confidence_threshold, look_back=look_back)


def load_data(data_config: str = "SOLUSDT_1h", start_date: str = "2022-10-01", end_date: str = "2024-09-30", verbose: bool = True):
    """
    載入數據 (優先使用本地數據)

    使用統一的data_loader模組,優先從本地載入已下載的數據,
    大幅提升測試速度,避免重複下載。
    """
    from shared.data_loader import load_data_with_fallback
    from data_fetchers.bybit_sol_1h_fetcher import fetch_bybit_sol_1h_data

    # 根據配置選擇數據源和fetcher
    if data_config == "SOLUSDT_1h":
        data_source = "1h"
        symbol = "SOLUSDT"
        fetcher_func = fetch_bybit_sol_1h_data
    else:
        raise ValueError(f"不支援的數據配置: {data_config}")

    return load_data_with_fallback(
        data_source=data_source,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fetcher_func=fetcher_func,
        verbose=verbose
    )


def run_mcpt(
    data_config: str = "SOLUSDT_1h",
    confidence_threshold: float = 0.01,
    look_back: int = 72,
    start_date: str = "2022-10-01",
    end_date: str = "2024-09-30",
    n_permutations: int = 1000,
    mode: str = 'normal',
    metric: str = 'profit_factor',
    transaction_cost: float = 0.002,
    slippage: float = 0.0005,
    start_index: int = 100,
    n_jobs: int = 1,
    verbose: bool = True
):
    """
    執行MCPT測試

    Args:
        data_config: 數據配置 (SOLUSDT_1h)
        confidence_threshold: 置信度閾值（normal模式使用）
        look_back: 回溯窗口（normal模式使用）
        start_date: 開始日期
        end_date: 結束日期
        n_permutations: 排列次數
        mode: 測試模式（'normal'或'walkforward'）
        metric: 評估指標
        transaction_cost: 交易成本
        slippage: 滑價
        start_index: 開始排列的索引
        n_jobs: 並行核心數
        verbose: 是否打印詳細信息

    Returns:
        Dict: MCPT測試結果字典
    """
    # 參數驗證
    if mode not in ['normal', 'walkforward']:
        raise ValueError(f"mode必須是'normal'或'walkforward'，當前值: {mode}")

    if start_index <= look_back:
        raise ValueError(
            f"start_index({start_index})必須大於look_back({look_back})，"
            f"以確保有足夠的歷史數據用於指標計算。"
            f"建議: start_index >= look_back + 10"
        )

    if n_permutations < 100:
        print(f"警告: n_permutations={n_permutations}較小，建議至少1000次以獲得可靠的p值")

    if look_back < 2:
        raise ValueError(f"look_back必須大於等於2，當前值: {look_back}")

    # 1小時線，一年 365*24 小時
    periods_per_year = 365 * 24

    if verbose:
        print("\n" + "=" * 70)
        print("LSTM SOL策略MCPT測試")
        print("=" * 70)
        print(f"參數設置:")
        print(f"  數據配置:         {data_config}")
        print(f"  置信度閾值:       {confidence_threshold}")
        print(f"  回溯窗口:         {look_back}")
        print(f"  排列次數:         {n_permutations}")
        print(f"  測試模式:         {mode}")
        print(f"  評估指標:         {metric}")
        print(f"  交易成本:         {transaction_cost*100:.2f}%")
        print(f"  滑價:             {slippage*100:.3f}%")
        print(f"  開始索引:         {start_index}")
        print(f"  並行核心:         {n_jobs}")
        print(f"  年化週期數:       {periods_per_year}")
        print("=" * 70)

    # 步驟1: 載入數據
    try:
        df = load_data(data_config, start_date, end_date, verbose)
    except Exception as e:
        raise RuntimeError(f"數據載入失敗: {e}")

    # 驗證數據質量
    if len(df) < start_index + look_back:
        raise ValueError(
            f"數據長度不足: 需要至少 {start_index + look_back} 根K棒，"
            f"實際只有 {len(df)} 根K棒"
        )

    if verbose:
        print(f"\n數據驗證通過:")
        print(f"  數據長度:         {len(df)}")
        print(f"  最小要求:         {start_index + look_back}")
        print(f"  日期範圍:         {df.index[0]} ~ {df.index[-1]}")

    # 步驟2: 設置策略參數
    if mode == 'normal':
        # Normal模式：使用固定參數
        strategy_params = {'confidence_threshold': confidence_threshold, 'look_back': look_back}
    else:
        # Walk-Forward模式：使用參數網格
        strategy_params = {'confidence_threshold': [confidence_threshold], 'look_back': [look_back]}

    # 步驟3: 創建MCPT測試器
    try:
        tester = MonteCarloTester(
            strategy_func=lstm_sol_strategy_func,
            data=df,
            strategy_params=strategy_params,
            start_index=start_index,
            n_permutations=n_permutations,
            mode=mode,
            metric=metric,
            transaction_cost=transaction_cost,
            slippage=slippage,
            n_jobs=n_jobs,
            periods_per_year=periods_per_year
        )
    except Exception as e:
        raise RuntimeError(f"MCPT測試器創建失敗: {e}")

    # 步驟4: 執行MCPT測試
    try:
        results = tester.run(verbose=verbose)
    except Exception as e:
        raise RuntimeError(f"MCPT測試執行失敗: {e}")

    # 步驟5: 保存結果
    try:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_source = f"MCPT_LSTMSOL_{data_config}_conf{confidence_threshold}_lb{look_back}_{date_str}"

        output_dir = Path(__file__).parent / "results"
        save_path = tester.save_results(data_source=data_source, output_dir=output_dir)
    except Exception as e:
        print(f"警告: 結果保存失敗: {e}")
        save_path = None

    # 步驟6: 繪製MCPT分佈圖（Medium風格）
    if save_path:
        try:
            if verbose:
                print("\n繪製MCPT分佈圖...")

            plot_save_path = save_path / "mcpt_distribution.png"

            # 使用Medium風格繪圖函數
            metric_display_names = {
                'profit_factor': 'Profit Factor',
                'sharpe_ratio': 'Sharpe Ratio',
                'win_rate': 'Win Rate',
                'total_return': 'Total Return',
                'max_drawdown': 'Max Drawdown',
                'calmar_ratio': 'Calmar Ratio',
                'sortino_ratio': 'Sortino Ratio',
                'profit_loss_ratio': 'Profit/Loss Ratio',
                'avg_trade': 'Average Trade',
                'num_trades': 'Number of Trades'
            }
            metric_name_english = metric_display_names.get(metric, metric.replace('_', ' ').title())

            plot_mcpt_distribution_medium_style(
                original_score=results['original_score'],
                permutation_scores=np.array(results['permutation_scores']),
                p_value=results['p_value'],
                metric_name=metric_name_english,
                title=f"MCPT - LSTM SOL Strategy",
                save_path=plot_save_path
            )
        except Exception as e:
            print(f"警告: 圖表繪製失敗: {e}")
        finally:
            # Ensure matplotlib resources are cleaned up
            import matplotlib.pyplot as plt
            plt.close('all')

    # 步驟7: 打印結論
    if verbose:
        print("\n" + "=" * 70)
        print("MCPT測試結論")
        print("=" * 70)
        if results['is_significant']:
            print("策略顯著優於隨機（p < 0.05）")
            print("結論: 策略可能具有真實的預測能力")
        else:
            print("策略未顯著優於隨機（p >= 0.05）")
            print("結論: 策略可能存在過度擬合")
        print("=" * 70)

    return {
        'tester': tester,
        'results': results,
        'save_path': save_path
    }


if __name__ == "__main__":
    # ========== 數據配置 ==========
    DATA_CONFIG = "SOLUSDT_1h"        # 數據配置: SOLUSDT_1h

    # ========== MCPT參數 ==========
    CONFIDENCE_THRESHOLD = 0.01       # 置信度閾值 (從優化得到)
    LOOK_BACK = 72                    # 回溯窗口 (從優化得到)
    START_DATE = "2022-10-01"         # 開始日期 (與優化相同)
    END_DATE = "2024-09-30"           # 結束日期 (與優化相同)
    N_PERMUTATIONS = 1000             # 排列次數（建議1000+）
    MODE = 'normal'                   # 測試模式
    METRIC = 'profit_factor'          # 評估指標
    TRANSACTION_COST = 0.002          # 交易成本 0.2%
    SLIPPAGE = 0.0005                 # 滑價 0.05%
    START_INDEX = 200                 # 開始排列的索引
    N_JOBS = 1                        # 並行核心數 (Windows多進程問題，暫時使用1)

    # 執行MCPT測試
    print("\n" + "=" * 70)
    print("LSTM SOL策略 - MCPT測試")
    print("=" * 70)

    try:
        mcpt_results = run_mcpt(
            data_config=DATA_CONFIG,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            look_back=LOOK_BACK,
            start_date=START_DATE,
            end_date=END_DATE,
            n_permutations=N_PERMUTATIONS,
            mode=MODE,
            metric=METRIC,
            transaction_cost=TRANSACTION_COST,
            slippage=SLIPPAGE,
            start_index=START_INDEX,
            n_jobs=N_JOBS,
            verbose=True
        )

        print("\n測試完成！")
        if mcpt_results['save_path']:
            print(f"結果已保存到: {mcpt_results['save_path']}")
        print(f"p-value: {mcpt_results['results']['p_value']:.4f}")
        print(f"顯著性: {'是' if mcpt_results['results']['is_significant'] else '否'}")

    except Exception as e:
        print(f"\n錯誤: MCPT測試失敗")
        print(f"原因: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
