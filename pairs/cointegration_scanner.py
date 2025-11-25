"""
協整性掃描工具

掃描所有可能的配對組合,找出協整的交易對
這是一個獨立的探索性分析工具
"""

import sys
from pathlib import Path
import warnings

# 添加項目根目錄到 Python 路徑
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from shared.data_loader import load_local_data
from factors.kalman_pairs.cointegration_utils import (
    calculate_hedge_ratio_ols,
    get_universe_symbols_for_date,
    test_stationarity_adf,
    calculate_hurst_exponent,
    calculate_variance_ratio,
    johansen_test
)

# 抑制 statsmodels 的 ValueWarning: No frequency information was provided
warnings.filterwarnings("ignore", message="No frequency information was provided")

logging.basicConfig(
    level=logging.INFO, # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_pair_report(result: dict):
    """打印單個配對的檢驗結果 (只顯示4個核心檢驗)"""
    
    pair_name = f"{result['asset1']}-{result['asset2']}"
    report = f"""
----------------------------------------------------------------------
配對: {pair_name} | 數據點: {result['data_points']}
----------------------------------------------------------------------

1. ADF 檢驗 (增廣迪基-福勒檢驗)
   - t-statistic: {result['adf_tstat']:.4f}
   - 臨界值 (95% 信心水準): {result['adf_critical_95']:.4f}
   - 判定: {'平穩 ✓' if result['adf_tstat'] < result['adf_critical_95'] else '非平穩 ✗'}
   - 說明: t-stat 越負 (越小於臨界值), 均值回歸的信心越強

2. 赫斯特指數 (Hurst Exponent)
   - H 值: {result['hurst']:.4f}
   - 判定: {'均值回歸 ✓' if result['hurst'] < 0.5 else ('隨機漫步 ✗' if abs(result['hurst'] - 0.5) < 0.05 else '趨勢型 ✗')}
   - 說明: H < 0.5 適合配對交易 | H = 0.5 隨機漫步 | H > 0.5 趨勢

3. 方差比檢驗 (Variance Ratio Test)
   - VR 值: {result['variance_ratio']:.4f}
   - 判定: {'均值回歸 ✓' if result['variance_ratio'] < 1 else ('隨機漫步 ✗' if abs(result['variance_ratio'] - 1) < 0.1 else '趨勢型 ✗')}
   - 說明: VR < 1 適合配對交易 | VR = 1 隨機漫步 | VR > 1 趨勢

4. 約翰森檢驗 (Johansen Trace Statistic)
   - Trace Statistic: {result['johansen_trace']:.4f}
   - 臨界值 (95% 信心水準): {result['johansen_critical_95']:.4f}
   - 判定: {'協整 ✓' if result['johansen_trace'] > result['johansen_critical_95'] else '非協整 ✗'}
   - 說明: Trace > Critical 表示存在協整關係

總體評估: {'通過所有檢驗 ✓✓✓' if result['passes_all_filters'] else '未通過所有檢驗'}
----------------------------------------------------------------------
"""
    logger.info(report)


def scan_all_pairs(
    universe_csv_path: str = "data/40.csv",
    universe_date: str = None,
    start_date: str = "2022-11-01",
    end_date: str = "2024-12-31",
    output_dir: str = None
):
    """
    掃描宇宙中所有可能的配對組合

    Args:
        universe_csv_path: 動態宇宙 CSV 路徑
        universe_date: 指定要使用的宇宙日期 (格式: "YYYY-MM-DD" 或 "DD/MM/YYYY")
                      如果為 None,使用最後一個日期
        start_date: 開始日期
        end_date: 結束日期
        output_dir: 輸出目錄,默認為 factors/kalman_pairs/results/CointegrationScan_<timestamp>/
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "factors" / "kalman_pairs" / "results" / f"CointegrationScan_{timestamp}"
        output_dir = str(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"=== 協整性掃描開始 ===")
    logger.info(f"時間範圍: {start_date} 到 {end_date}")
    logger.info(f"宇宙檔案: {universe_csv_path}")

    # 讀取宇宙
    logger.info("載入動態宇宙...")
    universe_df = pd.read_csv(universe_csv_path)
    universe_df['timestamp'] = pd.to_datetime(universe_df['timestamp'], dayfirst=True)

    # 確定要使用的宇宙日期
    if universe_date is None:
        target_date = universe_df['timestamp'].iloc[-1]
        logger.info(f"使用最後一個宇宙日期: {target_date.strftime('%Y-%m-%d')}")
    else:
        target_date = pd.to_datetime(universe_date, dayfirst=True)
        logger.info(f"使用指定宇宙日期: {target_date.strftime('%Y-%m-%d')}")

    symbols = get_universe_symbols_for_date(universe_df, target_date, top_n=40)

    logger.info(f"宇宙資產數量: {len(symbols)}")
    logger.info(f"可能配對數量: {len(symbols) * (len(symbols) - 1) // 2}")

    # 載入所有資產數據
    logger.info("載入價格數據...")
    price_data = {}
    for symbol in symbols:
        try:
            ticker = f"{symbol}USDT"
            df = load_local_data(
                symbol=ticker,
                data_source="1d",
                start_date=start_date,
                end_date=end_date
            )
            if df is not None and len(df) > 100:
                price_data[ticker] = df
            else:
                logger.warning(f"跳過 {ticker}: 數據不足")
        except Exception as e:
            logger.warning(f"載入 {symbol} 失敗: {e}")

    loaded_symbols = list(price_data.keys())
    logger.info(f"成功載入 {len(loaded_symbols)} 個資產")

    # 掃描所有配對
    logger.info("開始掃描配對...")
    results = []

    for i in range(len(loaded_symbols)):
        for j in range(i + 1, len(loaded_symbols)):
            asset1 = loaded_symbols[i]
            asset2 = loaded_symbols[j]

            try:
                price1 = price_data[asset1]['close']
                price2 = price_data[asset2]['close']

                common_idx = price1.index.intersection(price2.index)
                if len(common_idx) < 100:
                    continue

                price1_aligned = price1.loc[common_idx]
                price2_aligned = price2.loc[common_idx]

                # --- 只執行4個檢驗 ---
                # 計算對沖比率和價差
                hedge_ratio = calculate_hedge_ratio_ols(price1_aligned, price2_aligned)
                spread = price1_aligned - hedge_ratio * price2_aligned
                
                # 1. ADF 檢驗
                adf_pvalue, adf_is_stationary, adf_tstat, adf_critical_95 = test_stationarity_adf(spread)
                
                # 2. 赫斯特指數
                hurst = calculate_hurst_exponent(spread)
                
                # 3. 方差比檢驗
                variance_ratio = calculate_variance_ratio(spread, lag=2)
                
                # 4. 約翰森檢驗
                johansen_trace, johansen_critical_95, is_johansen_coint = johansen_test(price1_aligned, price2_aligned)

                # 記錄結果 (只保留4個檢驗的數據)
                result = {
                    'asset1': asset1,
                    'asset2': asset2,
                    'adf_tstat': adf_tstat,
                    'adf_critical_95': adf_critical_95,
                    'adf_pvalue': adf_pvalue,
                    'hurst': hurst,
                    'variance_ratio': variance_ratio,
                    'johansen_trace': johansen_trace,
                    'johansen_critical_95': johansen_critical_95,
                    'data_points': len(common_idx)
                }

                # 判斷是否通過所有篩選
                passes_adf = adf_tstat < adf_critical_95
                passes_hurst = hurst < 0.5 and not np.isnan(hurst)
                passes_vr = variance_ratio < 1 and not np.isnan(variance_ratio)
                passes_johansen = is_johansen_coint
                
                result['passes_all_filters'] = (
                    passes_adf and
                    passes_hurst and
                    passes_vr and
                    passes_johansen
                )

                results.append(result)

            except Exception as e:
                logger.debug(f"配對 {asset1}-{asset2} 分析失敗: {e}")
                continue

    logger.info(f"掃描完成,共分析 {len(results)} 個配對")

    if not results:
        logger.warning("未找到任何可分析的配對")
        return

    results_df = pd.DataFrame(results)

    # --- 顯示詳細報告 ---
    logger.info("\n\n\n=== 潛在交易對詳細報告 ===")
    
    # 根據多個條件對結果進行排序，找出最好的候選
    # 優先級: ADF t-stat -> Hurst -> 方差比
    sorted_df = results_df.sort_values(
        by=['adf_tstat', 'hurst', 'variance_ratio'], 
        ascending=[True, True, True]
    )
    
    # 獲取通過所有篩選的配對
    candidate_pairs = sorted_df[sorted_df['passes_all_filters']]
    
    if candidate_pairs.empty:
        logger.info("沒有找到任何通過所有篩選條件的交易對。")
        logger.info("篩選條件: ADF t-stat < Critical, Hurst < 0.5, VR < 1, Johansen Cointegrated")
        # 顯示前5個最接近的
        logger.info("\n顯示前 5 個最接近的候選配對:")
        top_5_candidates = sorted_df.head(5)
        for _, row in top_5_candidates.iterrows():
            print_pair_report(row.to_dict())
    else:
        logger.info(f"找到 {len(candidate_pairs)} 個通過所有篩選條件的交易對。")
        for _, row in candidate_pairs.iterrows():
            print_pair_report(row.to_dict())


    # --- 保存結果 ---
    logger.info("\n\n=== 保存結果 ===")
    # 保存所有配對的完整結果
    all_results_file = os.path.join(output_dir, "all_pairs_full_results.csv")
    results_df.to_csv(all_results_file, index=False)
    logger.info(f"所有配對的完整測試結果已保存: {all_results_file}")

    # 保存通過篩選的候選配對
    if not candidate_pairs.empty:
        candidate_file = os.path.join(output_dir, "candidate_pairs.csv")
        candidate_pairs.to_csv(candidate_file, index=False)
        logger.info(f"候選配對已保存: {candidate_file}")

    logger.info(f"\n所有結果已保存到: {output_dir}")
    logger.info("=== 掃描完成 ===")


if __name__ == "__main__":
    # 運行協整性掃描
    universe_path = project_root / "data" / "40.csv"

    scan_all_pairs(
        universe_csv_path=str(universe_path),
        universe_date="2022-10-30",
        start_date="2022-11-01",
        end_date="2024-12-31"
    )
