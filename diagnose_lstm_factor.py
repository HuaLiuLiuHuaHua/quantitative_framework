"""
LSTM 因子診斷工具

檢查你的因子是否有問題
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[0]))

import pandas as pd
import numpy as np
import torch
from shared.data_loader import load_local_data
from strategies.lstm_momentum.strategy import LSTMMomentumStrategy


def diagnose():
    print("=" * 70)
    print("LSTM 因子診斷工具")
    print("=" * 70)

    # 1. 檢查模型文件
    print("\n[1/5] 檢查模型文件...")
    models_dir = Path('strategies/lstm_momentum/models')

    if not models_dir.exists():
        print("  ❌ 模型目錄不存在: strategies/lstm_momentum/models/")
        print("  💡 解決: 運行 python strategies/lstm_momentum/train.py")
        return

    required_files = ['feature_extractor.pth', 'config.json', 'scaler.pkl']
    for fname in required_files:
        fpath = models_dir / fname
        if fpath.exists():
            size = fpath.stat().st_size / 1024 / 1024  # MB
            print(f"  ✅ {fname} ({size:.1f} MB)")
        else:
            print(f"  ❌ 缺少: {fname}")
            return

    # 2. 檢查數據
    print("\n[2/5] 檢查測試數據...")
    try:
        data = load_local_data(
            symbol='BTCUSDT',
            data_source='1h',
            start_date='2024-01-01',
            end_date='2024-12-31',
            lookback_days=100
        )
        if data is None or len(data) == 0:
            print("  ❌ 無法加載測試數據")
            print("  💡 解決: 確保 data/ 目錄有 BTCUSDT_1h_*.csv 文件")
            return

        print(f"  ✅ 加載成功 ({len(data)} 根 K 線)")
        print(f"     日期範圍: {data.index[0]} ~ {data.index[-1]}")
        print(f"     缺失值: {data.isna().sum().sum()} 個")
    except Exception as e:
        print(f"  ❌ 加載失敗: {e}")
        return

    # 3. 檢查因子計算
    print("\n[3/5] 檢查因子計算...")
    try:
        strategy = LSTMMomentumStrategy()
        factor_values = strategy.factor.calculate(data)

        print(f"  ✅ 因子計算成功")
        print(f"     形狀: {factor_values.shape}")
        print(f"     缺失值: {factor_values.isna().sum()}")
        print(f"     統計:")
        print(f"       平均值: {factor_values.mean():.4f}")
        print(f"       標準差: {factor_values.std():.4f}")
        print(f"       最小值: {factor_values.min():.4f}")
        print(f"       最大值: {factor_values.max():.4f}")

        # 檢查因子分佈
        if abs(factor_values.mean()) > 0.1:
            print(f"  ⚠️  因子平均值偏離 0: {factor_values.mean():.4f}")
            print(f"     💡 建議: 檢查標準化是否正確")
    except Exception as e:
        print(f"  ❌ 因子計算失敗: {e}")
        return

    # 4. 檢查信號生成
    print("\n[4/5] 檢查信號生成...")
    try:
        signals = strategy.generate_signals(data, threshold=0.0)

        n_long = (signals == 1).sum()
        n_short = (signals == -1).sum()
        n_neutral = (signals == 0).sum()

        print(f"  ✅ 信號生成成功")
        print(f"     做多: {n_long} ({n_long/len(signals)*100:.1f}%)")
        print(f"     做空: {n_short} ({n_short/len(signals)*100:.1f}%)")
        print(f"     空倉: {n_neutral} ({n_neutral/len(signals)*100:.1f}%)")

        # 檢查信號質量
        if n_long + n_short < len(signals) * 0.1:
            print(f"  ⚠️  信號太稀疏 (只有 {(n_long+n_short)/len(signals)*100:.1f}%)")
            print(f"     💡 建議: 降低 threshold 讓信號更頻繁")

        if n_long + n_short > len(signals) * 0.9:
            print(f"  ⚠️  信號太頻繁 (有 {(n_long+n_short)/len(signals)*100:.1f}%)")
            print(f"     💡 建議: 提高 threshold 讓信號更有選擇性")
    except Exception as e:
        print(f"  ❌ 信號生成失敗: {e}")
        return

    # 5. 檢查因子預測力 (IC)
    print("\n[5/5] 檢查因子預測力...")
    try:
        future_return = data['close'].pct_change(5).shift(-5)
        ic = factor_values.corr(future_return)

        print(f"  信息係數 (IC): {ic:.4f}")

        if ic > 0.05:
            print(f"  ✅ 因子有預測力！(IC > 0.05)")
        elif ic > 0.02:
            print(f"  ⚠️  因子預測力一般 (0.02 < IC < 0.05)")
            print(f"     💡 建議: 嘗試修改 LSTM 參數或添加更多基礎因子")
        else:
            print(f"  ❌ 因子基本沒有預測力 (IC < 0.02)")
            print(f"     💡 建議: 重新訓練模型或重新設計因子")
    except Exception as e:
        print(f"  ⚠️  無法計算 IC: {e}")

    # 總體診斷
    print("\n" + "=" * 70)
    print("診斷結果")
    print("=" * 70)

    if ic > 0.05:
        print("✅ 恭喜！你的因子看起來不錯！")
        print("\n下一步:")
        print("  1. 運行 python strategies/lstm_momentum/test_optimization_random.py")
        print("     (尋找最優的 threshold 參數)")
        print("  2. 運行 python strategies/lstm_momentum/test_walkforward_random.py")
        print("     (驗證樣本外效果)")
        print("  3. 運行 python strategies/lstm_momentum/test_mcpt.py")
        print("     (統計檢驗)")
    elif ic > 0.02:
        print("⚠️  因子有一定潛力，但還需要改進")
        print("\n建議:")
        print("  1. 嘗試調整 LSTM 參數")
        print("     - 增加 sequence_length (e.g., 120)")
        print("     - 增加 hidden_dim (e.g., 256)")
        print("  2. 添加更多基礎因子到決策樹")
        print("  3. 嘗試不同的資產進行訓練")
    else:
        print("❌ 因子基本沒有預測力")
        print("\n建議:")
        print("  1. 檢查訓練數據品質")
        print("  2. 驗證標籤構造是否正確")
        print("  3. 嘗試更複雜的模型 (Attention, Transformer)")
        print("  4. 使用多個資產組合訓練")

    print("\n詳細幫助請閱讀:")
    print("  - QUICK_START.md (快速開始)")
    print("  - LSTM_FACTOR_GUIDE.md (詳細指南)")


if __name__ == '__main__':
    diagnose()
