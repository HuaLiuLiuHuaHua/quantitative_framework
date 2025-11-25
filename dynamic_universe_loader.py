import pandas as pd
from pathlib import Path
from factors.momentum.factor import MomentumFactor

# 讀入 universe 名單
universe_path = Path('data/40.csv')
universe_df = pd.read_csv(universe_path)
universe_df['timestamp'] = pd.to_datetime(universe_df['timestamp'], dayfirst=True)

# 只示範第一個 rebalancing date
row = universe_df.iloc[1]  # 跳過 header row，取第一個有成分的日期
rebalance_date = row['timestamp']
symbols = [s for s in row[1:] if pd.notna(s) and s != '']

price_data_dict = {}
for symbol in symbols:
    # 假設你的日線價格檔案命名規則如下
    price_path = Path(f'data/{symbol}USDT_1d_2021-01-01_2025-10-13.csv')
    if price_path.exists():
        df = pd.read_csv(price_path, parse_dates=['date'])
        df = df.set_index('date')
        price_data_dict[symbol] = df
    else:
        print(f"[警告] 找不到價格檔案: {price_path}")

# 傳給 MomentumFactor 計算
factor = MomentumFactor()
factor_df = factor.calculate(price_data_dict, lookback_period=10)

print(factor_df.tail())
