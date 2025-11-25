"""
Performance Metrics Calculator
績效指標計算器 - Medium框架的13個指標

指標列表：
1. total_return - 總收益率
2. annualized_return - 年化收益率
3. volatility - 年化波動率
4. sharpe_ratio - 夏普比率
5. max_drawdown - 最大回撤
6. calmar_ratio - 卡瑪比率
7. win_rate - 勝率
8. avg_win - 平均獲利
9. avg_loss - 平均虧損
10. profit_factor - 獲利因子
11. total_trades - 總交易次數
12. winning_trades - 獲利交易次數
13. losing_trades - 虧損交易次數
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_all_metrics(
    returns: pd.Series,
    signals: pd.Series = None,
    initial_capital: float = 100000,
    periods_per_year: int = 8760,  # 1小時K線: 24*365 = 8760
) -> Dict[str, float]:
    """
    計算完整的績效指標（Medium框架的13個指標）

    Args:
        returns: 收益率序列（淨收益，已扣除交易成本）
        signals: 交易信號序列（可選，用於計算交易次數）
        initial_capital: 初始資金
        periods_per_year: 每年的時間週期數
            - 1小時K線: 24 * 365 = 8760
            - 日K線: 252 (交易日)
            - 分鐘K線: 需要根據實際情況調整

    Returns:
        Dict[str, float]: 包含13個績效指標的字典
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return _get_zero_metrics()

    # --- Portfolio-level metrics (Unaffected by the fix) ---
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns.mean()) ** periods_per_year - 1 if len(returns) > 0 else 0
    volatility = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # --- Trade-level metrics (FIXED LOGIC) ---
    if signals is not None and not signals.eq(0).all():
        # Use previous period's signal to determine position for the current period's return
        positions = signals.shift(1).fillna(0)
        
        # A trade is a sequence of identical non-zero positions.
        # A new trade starts when the position changes.
        trades = positions.diff().ne(0).cumsum()
        
        # We only analyze periods where a position was held
        active_positions = positions[positions != 0]
        
        if not active_positions.empty:
            active_returns = returns[positions != 0]
            active_trades = trades[positions != 0]

            # Calculate PnL for each individual trade by compounding its returns
            per_trade_pnl = active_returns.groupby(active_trades).apply(lambda x: (1 + x).prod() - 1)

            winning_trades_pnl = per_trade_pnl[per_trade_pnl > 0]
            losing_trades_pnl = per_trade_pnl[per_trade_pnl < 0]

            total_trades = len(per_trade_pnl)
            winning_trades = len(winning_trades_pnl)
            losing_trades = len(losing_trades_pnl)

            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            avg_win = winning_trades_pnl.mean() if winning_trades > 0 else 0
            avg_loss = losing_trades_pnl.mean() if losing_trades > 0 else 0
            
            # Profit factor is the sum of all positive *period returns* during active trades
            # divided by the absolute sum of all negative *period returns* during active trades.
            gross_profit = active_returns[active_returns > 0].sum()
            gross_loss = abs(active_returns[active_returns < 0].sum())

            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            else:
                profit_factor = np.inf if gross_profit > 0 else 0
        else:
            # No positions were ever held
            total_trades, winning_trades, losing_trades, win_rate, avg_win, avg_loss, profit_factor = (0, 0, 0, 0, 0, 0, 0)
    else:
        # No trades if no signals or all signals are zero
        total_trades, winning_trades, losing_trades, win_rate, avg_win, avg_loss, profit_factor = (0, 0, 0, 0, 0, 0, 0)

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar_ratio,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
    }


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    計算獲利因子（單獨函數，用於優化目標）

    Args:
        returns: 收益率序列

    Returns:
        float: 獲利因子 = 總獲利 / 總虧損
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return 0

    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]

    gross_profit = winning_returns.sum()
    gross_loss = abs(losing_returns.sum())

    if gross_loss > 0:
        return gross_profit / gross_loss
    else:
        return np.inf if gross_profit > 0 else 0


def calculate_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 8760
) -> float:
    """
    計算夏普比率（單獨函數，用於優化目標）

    Args:
        returns: 收益率序列
        periods_per_year: 每年的時間週期數

    Returns:
        float: 夏普比率
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return 0

    annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    volatility = returns.std() * np.sqrt(periods_per_year)

    if volatility > 0:
        return annualized_return / volatility
    else:
        return 0


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    計算最大回撤（單獨函數，用於優化目標）

    Args:
        returns: 收益率序列

    Returns:
        float: 最大回撤（負值）
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return 0

    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max

    return drawdown.min()


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 8760
) -> float:
    """
    計算卡瑪比率（單獨函數，用於優化目標）

    Args:
        returns: 收益率序列
        periods_per_year: 每年的時間週期數

    Returns:
        float: 卡瑪比率
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return 0

    annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    max_dd = calculate_max_drawdown(returns)

    if max_dd != 0:
        return annualized_return / abs(max_dd)
    else:
        return 0


def _get_zero_metrics() -> Dict[str, float]:
    """返回全零的指標字典（用於無效數據情況）"""
    return {
        'total_return': 0.0,
        'annualized_return': 0.0,
        'volatility': 0.0,
        'sharpe_ratio': 0.0,
        'max_drawdown': 0.0,
        'calmar_ratio': 0.0,
        'win_rate': 0.0,
        'avg_win': 0.0,
        'avg_loss': 0.0,
        'profit_factor': 0.0,
        'total_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
    }


def print_metrics(metrics: Dict[str, float], title: str = "績效指標"):
    """
    格式化打印績效指標

    Args:
        metrics: 績效指標字典
        title: 報告標題
    """
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"總收益率:      {metrics['total_return']*100:>10.2f}%")
    print(f"年化收益率:    {metrics['annualized_return']*100:>10.2f}%")
    print(f"年化波動率:    {metrics['volatility']*100:>10.2f}%")
    print(f"夏普比率:      {metrics['sharpe_ratio']:>10.3f}")
    print(f"最大回撤:      {metrics['max_drawdown']*100:>10.2f}%")
    print(f"卡瑪比率:      {metrics['calmar_ratio']:>10.3f}")
    print("-" * 60)
    print(f"勝率:          {metrics['win_rate']*100:>10.2f}%")
    print(f"平均獲利:      {metrics['avg_win']*100:>10.4f}%")
    print(f"平均虧損:      {metrics['avg_loss']*100:>10.4f}%")
    print(f"獲利因子:      {metrics['profit_factor']:>10.3f}")
    print("-" * 60)
    print(f"總交易次數:    {metrics['total_trades']:>10}")
    print(f"獲利交易:      {metrics['winning_trades']:>10}")
    print(f"虧損交易:      {metrics['losing_trades']:>10}")
    print("=" * 60)


if __name__ == "__main__":
    # 測試範例
    print("績效指標計算器測試\n")

    # 創建模擬收益率序列
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0001, 0.02, 1000))

    # 計算所有指標
    metrics = calculate_all_metrics(returns, periods_per_year=8760)

    # 打印指標
    print_metrics(metrics, "模擬策略績效")

    # 測試單獨函數
    print("\n單獨指標測試:")
    print(f"獲利因子: {calculate_profit_factor(returns):.3f}")
    print(f"夏普比率: {calculate_sharpe_ratio(returns):.3f}")
    print(f"最大回撤: {calculate_max_drawdown(returns)*100:.2f}%")
    print(f"卡瑪比率: {calculate_calmar_ratio(returns):.3f}")
