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

    Example:
        >>> metrics = calculate_all_metrics(
        ...     returns=strategy_returns,
        ...     signals=signals,
        ...     periods_per_year=8760  # 1小時K線
        ... )
        >>> print(f"夏普比率: {metrics['sharpe_ratio']:.3f}")
    """
    returns = returns.dropna()

    if len(returns) == 0:
        return _get_zero_metrics()

    # 1. 總收益率
    total_return = (1 + returns).prod() - 1

    # 2. 年化收益率
    if len(returns) > 0:
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    else:
        annualized_return = 0

    # 3. 年化波動率
    volatility = returns.std() * np.sqrt(periods_per_year)

    # 4. 夏普比率
    if volatility > 0:
        sharpe_ratio = annualized_return / volatility
    else:
        sharpe_ratio = 0

    # 5. 最大回撤
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 6. 卡瑪比率
    if max_drawdown != 0:
        calmar_ratio = annualized_return / abs(max_drawdown)
    else:
        calmar_ratio = 0

    # 7-10. 勝率和獲利因子相關指標
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]

    win_rate = len(winning_returns) / len(returns) if len(returns) > 0 else 0
    avg_win = winning_returns.mean() if len(winning_returns) > 0 else 0
    avg_loss = losing_returns.mean() if len(losing_returns) > 0 else 0

    # 10. 獲利因子
    gross_profit = winning_returns.sum()
    gross_loss = abs(losing_returns.sum())

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = np.inf if gross_profit > 0 else 0

    # 11-13. 交易次數
    if signals is not None:
        # 計算實際的交易次數（開倉或換倉）
        # 從0進入非0，或從非0切換到另一個非0都算作交易
        # 但0→0不算交易
        signal_changes = signals.diff()
        # 計算所有倉位變化（但排除0→0的情況）
        # 倉位變化且（當前非0或前一個非0）才算交易
        trades_mask = (signal_changes != 0) & ((signals != 0) | (signals.shift(1) != 0))
        total_trades = trades_mask.sum()

        # 更精確的獲利/虧損交易計算
        winning_trades = len(winning_returns)
        losing_trades = len(losing_returns)
    else:
        # 如果沒有提供signals，使用returns作為代理
        total_trades = len(returns)
        winning_trades = len(winning_returns)
        losing_trades = len(losing_returns)

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
