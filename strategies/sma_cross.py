from typing import Dict, Any, List, Optional
import numpy as np
from strategies.base import BaseStrategy

class SmaCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy:
    - Calculates Fast MA (e.g. 9 period) and Slow MA (e.g. 21 period).
    - Buy Signal: Fast MA crosses above Slow MA (Golden Cross).
    - Sell Signal: Fast MA crosses below Slow MA (Death Cross).
    """

    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        super().__init__(name="SMA_CROSSOVER")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def evaluate(self, candles: List[Dict[str, Any]], current_price: float, position: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if len(candles) < self.slow_period + 2:
            return {
                "signal": "HOLD",
                "reason": f"Need at least {self.slow_period + 2} candles",
                "indicators": {}
            }

        closes = np.array([c["close"] for c in candles], dtype=float)
        
        fast_prev = np.mean(closes[-self.fast_period - 1 : -1])
        fast_curr = np.mean(closes[-self.fast_period :])
        slow_prev = np.mean(closes[-self.slow_period - 1 : -1])
        slow_curr = np.mean(closes[-self.slow_period :])

        indicators = {
            "fast_ma": round(fast_curr, 2),
            "slow_ma": round(slow_curr, 2),
            "price": current_price
        }

        # Golden Cross
        if fast_prev <= slow_prev and fast_curr > slow_curr:
            return {
                "signal": "BUY",
                "reason": f"Golden Cross: Fast MA ({fast_curr:.2f}) crossed above Slow MA ({slow_curr:.2f})",
                "indicators": indicators
            }
        # Death Cross
        elif fast_prev >= slow_prev and fast_curr < slow_curr:
            return {
                "signal": "SELL",
                "reason": f"Death Cross: Fast MA ({fast_curr:.2f}) crossed below Slow MA ({slow_curr:.2f})",
                "indicators": indicators
            }

        trend = "Bullish" if fast_curr > slow_curr else "Bearish"
        return {
            "signal": "HOLD",
            "reason": f"Trend: {trend} (Fast: {fast_curr:.2f}, Slow: {slow_curr:.2f})",
            "indicators": indicators
        }
