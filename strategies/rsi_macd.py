from typing import Dict, Any, List, Optional
import numpy as np
from strategies.base import BaseStrategy

class RsiMacdStrategy(BaseStrategy):
    """
    RSI + MACD Trend & Momentum Strategy:
    - Calculates RSI (Relative Strength Index)
    - Calculates MACD Line, Signal Line, and Histogram
    - Buy Signal: RSI is coming out of oversold (< 35) or crossing up, AND MACD line crosses above signal line.
    - Sell Signal: RSI is overbought (> 65) or MACD crosses below signal line.
    """

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 35.0,
        rsi_overbought: float = 65.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ):
        super().__init__(name="RSI_MACD")
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        if len(prices) <= period:
            return np.full(len(prices), np.nan)
        
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs) if down != 0 else 100.0

        up_val = up
        down_val = down
        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                up_val = (up_val * (period - 1) + delta) / period
                down_val = (down_val * (period - 1)) / period
            else:
                up_val = (up_val * (period - 1)) / period
                down_val = (down_val * (period - 1) - delta) / period
            rs = up_val / down_val if down_val != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs) if down_val != 0 else 100.0

        return rsi

    @staticmethod
    def calculate_ema(series: np.ndarray, span: int) -> np.ndarray:
        alpha = 2 / (span + 1)
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for i in range(1, len(series)):
            ema[i] = alpha * series[i] + (1 - alpha) * ema[i - 1]
        return ema

    def evaluate(self, candles: List[Dict[str, Any]], current_price: float, position: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        min_candles = max(self.macd_slow + self.macd_signal, self.rsi_period) + 5
        if len(candles) < min_candles:
            return {
                "signal": "HOLD",
                "reason": f"Insufficient candles (need {min_candles}, got {len(candles)})",
                "indicators": {}
            }

        closes = np.array([c["close"] for c in candles], dtype=float)

        # 1. RSI
        rsi_series = self.calculate_rsi(closes, period=self.rsi_period)
        curr_rsi = rsi_series[-1]
        prev_rsi = rsi_series[-2]

        # 2. MACD
        fast_ema = self.calculate_ema(closes, span=self.macd_fast)
        slow_ema = self.calculate_ema(closes, span=self.macd_slow)
        macd_line = fast_ema - slow_ema
        signal_line = self.calculate_ema(macd_line, span=self.macd_signal)
        macd_hist = macd_line - signal_line

        curr_macd = macd_line[-1]
        curr_sig = signal_line[-1]
        prev_macd = macd_line[-2]
        prev_sig = signal_line[-2]
        curr_hist = macd_hist[-1]

        macd_cross_up = (prev_macd <= prev_sig) and (curr_macd > curr_sig)
        macd_cross_down = (prev_macd >= prev_sig) and (curr_macd < curr_sig)

        indicators = {
            "rsi": round(curr_rsi, 2),
            "macd": round(curr_macd, 4),
            "macd_signal": round(curr_sig, 4),
            "macd_hist": round(curr_hist, 4),
            "price": current_price
        }

        # Signal logic
        if curr_rsi < self.rsi_oversold or (macd_cross_up and curr_rsi < 55):
            return {
                "signal": "BUY",
                "reason": f"Bullish setup: RSI {curr_rsi:.1f} & MACD bullish momentum (cross={macd_cross_up})",
                "indicators": indicators
            }
        elif curr_rsi > self.rsi_overbought or (macd_cross_down and curr_rsi > 45):
            return {
                "signal": "SELL",
                "reason": f"Bearish setup: RSI {curr_rsi:.1f} & MACD bearish momentum (cross={macd_cross_down})",
                "indicators": indicators
            }

        return {
            "signal": "HOLD",
            "reason": f"Neutral: RSI {curr_rsi:.1f}, MACD {curr_macd:.2f}/{curr_sig:.2f}",
            "indicators": indicators
        }
