from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseStrategy(ABC):
    """
    Abstract Base Class for Trading Strategies.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, candles: List[Dict[str, Any]], current_price: float, position: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze candlestick data and return a signal.
        Expected return format:
        {
            "signal": "BUY" | "SELL" | "HOLD",
            "reason": str,
            "indicators": Dict[str, Any]
        }
        """
        pass
