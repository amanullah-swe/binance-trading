import time
from typing import Dict, Any, Optional
from strategies.base import BaseStrategy

class DcaValueAveragingStrategy(BaseStrategy):
    """
    Dollar-Cost Averaging (DCA) and Value Averaging (VA) Execution Strategy.
    
    Supports:
    1. Pure DCA: Fixed USD purchase at regular intervals.
    2. Dynamic Fear & Greed DCA: Multiplies purchase size (e.g. 2x) when F&G < 25 (Extreme Fear).
    3. Value Averaging (VA): Calculates target portfolio path: Target = step * target_increment.
       Adjusts contribution size: Buy Amount = Target Value - Current Holdings Value.
    """

    def __init__(
        self,
        mode: str = "dynamic_fng",   # "pure_dca", "dynamic_fng", "value_averaging"
        base_order_usdt: float = 50.0,
        interval_seconds: int = 3600, # e.g. 1 hour or 24 hours
        fng_multiplier: float = 2.0,
        target_increment_usdt: float = 50.0
    ):
        super().__init__(name=f"DCA_{mode.upper()}")
        self.mode = mode
        self.base_order_usdt = base_order_usdt
        self.interval_seconds = interval_seconds
        self.fng_multiplier = fng_multiplier
        self.target_increment_usdt = target_increment_usdt
        self.last_buy_time = 0.0
        self.step_counter = 0

    def evaluate(self, candles: list, current_price: float, position: Optional[Dict[str, Any]] = None, current_fng: int = 50) -> Dict[str, Any]:
        now = time.time()
        time_elapsed = now - self.last_buy_time
        
        # Check schedule
        if self.last_buy_time > 0 and time_elapsed < self.interval_seconds:
            remaining = int(self.interval_seconds - time_elapsed)
            return {
                "signal": "HOLD",
                "reason": f"DCA interval pending. Next buy in {remaining}s",
                "indicators": {"fng": current_fng, "price": current_price}
            }

        # Time to buy!
        self.step_counter += 1
        
        if self.mode == "pure_dca":
            buy_amount = self.base_order_usdt
            reason = f"Scheduled Pure DCA buy (${buy_amount:.2f} USDT)"
            
        elif self.mode == "dynamic_fng":
            if current_fng < 25:
                buy_amount = self.base_order_usdt * self.fng_multiplier
                reason = f"Extreme Fear Dynamic Buy (F&G={current_fng} < 25) -> {self.fng_multiplier}x buy (${buy_amount:.2f} USDT)"
            else:
                buy_amount = self.base_order_usdt
                reason = f"Standard DCA Buy (F&G={current_fng}) -> 1x buy (${buy_amount:.2f} USDT)"
                
        elif self.mode == "value_averaging":
            # Target portfolio value at this step
            target_value = self.step_counter * self.target_increment_usdt
            current_holdings_qty = position["qty"] if position else 0.0
            current_value = current_holdings_qty * current_price
            deficit = target_value - current_value
            
            if deficit >= 10.0:  # Min notional
                buy_amount = deficit
                reason = f"Value Averaging step #{self.step_counter}: Target ${target_value:.2f}, Current ${current_value:.2f} -> Buying deficit ${buy_amount:.2f} USDT"
            else:
                return {
                    "signal": "HOLD",
                    "reason": f"Value Averaging step #{self.step_counter}: Target ${target_value:.2f} already met by current value (${current_value:.2f}). No buy needed.",
                    "indicators": {"target_value": target_value, "current_value": current_value}
                }
        else:
            buy_amount = self.base_order_usdt
            reason = f"DCA Buy (${buy_amount:.2f} USDT)"

        self.last_buy_time = now
        return {
            "signal": "BUY",
            "buy_amount_usdt": buy_amount,
            "reason": reason,
            "indicators": {"fng": current_fng, "price": current_price, "step": self.step_counter}
        }
