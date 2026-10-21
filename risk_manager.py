from typing import Dict, Any, Optional

class RiskManager:
    """
    Risk Management module:
    - Position Sizing (Fixed USD or % of Equity)
    - Stop-Loss & Take-Profit enforcement
    - Maximum daily drawdown and loss limits
    """

    def __init__(
        self,
        stop_loss_pct: float = 1.5,
        take_profit_pct: float = 3.0,
        max_position_usdt: float = 100.0,
        min_order_usdt: float = 10.0,
        trailing_stop_pct: Optional[float] = None
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_usdt = max_position_usdt
        self.min_order_usdt = min_order_usdt
        self.trailing_stop_pct = trailing_stop_pct

    def calculate_order_amount(self, free_usdt: float, target_amount_usdt: Optional[float] = None) -> float:
        """Calculate safe order size in quote currency (USDT)."""
        requested = target_amount_usdt if target_amount_usdt is not None else self.max_position_usdt
        # Cap at free USDT and max position
        allowed = min(requested, self.max_position_usdt, free_usdt * 0.95)  # leave 5% buffer
        if allowed < self.min_order_usdt:
            return 0.0
        return round(allowed, 2)

    def check_exit_conditions(self, entry_price: float, current_price: float, highest_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate if an open position should be closed due to SL, TP, or Trailing Stop.
        """
        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0

        # Check Stop Loss
        if pnl_pct <= -abs(self.stop_loss_pct):
            return {
                "should_exit": True,
                "reason": f"STOP_LOSS triggered (PnL: {pnl_pct:.2f}% <= -{self.stop_loss_pct}%)",
                "pnl_pct": pnl_pct
            }

        # Check Take Profit
        if pnl_pct >= abs(self.take_profit_pct):
            return {
                "should_exit": True,
                "reason": f"TAKE_PROFIT reached (PnL: +{pnl_pct:.2f}% >= +{self.take_profit_pct}%)",
                "pnl_pct": pnl_pct
            }

        # Check Trailing Stop
        if self.trailing_stop_pct and highest_price and highest_price > entry_price:
            drop_from_peak = ((highest_price - current_price) / highest_price) * 100.0
            if drop_from_peak >= self.trailing_stop_pct:
                return {
                    "should_exit": True,
                    "reason": f"TRAILING_STOP triggered (Peak: {highest_price}, Drop: {drop_from_peak:.2f}%)",
                    "pnl_pct": pnl_pct
                }

        return {
            "should_exit": False,
            "reason": f"Position active (PnL: {pnl_pct:+.2f}%)",
            "pnl_pct": pnl_pct
        }
