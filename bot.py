import time
import math
import logging
from typing import Dict, Any, Optional
from client import BinanceDemoClient
from risk_manager import RiskManager
from strategies.base import BaseStrategy
from strategies.rsi_macd import RsiMacdStrategy
from strategies.sma_cross import SmaCrossoverStrategy
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TradingBot")

class BinanceTradingBot:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        strategy: Optional[BaseStrategy] = None,
        trade_amount_usdt: float = 50.0,
        stop_loss_pct: float = 1.5,
        take_profit_pct: float = 3.0,
        timeframe: str = "1m",
        client: Optional[BinanceDemoClient] = None
    ):
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.trade_amount_usdt = trade_amount_usdt
        self.client = client or BinanceDemoClient()
        self.strategy = strategy or RsiMacdStrategy()
        self.risk_manager = RiskManager(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            max_position_usdt=trade_amount_usdt
        )
        
        # Symbol filter precision
        self.base_asset = ""
        self.quote_asset = ""
        self.step_size = 0.00001
        self.tick_size = 0.01
        self.min_qty = 0.00001
        self.min_notional = 5.0
        self._load_symbol_filters()

        # Position tracking
        self.position: Optional[Dict[str, Any]] = None
        self._init_position_from_balance()

    def _load_symbol_filters(self):
        try:
            info = self.client.get_exchange_info(self.symbol)
            s_info = info["symbols"][0]
            self.base_asset = s_info["baseAsset"]
            self.quote_asset = s_info["quoteAsset"]

            for f in s_info.get("filters", []):
                ftype = f["filterType"]
                if ftype == "LOT_SIZE":
                    self.step_size = float(f["stepSize"])
                    self.min_qty = float(f["minQty"])
                elif ftype == "PRICE_FILTER":
                    self.tick_size = float(f["tickSize"])
                elif ftype in ("NOTIONAL", "MIN_NOTIONAL"):
                    self.min_notional = float(f.get("minNotional", 5.0))
            logger.info(f"Loaded filters for {self.symbol}: Base={self.base_asset}, Quote={self.quote_asset}, stepSize={self.step_size}, minNotional={self.min_notional}")
        except Exception as e:
            logger.warning(f"Could not load symbol filters: {e}. Using defaults.")

    def _round_step(self, value: float, step: float) -> float:
        if step <= 0:
            return value
        precision = max(0, int(round(-math.log10(step))))
        qty = math.floor(value / step) * step
        return round(qty, precision)

    def _init_position_from_balance(self):
        try:
            balances = self.client.get_balances(non_zero_only=True)
            base_bal = balances.get(self.base_asset, {}).get("free", 0.0)
            price = self.client.get_ticker_price(self.symbol)
            notional = base_bal * price

            if notional >= self.min_notional:
                self.position = {
                    "symbol": self.symbol,
                    "qty": base_bal,
                    "entry_price": price,
                    "highest_price": price,
                    "entry_time": time.time()
                }
                logger.info(f"Detected existing position: {base_bal} {self.base_asset} (~${notional:.2f} USDT) at est. price ${price:,.2f}")
            else:
                self.position = None
        except Exception as e:
            logger.warning(f"Failed to check initial balance: {e}")

    def execute_buy(self, current_price: float, reason: str):
        balances = self.client.get_balances(non_zero_only=True)
        quote_bal = balances.get(self.quote_asset, {}).get("free", 0.0)
        
        target_usdt = self.risk_manager.calculate_order_amount(quote_bal, self.trade_amount_usdt)
        if target_usdt < self.min_notional:
            logger.warning(f"Insufficient {self.quote_asset} balance (${quote_bal:.2f}) to place order (need min ${self.min_notional}).")
            return

        raw_qty = target_usdt / current_price
        order_qty = self._round_step(raw_qty, self.step_size)

        if order_qty < self.min_qty or (order_qty * current_price) < self.min_notional:
            logger.warning(f"Calculated quantity {order_qty} is below minimum limits.")
            return

        logger.info(f">>> EXECUTING BUY: {order_qty} {self.base_asset} (~${target_usdt:.2f} {self.quote_asset}) | Reason: {reason}")
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side="BUY",
                order_type="MARKET",
                quantity=order_qty
            )
            logger.info(f"[SUCCESS] BUY Order Executed! OrderID: {order.get('orderId')}, Status: {order.get('status')}")
            
            # Fill details
            executed_qty = float(order.get("executedQty", order_qty))
            cummulative_quote = float(order.get("cummulativeQuoteQty", target_usdt))
            fill_price = cummulative_quote / executed_qty if executed_qty > 0 else current_price

            self.position = {
                "symbol": self.symbol,
                "qty": executed_qty,
                "entry_price": fill_price,
                "highest_price": fill_price,
                "entry_time": time.time()
            }
        except Exception as e:
            logger.error(f"[FAILED] Error placing BUY order: {e}")

    def execute_sell(self, current_price: float, reason: str):
        if not self.position or self.position["qty"] <= 0:
            logger.info("No open position to sell.")
            return

        balances = self.client.get_balances(non_zero_only=True)
        base_bal = balances.get(self.base_asset, {}).get("free", 0.0)
        available_qty = min(self.position["qty"], base_bal)
        order_qty = self._round_step(available_qty, self.step_size)

        if order_qty < self.min_qty or (order_qty * current_price) < self.min_notional:
            logger.warning(f"Sell quantity {order_qty} is below minimum notional. Clearing tracked position.")
            self.position = None
            return

        logger.info(f">>> EXECUTING SELL: {order_qty} {self.base_asset} (~${order_qty * current_price:.2f} {self.quote_asset}) | Reason: {reason}")
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side="SELL",
                order_type="MARKET",
                quantity=order_qty
            )
            logger.info(f"[SUCCESS] SELL Order Executed! OrderID: {order.get('orderId')}, Status: {order.get('status')}")
            
            pnl_pct = ((current_price - self.position["entry_price"]) / self.position["entry_price"]) * 100.0
            pnl_val = (current_price - self.position["entry_price"]) * order_qty
            logger.info(f"Trade Closed. Realized PnL: {pnl_pct:+.2f}% (${pnl_val:+.2f} {self.quote_asset})")
            self.position = None
        except Exception as e:
            logger.error(f"[FAILED] Error placing SELL order: {e}")

    def tick(self) -> Dict[str, Any]:
        """Perform one iteration of evaluation: fetch candles, evaluate strategy & risk, act."""
        try:
            current_price = self.client.get_ticker_price(self.symbol)
            candles = self.client.get_klines(self.symbol, interval=self.timeframe, limit=100)

            # Update highest price if position open
            if self.position:
                if current_price > self.position["highest_price"]:
                    self.position["highest_price"] = current_price
                
                # Check Risk Manager exits first (Stop Loss / Take Profit)
                exit_check = self.risk_manager.check_exit_conditions(
                    entry_price=self.position["entry_price"],
                    current_price=current_price,
                    highest_price=self.position["highest_price"]
                )
                if exit_check["should_exit"]:
                    logger.info(f"[RISK ALERT] {exit_check['reason']}")
                    self.execute_sell(current_price, reason=exit_check["reason"])
                    return {
                        "action": "SELL",
                        "price": current_price,
                        "reason": exit_check["reason"]
                    }

            # Evaluate Strategy
            eval_res = self.strategy.evaluate(candles, current_price, self.position)
            signal = eval_res.get("signal", "HOLD")
            reason = eval_res.get("reason", "")
            indicators = eval_res.get("indicators", {})

            pnl_str = ""
            if self.position:
                pnl = ((current_price - self.position["entry_price"]) / self.position["entry_price"]) * 100.0
                pnl_str = f" | Open PnL: {pnl:+.2f}%"

            logger.info(f"[{self.symbol} @ ${current_price:,.2f}] Signal: {signal:4s}{pnl_str} | Reason: {reason}")

            if signal == "BUY" and self.position is None:
                self.execute_buy(current_price, reason)
                return {"action": "BUY", "price": current_price, "reason": reason, "indicators": indicators}
            elif signal == "SELL" and self.position is not None:
                self.execute_sell(current_price, reason)
                return {"action": "SELL", "price": current_price, "reason": reason, "indicators": indicators}

            return {"action": "HOLD", "price": current_price, "reason": reason, "indicators": indicators}

        except Exception as e:
            logger.error(f"Error during tick evaluation: {e}")
            return {"action": "ERROR", "error": str(e)}

    def run(self, poll_interval_seconds: int = 15, max_ticks: Optional[int] = None):
        """Run bot continuous loop."""
        logger.info(f"Starting trading bot loop for {self.symbol} using {self.strategy.name} strategy.")
        logger.info(f"Trade size: ${self.trade_amount_usdt} USDT | Poll interval: {poll_interval_seconds}s | Timeframe: {self.timeframe}")
        logger.info(f"Risk Limits: Stop Loss: -{self.risk_manager.stop_loss_pct}% | Take Profit: +{self.risk_manager.take_profit_pct}%")
        
        ticks = 0
        try:
            while True:
                self.tick()
                ticks += 1
                if max_ticks and ticks >= max_ticks:
                    logger.info(f"Completed {ticks} ticks. Exiting loop.")
                    break
                time.sleep(poll_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
