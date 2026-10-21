#!/usr/bin/env python3
"""
Autonomous Adaptive Profit-Generating Trading Engine (Buys & Sells).
- Uses multi-indicator signals (RSI + MACD + EMA Trend + Fear/Greed)
- Executes real BUY and SELL orders on Binance Demo
- Employs dynamic Take-Profit, Trailing Stop, and Stop-Loss
- Feeds round-trip trade results to TradeLearningEngine to learn and adapt
- Robust exception handling and automatic network recovery
"""

import os
import sys
import time
import math
import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from client import BinanceDemoClient
from learning_engine import TradeLearningEngine
from dca_analytics import fetch_fear_and_greed

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

STATE_FILE = DATA_DIR / "auto_trader_state.json"
LOG_FILE = LOGS_DIR / "auto_trader.log"

# Setup dual logging
logger = logging.getLogger("AutoTrader")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

c_handler = logging.StreamHandler(sys.stdout)
c_handler.setFormatter(formatter)
logger.addHandler(c_handler)

f_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
f_handler.setFormatter(formatter)
logger.addHandler(f_handler)

class AutoProfitTrader:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        trade_amount_usdt: float = 50.0,
        timeframe: str = "1m",
        poll_interval: int = 10,
        max_position_usdt: float = 250.0
    ):
        self.symbol = symbol.upper()
        self.trade_amount_usdt = trade_amount_usdt
        self.timeframe = timeframe
        self.poll_interval = poll_interval
        self.max_position_usdt = max_position_usdt
        
        self.client = BinanceDemoClient()
        self.learner = TradeLearningEngine()
        
        # Load exchange precision filters
        self._load_symbol_filters()
        
        # Open position state
        self.active_position: Optional[Dict[str, Any]] = None
        self._restore_position_from_state()

    def _load_symbol_filters(self):
        try:
            info = self.client.get_exchange_info(self.symbol)
            s = info["symbols"][0]
            self.base_asset = s["baseAsset"]
            self.quote_asset = s["quoteAsset"]
            self.step_size = 0.00001
            self.min_qty = 0.00001
            self.min_notional = 5.0
            for f in s.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    self.step_size = float(f["stepSize"])
                    self.min_qty = float(f["minQty"])
                elif f["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
                    self.min_notional = float(f.get("minNotional", 5.0))
            self.precision = max(0, int(round(-math.log10(self.step_size)))) if self.step_size > 0 else 5
            logger.info(f"Loaded filters for {self.symbol}: Base={self.base_asset}, Quote={self.quote_asset}, step={self.step_size}, minNotional={self.min_notional}")
        except Exception as e:
            logger.error(f"Error loading filters: {e}")
            self.base_asset = "BTC"
            self.quote_asset = "USDT"
            self.step_size = 0.00001
            self.precision = 5
            self.min_notional = 5.0

    def _restore_position_from_state(self):
        # Check current balance on exchange
        try:
            balances = self.client.get_balances(non_zero_only=True)
            base_bal = balances.get(self.base_asset, {}).get("free", 0.0)
            price = self.client.get_ticker_price(self.symbol)
            notional = base_bal * price

            if STATE_FILE.exists():
                with open(STATE_FILE, "r") as f:
                    s = json.load(f)
                    pos = s.get("active_position")
                    if pos and base_bal >= self.min_qty:
                        self.active_position = pos
                        logger.info(f"Restored open position from state: {pos['qty']} {self.base_asset} @ entry ${pos['entry_price']:,.2f}")
                        return

            if notional >= self.min_notional:
                self.active_position = {
                    "symbol": self.symbol,
                    "qty": base_bal,
                    "entry_price": price,
                    "cost_usdt": notional,
                    "highest_price": price,
                    "entry_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "entry_indicators": {"restored": True}
                }
                logger.info(f"Initialized position from existing exchange balance: {base_bal} {self.base_asset} (~${notional:.2f})")
        except Exception as e:
            logger.warning(f"Could not restore position: {e}")

    def _round_step(self, val: float) -> float:
        qty = math.floor(val / self.step_size) * self.step_size
        return round(qty, self.precision)

    def calculate_indicators(self, candles: list) -> Dict[str, Any]:
        """Compute RSI, MACD, EMA(9), EMA(21)."""
        closes = np.array([c["close"] for c in candles], dtype=float)
        
        # 1. RSI (14)
        period = 14
        deltas = np.diff(closes)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(closes)
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs) if down != 0 else 100.0
        up_val, down_val = up, down
        for i in range(period, len(closes)):
            delta = deltas[i - 1]
            if delta > 0:
                up_val = (up_val * (period - 1) + delta) / period
                down_val = (down_val * (period - 1)) / period
            else:
                up_val = (up_val * (period - 1)) / period
                down_val = (down_val * (period - 1) - delta) / period
            rs = up_val / down_val if down_val != 0 else 0
            rsi[i] = 100.0 - 100.0 / (1.0 + rs) if down_val != 0 else 100.0
        curr_rsi = rsi[-1]

        # 2. EMAs
        def ema(arr, span):
            alpha = 2 / (span + 1)
            res = np.zeros_like(arr)
            res[0] = arr[0]
            for i in range(1, len(arr)):
                res[i] = alpha * arr[i] + (1 - alpha) * res[i - 1]
            return res

        ema9 = ema(closes, 9)[-1]
        ema21 = ema(closes, 21)[-1]
        
        # 3. MACD
        fast_ema = ema(closes, 12)
        slow_ema = ema(closes, 26)
        macd_line = fast_ema - slow_ema
        sig_line = ema(macd_line, 9)
        macd_hist = macd_line - sig_line

        return {
            "rsi": round(curr_rsi, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "macd": round(macd_line[-1], 4),
            "macd_sig": round(sig_line[-1], 4),
            "macd_hist": round(macd_hist[-1], 4),
            "macd_hist_prev": round(macd_hist[-2], 4),
            "price": closes[-1]
        }

    def evaluate_entry(self, indicators: Dict[str, Any], params: Dict[str, Any], current_fng: int) -> Dict[str, Any]:
        """Check if multi-factor buy conditions are satisfied."""
        rsi = indicators["rsi"]
        hist = indicators["macd_hist"]
        hist_prev = indicators["macd_hist_prev"]
        ema9 = indicators["ema9"]
        ema21 = indicators["ema21"]
        price = indicators["price"]

        rsi_oversold_threshold = params.get("rsi_oversold", 32.0)
        
        reasons = []
        score = 0

        # Factor 1: RSI in oversold territory or turning up from oversold
        if rsi <= rsi_oversold_threshold:
            score += 2
            reasons.append(f"RSI oversold ({rsi} <= {rsi_oversold_threshold})")
        elif rsi <= 45 and hist > hist_prev:
            score += 1
            reasons.append(f"RSI low & rising momentum ({rsi})")

        # Factor 2: MACD momentum turnaround
        if hist > 0 and hist_prev <= 0:
            score += 2
            reasons.append("MACD bullish cross")
        elif hist > hist_prev and hist < 0:
            score += 1
            reasons.append("MACD histogram ticking up (seller exhaustion)")

        # Factor 3: EMA trend alignment
        if price > ema9 and ema9 > ema21:
            score += 1
            reasons.append("Price above 9 EMA & Golden alignment")

        # Factor 4: Extreme Fear discount boost
        if current_fng < 25:
            score += 1
            reasons.append(f"Extreme Fear discount (F&G={current_fng})")

        should_buy = score >= params.get("min_confirmation_score", 2)
        return {
            "should_buy": should_buy,
            "score": score,
            "reason": " + ".join(reasons) if should_buy else "Awaiting setup confirmation"
        }

    def evaluate_exit(self, position: Dict[str, Any], current_price: float, indicators: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if open position should be sold for profit or protected by stop loss."""
        entry_price = position["entry_price"]
        highest_price = position.get("highest_price", entry_price)
        
        pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
        
        tp_pct = params.get("take_profit_pct", 1.2)
        sl_pct = params.get("stop_loss_pct", 1.0)
        trailing_pct = params.get("trailing_stop_pct", 0.5)

        # 1. Take-Profit hit
        if pnl_pct >= tp_pct:
            return {
                "should_sell": True,
                "reason": f"TAKE_PROFIT target reached (+{pnl_pct:.2f}% >= +{tp_pct}%)",
                "pnl_pct": pnl_pct
            }

        # 2. Trailing Stop (locks profit if pulled back after reaching peak profit)
        if pnl_pct >= 0.6 and highest_price > entry_price:
            drop_from_peak = ((highest_price - current_price) / highest_price) * 100.0
            if drop_from_peak >= trailing_pct:
                return {
                    "should_sell": True,
                    "reason": f"TRAILING_STOP triggered (Peak: ${highest_price:,.2f}, Pullback: -{drop_from_peak:.2f}%)",
                    "pnl_pct": pnl_pct
                }

        # 3. Technical exit (Overbought RSI + Bearish MACD divergence)
        rsi = indicators.get("rsi", 50)
        overbought_thresh = params.get("rsi_overbought", 68.0)
        if rsi >= overbought_thresh and pnl_pct > 0.4:
            return {
                "should_sell": True,
                "reason": f"TECHNICAL_OVERBOUGHT exit (RSI: {rsi} >= {overbought_thresh}, PnL: +{pnl_pct:.2f}%)",
                "pnl_pct": pnl_pct
            }

        # 4. Stop-Loss defense
        if pnl_pct <= -abs(sl_pct):
            return {
                "should_sell": True,
                "reason": f"STOP_LOSS protection hit ({pnl_pct:.2f}% <= -{sl_pct}%)",
                "pnl_pct": pnl_pct
            }

        return {
            "should_sell": False,
            "reason": f"Position healthy (PnL: {pnl_pct:+.2f}%)",
            "pnl_pct": pnl_pct
        }

    def execute_buy(self, current_price: float, reason: str, indicators: Dict[str, Any]):
        balances = self.client.get_balances(non_zero_only=True)
        quote_bal = balances.get(self.quote_asset, {}).get("free", 0.0)
        
        buy_amount = min(self.trade_amount_usdt, quote_bal * 0.95)
        if buy_amount < self.min_notional:
            logger.warning(f"Insufficient {self.quote_asset} balance (${quote_bal:.2f}) to place order.")
            return

        raw_qty = buy_amount / current_price
        order_qty = self._round_step(raw_qty)

        if order_qty * current_price < self.min_notional:
            logger.warning(f"Order notional below minimum ${self.min_notional}")
            return

        logger.info(f">>> [BUY SIGNAL] Placed: {order_qty} {self.base_asset} (~${buy_amount:.2f} {self.quote_asset}) | {reason}")
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side="BUY",
                order_type="MARKET",
                quantity=order_qty
            )
            order_id = order.get("orderId")
            executed_qty = float(order.get("executedQty", order_qty))
            cummulative_quote = float(order.get("cummulativeQuoteQty", buy_amount))
            fill_price = cummulative_quote / executed_qty if executed_qty > 0 else current_price

            logger.info(f"[SUCCESS BUY] Order #{order_id} FILLED: {executed_qty} @ ${fill_price:,.2f}")

            self.active_position = {
                "symbol": self.symbol,
                "order_id": order_id,
                "qty": executed_qty,
                "entry_price": fill_price,
                "cost_usdt": cummulative_quote,
                "highest_price": fill_price,
                "entry_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "entry_indicators": indicators
            }
        except Exception as e:
            logger.error(f"[ERROR BUY] {e}")

    def execute_sell(self, current_price: float, exit_reason: str, indicators: Dict[str, Any]):
        if not self.active_position:
            return

        balances = self.client.get_balances(non_zero_only=True)
        base_bal = balances.get(self.base_asset, {}).get("free", 0.0)
        available_qty = min(self.active_position["qty"], base_bal)
        order_qty = self._round_step(available_qty)

        if order_qty * current_price < self.min_notional:
            logger.warning(f"Sell order notional below minimum. Clearing position tracking.")
            self.active_position = None
            return

        logger.info(f">>> [SELL SIGNAL] Executing SELL: {order_qty} {self.base_asset} (~${order_qty * current_price:.2f} {self.quote_asset}) | {exit_reason}")
        try:
            order = self.client.create_order(
                symbol=self.symbol,
                side="SELL",
                order_type="MARKET",
                quantity=order_qty
            )
            order_id = order.get("orderId")
            executed_qty = float(order.get("executedQty", order_qty))
            cummulative_quote = float(order.get("cummulativeQuoteQty", order_qty * current_price))
            fill_price = cummulative_quote / executed_qty if executed_qty > 0 else current_price

            logger.info(f"[SUCCESS SELL] Order #{order_id} FILLED: {executed_qty} @ ${fill_price:,.2f}")

            # Record round-trip trade into TradeLearningEngine
            record = self.learner.record_completed_trade(
                symbol=self.symbol,
                entry_price=self.active_position["entry_price"],
                exit_price=fill_price,
                qty=executed_qty,
                cost_usdt=self.active_position["cost_usdt"],
                revenue_usdt=cummulative_quote,
                entry_time=self.active_position["entry_time"],
                exit_time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                exit_reason=exit_reason,
                entry_indicators=self.active_position.get("entry_indicators", {}),
                exit_indicators=indicators
            )

            pnl_str = f"+${record['net_pnl_usdt']:.2f}" if record['is_win'] else f"-${abs(record['net_pnl_usdt']):.2f}"
            logger.info(f"[LEARNING UPDATE] Trade Realized Net PnL: {pnl_str} ({record['net_pnl_pct']:+.2f}%) | Result: {'WIN' if record['is_win'] else 'LOSS'}")
            
            self.active_position = None
        except Exception as e:
            logger.error(f"[ERROR SELL] {e}")

    def tick(self):
        try:
            price = self.client.get_ticker_price(self.symbol)
            candles = self.client.get_klines(self.symbol, interval=self.timeframe, limit=50)
            indicators = self.calculate_indicators(candles)
            params = self.learner.get_params()

            # Fear & Greed index
            fng_map = fetch_fear_and_greed(limit=1)
            current_fng = list(fng_map.values())[0]["value"] if fng_map else 50

            # 1. Evaluate open position exit (SELL)
            if self.active_position:
                if price > self.active_position.get("highest_price", 0):
                    self.active_position["highest_price"] = price
                
                exit_check = self.evaluate_exit(self.active_position, price, indicators, params)
                pnl_pct = exit_check.get("pnl_pct", 0.0)
                
                logger.info(f"[{self.symbol} @ ${price:,.2f}] OPEN POSITION PnL: {pnl_pct:+.2f}% | Highest: ${self.active_position['highest_price']:,.2f} | Status: {exit_check['reason']}")
                
                if exit_check["should_sell"]:
                    self.execute_sell(price, exit_check["reason"], indicators)
                    return
            else:
                # 2. Evaluate new trade entry (BUY)
                entry_check = self.evaluate_entry(indicators, params, current_fng)
                logger.info(f"[{self.symbol} @ ${price:,.2f}] RSI: {indicators['rsi']} | MACD Hist: {indicators['macd_hist']:+.2f} | Score: {entry_check['score']} | {entry_check['reason']}")
                
                if entry_check["should_buy"]:
                    self.execute_buy(price, entry_check["reason"], indicators)

            # Save state
            perf = self.learner.get_performance_summary()
            balances = self.client.get_balances(non_zero_only=True)
            state_data = {
                "status": "RUNNING",
                "pid": os.getpid(),
                "symbol": self.symbol,
                "current_price": price,
                "indicators": indicators,
                "fng": current_fng,
                "balances": balances,
                "active_position": self.active_position,
                "performance": perf,
                "adaptive_params": params,
                "last_heartbeat": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error during AutoTrader tick: {e}")

    def run(self, max_ticks: Optional[int] = None):
        logger.info(f"Starting Autonomous Profit Trader for {self.symbol} (Timeframe: {self.timeframe}).")
        logger.info(f"Adaptive parameters: {self.learner.get_params()}")
        
        ticks = 0
        while True:
            self.tick()
            ticks += 1
            if max_ticks and ticks >= max_ticks:
                break
            time.sleep(self.poll_interval)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Autonomous Profit-Generating Trading Bot")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair")
    parser.add_argument("--amount", type=float, default=50.0, help="Trade amount in USDT")
    parser.add_argument("--timeframe", type=str, default="1m", help="Candle timeframe")
    parser.add_argument("--poll", type=int, default=10, help="Polling interval seconds")
    parser.add_argument("--ticks", type=int, default=None, help="Max ticks")
    args = parser.parse_args()

    trader = AutoProfitTrader(
        symbol=args.symbol,
        trade_amount_usdt=args.amount,
        timeframe=args.timeframe,
        poll_interval=args.poll
    )
    trader.run(max_ticks=args.ticks)

if __name__ == "__main__":
    main()
