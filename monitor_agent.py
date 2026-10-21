#!/usr/bin/env python3
"""
Trading Bot Monitor & Health Check Agent.
Monitors the AutoProfitTrader or DCA bot process, reads state, validates exchange sync,
inspects trade journals and adaptive learning parameters, and generates real-time health reports.
"""

import os
import sys
import json
import psutil
import datetime
from pathlib import Path
from client import BinanceDemoClient
from learning_engine import TradeLearningEngine

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

AUTO_STATE_FILE = DATA_DIR / "auto_trader_state.json"
DCA_STATE_FILE = DATA_DIR / "bot_state.json"
JOURNAL_FILE = DATA_DIR / "trade_journal.json"
AUTO_LOG_FILE = LOGS_DIR / "auto_trader.log"
DCA_LOG_FILE = LOGS_DIR / "dca_bot.log"

def check_process_alive(pid: int) -> bool:
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def generate_health_report() -> dict:
    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "bot_alive": False,
        "api_connected": False,
        "details": {}
    }

    # 1. API Connectivity & Balances
    try:
        c = BinanceDemoClient()
        report["api_connected"] = c.ping()
        report["exchange_balances"] = c.get_balances(non_zero_only=True)
    except Exception as e:
        report["api_error"] = str(e)

    # 2. Check Auto Trader or DCA state
    state_file = AUTO_STATE_FILE if AUTO_STATE_FILE.exists() else DCA_STATE_FILE
    log_file = AUTO_LOG_FILE if AUTO_LOG_FILE.exists() else DCA_LOG_FILE

    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)
            report["state"] = state
            pid = state.get("pid")
            if pid and check_process_alive(pid):
                report["bot_alive"] = True
            else:
                report["bot_alive"] = False
        except Exception as e:
            report["state_read_error"] = str(e)

    # 3. Learning Performance Summary
    try:
        learner = TradeLearningEngine()
        report["learning_summary"] = learner.get_performance_summary()
    except Exception as e:
        report["learning_error"] = str(e)

    # 4. Recent logs
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                report["recent_logs"] = [l.strip() for l in lines[-6:]]
        except Exception:
            pass

    return report

def print_report():
    rep = generate_health_report()
    print("=" * 65)
    print("      AUTONOMOUS TRADING BOT SENTINEL REPORT")
    print("=" * 65)
    print(f"Timestamp:       {rep['timestamp']}")
    print(f"Bot Engine:      {'HEALTHY (ACTIVE)' if rep['bot_alive'] else 'OFFLINE / STANDBY'}")
    print(f"Binance API:     {'CONNECTED (ONLINE)' if rep['api_connected'] else 'DISCONNECTED'}")
    
    state = rep.get("state", {})
    if state:
        print("\n--- Live Market & State ---")
        print(f"Symbol:          {state.get('symbol')} | Price: ${state.get('current_price', 0):,.2f}")
        ind = state.get("indicators", {})
        if ind:
            print(f"Indicators:      RSI: {ind.get('rsi')} | MACD Hist: {ind.get('macd_hist', 0):+.2f} | 9 EMA: ${ind.get('ema9', 0):,.2f}")
        
        pos = state.get("active_position")
        if pos:
            pnl_pct = ((state.get('current_price', 0) - pos.get('entry_price', 1)) / pos.get('entry_price', 1)) * 100.0
            print(f"Active Position: {pos.get('qty')} {state.get('symbol')} @ Entry ${pos.get('entry_price', 0):,.2f} | PnL: {pnl_pct:+.2f}%")
        else:
            print("Active Position: None (Scanning for high-probability entry)")

    learn = rep.get("learning_summary", {})
    if learn:
        print("\n--- Learning & Performance Metrics ---")
        print(f"Total Closed Trades: {learn.get('total_trades', 0)} (Wins: {learn.get('wins', 0)}, Losses: {learn.get('losses', 0)})")
        print(f"Win Rate:            {learn.get('win_rate_pct', 0.0)}%")
        print(f"Realized Net PnL:    ${learn.get('total_net_pnl_usdt', 0.0):+.2f} USDT")
        print(f"Profit Factor:       {learn.get('profit_factor', 0.0)}")
        params = learn.get("adaptive_params", {})
        if params:
            print(f"Adaptive Targets:    Take-Profit: +{params.get('take_profit_pct')}% | Stop-Loss: -{params.get('stop_loss_pct')}% | RSI Entry: <={params.get('rsi_oversold')}")

    if rep.get("recent_logs"):
        print("\n--- Latest Log Entries ---")
        for line in rep["recent_logs"]:
            print(f"  {line}")

    print("=" * 65)

if __name__ == "__main__":
    print_report()
