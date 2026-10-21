#!/usr/bin/env python3
"""
Comprehensive Dollar-Cost Averaging (DCA) and Value Averaging (VA) Research Engine.

Collects:
- Daily OHLCV for BTC and ETH from Binance
- Live USD/INR conversion rate
- 7d, 30d, 90d rolling returns
- 30d rolling realized volatility (annualized)
- 90d maximum drawdown
- Daily Fear and Greed Index
- BTC Dominance
- Exchange fees per trade (0.10% spot taker)
- 3-Month Volume Weighted Average Price (VWAP)

Simulates & Compares:
1. Lump Sum on Day 1
2. Daily DCA
3. Weekly DCA (every 7 days)
4. Bi-Weekly DCA (every 14 days)
5. Fear & Greed Dynamic DCA (Daily, 2x when F&G < 25)
6. Fear & Greed Dynamic DCA (Weekly, 2x when F&G < 25)
7. Value Averaging (Weekly)
"""

import json
import math
import datetime
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHARTS_DIR = BASE_DIR / "charts"
DATA_DIR.mkdir(exist_ok=True)
CHARTS_DIR.mkdir(exist_ok=True)

SPOT_FEE_RATE = 0.0010  # 0.10% Binance spot trading fee

def fetch_usd_inr_rate() -> float:
    """Fetch live USD to INR exchange rate."""
    try:
        req = urllib.request.Request("https://open.er-api.com/v6/latest/USD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            return float(json.loads(r.read().decode())["rates"]["INR"])
    except Exception as e:
        print(f"[Warning] Failed to fetch live USD/INR ({e}), defaulting to 87.50")
        return 87.50

def fetch_btc_dominance() -> float:
    """Fetch global BTC dominance percentage from CoinGecko."""
    try:
        req = urllib.request.Request("https://api.coingecko.com/api/v3/global", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
            return float(data["data"]["market_cap_percentage"]["btc"])
    except Exception as e:
        print(f"[Warning] Failed to fetch BTC dominance ({e}), defaulting to 58.5%")
        return 58.50

def fetch_fear_and_greed(limit: int = 90) -> Dict[str, Dict[str, Any]]:
    """Fetch daily Fear and Greed index history."""
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = json.loads(r.read().decode())["data"]
            fng_by_date = {}
            for item in raw:
                dt_str = datetime.datetime.fromtimestamp(int(item["timestamp"]), datetime.timezone.utc).strftime("%Y-%m-%d")
                fng_by_date[dt_str] = {
                    "value": int(item["value"]),
                    "classification": item["value_classification"]
                }
            return fng_by_date
    except Exception as e:
        print(f"[Warning] Failed to fetch Fear & Greed index: {e}")
        return {}

def fetch_binance_daily_klines(symbol: str, limit: int = 90) -> pd.DataFrame:
    """Fetch daily OHLCV candlestick data from Binance."""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        raw = json.loads(r.read().decode())
    
    rows = []
    for k in raw:
        dt = datetime.datetime.fromtimestamp(int(k[0]) / 1000.0, datetime.timezone.utc).strftime("%Y-%m-%d")
        rows.append({
            "date": dt,
            "timestamp": int(k[0]),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "quote_volume": float(k[7]),
            "trades": int(k[8])
        })
    df = pd.DataFrame(rows)
    return df

def process_market_data(symbol: str, df: pd.DataFrame, usd_inr: float, fng_map: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compute rolling returns, volatility, max drawdown, VWAP, and F&G integration."""
    df["close_inr"] = df["close"] * usd_inr
    
    # Typical price for VWAP: (High + Low + Close) / 3
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["pv"] = df["typical_price"] * df["volume"]
    df["cum_pv"] = df["pv"].cumsum()
    df["cum_vol"] = df["volume"].cumsum()
    df["cum_vwap"] = df["cum_pv"] / df["cum_vol"]

    # Rolling returns
    df["return_7d"] = df["close"].pct_change(7) * 100.0
    df["return_30d"] = df["close"].pct_change(30) * 100.0
    df["return_90d"] = (df["close"] / df["close"].iloc[0] - 1.0) * 100.0

    # Daily log returns & 30-day rolling realized volatility (annualized)
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility_30d"] = df["log_ret"].rolling(window=30).std() * np.sqrt(365) * 100.0

    # Running maximum & drawdown
    df["cum_max"] = df["close"].cummax()
    df["drawdown"] = (df["close"] - df["cum_max"]) / df["cum_max"] * 100.0

    # Map Fear & Greed
    df["fng_value"] = df["date"].map(lambda d: fng_map.get(d, {}).get("value", np.nan))
    df["fng_class"] = df["date"].map(lambda d: fng_map.get(d, {}).get("classification", "Unknown"))
    
    # Fill any missing F&G values with forward-fill / backward-fill
    df["fng_value"] = df["fng_value"].ffill().bfill().fillna(50)

    return df

def simulate_strategies(df: pd.DataFrame, total_budget: float, usd_inr: float, symbol_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates all DCA / VA variants across the dataset.
    Budget: total USD to invest over the window.
    """
    n_days = len(df)
    results = {}
    
    three_month_vwap = df["cum_vwap"].iloc[-1]
    final_price = df["close"].iloc[-1]
    first_price = df["close"].iloc[0]
    
    # -------------------------------------------------------------
    # 1. Lump Sum on Day 1
    # -------------------------------------------------------------
    lump_fee = total_budget * SPOT_FEE_RATE
    lump_net_usd = total_budget - lump_fee
    lump_coins = lump_net_usd / first_price
    lump_cost_basis = total_budget / lump_coins
    lump_final_val = lump_coins * final_price
    
    results["Lump Sum (Day 1)"] = {
        "Total Invested (USD)": total_budget,
        "Total Invested (INR)": total_budget * usd_inr,
        "Coins Accumulated": lump_coins,
        "Avg Cost Basis (USD)": lump_cost_basis,
        "Avg Cost Basis (INR)": lump_cost_basis * usd_inr,
        "Final Value (USD)": lump_final_val,
        "Final Value (INR)": lump_final_val * usd_inr,
        "Net Profit (USD)": lump_final_val - total_budget,
        "Net Return (%)": ((lump_final_val - total_budget) / total_budget) * 100.0,
        "Total Fees (USD)": lump_fee,
        "Vs 3M VWAP (%)": ((lump_cost_basis - three_month_vwap) / three_month_vwap) * 100.0,
        "Trades Count": 1
    }

    # -------------------------------------------------------------
    # 2. Daily DCA (90 buys)
    # -------------------------------------------------------------
    daily_budget = total_budget / n_days
    daily_coins = 0.0
    daily_fees = 0.0
    daily_invested = 0.0
    
    for i in range(n_days):
        price = df["close"].iloc[i]
        fee = daily_budget * SPOT_FEE_RATE
        net_usd = daily_budget - fee
        coins = net_usd / price
        daily_coins += coins
        daily_fees += fee
        daily_invested += daily_budget
        
    daily_cost_basis = daily_invested / daily_coins
    daily_final_val = daily_coins * final_price
    
    results["Daily DCA"] = {
        "Total Invested (USD)": daily_invested,
        "Total Invested (INR)": daily_invested * usd_inr,
        "Coins Accumulated": daily_coins,
        "Avg Cost Basis (USD)": daily_cost_basis,
        "Avg Cost Basis (INR)": daily_cost_basis * usd_inr,
        "Final Value (USD)": daily_final_val,
        "Final Value (INR)": daily_final_val * usd_inr,
        "Net Profit (USD)": daily_final_val - daily_invested,
        "Net Return (%)": ((daily_final_val - daily_invested) / daily_invested) * 100.0,
        "Total Fees (USD)": daily_fees,
        "Vs 3M VWAP (%)": ((daily_cost_basis - three_month_vwap) / three_month_vwap) * 100.0,
        "Trades Count": n_days
    }

    # -------------------------------------------------------------
    # 3. Weekly DCA (Every 7 days)
    # -------------------------------------------------------------
    weekly_indices = list(range(0, n_days, 7))
    weekly_budget = total_budget / len(weekly_indices)
    weekly_coins = 0.0
    weekly_fees = 0.0
    weekly_invested = 0.0
    
    for idx in weekly_indices:
        price = df["close"].iloc[idx]
        fee = weekly_budget * SPOT_FEE_RATE
        net_usd = weekly_budget - fee
        coins = net_usd / price
        weekly_coins += coins
        weekly_fees += fee
        weekly_invested += weekly_budget
        
    weekly_cost_basis = weekly_invested / weekly_coins
    weekly_final_val = weekly_coins * final_price
    
    results["Weekly DCA (7d)"] = {
        "Total Invested (USD)": weekly_invested,
        "Total Invested (INR)": weekly_invested * usd_inr,
        "Coins Accumulated": weekly_coins,
        "Avg Cost Basis (USD)": weekly_cost_basis,
        "Avg Cost Basis (INR)": weekly_cost_basis * usd_inr,
        "Final Value (USD)": weekly_final_val,
        "Final Value (INR)": weekly_final_val * usd_inr,
        "Net Profit (USD)": weekly_final_val - weekly_invested,
        "Net Return (%)": ((weekly_final_val - weekly_invested) / weekly_invested) * 100.0,
        "Total Fees (USD)": weekly_fees,
        "Vs 3M VWAP (%)": ((weekly_cost_basis - three_month_vwap) / three_month_vwap) * 100.0,
        "Trades Count": len(weekly_indices)
    }

    # -------------------------------------------------------------
    # 4. Bi-Weekly DCA (Every 14 days)
    # -------------------------------------------------------------
    biweekly_indices = list(range(0, n_days, 14))
    biweekly_budget = total_budget / len(biweekly_indices)
    biweekly_coins = 0.0
    biweekly_fees = 0.0
    biweekly_invested = 0.0
    
    for idx in biweekly_indices:
        price = df["close"].iloc[idx]
        fee = biweekly_budget * SPOT_FEE_RATE
        net_usd = biweekly_budget - fee
        coins = net_usd / price
        biweekly_coins += coins
        biweekly_fees += fee
        biweekly_invested += biweekly_budget
        
    biweekly_cost_basis = biweekly_invested / biweekly_coins
    biweekly_final_val = biweekly_coins * final_price
    
    results["Bi-Weekly DCA (14d)"] = {
        "Total Invested (USD)": biweekly_invested,
        "Total Invested (INR)": biweekly_invested * usd_inr,
        "Coins Accumulated": biweekly_coins,
        "Avg Cost Basis (USD)": biweekly_cost_basis,
        "Avg Cost Basis (INR)": biweekly_cost_basis * usd_inr,
        "Final Value (USD)": biweekly_final_val,
        "Final Value (INR)": biweekly_final_val * usd_inr,
        "Net Profit (USD)": biweekly_final_val - biweekly_invested,
        "Net Return (%)": ((biweekly_final_val - biweekly_invested) / biweekly_invested) * 100.0,
        "Total Fees (USD)": biweekly_fees,
        "Vs 3M VWAP (%)": ((biweekly_cost_basis - three_month_vwap) / three_month_vwap) * 100.0,
        "Trades Count": len(biweekly_indices)
    }

    # -------------------------------------------------------------
    # 5. Dynamic Fear & Greed DCA (Daily: 2x buy when F&G < 25)
    # -------------------------------------------------------------
    # Weights: 2 if F&G < 25 else 1
    daily_weights = [2.0 if df["fng_value"].iloc[i] < 25 else 1.0 for i in range(n_days)]
    unit_daily_usd = total_budget / sum(daily_weights)
    dyn_daily_coins = 0.0
    dyn_daily_fees = 0.0
    dyn_daily_invested = 0.0
    
    for i in range(n_days):
        w = daily_weights[i]
        b = unit_daily_usd * w
        price = df["close"].iloc[i]
        fee = b * SPOT_FEE_RATE
        net_usd = b - fee
        coins = net_usd / price
        dyn_daily_coins += coins
        dyn_daily_fees += fee
        dyn_daily_invested += b
        
    dyn_daily_cost_basis = dyn_daily_invested / dyn_daily_coins
    dyn_daily_final_val = dyn_daily_coins * final_price
    
    results["Dynamic F&G DCA (Daily, 2x < 25)"] = {
        "Total Invested (USD)": dyn_daily_invested,
        "Total Invested (INR)": dyn_daily_invested * usd_inr,
        "Coins Accumulated": dyn_daily_coins,
        "Avg Cost Basis (USD)": dyn_daily_cost_basis,
        "Avg Cost Basis (INR)": dyn_daily_cost_basis * usd_inr,
        "Final Value (USD)": dyn_daily_final_val,
        "Final Value (INR)": dyn_daily_final_val * usd_inr,
        "Net Profit (USD)": dyn_daily_final_val - dyn_daily_invested,
        "Net Return (%)": ((dyn_daily_final_val - dyn_daily_invested) / dyn_daily_invested) * 100.0,
        "Total Fees (USD)": dyn_daily_fees,
        "Vs 3M VWAP (%)": ((dyn_daily_cost_basis - three_month_vwap) / three_month_vwap) * 100.0,
        "Trades Count": n_days
    }

    # -------------------------------------------------------------
    # 6. Dynamic Fear & Greed DCA (Weekly: 2x buy when F&G < 25)
    # -------------------------------------------------------------
    weekly_weights = [2.0 if df["fng_value"].iloc[idx] < 25 else 1.0 for idx in weekly_indices]
    unit_weekly_usd = total_budget / sum(weekly_weights)
    dyn_weekly_coins = 0.0
    dyn_weekly_fees = 0.0
    dyn_weekly_invested = 0.0
    
    for idx, w in zip(weekly_indices, weekly_weights):
        b = unit_weekly_usd * w
        price = df["close"].iloc[idx]
        fee = b * SPOT_FEE_RATE
        net_usd = b - fee
        coins = net_usd / price
        dyn_weekly_coins += coins
        dyn_weekly_fees += fee
        dyn_weekly_invested += b
        
    dyn_weekly_cost_basis = dyn_weekly_invested / dyn_weekly_coins
    dyn_weekly_final_val = dyn_weekly_coins * final_price
    
    results["Dynamic F&G DCA (Weekly, 2x < 25)"] = {
        "Total Invested (USD)": dyn_weekly_invested,
        "Total Invested (INR)": dyn_weekly_invested * usd_inr,
        "Coins Accumulated": dyn_weekly_coins,
        "Avg Cost Basis (USD)": dyn_weekly_cost_basis,
        "Avg Cost Basis (INR)": dyn_weekly_cost_basis * usd_inr,
        "Final Value (USD)": dyn_weekly_final_val,
        "Final Value (INR)": dyn_weekly_final_val * usd_inr,
        "Net Profit (USD)": dyn_weekly_final_val - dyn_weekly_invested,
        "Net Return (%)": ((dyn_weekly_final_val - dyn_weekly_invested) / dyn_weekly_invested) * 100.0,
        "Total Fees (USD)": dyn_weekly_fees,
        "Vs 3M VWAP (%)": ((dyn_weekly_cost_basis - three_month_vwap) / three_month_vwap) * 100.0,
        "Trades Count": len(weekly_indices)
    }

    # -------------------------------------------------------------
    # 7. Value Averaging (Weekly)
    # -------------------------------------------------------------
    # Target path: portfolio value should reach V_k = k * target_step
    # If portfolio is below target, buy the difference. (No short selling / no sell on crypto DCA accumulation)
    num_steps = len(weekly_indices)
    target_step_val = total_budget / num_steps
    va_coins = 0.0
    va_invested = 0.0
    va_fees = 0.0
    va_trades = 0
    
    for k, idx in enumerate(weekly_indices, 1):
        price = df["close"].iloc[idx]
        target_val = k * target_step_val
        curr_val = va_coins * price
        needed_usd = target_val - curr_val
        
        # In accumulation mode, if current value exceeds target, contribute minimum or 0
        if needed_usd > 5.0:
            fee = needed_usd * SPOT_FEE_RATE
            net_usd = needed_usd - fee
            bought = net_usd / price
            va_coins += bought
            va_fees += fee
            va_invested += needed_usd
            va_trades += 1

    va_cost_basis = va_invested / va_coins if va_coins > 0 else 0
    va_final_val = va_coins * final_price

    results["Value Averaging (Weekly)"] = {
        "Total Invested (USD)": va_invested,
        "Total Invested (INR)": va_invested * usd_inr,
        "Coins Accumulated": va_coins,
        "Avg Cost Basis (USD)": va_cost_basis,
        "Avg Cost Basis (INR)": va_cost_basis * usd_inr,
        "Final Value (USD)": va_final_val,
        "Final Value (INR)": va_final_val * usd_inr,
        "Net Profit (USD)": va_final_val - va_invested,
        "Net Return (%)": ((va_final_val - va_invested) / va_invested) * 100.0 if va_invested > 0 else 0,
        "Total Fees (USD)": va_fees,
        "Vs 3M VWAP (%)": ((va_cost_basis - three_month_vwap) / three_month_vwap) * 100.0 if three_month_vwap > 0 else 0,
        "Trades Count": va_trades
    }

    res_df = pd.DataFrame(results).T
    
    # Calculate difference vs Lump Sum Cost Basis
    res_df["Vs Lump Sum Basis (%)"] = ((res_df["Avg Cost Basis (USD)"] - lump_cost_basis) / lump_cost_basis) * 100.0

    metadata = {
        "symbol": symbol_name,
        "days": n_days,
        "first_date": df["date"].iloc[0],
        "last_date": df["date"].iloc[-1],
        "start_price": first_price,
        "final_price": final_price,
        "min_price": df["low"].min(),
        "max_price": df["high"].max(),
        "vwap_3m": three_month_vwap,
        "max_drawdown": df["drawdown"].min(),
        "current_30d_vol": df["volatility_30d"].iloc[-1],
        "return_90d": df["return_90d"].iloc[-1]
    }
    
    return res_df, metadata

def plot_comprehensive_report(btc_df: pd.DataFrame, eth_df: pd.DataFrame, btc_res: pd.DataFrame, eth_res: pd.DataFrame, output_path: Path):
    """Generate high quality multi-panel comparison chart."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14), gridspec_kw={"height_ratios": [1.2, 1, 1.2]})
    fig.patch.set_facecolor("#0b0e14")

    for row in axes:
        for ax in row:
            ax.set_facecolor("#12161f")
            ax.tick_params(colors="#c0c0c0")
            for spine in ax.spines.values():
                spine.set_color("#2d3748")

    # 1. Price & 3M VWAP (BTC)
    ax0 = axes[0, 0]
    ax0.plot(pd.to_datetime(btc_df["date"]), btc_df["close"], label="BTC Daily Close", color="#f7931a", lw=2)
    ax0.plot(pd.to_datetime(btc_df["date"]), btc_df["cum_vwap"], label="3M Cumulative VWAP", color="#38bdf8", lw=1.8, linestyle="--")
    ax0.set_title("BTC/USDT Price vs 3-Month VWAP", color="white", fontsize=12, fontweight="bold")
    ax0.set_ylabel("USD", color="white")
    ax0.legend(loc="upper left", facecolor="#1e222d", edgecolor="#333", labelcolor="white")
    ax0.grid(True, color="#222b3a", linestyle=":")

    # 2. Price & 3M VWAP (ETH)
    ax1 = axes[0, 1]
    ax1.plot(pd.to_datetime(eth_df["date"]), eth_df["close"], label="ETH Daily Close", color="#627eea", lw=2)
    ax1.plot(pd.to_datetime(eth_df["date"]), eth_df["cum_vwap"], label="3M Cumulative VWAP", color="#38bdf8", lw=1.8, linestyle="--")
    ax1.set_title("ETH/USDT Price vs 3-Month VWAP", color="white", fontsize=12, fontweight="bold")
    ax1.set_ylabel("USD", color="white")
    ax1.legend(loc="upper left", facecolor="#1e222d", edgecolor="#333", labelcolor="white")
    ax1.grid(True, color="#222b3a", linestyle=":")

    # 3. Fear & Greed Index & 30d Realized Volatility
    ax2 = axes[1, 0]
    dates = pd.to_datetime(btc_df["date"])
    ax2.plot(dates, btc_df["fng_value"], label="Fear & Greed Index", color="#10b981", lw=1.8)
    ax2.axhline(25, color="#ef4444", linestyle="--", alpha=0.8, label="Extreme Fear (<25)")
    ax2.axhline(75, color="#f59e0b", linestyle="--", alpha=0.8, label="Extreme Greed (>75)")
    ax2.set_title("Crypto Fear & Greed Index (Daily)", color="white", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Index (0-100)", color="white")
    ax2.legend(loc="upper left", facecolor="#1e222d", edgecolor="#333", labelcolor="white")
    ax2.grid(True, color="#222b3a", linestyle=":")

    ax3 = axes[1, 1]
    ax3.plot(dates, btc_df["volatility_30d"], label="BTC 30d Realized Vol (%)", color="#f7931a", lw=1.8)
    ax3.plot(dates, eth_df["volatility_30d"], label="ETH 30d Realized Vol (%)", color="#627eea", lw=1.8)
    ax3.set_title("30-Day Realized Annualized Volatility", color="white", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Volatility %", color="white")
    ax3.legend(loc="upper left", facecolor="#1e222d", edgecolor="#333", labelcolor="white")
    ax3.grid(True, color="#222b3a", linestyle=":")

    # 4. Strategy Average Cost Basis Comparison (BTC)
    ax4 = axes[2, 0]
    btc_costs = btc_res["Avg Cost Basis (USD)"]
    vwap_btc = btc_df["cum_vwap"].iloc[-1]
    bars4 = ax4.barh(btc_res.index, btc_costs, color=["#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1"])
    ax4.axvline(vwap_btc, color="#ef4444", linestyle="--", lw=2, label=f"3M VWAP: ${vwap_btc:,.0f}")
    ax4.set_title("BTC Strategy Average Cost Basis vs VWAP", color="white", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Average Cost ($/BTC)", color="white")
    ax4.legend(loc="lower right", facecolor="#1e222d", edgecolor="#333", labelcolor="white")
    ax4.grid(True, color="#222b3a", linestyle=":")
    for bar in bars4:
        w = bar.get_width()
        ax4.text(w * 0.75, bar.get_y() + bar.get_height()/2, f"${w:,.0f}", va="center", ha="center", color="white", fontweight="bold", fontsize=9)

    # 5. Strategy Average Cost Basis Comparison (ETH)
    ax5 = axes[2, 1]
    eth_costs = eth_res["Avg Cost Basis (USD)"]
    vwap_eth = eth_df["cum_vwap"].iloc[-1]
    bars5 = ax5.barh(eth_res.index, eth_costs, color=["#f59e0b", "#3b82f6", "#10b981", "#8b5cf6", "#ec4899", "#14b8a6", "#6366f1"])
    ax5.axvline(vwap_eth, color="#ef4444", linestyle="--", lw=2, label=f"3M VWAP: ${vwap_eth:,.0f}")
    ax5.set_title("ETH Strategy Average Cost Basis vs VWAP", color="white", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Average Cost ($/ETH)", color="white")
    ax5.legend(loc="lower right", facecolor="#1e222d", edgecolor="#333", labelcolor="white")
    ax5.grid(True, color="#222b3a", linestyle=":")
    for bar in bars5:
        w = bar.get_width()
        ax5.text(w * 0.75, bar.get_y() + bar.get_height()/2, f"${w:,.0f}", va="center", ha="center", color="white", fontweight="bold", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

def main():
    print("=" * 70)
    print("   DCA & VALUE AVERAGING 90-DAY STRATEGY RESEARCH STUDY")
    print("=" * 70)

    # 1. Macro & Market context
    usd_inr = fetch_usd_inr_rate()
    btc_dominance = fetch_btc_dominance()
    print(f"[*] Base Currency FX: 1 USD = ₹{usd_inr:.2f} INR")
    print(f"[*] BTC Dominance: {btc_dominance:.2f}%")
    print(f"[*] Binance Spot Fee per Trade: {SPOT_FEE_RATE * 100:.2f}%")

    # 2. Fetch Fear and Greed
    fng_map = fetch_fear_and_greed(limit=90)
    print(f"[*] Fear & Greed Data Points Collected: {len(fng_map)}")

    # 3. Fetch Binance OHLCV
    print("[*] Fetching 90-Day Daily OHLCV for BTCUSDT...")
    btc_raw = fetch_binance_daily_klines("BTCUSDT", limit=90)
    btc_df = process_market_data("BTCUSDT", btc_raw, usd_inr, fng_map)

    print("[*] Fetching 90-Day Daily OHLCV for ETHUSDT...")
    eth_raw = fetch_binance_daily_klines("ETHUSDT", limit=90)
    eth_df = process_market_data("ETHUSDT", eth_raw, usd_inr, fng_map)

    # Save processed dataset to CSV
    combined_csv_path = DATA_DIR / "crypto_90d_market_data.csv"
    btc_df["asset"] = "BTC"
    eth_df["asset"] = "ETH"
    pd.concat([btc_df, eth_df]).to_csv(combined_csv_path, index=False)
    print(f"[+] Saved 90-day time series data to: {combined_csv_path}")

    # 4. Run simulations with $3,000 budget per asset
    BUDGET = 3000.0  # USD
    print(f"\n[*] Running DCA & Value Averaging Simulations (Budget: ${BUDGET:,.2f} USD / ₹{BUDGET * usd_inr:,.2f} INR)...")
    btc_res, btc_meta = simulate_strategies(btc_df, BUDGET, usd_inr, "BTC")
    eth_res, eth_meta = simulate_strategies(eth_df, BUDGET, usd_inr, "ETH")

    # Save results to CSV
    btc_res.to_csv(DATA_DIR / "btc_dca_simulation_results.csv")
    eth_res.to_csv(DATA_DIR / "eth_dca_simulation_results.csv")

    # 5. Plot visualization chart
    chart_path = CHARTS_DIR / "dca_vs_va_comparison.png"
    plot_comprehensive_report(btc_df, eth_df, btc_res, eth_res, chart_path)
    print(f"[+] Generated comprehensive visual report at: {chart_path}")

    # Print summary tables to stdout
    print("\n" + "=" * 70)
    print("   BITCOIN (BTC) 90-DAY DCA & VALUE AVERAGING RESULTS")
    print("=" * 70)
    print(f"90-Day Period: {btc_meta['first_date']} to {btc_meta['last_date']}")
    print(f"Start Price: ${btc_meta['start_price']:,.2f} | Final Price: ${btc_meta['final_price']:,.2f} ({btc_meta['return_90d']:+.2f}%)")
    print(f"Price Range: ${btc_meta['min_price']:,.2f} - ${btc_meta['max_price']:,.2f} | Max Drawdown: {btc_meta['max_drawdown']:.2f}%")
    print(f"3-Month VWAP: ${btc_meta['vwap_3m']:,.2f} | Current 30d Realized Volatility: {btc_meta['current_30d_vol']:.2f}%")
    print("-" * 70)
    
    cols_display = ["Avg Cost Basis (USD)", "Avg Cost Basis (INR)", "Coins Accumulated", "Vs 3M VWAP (%)", "Vs Lump Sum Basis (%)", "Net Return (%)", "Total Fees (USD)"]
    print(btc_res[cols_display].to_string())

    print("\n" + "=" * 70)
    print("   ETHEREUM (ETH) 90-DAY DCA & VALUE AVERAGING RESULTS")
    print("=" * 70)
    print(f"90-Day Period: {eth_meta['first_date']} to {eth_meta['last_date']}")
    print(f"Start Price: ${eth_meta['start_price']:,.2f} | Final Price: ${eth_meta['final_price']:,.2f} ({eth_meta['return_90d']:+.2f}%)")
    print(f"Price Range: ${eth_meta['min_price']:,.2f} - ${eth_meta['max_price']:,.2f} | Max Drawdown: {eth_meta['max_drawdown']:.2f}%")
    print(f"3-Month VWAP: ${eth_meta['vwap_3m']:,.2f} | Current 30d Realized Volatility: {eth_meta['current_30d_vol']:.2f}%")
    print("-" * 70)
    print(eth_res[cols_display].to_string())
    print("=" * 70)

if __name__ == "__main__":
    main()
