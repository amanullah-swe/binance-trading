#!/usr/bin/env python3
"""
Quick utility script to check Binance Demo account connection, balances, and market prices.
"""
from client import BinanceDemoClient
from config import config

def main():
    print("=" * 60)
    print("   Binance Spot Demo Trading Bot - Account Inspector")
    print("=" * 60)
    print(f"Connecting to: {config.base_url}")
    
    try:
        client = BinanceDemoClient()
        ping_ok = client.ping()
        print(f"[*] API Connectivity: {'ONLINE' if ping_ok else 'OFFLINE'}")
        
        server_time = client.get_server_time()
        print(f"[*] Server Time: {server_time} (offset: {client.time_offset}ms)")
        
        account = client.get_account()
        can_trade = account.get("canTrade", False)
        account_type = account.get("accountType", "UNKNOWN")
        print(f"[*] Account Type: {account_type} | Can Trade: {can_trade}")
        
        print("\n--- Current Demo Balances ---")
        balances = client.get_balances(non_zero_only=True)
        if balances:
            for asset, bal in balances.items():
                print(f"  • {asset:8s} Free: {bal['free']:>14.4f} | Locked: {bal['locked']:>10.4f} | Total: {bal['total']:>14.4f}")
        else:
            print("  (No balances found)")

        print("\n--- Live Market Check ---")
        for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
            try:
                price = client.get_ticker_price(sym)
                print(f"  • {sym:10s} : ${price:,.2f}")
            except Exception as e:
                print(f"  • {sym:10s} : Error ({e})")
                
        print("\n--- Open Orders ---")
        open_orders = client.get_open_orders()
        print(f"  Total open orders: {len(open_orders)}")
        for o in open_orders:
            print(f"  • #{o['orderId']} {o['symbol']} {o['side']} {o['type']} Qty: {o['origQty']} Price: {o['price']}")

        print("=" * 60)
        print("Ready for automated trading!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")

if __name__ == "__main__":
    main()
