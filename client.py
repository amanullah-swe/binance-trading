import hmac
import hashlib
import time
import json
import urllib.parse
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List
from config import config

class BinanceDemoClient:
    """
    Client for interacting with Binance Spot Demo API (https://demo-api.binance.com).
    Handles HMAC SHA256 authentication, timestamp synchronization, and request signing.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or config.api_key
        self.api_secret = api_secret or config.api_secret
        self.base_url = (base_url or config.base_url).rstrip("/")
        self.time_offset = 0
        self.sync_time()

    def sync_time(self):
        """Calculate time offset between local machine and Binance server to prevent timestamp drift."""
        try:
            res = self._request("GET", "/api/v3/time", signed=False)
            server_time = res["serverTime"]
            local_time = int(time.time() * 1000)
            self.time_offset = server_time - local_time
        except Exception as e:
            print(f"[Warning] Failed to sync server time: {e}")
            self.time_offset = 0

    def _get_timestamp(self) -> int:
        return int(time.time() * 1000) + self.time_offset

    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, signed: bool = False) -> Dict[str, Any]:
        params = params or {}
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "BinanceDemoTradingBot/1.0"
        }

        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("API Key and Secret Key are required for signed endpoints.")
            headers["X-MBX-APIKEY"] = self.api_key
            params["timestamp"] = self._get_timestamp()
            params["recvWindow"] = 5000
            
            query_string = urllib.parse.urlencode(params)
            signature = self._generate_signature(query_string)
            query_string += f"&signature={signature}"
        else:
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            query_string = urllib.parse.urlencode(params)

        url = f"{self.base_url}{endpoint}"
        if method == "GET":
            if query_string:
                url = f"{url}?{query_string}"
            req = urllib.request.Request(url, headers=headers, method="GET")
            data = None
        else:
            data = query_string.encode("utf-8")
            req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                err_json = json.loads(error_body)
                raise RuntimeError(f"Binance API Error [{err_json.get('code')}]: {err_json.get('msg')}") from e
            except json.JSONDecodeError:
                raise RuntimeError(f"HTTP Error {e.code}: {error_body}") from e
        except Exception as e:
            raise RuntimeError(f"Network error communicating with Binance Demo: {e}") from e

    # --- Market Data Endpoints ---

    def ping(self) -> bool:
        """Test connectivity."""
        self._request("GET", "/api/v3/ping")
        return True

    def get_server_time(self) -> int:
        """Get Binance server time."""
        res = self._request("GET", "/api/v3/time")
        return res["serverTime"]

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/api/v3/exchangeInfo", params=params)

    def get_ticker_price(self, symbol: str) -> float:
        """Get latest price for a symbol."""
        res = self._request("GET", "/api/v3/ticker/price", params={"symbol": symbol.upper()})
        return float(res["price"])

    def get_order_book(self, symbol: str, limit: int = 10) -> Dict[str, Any]:
        """Get order book depth."""
        return self._request("GET", "/api/v3/depth", params={"symbol": symbol.upper(), "limit": limit})

    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch OHLCV candlestick data.
        Returns list of parsed candles with timestamp, open, high, low, close, volume.
        """
        raw = self._request("GET", "/api/v3/klines", params={
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit
        })
        candles = []
        for k in raw:
            candles.append({
                "open_time": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": int(k[6]),
                "trades": int(k[8])
            })
        return candles

    # --- Account & Trading Endpoints ---

    def get_account(self) -> Dict[str, Any]:
        """Fetch account balance and permissions."""
        return self._request("GET", "/api/v3/account", signed=True)

    def get_balances(self, non_zero_only: bool = True) -> Dict[str, Dict[str, float]]:
        """Fetch dictionary of asset balances."""
        account = self.get_account()
        balances = {}
        for b in account.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            total = free + locked
            if not non_zero_only or total > 0:
                balances[b["asset"]] = {
                    "free": free,
                    "locked": locked,
                    "total": total
                }
        return balances

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Optional[float] = None,
        quote_order_qty: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: str = "GTC"
    ) -> Dict[str, Any]:
        """
        Place an order on the demo exchange.
        Side: BUY | SELL
        Order Type: MARKET | LIMIT
        """
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
        }

        if order_type.upper() == "LIMIT":
            if price is None or quantity is None:
                raise ValueError("Price and quantity are required for LIMIT orders.")
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["quantity"] = f"{quantity:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = time_in_force
        elif order_type.upper() == "MARKET":
            if quantity is not None:
                params["quantity"] = f"{quantity:.8f}".rstrip("0").rstrip(".")
            elif quote_order_qty is not None:
                params["quoteOrderQty"] = f"{quote_order_qty:.8f}".rstrip("0").rstrip(".")
            else:
                raise ValueError("Either quantity or quote_order_qty must be provided for MARKET orders.")

        return self._request("POST", "/api/v3/order", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        """Cancel an active order."""
        return self._request("DELETE", "/api/v3/order", params={
            "symbol": symbol.upper(),
            "orderId": order_id
        }, signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """List open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        return self._request("GET", "/api/v3/openOrders", params=params, signed=True)

    def get_my_trades(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trades history."""
        return self._request("GET", "/api/v3/myTrades", params={
            "symbol": symbol.upper(),
            "limit": limit
        }, signed=True)
