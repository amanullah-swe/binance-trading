import os
from pathlib import Path
from dataclasses import dataclass

# Fallback naive env loader if python-dotenv is not installed
def load_dotenv_fallback(dotenv_path: Path):
    if not dotenv_path.exists():
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    env_path = Path(__file__).parent / ".env"
    load_dotenv_fallback(env_path)

@dataclass
class BinanceConfig:
    api_key: str = os.getenv("BINANCE_API_KEY", "")
    api_secret: str = os.getenv("BINANCE_API_SECRET", "")
    base_url: str = os.getenv("BINANCE_BASE_URL", "https://demo-api.binance.com")
    ws_url: str = os.getenv("BINANCE_WS_URL", "wss://demo-stream.binance.com/ws")
    default_symbol: str = os.getenv("DEFAULT_SYMBOL", "BTCUSDT")
    default_timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "1m")
    default_trade_amount_usdt: float = float(os.getenv("DEFAULT_TRADE_AMOUNT_USDT", "50.0"))
    slippage_tolerance: float = float(os.getenv("SLIPPAGE_TOLERANCE", "0.001"))
    max_risk_per_trade_percent: float = float(os.getenv("MAX_RISK_PER_TRADE_PERCENT", "2.0"))

config = BinanceConfig()
