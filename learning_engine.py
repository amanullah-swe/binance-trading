import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

logger = logging.getLogger("LearningEngine")

class TradeLearningEngine:
    """
    Self-Improving Trade Journal & Learning Engine.
    Records every round-trip trade, tracks performance metrics,
    and dynamically adapts strategy parameters to optimize profit.
    """

    def __init__(self, journal_file: Optional[Path] = None, learner_file: Optional[Path] = None):
        base_dir = Path(__file__).parent
        data_dir = base_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        self.journal_file = journal_file or (data_dir / "trade_journal.json")
        self.learner_file = learner_file or (data_dir / "learner_state.json")
        
        self.trades: List[Dict[str, Any]] = self._load_journal()
        self.params: Dict[str, Any] = self._load_params()

    def _load_journal(self) -> List[Dict[str, Any]]:
        if self.journal_file.exists():
            try:
                with open(self.journal_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load journal: {e}")
        return []

    def _save_journal(self):
        try:
            with open(self.journal_file, "w", encoding="utf-8") as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save journal: {e}")

    def _load_params(self) -> Dict[str, Any]:
        default_params = {
            "rsi_oversold": 32.0,
            "rsi_overbought": 68.0,
            "take_profit_pct": 1.2,
            "stop_loss_pct": 1.0,
            "trailing_stop_pct": 0.5,
            "min_confirmation_score": 2,
            "consecutive_losses": 0,
            "consecutive_wins": 0,
            "total_pnl_usd": 0.0,
            "win_rate_pct": 0.0,
            "last_learning_update": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        if self.learner_file.exists():
            try:
                with open(self.learner_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                    default_params.update(saved)
            except Exception:
                pass
        return default_params

    def _save_params(self):
        try:
            with open(self.learner_file, "w", encoding="utf-8") as f:
                json.dump(self.params, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save params: {e}")

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def record_completed_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        cost_usdt: float,
        revenue_usdt: float,
        entry_time: str,
        exit_time: str,
        exit_reason: str,
        entry_indicators: Dict[str, Any],
        exit_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Record a round-trip buy -> sell trade, compute metrics, and adapt parameters.
        """
        fee_rate = 0.0010  # 0.1% spot fee per side
        total_fees = (cost_usdt + revenue_usdt) * fee_rate
        gross_pnl = revenue_usdt - cost_usdt
        net_pnl = gross_pnl - total_fees
        net_pnl_pct = (net_pnl / cost_usdt) * 100.0 if cost_usdt > 0 else 0.0
        is_win = net_pnl > 0

        trade_record = {
            "trade_id": len(self.trades) + 1,
            "symbol": symbol,
            "qty": qty,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "cost_usdt": round(cost_usdt, 2),
            "revenue_usdt": round(revenue_usdt, 2),
            "total_fees_usdt": round(total_fees, 4),
            "net_pnl_usdt": round(net_pnl, 2),
            "net_pnl_pct": round(net_pnl_pct, 2),
            "is_win": is_win,
            "exit_reason": exit_reason,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_indicators": entry_indicators,
            "exit_indicators": exit_indicators
        }

        self.trades.append(trade_record)
        self._save_journal()

        # Adapt strategy based on trade result
        self._learn_from_trade(trade_record)
        return trade_record

    def _learn_from_trade(self, trade: Dict[str, Any]):
        """
        Reinforcement & parameter adaptation loop:
        - If winning trade: solidify entry parameters, allow slightly looser profit taking.
        - If losing trade: tighten entry conditions (demand deeper RSI discount), tighten stop loss.
        """
        pnl = trade["net_pnl_usdt"]
        self.params["total_pnl_usd"] = round(self.params["total_pnl_usd"] + pnl, 2)

        if trade["is_win"]:
            self.params["consecutive_wins"] += 1
            self.params["consecutive_losses"] = 0
            logger.info(f"[*] Trade WIN (+${pnl:.2f}). Consecutive wins: {self.params['consecutive_wins']}")
            
            # If on a win streak, slightly reward system by maintaining or fine-tuning TP
            if self.params["consecutive_wins"] >= 2:
                # Target slightly higher profit if market momentum is favorable
                self.params["take_profit_pct"] = min(2.5, round(self.params["take_profit_pct"] + 0.1, 2))
        else:
            self.params["consecutive_losses"] += 1
            self.params["consecutive_wins"] = 0
            logger.warning(f"[!] Trade LOSS (-${abs(pnl):.2f}). Consecutive losses: {self.params['consecutive_losses']}")
            
            # Adaptive defensive mechanism:
            # 1. Require deeper discount before buying (lower RSI oversold target)
            self.params["rsi_oversold"] = max(25.0, round(self.params["rsi_oversold"] - 1.0, 1))
            # 2. Tighten stop loss to cut future losses faster
            self.params["stop_loss_pct"] = max(0.8, round(self.params["stop_loss_pct"] - 0.1, 2))
            logger.info(f"[Adaptive Action] Tightened entry: RSI oversold={self.params['rsi_oversold']}, SL={self.params['stop_loss_pct']}%")

        # Update overall win rate
        wins = sum(1 for t in self.trades if t.get("is_win", False))
        self.params["win_rate_pct"] = round((wins / len(self.trades)) * 100.0, 1) if self.trades else 0.0
        self.params["last_learning_update"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        self._save_params()

    def get_performance_summary(self) -> Dict[str, Any]:
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "total_pnl_usdt": 0.0,
                "profit_factor": 0.0
            }

        wins = [t for t in self.trades if t["is_win"]]
        losses = [t for t in self.trades if not t["is_win"]]
        
        total_win_pnl = sum(t["net_pnl_usdt"] for t in wins)
        total_loss_pnl = abs(sum(t["net_pnl_usdt"] for t in losses))
        
        pf = (total_win_pnl / total_loss_pnl) if total_loss_pnl > 0 else (99.0 if total_win_pnl > 0 else 1.0)
        
        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round((len(wins) / len(self.trades)) * 100.0, 1),
            "total_net_pnl_usdt": round(sum(t["net_pnl_usdt"] for t in self.trades), 2),
            "profit_factor": round(pf, 2),
            "avg_win_usdt": round(total_win_pnl / len(wins), 2) if wins else 0.0,
            "avg_loss_usdt": round(total_loss_pnl / len(losses), 2) if losses else 0.0,
            "adaptive_params": self.params
        }
