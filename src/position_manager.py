"""Active position management: breakeven, trailing stop, TP tightening, partial TP.

Runs every cycle BEFORE the LLM is asked for new decisions. The goal is to lock
in gains on winning trades WITHOUT relying on the LLM — this is the primary
fix for "trade went into profit but flipped before TP was hit".

All thresholds are expressed in R-multiples (unrealised P&L distance / initial
SL distance) so behaviour is independent of symbol volatility.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from src.config_loader import CONFIG

logger = logging.getLogger(__name__)


class PositionManager:
    """Per-ticket state tracker + deterministic post-entry exit manager.

    State that must survive between cycles (stored in-memory per ticket):
      - entry_price
      - initial_sl_distance  (abs(entry - initial_sl), in price units)
      - atr_at_entry         (5m ATR14 when the trade was opened)
      - is_buy
      - partial_taken        (True once a partial close has executed)
      - breakeven_done       (True once SL has been moved to breakeven)
      - trailing_active      (True once trailing SL has been engaged)
    """

    def __init__(self, mt5_api, diary_path: str = "diary.jsonl"):
        self.mt5_api = mt5_api
        self.diary_path = diary_path

        self.enabled = bool(CONFIG.get("enable_position_manager", True))

        self.breakeven_activate_r = float(CONFIG.get("breakeven_activate_r") or 0.0)
        self.breakeven_buffer_atr = float(CONFIG.get("breakeven_buffer_atr") or 0.0)
        self.trail_activate_r = float(CONFIG.get("trail_activate_r") or 0.0)
        self.trail_atr_mult = float(CONFIG.get("trail_atr_mult") or 1.0)
        self.tighten_tp_r = float(CONFIG.get("tighten_tp_r") or 0.0)
        self.tighten_tp_atr_mult = float(CONFIG.get("tighten_tp_atr_mult") or 1.0)
        self.partial_tp_r = float(CONFIG.get("partial_tp_r") or 0.0)
        self.partial_tp_fraction = float(CONFIG.get("partial_tp_fraction") or 0.0)

        # ticket -> tracking dict
        self._state: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_new_trade(
        self,
        ticket: int,
        symbol: str,
        entry_price: float,
        initial_sl: float,
        initial_tp: Optional[float],
        is_buy: bool,
        atr_at_entry: Optional[float],
    ) -> None:
        """Called by main.py immediately after a successful order_send."""
        if not ticket:
            return
        sl_dist = abs(entry_price - initial_sl) if initial_sl else 0.0
        self._state[ticket] = {
            "symbol": symbol,
            "entry_price": entry_price,
            "initial_sl": initial_sl,
            "initial_tp": initial_tp,
            "initial_sl_distance": sl_dist,
            "atr_at_entry": atr_at_entry,
            "is_buy": is_buy,
            "partial_taken": False,
            "breakeven_done": False,
            "trailing_active": False,
            "tp_tightened": False,
            "opened_at": datetime.now(timezone.utc).isoformat(),
        }
        logger.info(
            "PositionManager: registered ticket=%d %s entry=%.5f sl=%.5f tp=%s "
            "sl_dist=%.5f atr=%s",
            ticket, symbol, entry_price, initial_sl, initial_tp, sl_dist, atr_at_entry,
        )

    # ------------------------------------------------------------------
    # Cold-start / reconciliation
    # ------------------------------------------------------------------

    def reconcile(self, live_positions: list[dict], current_atrs: dict[str, float]) -> None:
        """Ensure every open live position has a tracking entry.

        Called once per cycle before `manage`. For positions opened in a prior
        run (bot restart) we infer initial_sl_distance from the currently-set SL
        on the MT5 side, which is the best estimate available.
        """
        live_tickets = {int(p.get("ticket")) for p in live_positions if p.get("ticket")}

        # Drop state for closed positions
        stale = [t for t in self._state if t not in live_tickets]
        for t in stale:
            logger.info("PositionManager: dropping state for closed ticket %d", t)
            self._state.pop(t, None)

        for pos in live_positions:
            ticket = int(pos.get("ticket") or 0)
            if not ticket or ticket in self._state:
                continue
            symbol = pos.get("symbol")
            entry_price = float(pos.get("price_open") or 0)
            sl = float(pos.get("sl") or 0)
            tp = float(pos.get("tp") or 0)
            is_buy = str(pos.get("type", "BUY")).upper() == "BUY"
            atr = current_atrs.get(symbol)
            sl_dist = abs(entry_price - sl) if sl else (
                float(atr) if atr else 0.0
            )
            self._state[ticket] = {
                "symbol": symbol,
                "entry_price": entry_price,
                "initial_sl": sl,
                "initial_tp": tp,
                "initial_sl_distance": sl_dist,
                "atr_at_entry": atr,
                "is_buy": is_buy,
                "partial_taken": False,
                "breakeven_done": False,
                "trailing_active": False,
                "tp_tightened": False,
                "opened_at": None,
                "from_reconcile": True,
            }
            logger.info(
                "PositionManager: reconciled pre-existing ticket=%d %s "
                "entry=%.5f sl=%.5f tp=%.5f sl_dist=%.5f",
                ticket, symbol, entry_price, sl, tp, sl_dist,
            )

    # ------------------------------------------------------------------
    # Diary helper
    # ------------------------------------------------------------------

    def _write_diary(self, entry: dict) -> None:
        try:
            entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
            with open(self.diary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as err:
            logger.warning("PositionManager: diary write failed: %s", err)

    # ------------------------------------------------------------------
    # Main entry point — runs each cycle
    # ------------------------------------------------------------------

    def manage(self, live_positions: list[dict], current_atrs: dict[str, float]) -> None:
        """Iterate through all open live positions and apply management rules."""
        if not self.enabled:
            return

        self.reconcile(live_positions, current_atrs)

        for pos in live_positions:
            try:
                self._manage_one(pos, current_atrs)
            except Exception as err:
                logger.exception(
                    "PositionManager: error managing %s ticket=%s: %s",
                    pos.get("symbol"), pos.get("ticket"), err,
                )

    def _manage_one(self, pos: dict, current_atrs: dict[str, float]) -> None:
        ticket = int(pos.get("ticket") or 0)
        if not ticket:
            return
        state = self._state.get(ticket)
        if not state:
            return

        symbol = pos.get("symbol")
        is_buy = state["is_buy"]
        entry = float(state["entry_price"])
        sl_dist = float(state["initial_sl_distance"] or 0)
        current_price = float(pos.get("current_price") or 0)
        cur_sl = float(pos.get("sl") or 0)
        cur_tp = float(pos.get("tp") or 0)
        # Prefer live ATR (volatility may have changed since entry)
        atr = current_atrs.get(symbol) or state.get("atr_at_entry") or 0.0
        atr = float(atr or 0)

        if current_price <= 0 or sl_dist <= 0:
            return  # Can't compute R without valid reference

        # R-multiple of current unrealised P&L
        if is_buy:
            unrealised_R = (current_price - entry) / sl_dist
        else:
            unrealised_R = (entry - current_price) / sl_dist

        # --- Phase 3: partial profit taking -------------------------------
        if (
            not state["partial_taken"]
            and self.partial_tp_fraction > 0
            and self.partial_tp_r > 0
            and unrealised_R >= self.partial_tp_r
        ):
            self._partial_close(pos, state, unrealised_R)

        # --- Phase 2a: tighten TP -----------------------------------------
        if (
            not state["tp_tightened"]
            and self.tighten_tp_r > 0
            and self.tighten_tp_atr_mult > 0
            and atr > 0
            and unrealised_R >= self.tighten_tp_r
        ):
            dist = self.tighten_tp_atr_mult * atr
            new_tp = round(current_price + dist, 5) if is_buy else round(current_price - dist, 5)
            # Only tighten (i.e. bring TP closer to current price), never push it out.
            should_update = False
            if cur_tp <= 0:
                should_update = True
            elif is_buy and new_tp < cur_tp:
                should_update = True
            elif (not is_buy) and new_tp > cur_tp:
                should_update = True
            if should_update:
                self._modify_sltp(pos, new_sl=None, new_tp=new_tp, reason="tighten_tp",
                                  unrealised_R=unrealised_R)
                state["tp_tightened"] = True
                cur_tp = new_tp

        # --- Phase 2b: breakeven move -------------------------------------
        if (
            not state["breakeven_done"]
            and self.breakeven_activate_r > 0
            and unrealised_R >= self.breakeven_activate_r
        ):
            buffer = (self.breakeven_buffer_atr or 0) * atr
            be_sl = round(entry + buffer, 5) if is_buy else round(entry - buffer, 5)
            # Only move SL if it's a tighter (better) stop than current
            if self._is_tighter_sl(cur_sl, be_sl, is_buy):
                self._modify_sltp(pos, new_sl=be_sl, new_tp=None, reason="breakeven",
                                  unrealised_R=unrealised_R)
                state["breakeven_done"] = True
                cur_sl = be_sl

        # --- Phase 2c: trailing stop --------------------------------------
        if (
            self.trail_activate_r > 0
            and self.trail_atr_mult > 0
            and atr > 0
            and unrealised_R >= self.trail_activate_r
        ):
            trail_dist = self.trail_atr_mult * atr
            trail_sl = round(current_price - trail_dist, 5) if is_buy else round(current_price + trail_dist, 5)
            if self._is_tighter_sl(cur_sl, trail_sl, is_buy):
                self._modify_sltp(pos, new_sl=trail_sl, new_tp=None, reason="trail",
                                  unrealised_R=unrealised_R)
                state["trailing_active"] = True
                cur_sl = trail_sl

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_tighter_sl(current_sl: float, proposed_sl: float, is_buy: bool) -> bool:
        """True if proposed SL is a *tighter* (more protective) stop than current."""
        if proposed_sl <= 0:
            return False
        if is_buy:
            # BUY: SL is below entry. Tighter = higher.
            return current_sl <= 0 or proposed_sl > current_sl
        # SELL: SL is above entry. Tighter = lower.
        return current_sl <= 0 or proposed_sl < current_sl

    def _modify_sltp(
        self,
        pos: dict,
        new_sl: Optional[float],
        new_tp: Optional[float],
        reason: str,
        unrealised_R: float,
    ) -> None:
        symbol = pos.get("symbol")
        is_buy = str(pos.get("type", "BUY")).upper() == "BUY"
        ticket = pos.get("ticket")

        # Only call MT5 with a non-None field
        result = None
        try:
            if new_sl is not None and new_tp is not None:
                # Single call with both — _modify_position_sltp accepts both
                result = self.mt5_api._modify_position_sltp(symbol, is_buy, tp=new_tp, sl=new_sl)
            elif new_sl is not None:
                result = self.mt5_api.place_stop_loss(symbol, is_buy, None, new_sl)
            elif new_tp is not None:
                result = self.mt5_api.place_take_profit(symbol, is_buy, None, new_tp)
        except Exception as err:
            logger.exception("PositionManager: SLTP modify error %s %s: %s", symbol, reason, err)
            return

        retcode = (result or {}).get("retcode")
        success = retcode == 10009
        logger.info(
            "PositionManager: %s %s ticket=%s R=%.2f new_sl=%s new_tp=%s retcode=%s",
            reason, symbol, ticket, unrealised_R, new_sl, new_tp, retcode,
        )
        self._write_diary({
            "asset": symbol,
            "action": f"pm_{reason}",
            "ticket": ticket,
            "unrealised_R": round(unrealised_R, 3),
            "new_sl": new_sl,
            "new_tp": new_tp,
            "retcode": retcode,
            "success": success,
        })

    def _partial_close(self, pos: dict, state: dict, unrealised_R: float) -> None:
        symbol = pos.get("symbol")
        ticket = pos.get("ticket")
        is_buy = str(pos.get("type", "BUY")).upper() == "BUY"
        volume = float(pos.get("volume") or 0)
        if volume <= 0:
            return

        raw_close = volume * self.partial_tp_fraction
        # Respect broker volume_step by letting MT5 close what it can; we still
        # floor to 3 decimals as a sane default. The broker rejects <volume_min.
        try:
            info = self.mt5_api._symbol_info(symbol)
            step = getattr(info, "volume_step", 0.01) or 0.01
            vol_min = getattr(info, "volume_min", 0.01) or 0.01
        except Exception:
            step = 0.01
            vol_min = 0.01

        close_vol = math.floor(raw_close / step) * step
        close_vol = round(close_vol, 8)
        remaining = round(volume - close_vol, 8)
        if close_vol < vol_min or remaining < vol_min:
            logger.info(
                "PositionManager: skipping partial close %s — close_vol=%.4f "
                "remaining=%.4f below volume_min=%.4f",
                symbol, close_vol, remaining, vol_min,
            )
            state["partial_taken"] = True  # Don't retry every cycle
            return

        try:
            result = self.mt5_api.close_position(symbol, ticket, close_vol, is_buy)
        except Exception as err:
            logger.exception("PositionManager: partial close error %s: %s", symbol, err)
            return

        retcode = (result or {}).get("retcode")
        success = retcode == 10009
        logger.info(
            "PositionManager: partial_close %s ticket=%s vol=%.4f (of %.4f) R=%.2f retcode=%s",
            symbol, ticket, close_vol, volume, unrealised_R, retcode,
        )
        self._write_diary({
            "asset": symbol,
            "action": "pm_partial_close",
            "ticket": ticket,
            "closed_volume": close_vol,
            "remaining_volume": remaining,
            "unrealised_R": round(unrealised_R, 3),
            "retcode": retcode,
            "success": success,
        })
        if success:
            state["partial_taken"] = True
