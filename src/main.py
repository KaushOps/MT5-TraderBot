"""Entry-point script — wires together the MT5 trading agent, data feeds, and API.

Adapted from the Hyperliquid AI agent for Exness MetaTrader 5.

Key differences from the Hyperliquid version:
  - MT5API replaces HyperliquidAPI (same interface; synchronous calls
    are dispatched with asyncio.to_thread to keep the event loop alive)
  - TP/SL are embedded in the market order via mt5_api (no separate
    trigger orders to track)
  - Funding rate → swap rate in context sent to Claude
  - MT5 connect/disconnect lifecycle at startup and shutdown
"""

import sys
import argparse
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.agent.decision_maker import TradingAgent
from src.indicators.local_indicators import compute_all, last_n, latest
from src.market_hours import MarketHours
from src.position_manager import PositionManager
from src.risk_manager import RiskManager
from src.trading.mt5_api import MT5API
import asyncio
import logging
from collections import deque, OrderedDict
from datetime import datetime, timezone
import math
import os
import json

from dotenv import load_dotenv
from aiohttp import web

from src.utils.formatting import format_number as fmt, format_size as fmt_sz
from src.utils.prompt_utils import json_default, round_or_none, round_series

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clear_terminal():
    """Clear the terminal screen on Windows or POSIX systems."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_interval_seconds(interval_str: str) -> int:
    """Convert interval strings like '5m' or '1h' to seconds."""
    if interval_str.endswith('m'):
        return int(interval_str[:-1]) * 60
    elif interval_str.endswith('h'):
        return int(interval_str[:-1]) * 3600
    elif interval_str.endswith('d'):
        return int(interval_str[:-1]) * 86400
    else:
        raise ValueError(f"Unsupported interval: {interval_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse CLI args, bootstrap dependencies, and launch the trading loop."""
    clear_terminal()
    parser = argparse.ArgumentParser(description="LLM-based Trading Agent on Exness MT5")
    parser.add_argument("--assets", type=str, nargs="+", required=False,
                        help="Symbols to trade, e.g. BTCUSD ETHUSD XAUUSD")
    parser.add_argument("--interval", type=str, required=False,
                        help="Interval period, e.g. 5m, 1h")
    args = parser.parse_args()

    from src.config_loader import CONFIG
    assets_env  = CONFIG.get("assets")
    interval_env = CONFIG.get("interval")

    if (not args.assets or len(args.assets) == 0) and assets_env:
        if "," in assets_env:
            args.assets = [a.strip() for a in assets_env.split(",") if a.strip()]
        else:
            args.assets = [a.strip() for a in assets_env.split(" ") if a.strip()]
    if not args.interval and interval_env:
        args.interval = interval_env

    if not args.assets or not args.interval:
        parser.error("Please provide --assets and --interval, or set ASSETS and INTERVAL in .env")

    # --- MT5 connection -------------------------------------------------------
    mt5_api = MT5API()
    if not mt5_api.connect():
        logging.error("Cannot connect to MT5 terminal. Ensure MT5 is running and credentials are correct.")
        sys.exit(1)

    # Pre-resolve symbols so bad configs fail fast
    for sym in args.assets:
        resolved = mt5_api.resolve_symbol(sym)
        if resolved is None:
            logging.error("Symbol '%s' not available on this broker — remove it from ASSETS.", sym)
            mt5_api.disconnect()
            sys.exit(1)

    agent    = TradingAgent(mt5_api=mt5_api)
    risk_mgr = RiskManager(mt5_api=mt5_api)
    diary_path = "diary.jsonl"
    pos_mgr  = PositionManager(mt5_api=mt5_api, diary_path=diary_path)
    market_hours = MarketHours(mt5_api=mt5_api)

    start_time           = datetime.now(timezone.utc)
    invocation_count     = 0
    trade_log            = []     # list of trade return dicts for Sharpe calc
    active_trades        = []     # {asset, is_long, amount, entry_price, ticket, exit_plan, opened_at}
    initial_account_value = None
    price_history         = {}    # {symbol: deque of {t, mid}}

    print(f"Starting MT5 trading agent | assets: {args.assets} | interval: {args.interval}")

    def add_event(msg: str):
        logging.info(msg)

    # -------------------------------------------------------------------------
    # Async wrappers — keep the event loop clean while MT5 runs sync calls
    # -------------------------------------------------------------------------

    async def _t(fn, *args, **kwargs):
        """Run a synchronous MT5 call in a threadpool executor."""
        return await asyncio.to_thread(fn, *args, **kwargs)

    # -------------------------------------------------------------------------
    # Core trading loop
    # -------------------------------------------------------------------------

    async def run_loop():
        nonlocal invocation_count, initial_account_value

        while True:
            invocation_count += 1
            minutes_since_start = (datetime.now(timezone.utc) - start_time).total_seconds() / 60

            # --- Account state ------------------------------------------------
            state = await _t(mt5_api.get_user_state)
            account_value = state.get("total_value") or state["balance"]
            if initial_account_value is None:
                initial_account_value = account_value
            total_return_pct = (
                ((account_value - initial_account_value) / initial_account_value * 100.0)
                if initial_account_value else 0.0
            )
            sharpe = _calculate_sharpe(trade_log)

            # Normalise position shape for risk_manager (which expects certain keys)
            positions_normalised = []
            for pos in state["positions"]:
                positions_normalised.append({
                    # RiskManager reads these keys
                    "symbol":        pos.get("symbol"),
                    "coin":          pos.get("symbol"),       # compat alias
                    "volume":        pos.get("volume", 0),
                    "szi":           pos.get("volume", 0),    # compat alias
                    "price_open":    pos.get("price_open", 0),
                    "entryPx":       pos.get("price_open", 0),
                    "profit":        pos.get("profit", 0),
                    "pnl":           pos.get("pnl", 0),
                    "contract_size": pos.get("contract_size", 1.0),
                    "type":          pos.get("type", "BUY"),
                    "ticket":        pos.get("ticket"),
                    "current_price": pos.get("current_price", 0),
                    "sl":            pos.get("sl", 0),
                    "tp":            pos.get("tp", 0),
                })

            # --- RISK: force-close over-loss positions -------------------------
            try:
                positions_to_close = risk_mgr.check_losing_positions(positions_normalised)
                for ptc in positions_to_close:
                    sym    = ptc["symbol"]
                    vol    = ptc["volume"]
                    ticket = ptc.get("ticket")
                    is_buy = ptc["is_buy"]
                    add_event(f"RISK FORCE-CLOSE: {sym} ticket={ticket} at {ptc['loss_pct']}% loss (P&L: ${ptc['profit']})")
                    try:
                        res = await _t(mt5_api.close_position, sym, ticket, vol, is_buy)
                        await _t(mt5_api.cancel_all_orders, sym)
                        for tr in active_trades[:]:
                            if tr.get("asset") == sym:
                                active_trades.remove(tr)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset":     sym,
                                "action":    "risk_force_close",
                                "loss_pct":  ptc["loss_pct"],
                                "profit":    ptc["profit"],
                                "retcode":   res.get("retcode"),
                            }) + "\n")
                    except Exception as fc_err:
                        add_event(f"Force-close error for {sym}: {fc_err}")
            except Exception as risk_err:
                add_event(f"Risk check error: {risk_err}")

            # --- Recent diary entries (for LLM context) -----------------------
            recent_diary = []
            try:
                with open(diary_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[-4:]:
                        entry = json.loads(line)
                        entry.pop("rationale", None)
                        entry.pop("exit_plan", None)
                        recent_diary.append(entry)
            except Exception:
                pass

            # --- Open orders --------------------------------------------------
            open_orders_struct = []
            try:
                open_orders = await _t(mt5_api.get_open_orders)
                for o in open_orders[:50]:
                    open_orders_struct.append({
                        "coin":       o.get("symbol"),
                        "oid":        o.get("ticket"),
                        "is_buy":     o.get("is_buy"),
                        "size":       round_or_none(o.get("size"), 5),
                        "price":      round_or_none(o.get("price"), 5),
                        "sl":         round_or_none(o.get("sl"), 5),
                        "tp":         round_or_none(o.get("tp"), 5),
                        "order_type": o.get("order_type"),
                    })
            except Exception:
                open_orders = []

            # --- Reconcile active_trades vs real positions --------------------
            try:
                live_symbols = {p.get("symbol") for p in state["positions"]}
                order_symbols = {o.get("symbol") for o in (open_orders or [])}
                for tr in active_trades[:]:
                    asset = tr.get("asset")
                    if asset not in live_symbols and asset not in order_symbols:
                        add_event(f"Reconciling stale active trade for {asset}")
                        active_trades.remove(tr)
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset":     asset,
                                "action":    "reconcile_close",
                                "reason":    "no_position_no_orders",
                                "opened_at": tr.get("opened_at"),
                            }) + "\n")
            except Exception:
                pass

            # --- Recent fills -------------------------------------------------
            recent_fills_struct = []
            try:
                fills = await _t(mt5_api.get_recent_fills, 50)
                for f_entry in fills[-4:]:
                    recent_fills_struct.append({
                        "timestamp": f_entry.get("timestamp"),
                        "coin":      f_entry.get("symbol"),
                        "is_buy":    f_entry.get("is_buy"),
                        "size":      round_or_none(f_entry.get("size"), 5),
                        "price":     round_or_none(f_entry.get("price"), 5),
                        "profit":    round_or_none(f_entry.get("profit"), 2),
                    })
            except Exception:
                pass

            # --- Dashboard snapshot -------------------------------------------
            dashboard = {
                "total_return_pct": round(total_return_pct, 2),
                "balance":          round_or_none(state["balance"], 2),
                "account_value":    round_or_none(account_value, 2),
                "sharpe_ratio":     round_or_none(sharpe, 3),
                "positions": [
                    {
                        "symbol":        p.get("symbol"),
                        "type":          p.get("type"),
                        "volume":        round_or_none(p.get("volume"), 5),
                        "price_open":    round_or_none(p.get("price_open"), 5),
                        "current_price": round_or_none(p.get("current_price"), 5),
                        "sl":            round_or_none(p.get("sl"), 5),
                        "tp":            round_or_none(p.get("tp"), 5),
                        "profit":        round_or_none(p.get("profit"), 2),
                        "swap":          round_or_none(p.get("swap"), 2),
                    }
                    for p in state["positions"]
                ],
                "active_trades": [
                    {
                        "asset":       tr.get("asset"),
                        "is_long":     tr.get("is_long"),
                        "amount":      round_or_none(tr.get("amount"), 5),
                        "entry_price": round_or_none(tr.get("entry_price"), 5),
                        "ticket":      tr.get("ticket"),
                        "exit_plan":   tr.get("exit_plan"),
                        "opened_at":   tr.get("opened_at"),
                    }
                    for tr in active_trades
                ],
                "open_orders":   open_orders_struct,
                "recent_diary":  recent_diary,
                "recent_fills":  recent_fills_struct,
            }

            # --- Smart Polling: Filter Assets ---------------------------------
            # Always evaluate (at least data-gather) any asset that currently
            # has an open live position — the PositionManager needs fresh ATR.
            live_symbols_for_poll = {p.get("symbol") for p in state["positions"]}
            assets_to_evaluate = []
            auto_hold_decisions = []
            closed_markets = {}   # asset -> reason (passed to execution dispatch)
            for asset in args.assets:
                has_active = (
                    any(tr.get("asset") == asset for tr in active_trades)
                    or asset in live_symbols_for_poll
                )
                
                # Market hours check — skip closed symbols entirely unless we
                # have an open position (we still need to manage it).
                tradable, mh_reason = market_hours.is_tradable(asset)
                if not tradable and not has_active:
                    closed_markets[asset] = mh_reason
                    auto_hold_decisions.append({
                        "asset": asset,
                        "action": "hold",
                        "rationale": f"Market closed: {mh_reason}",
                        "allocation_usd": 0.0,
                    })
                    continue

                # ── ALWAYS evaluate if we have an open position ──
                if has_active:
                    assets_to_evaluate.append(asset)
                    continue

                # ── Smart polling for assets WITHOUT positions ──
                if invocation_count % 3 != 1:
                    auto_hold_decisions.append({
                        "asset": asset,
                        "action": "hold",
                        "rationale": "Smart Polling: Skipped cold asset cycle.",
                        "allocation_usd": 0.0,
                    })
                    continue
                
                assets_to_evaluate.append(asset)

            if not assets_to_evaluate:
                add_event("Smart Polling: All assets cold, skipping LLM evaluation this cycle.")
                outputs = {"trade_decisions": auto_hold_decisions, "reasoning": "Skipped due to cold asset logic"}
                goto_execution = True
            else:
                goto_execution = False

            # --- Gather market data for all active assets ---------------------
            market_sections = []
            asset_prices    = {}
            asset_intraday_atr = {}   # 5m ATR14 per asset, used by RiskManager
            asset_intraday_data = {}  # Store for ADX check later
            asset_1h_atr = {}

            for asset in args.assets:
                # Update prices to memory regardless
                try:
                    current_price = await _t(mt5_api.get_current_price, asset)
                    asset_prices[asset] = current_price
                    if asset not in price_history:
                        price_history[asset] = deque(maxlen=60)
                    price_history[asset].append({
                        "t":   datetime.now(timezone.utc).isoformat(),
                        "mid": round_or_none(current_price, 5),
                    })
                except Exception as e:
                    add_event(f"Price fetch error {asset}: {e}")
                    continue
                
                # Only gather heavy data for evaluated assets
                if asset not in assets_to_evaluate:
                    continue
                
                try:
                    swap_rate = await _t(mt5_api.get_funding_rate, asset)   # actually swap

                    candles_5m = await _t(mt5_api.get_candles, asset, "5m", 100)
                    candles_4h = await _t(mt5_api.get_candles, asset, "4h", 100)
                    candles_1h = await _t(mt5_api.get_candles, asset, "1h", 100)

                    intra = compute_all(candles_5m)
                    lt    = compute_all(candles_4h)
                    tf_1h = compute_all(candles_1h)
                    
                    asset_intraday_data[asset] = intra

                    recent_mids = [e["mid"] for e in list(price_history.get(asset, []))[-3:]]
                    swap_annualized = round(swap_rate * 365 * 100, 4) if swap_rate else None

                    intraday_atr14 = latest(intra.get("atr14", []))
                    intraday_atr3  = latest(intra.get("atr3", []))
                    asset_intraday_atr[asset] = intraday_atr14
                    
                    atr14_1h = latest(tf_1h.get("atr14", []))
                    asset_1h_atr[asset] = atr14_1h

                    market_sections.append({
                        "asset":         asset,
                        "current_price": round_or_none(current_price, 5),
                        "intraday": {
                            "ema20":  round_or_none(latest(intra.get("ema20", [])), 5),
                            "macd":   round_or_none(latest(intra.get("macd", [])), 5),
                            "rsi7":   round_or_none(latest(intra.get("rsi7", [])), 2),
                            "rsi14":  round_or_none(latest(intra.get("rsi14", [])), 2),
                            "atr3":   round_or_none(intraday_atr3, 5),
                            "atr14":  round_or_none(intraday_atr14, 5),
                            "atr14_pct_of_price": (
                                round((intraday_atr14 / current_price) * 100.0, 3)
                                if intraday_atr14 and current_price else None
                            ),
                            "series": {
                                "ema20": round_series(last_n(intra.get("ema20", []), 3), 5),
                                "macd":  round_series(last_n(intra.get("macd", []), 3), 3),
                                "rsi7":  round_series(last_n(intra.get("rsi7", []), 3), 2),
                                "rsi14": round_series(last_n(intra.get("rsi14", []), 3), 2),
                                "atr14": round_series(last_n(intra.get("atr14", []), 3), 5),
                            },
                        },
                        "long_term": {
                            "ema20":       round_or_none(latest(lt.get("ema20", [])), 5),
                            "ema50":       round_or_none(latest(lt.get("ema50", [])), 5),
                            "atr3":        round_or_none(latest(lt.get("atr3", [])), 5),
                            "atr14":       round_or_none(latest(lt.get("atr14", [])), 5),
                            "macd_series": round_series(last_n(lt.get("macd", []), 3), 5),
                            "rsi_series":  round_series(last_n(lt.get("rsi14", []), 3), 2),
                        },
                        "open_interest":         None,   # not available in MT5
                        "swap_rate":             round_or_none(swap_rate, 8),
                        "swap_annualized_pct":   swap_annualized,
                        "recent_mid_prices":     recent_mids,
                    })
                except Exception as e:
                    add_event(f"Data gather error {asset}: {e}")
                    continue

            # --- Active position management (Phase 2 & 3) --------------------
            # Runs BEFORE the LLM call so breakeven / trail / partial-TP are
            # driven purely by deterministic logic, not model latency.
            try:
                await asyncio.to_thread(
                    pos_mgr.manage, state["positions"], asset_1h_atr
                )
            except Exception as pm_err:
                add_event(f"PositionManager error: {pm_err}")

            # --- Single LLM call with all assets ------------------------------
            if not goto_execution:
                context_payload = OrderedDict([
                    ("invocation", {
                        "minutes_since_start": round(minutes_since_start, 2),
                        "current_time":        datetime.now(timezone.utc).isoformat(),
                        "invocation_count":    invocation_count,
                    }),
                    ("account",     dashboard),
                    ("risk_limits", risk_mgr.get_risk_summary()),
                    ("market_data", market_sections),
                    ("instructions", {
                        "assets":      assets_to_evaluate,
                        "requirement": "Decide actions for all assets and return a strict JSON object matching the schema.",
                    }),
                ])
                context = json.dumps(context_payload, default=json_default)
                add_event(f"Combined prompt length: {len(context)} chars for {len(assets_to_evaluate)} assets")
                with open("prompts.log", "a") as f:
                    f.write(f"\n\n--- {datetime.now()} - ALL ASSETS ---\n"
                            f"{json.dumps(context_payload, indent=2, default=json_default)}\n")

                def _is_failed_outputs(outs):
                    if not isinstance(outs, dict):
                        return True
                    decisions = outs.get("trade_decisions")
                    if not isinstance(decisions, list) or not decisions:
                        return True
                    try:
                        return all(
                            isinstance(o, dict)
                            and (o.get("action") == "hold")
                            and ("parse error" in (o.get("rationale", "").lower()))
                            for o in decisions
                        )
                    except Exception:
                        return True

                # Run LLM in thread (it blocks on HTTP)
                try:
                    outputs = await asyncio.to_thread(agent.decide_trade, assets_to_evaluate, context)
                    if not isinstance(outputs, dict):
                        add_event(f"Invalid output format: {outputs}")
                        outputs = {}
                except Exception as e:
                    import traceback
                    add_event(f"Agent error: {e}\n{traceback.format_exc()}")
                    outputs = {}

                # Retry once if output is clearly bad
                if _is_failed_outputs(outputs):
                    add_event("Retrying LLM once due to invalid/parse-error output")
                    retry_payload = OrderedDict([
                        ("retry_instruction", "Return ONLY the JSON array per schema with no prose."),
                        ("original_context",  context_payload),
                    ])
                    retry_context = json.dumps(retry_payload, default=json_default)
                    try:
                        outputs = await asyncio.to_thread(agent.decide_trade, assets_to_evaluate, retry_context)
                        if not isinstance(outputs, dict):
                            outputs = {}
                    except Exception as e:
                        add_event(f"Retry agent error: {e}")
                        outputs = {}

                # Inject auto-holds
                if isinstance(outputs, dict) and "trade_decisions" in outputs:
                    outputs["trade_decisions"].extend(auto_hold_decisions)
                else:
                    outputs = {"trade_decisions": auto_hold_decisions, "reasoning": "Failed parse, auto holds added"}

                reasoning_text = outputs.get("reasoning", "") if isinstance(outputs, dict) else ""
                if reasoning_text:
                    add_event(f"LLM reasoning summary: {reasoning_text[:500]}")

            # Log cycle decisions
            cycle_decisions = []
            for d in (outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []):
                cycle_decisions.append({
                    "asset":          d.get("asset"),
                    "action":         d.get("action", "hold"),
                    "allocation_usd": d.get("allocation_usd", 0),
                    "rationale":      d.get("rationale", ""),
                })
            try:
                with open("decisions.jsonl", "a") as f:
                    f.write(json.dumps({
                        "timestamp":       datetime.now(timezone.utc).isoformat(),
                        "cycle":           invocation_count,
                        "reasoning":       reasoning_text[:2000] if reasoning_text else "",
                        "decisions":       cycle_decisions,
                        "account_value":   round_or_none(account_value, 2),
                        "balance":         round_or_none(state["balance"], 2),
                        "positions_count": len(state["positions"]),
                    }) + "\n")
            except Exception:
                pass

            # --- Execute trades -----------------------------------------------
            for output in (outputs.get("trade_decisions", []) if isinstance(outputs, dict) else []):
                try:
                    asset = output.get("asset")
                    if not asset or asset not in args.assets:
                        continue

                    action        = output.get("action", "hold")
                    current_price = asset_prices.get(asset, 0)
                    rationale     = output.get("rationale", "")

                    if rationale:
                        add_event(f"Decision rationale for {asset}: {rationale}")

                    if action in ("buy", "sell"):
                        is_buy    = action == "buy"
                        alloc_usd = float(output.get("allocation_usd", 0.0))
                        if alloc_usd <= 0:
                            add_event(f"Holding {asset}: zero/negative allocation")
                            continue

                        # --- Market hours gate on new entries ---------------
                        # Closed market → reject outright.
                        mh_reason = closed_markets.get(asset)
                        if mh_reason:
                            add_event(f"MARKET CLOSED {asset}: skipping new {action} ({mh_reason})")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset":     asset,
                                    "action":    "market_closed_skip",
                                    "reason":    mh_reason,
                                }) + "\n")
                            continue
                        # About-to-close → block new entries (overnight gap risk).
                        near_close, secs_left = market_hours.is_near_close(asset)
                        if near_close:
                            add_event(
                                f"SESSION CLOSE BLOCK {asset}: {secs_left}s until close — "
                                f"skipping new {action}"
                            )
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset":     asset,
                                    "action":    "pre_close_block",
                                    "seconds_to_close": secs_left,
                                }) + "\n")
                            continue

                        # --- ADX Regime Filter ---
                        adx_val = latest(asset_intraday_data.get(asset, {}).get("adx", []))
                        ok, adx_reason = risk_mgr.check_trend_strength({
                            "adx": adx_val
                        })
                        if not ok:
                            add_event(f"ADX REGIME BLOCK {asset}: {adx_reason}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp":          datetime.now(timezone.utc).isoformat(),
                                    "asset":              asset,
                                    "action":             "adx_blocked",
                                    "reason":             adx_reason,
                                }) + "\n")
                            continue

                        # --- Risk validation ---------------------------------
                        output["current_price"] = current_price
                        output["atr_at_entry"] = asset_intraday_atr.get(asset)
                        output["atr_1h"] = asset_1h_atr.get(asset)
                        account_state_for_risk = {
                            "balance":     state["balance"],
                            "total_value": account_value,
                            "equity":      account_value,
                            "positions":   positions_normalised,
                        }
                        allowed, reason, output = risk_mgr.validate_trade(
                            output, account_state_for_risk, initial_account_value or 0
                        )
                        if not allowed:
                            add_event(f"RISK BLOCKED {asset}: {reason}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp":          datetime.now(timezone.utc).isoformat(),
                                    "asset":              asset,
                                    "action":             "risk_blocked",
                                    "reason":             reason,
                                    "original_alloc_usd": alloc_usd,
                                }) + "\n")
                            continue

                        alloc_usd  = float(output.get("allocation_usd", alloc_usd))
                        tp_price   = float(output["tp_price"]) if output.get("tp_price") else 0.0
                        sl_price   = float(output["sl_price"]) if output.get("sl_price") else 0.0
                        order_type = output.get("order_type", "market")
                        limit_price = output.get("limit_price")

                        # --- Place order -------------------------------------
                        if order_type == "limit" and limit_price:
                            limit_price = float(limit_price)
                            if is_buy:
                                order_res = await _t(mt5_api.place_limit_buy, asset, alloc_usd, limit_price, tp_price, sl_price)
                            else:
                                order_res = await _t(mt5_api.place_limit_sell, asset, alloc_usd, limit_price, tp_price, sl_price)
                            add_event(f"LIMIT {action.upper()} {asset} ${alloc_usd:.2f} @ {limit_price}")
                        else:
                            if is_buy:
                                order_res = await _t(mt5_api.place_buy_order, asset, alloc_usd, tp_price, sl_price)
                            else:
                                order_res = await _t(mt5_api.place_sell_order, asset, alloc_usd, tp_price, sl_price)

                        retcode = order_res.get("retcode", -1)
                        ticket  = order_res.get("order", 0) or order_res.get("deal", 0)
                        success = retcode == 10009  # TRADE_RETCODE_DONE

                        # If market order succeeds, forcefully re-attach the TP and SL 
                        # just in case this broker enforces "Market Execution" (which strips initial TP/SL natively).
                        if success and order_type != "limit" and (tp_price > 0 or sl_price > 0):
                            await asyncio.sleep(0.5)  # Let MT5 register the trade first
                            try:
                                await _t(mt5_api.force_modify_sltp_on_market_order, asset, is_buy, tp_price, sl_price)
                            except Exception as er:
                                add_event(f"Fail attaching SL/TP to Market Exec {asset}: {er}")

                        if not success:
                            add_event(f"Order failed {asset}: retcode={retcode} comment={order_res.get('comment')}")
                        else:
                            add_event(f"{action.upper()} {asset} ${alloc_usd:.2f} ticket={ticket}")

                        # Reconcile active_trades
                        for existing in active_trades[:]:
                            if existing.get("asset") == asset:
                                try:
                                    active_trades.remove(existing)
                                except ValueError:
                                    pass
                        if success and order_type != "limit":
                            # Register with PositionManager so breakeven / trail /
                            # partial-TP work on this position from the next cycle.
                            try:
                                pos_mgr.register_new_trade(
                                    ticket=int(ticket) if ticket else 0,
                                    symbol=asset,
                                    entry_price=current_price,
                                    initial_sl=float(output.get("sl_price") or 0),
                                    initial_tp=(float(output["tp_price"])
                                                if output.get("tp_price") else None),
                                    is_buy=is_buy,
                                    atr_at_entry=asset_1h_atr.get(asset),
                                )
                            except Exception as reg_err:
                                add_event(f"PositionManager register error {asset}: {reg_err}")

                        if success:
                            active_trades.append({
                                "asset":       asset,
                                "is_long":     is_buy,
                                "amount":      alloc_usd,
                                "entry_price": current_price,
                                "ticket":      ticket,
                                "exit_plan":   output.get("exit_plan", ""),
                                "opened_at":   datetime.now().isoformat(),
                            })

                        trade_log.append({
                            "type":      action,
                            "price":     current_price,
                            "amount":    alloc_usd,
                            "exit_plan": output.get("exit_plan", ""),
                            "filled":    success,
                        })

                        # Diary entry
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp":    datetime.now(timezone.utc).isoformat(),
                                "asset":        asset,
                                "action":       action,
                                "order_type":   order_type,
                                "limit_price":  limit_price if order_type == "limit" else None,
                                "allocation_usd": alloc_usd,
                                "entry_price":  current_price,
                                "tp_price":     tp_price or None,
                                "sl_price":     sl_price or None,
                                "exit_plan":    output.get("exit_plan", ""),
                                "rationale":    output.get("rationale", ""),
                                "ticket":       ticket,
                                "retcode":      retcode,
                                "filled":       success,
                                "opened_at":    datetime.now(timezone.utc).isoformat(),
                            }) + "\n")
                    elif action == "adjust":
                        # --- LLM-driven TP/SL modification (Phase 4) ---------
                        raw_tp = output.get("tp_price")
                        raw_sl = output.get("sl_price")
                        new_tp = float(raw_tp) if raw_tp not in (None, "") else None
                        new_sl = float(raw_sl) if raw_sl not in (None, "") else None

                        target_pos = next(
                            (p for p in positions_normalised if p.get("symbol") == asset),
                            None,
                        )
                        if target_pos is None:
                            add_event(f"Adjust skipped {asset}: no open position")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset":     asset,
                                    "action":    "adjust_skipped",
                                    "reason":    "no_position",
                                    "rationale": rationale,
                                }) + "\n")
                            continue

                        allowed, reason, adj_tp, adj_sl = risk_mgr.validate_adjust(
                            asset, target_pos, new_tp, new_sl,
                            asset_intraday_atr.get(asset),
                            asset_1h_atr.get(asset),
                        )
                        if not allowed:
                            add_event(f"RISK BLOCKED adjust {asset}: {reason}")
                            with open(diary_path, "a") as f:
                                f.write(json.dumps({
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "asset":     asset,
                                    "action":    "adjust_blocked",
                                    "reason":    reason,
                                    "rationale": rationale,
                                }) + "\n")
                            continue

                        is_buy_pos = str(target_pos.get("type", "BUY")).upper() == "BUY"
                        try:
                            modify_res = await _t(
                                mt5_api._modify_position_sltp,
                                asset, is_buy_pos,
                                adj_tp, adj_sl,
                            )
                        except Exception as mod_err:
                            add_event(f"Adjust error {asset}: {mod_err}")
                            continue

                        adj_retcode = (modify_res or {}).get("retcode")
                        adj_ok = adj_retcode == 10009
                        add_event(
                            f"ADJUST {asset} ticket={target_pos.get('ticket')} "
                            f"new_tp={adj_tp} new_sl={adj_sl} retcode={adj_retcode}"
                        )
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "asset":     asset,
                                "action":    "adjust",
                                "ticket":    target_pos.get("ticket"),
                                "new_tp":    adj_tp,
                                "new_sl":    adj_sl,
                                "retcode":   adj_retcode,
                                "filled":    adj_ok,
                                "rationale": rationale,
                            }) + "\n")
                    else:
                        add_event(f"Hold {asset}: {rationale}")
                        with open(diary_path, "a") as f:
                            f.write(json.dumps({
                                "timestamp": datetime.now().isoformat(),
                                "asset":     asset,
                                "action":    "hold",
                                "rationale": rationale,
                            }) + "\n")

                except Exception as e:
                    import traceback
                    add_event(f"Execution error {asset}: {e}\n{traceback.format_exc()}")

            await asyncio.sleep(get_interval_seconds(args.interval))

    # -------------------------------------------------------------------------
    # HTTP API
    # -------------------------------------------------------------------------

    async def handle_diary(request):
        """Return diary entries as JSON or newline-delimited text."""
        try:
            raw      = request.query.get("raw")
            download = request.query.get("download")
            if raw or download:
                if not os.path.exists(diary_path):
                    return web.Response(text="", content_type="text/plain")
                with open(diary_path, "r") as f:
                    data = f.read()
                headers = {}
                if download:
                    headers["Content-Disposition"] = "attachment; filename=diary.jsonl"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(request.query.get("limit", "200"))
            with open(diary_path, "r") as f:
                lines = f.readlines()
            start = max(0, len(lines) - limit)
            entries = [json.loads(l) for l in lines[start:]]
            return web.json_response({"entries": entries})
        except FileNotFoundError:
            return web.json_response({"entries": []})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_logs(request):
        """Stream log files with optional download or tailing behaviour."""
        try:
            path     = request.query.get("path", "llm_requests.log")
            download = request.query.get("download")
            limit_p  = request.query.get("limit")
            if not os.path.exists(path):
                return web.Response(text="", content_type="text/plain")
            with open(path, "r") as f:
                data = f.read()
            if download or (limit_p and (limit_p.lower() == "all" or limit_p == "-1")):
                headers = {}
                if download:
                    headers["Content-Disposition"] = f"attachment; filename={os.path.basename(path)}"
                return web.Response(text=data, content_type="text/plain", headers=headers)
            limit = int(limit_p) if limit_p else 2000
            return web.Response(text=data[-limit:], content_type="text/plain")
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def main_async():
        """Start the aiohttp server and kick off the trading loop."""
        app = web.Application()
        app.router.add_get("/diary", handle_diary)
        app.router.add_get("/logs", handle_logs)

        from src.config_loader import CONFIG as CFG
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, CFG.get("api_host"), int(CFG.get("api_port")))
        await site.start()
        add_event(f"API server started on {CFG.get('api_host')}:{CFG.get('api_port')}")

        try:
            await run_loop()
        finally:
            mt5_api.disconnect()

    # -------------------------------------------------------------------------
    # Utilities (inner scope)
    # -------------------------------------------------------------------------

    def _calculate_sharpe(returns: list) -> float:
        """Compute a naive Sharpe-like ratio from the trade log."""
        if not returns:
            return 0
        vals = [r.get("pnl", 0) if "pnl" in r else 0 for r in returns]
        if not vals:
            return 0
        mean = sum(vals) / len(vals)
        var  = sum((v - mean) ** 2 for v in vals) / len(vals)
        std  = math.sqrt(var) if var > 0 else 0
        return mean / std if std > 0 else 0

    asyncio.run(main_async())


if __name__ == "__main__":
    main()
