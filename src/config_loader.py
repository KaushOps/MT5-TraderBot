"""Centralized environment variable loading for the MT5 trading agent configuration."""

import json
import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Fetch an environment variable with optional default and required validation."""
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int | None = None, required: bool = False) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        if required:
            raise RuntimeError(f"Missing required environment variable: {name}")
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for {name}: {raw}") from exc


def _get_json(name: str, default: dict | None = None) -> dict | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Environment variable {name} must be a JSON object")
        return parsed
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON for {name}: {raw}") from exc


CONFIG = {
    # ── MetaTrader 5 credentials ─────────────────────────────────────────────
    "mt5_login":    _get_int("MT5_LOGIN", required=True),
    "mt5_password": _get_env("MT5_PASSWORD", required=True),
    "mt5_server":   _get_env("MT5_SERVER", required=True),
    "mt5_magic":    _get_int("MT5_MAGIC", 20260419),   # tag all bot orders
    "mt5_slippage": _get_int("MT5_SLIPPAGE", 20),       # points of acceptable slippage

    # ── LLM Configuration ────────────────────────────────────────────────────
    "llm_api_key":          _get_env("LLM_API_KEY", required=True),
    "llm_model":            _get_env("LLM_MODEL", "llama-3.3-70b-versatile"),
    "llm_base_url":         _get_env("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
    "fallback_llm_api_key":   _get_env("FALLBACK_LLM_API_KEY"),
    "fallback_llm_base_url":  _get_env("FALLBACK_LLM_BASE_URL", "https://openrouter.ai/api/v1"),
    "enable_tool_calling":    _get_bool("ENABLE_TOOL_CALLING", True),

    # ── Runtime controls ─────────────────────────────────────────────────────
    "assets":   _get_env("ASSETS"),    # e.g. "BTCUSD ETHUSD XAUUSD EURUSD USOIL"
    "interval": _get_env("INTERVAL"),  # e.g. "5m", "1h"

    # ── Risk management ──────────────────────────────────────────────────────
    "max_position_pct":             _get_env("MAX_POSITION_PCT", "10"),
    "max_loss_per_position_pct":    _get_env("MAX_LOSS_PER_POSITION_PCT", "20"),
    "max_leverage":                 _get_env("MAX_LEVERAGE", "10"),
    "max_total_exposure_pct":       _get_env("MAX_TOTAL_EXPOSURE_PCT", "50"),
    "daily_loss_circuit_breaker_pct": _get_env("DAILY_LOSS_CIRCUIT_BREAKER_PCT", "10"),
    "mandatory_sl_pct":             _get_env("MANDATORY_SL_PCT", "2"),
    "max_concurrent_positions":     _get_env("MAX_CONCURRENT_POSITIONS", "5"),
    "min_balance_reserve_pct":      _get_env("MIN_BALANCE_RESERVE_PCT", "20"),

    # ── ATR-anchored TP/SL (Phase 1) ─────────────────────────────────────────
    # All bounds are expressed as multiples of the 5m ATR14 at entry time.
    "tp_atr_mult_min":              _get_env("TP_ATR_MULT_MIN", "1.5"),
    "tp_atr_mult_max":              _get_env("TP_ATR_MULT_MAX", "3.0"),
    "sl_atr_mult_min":              _get_env("SL_ATR_MULT_MIN", "1.2"),
    "sl_atr_mult_max":              _get_env("SL_ATR_MULT_MAX", "2.0"),
    "min_rr":                       _get_env("MIN_RR", "1.5"),
    "max_rr":                       _get_env("MAX_RR", "3.5"),
    # Fallback mandatory SL (used only if ATR is unavailable for the symbol)
    "default_sl_atr_mult":          _get_env("DEFAULT_SL_ATR_MULT", "1.2"),
    "default_tp_atr_mult":          _get_env("DEFAULT_TP_ATR_MULT", "2.0"),

    # Risk-based position sizing & ADX regime filter
    "risk_per_trade_pct":           _get_env("RISK_PER_TRADE_PCT", "1.0"),
    "adx_min_for_entry":            _get_env("ADX_MIN_FOR_ENTRY", "20"),

    # ── Active position management (Phase 2 / 3) ─────────────────────────────
    "enable_position_manager":      _get_bool("ENABLE_POSITION_MANAGER", True),
    # Breakeven move: once unrealised P&L in R-multiples ≥ this value,
    # move SL to entry + (buffer_atr × entry ATR). Set <=0 to disable.
    "breakeven_activate_r":         _get_env("BREAKEVEN_ACTIVATE_R", "1.5"),
    "breakeven_buffer_atr":         _get_env("BREAKEVEN_BUFFER_ATR", "0.15"),
    # Trailing stop: once unrealised R ≥ trail_activate_r, trail SL at
    # (trail_atr_mult × current ATR) behind current price. Set <=0 to disable.
    "trail_activate_r":             _get_env("TRAIL_ACTIVATE_R", "2.0"),
    "trail_atr_mult":               _get_env("TRAIL_ATR_MULT", "1.5"),
    # TP tightening: once unrealised R ≥ tighten_tp_r, pull TP in to
    # current price + (tighten_tp_atr_mult × ATR). Set <=0 to disable.
    "tighten_tp_r":                 _get_env("TIGHTEN_TP_R", "1.5"),
    "tighten_tp_atr_mult":          _get_env("TIGHTEN_TP_ATR_MULT", "1.5"),
    # Partial profit taking: once unrealised R ≥ partial_tp_r, close
    # partial_tp_fraction of the position. Set partial_tp_fraction<=0 to disable.
    "partial_tp_r":                 _get_env("PARTIAL_TP_R", "1.5"),
    "partial_tp_fraction":          _get_env("PARTIAL_TP_FRACTION", "0.33"),

    # ── Market hours filter ──────────────────────────────────────────────────
    # Skip evaluation / block new entries when the market is closed or about
    # to close. Crypto (BTCUSD, ETHUSD) is 24/7 and unaffected.
    "enable_market_hours":          _get_bool("ENABLE_MARKET_HOURS", True),
    "max_tick_age_sec":             _get_env("MAX_TICK_AGE_SEC", "120"),
    "pre_close_block_sec":          _get_env("PRE_CLOSE_BLOCK_SEC", "900"),

    # ── API server ───────────────────────────────────────────────────────────
    "api_host": _get_env("API_HOST", "0.0.0.0"),
    "api_port": _get_env("APP_PORT") or _get_env("API_PORT") or "3000",
}
