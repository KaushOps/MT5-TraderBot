"""JSON serialisation helpers — unchanged from original."""

import math


def json_default(obj):
    """Handle non-serialisable types for json.dumps(default=...)."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def round_or_none(value, decimals: int = 2):
    """Round a numeric value to ``decimals`` dp, returning None if input is None."""
    if value is None:
        return None
    try:
        return round(float(value), decimals)
    except (TypeError, ValueError):
        return None


def round_series(series: list, decimals: int = 2) -> list:
    """Round each value in a series, preserving None entries."""
    return [round_or_none(v, decimals) for v in series]
