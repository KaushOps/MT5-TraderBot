"""Number formatting helpers — unchanged from original."""


def format_number(value, decimals: int = 2) -> str:
    """Format a float to ``decimals`` decimal places, or return '--' for None."""
    if value is None:
        return "--"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


# Alias used in main.py
format_size = format_number
