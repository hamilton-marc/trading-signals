"""Matplotlib candlestick and overlay helpers for notebook charts."""

from __future__ import annotations

from datetime import date
import math
from typing import Any

import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


def _parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def build_ohlc_arrays(rows: list[dict[str, str]]) -> dict[str, list]:
    """Convert CSV-like dict rows to sorted OHLC arrays.

    Malformed rows are skipped. `Open`, `High`, or `Low` missing values are
    safely imputed from `Close`.
    """

    parsed: list[tuple[date, float, float, float, float, int]] = []

    for row in rows:
        day = _parse_date(row.get("Date"))
        close = _parse_float(row.get("Close"))
        if day is None or close is None:
            continue

        open_price = _parse_float(row.get("Open"))
        high_price = _parse_float(row.get("High"))
        low_price = _parse_float(row.get("Low"))
        volume_raw = (row.get("Volume") or "").strip()

        if open_price is None:
            open_price = close
        if high_price is None:
            high_price = close
        if low_price is None:
            low_price = close

        if high_price < low_price:
            high_price, low_price = low_price, high_price

        try:
            volume = int(float(volume_raw)) if volume_raw else 0
        except ValueError:
            volume = 0

        parsed.append((day, open_price, high_price, low_price, close, volume))

    parsed.sort(key=lambda item: item[0])

    return {
        "dates": [item[0] for item in parsed],
        "opens": [item[1] for item in parsed],
        "highs": [item[2] for item in parsed],
        "lows": [item[3] for item in parsed],
        "closes": [item[4] for item in parsed],
        "volumes": [item[5] for item in parsed],
    }


def plot_candlesticks(
    ax: Any,
    dates: list[date],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    width_days: float = 0.6,
    up_color: str = "#26a69a",
    down_color: str = "#ef5350",
    wick_color: str = "#666666",
    alpha: float = 0.9,
) -> None:
    """Draw candlesticks on a Matplotlib axis."""

    x = mdates.date2num(dates)
    half = width_days / 2.0

    for idx, x_val in enumerate(x):
        open_price = opens[idx]
        high_price = highs[idx]
        low_price = lows[idx]
        close_price = closes[idx]
        color = up_color if close_price >= open_price else down_color

        ax.vlines(x_val, low_price, high_price, color=wick_color, linewidth=0.9, alpha=alpha, zorder=2)

        lower = min(open_price, close_price)
        height = abs(close_price - open_price)
        if math.isclose(height, 0.0, abs_tol=1e-12):
            height = 1e-9

        ax.add_patch(
            Rectangle(
                (x_val - half, lower),
                width_days,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=alpha,
                zorder=3,
            )
        )

    ax.xaxis_date()


def plot_volume(
    ax: Any,
    dates: list[date],
    opens: list[float],
    closes: list[float],
    volumes: list[int],
    *,
    width_days: float = 0.6,
    up_color: str = "#26a69a",
    down_color: str = "#ef5350",
    alpha: float = 0.35,
) -> None:
    """Draw volume bars colored by candle direction."""

    x = mdates.date2num(dates)
    colors = [up_color if close >= open_price else down_color for open_price, close in zip(opens, closes, strict=True)]
    ax.bar(x, volumes, width=width_days, color=colors, alpha=alpha, align="center", linewidth=0)
    ax.xaxis_date()


def overlay_ema(
    ax: Any,
    dates: list[date],
    ema_values: list[float | None],
    *,
    label: str,
    color: str,
    linewidth: float = 1.2,
) -> None:
    """Overlay an EMA line, skipping missing values."""

    series = [float("nan") if value is None else value for value in ema_values]
    ax.plot(dates, series, label=label, color=color, linewidth=linewidth, zorder=4)


def overlay_event_markers(
    ax: Any,
    dates: list[date],
    closes: list[float],
    events: list[str],
    *,
    marker_map: dict[str, dict[str, Any]],
    highs: list[float] | None = None,
    lows: list[float] | None = None,
) -> None:
    """Overlay event markers mapped by event name.

    Marker styles can optionally include:
    - ``anchor``: ``"close"`` (default), ``"high"``, or ``"low"``
    - ``y_offset_frac``: vertical offset as a fraction of the visible price span
    """

    price_values: list[float] = []
    if highs is not None:
        price_values.extend(highs)
    if lows is not None:
        price_values.extend(lows)
    price_values.extend(closes)
    if price_values:
        price_span = max(price_values) - min(price_values)
    else:
        price_span = 1.0
    if math.isclose(price_span, 0.0, abs_tol=1e-12):
        baseline = abs(closes[-1]) if closes else 1.0
        price_span = max(1.0, baseline * 0.01)

    seen: set[str] = set()
    for event_name, style in marker_map.items():
        xs: list[date] = []
        ys: list[float] = []
        anchor = str(style.get("anchor", "close")).lower()
        try:
            y_offset_frac = float(style.get("y_offset_frac", 0.0))
        except (TypeError, ValueError):
            y_offset_frac = 0.0

        for idx, (day, event) in enumerate(zip(dates, events, strict=True)):
            if event != event_name:
                continue
            if anchor == "high" and highs is not None:
                base_price = highs[idx]
            elif anchor == "low" and lows is not None:
                base_price = lows[idx]
            else:
                base_price = closes[idx]
            xs.append(day)
            ys.append(base_price + (price_span * y_offset_frac))

        if not xs:
            continue
        label = style.get("label", event_name)
        legend_label = label if label not in seen else "_nolegend_"
        seen.add(label)
        ax.scatter(
            xs,
            ys,
            marker=style.get("marker", "o"),
            s=style.get("size", 28),
            color=style.get("color", "#1f77b4"),
            alpha=style.get("alpha", 0.85),
            edgecolors=style.get("edgecolor", "none"),
            linewidths=style.get("linewidth", 0.0),
            label=legend_label,
            zorder=6,
        )
