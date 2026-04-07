"""
Minimal Risk Engine Adapter (Contract-Preserving)
=================================================
This module intentionally keeps risk logic lightweight for environment-focused work.

Design goal:
- Keep the environment/client/inference contract stable.
- Avoid heavy forensic-model complexity.
- Return the same evaluate_company() output shape used by the server wrapper.

Returned keys from evaluate_company(data):
- flags: list[str]
- severity: dict[str, dict]
- risk_level: str
- total_risk: float in [0, 1]
- risk_breakdown: dict[str, float] with 5 dimensions in [0, 1]
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


def _to_series(values) -> List[float]:
    if values is None:
        return []
    out: List[float] = []
    for item in values:
        try:
            value = float(item)
        except (TypeError, ValueError):
            value = 0.0
        if not math.isfinite(value):
            value = 0.0
        out.append(value)
    return out


def _last(values: List[float], default: float = 0.0) -> float:
    return values[-1] if values else default


def _prev(values: List[float], default: float = 0.0) -> float:
    return values[-2] if len(values) >= 2 else default


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, 1.0)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


# ---------------------------------------------------------------------------
# Minimal dimensions (lightweight, deterministic)
# ---------------------------------------------------------------------------

def earnings_quality_risk(data: dict) -> float:
    profit = _to_series(data.get("profit", []))
    cash = _to_series(data.get("cash", []))
    assets = _to_series(data.get("assets", []))

    if len(cash) < 2 or not assets:
        return 0.0

    delta_cash = _last(cash) - _prev(cash)
    accrual_gap = max(_last(profit) - delta_cash, 0.0)
    accrual_ratio = _safe_ratio(accrual_gap, _last(assets, 1.0))
    return _clamp(accrual_ratio / 0.20)


def channel_stuffing_risk(data: dict) -> float:
    receivables = _to_series(data.get("receivables", []))
    revenue = _to_series(data.get("revenue", []))

    if len(receivables) < 2 or len(revenue) < 2:
        return 0.0

    latest_ratio = _safe_ratio(_last(receivables), _last(revenue, 1.0))
    prev_ratio = _safe_ratio(_prev(receivables), _prev(revenue, 1.0))

    spike = max(latest_ratio - prev_ratio, 0.0)
    absolute_score = _clamp((latest_ratio - 0.35) / 0.35)
    spike_score = _clamp(spike / 0.15)
    return max(absolute_score, spike_score)


def leverage_risk(data: dict) -> float:
    debt = _to_series(data.get("debt", []))
    assets = _to_series(data.get("assets", []))
    if not debt or not assets:
        return 0.0

    ratio = _safe_ratio(_last(debt), _last(assets, 1.0))
    return _clamp((ratio - 0.50) / 0.50)


def liquidity_risk(data: dict) -> float:
    cash = _to_series(data.get("cash", []))
    assets = _to_series(data.get("assets", []))
    if not cash or not assets:
        return 0.0

    cash_ratio = _safe_ratio(_last(cash), _last(assets, 1.0))
    if cash_ratio >= 0.08:
        return 0.0
    return _clamp((0.08 - cash_ratio) / 0.08)


def profitability_risk(data: dict) -> float:
    profit = _to_series(data.get("profit", []))
    assets = _to_series(data.get("assets", []))
    if not profit or not assets:
        return 0.0

    roa = _safe_ratio(_last(profit), _last(assets, 1.0))
    if roa >= 0.0:
        return 0.0
    return _clamp(abs(roa) / 0.20)


# ---------------------------------------------------------------------------
# Composite, flags, and API
# ---------------------------------------------------------------------------

_WEIGHTS: Dict[str, float] = {
    "earnings_quality": 0.30,
    "channel_stuffing": 0.25,
    "leverage": 0.15,
    "liquidity": 0.15,
    "profitability": 0.15,
}


def compute_risks(data: dict) -> Tuple[Dict[str, float], float]:
    risks = {
        "earnings_quality": earnings_quality_risk(data),
        "channel_stuffing": channel_stuffing_risk(data),
        "leverage": leverage_risk(data),
        "liquidity": liquidity_risk(data),
        "profitability": profitability_risk(data),
    }
    total = sum(_WEIGHTS[k] * v for k, v in risks.items())
    return risks, _clamp(total)


_FLAG_RULES: List[Tuple[str, float, str, str]] = [
    (
        "earnings_quality",
        0.70,
        "high_accrual_bloat",
        "Profits appear disconnected from cash-flow dynamics.",
    ),
    (
        "channel_stuffing",
        0.70,
        "receivables_spike",
        "Receivables growth appears inconsistent with revenue behavior.",
    ),
    (
        "leverage",
        0.70,
        "critical_debt_load",
        "Debt burden is high relative to assets.",
    ),
    (
        "liquidity",
        0.70,
        "severe_cash_depletion",
        "Cash reserves are low relative to assets.",
    ),
    (
        "profitability",
        0.70,
        "deep_margin_collapse",
        "Profitability appears materially negative.",
    ),
]


def generate_flags(risks: Dict[str, float]) -> Tuple[List[str], Dict[str, Dict]]:
    flags: List[str] = []
    severity: Dict[str, Dict] = {}
    for dimension, threshold, flag_name, description in _FLAG_RULES:
        score = risks.get(dimension, 0.0)
        if score >= threshold:
            flags.append(flag_name)
            severity[flag_name] = {
                "score": round(score, 4),
                "description": description,
                "dimension": dimension,
            }
    return flags, severity


def classify_risk(total_risk: float) -> str:
    if total_risk < 0.30:
        return "LOW"
    if total_risk < 0.55:
        return "MEDIUM"
    if total_risk < 0.75:
        return "HIGH"
    return "CRITICAL"


def evaluate_company(data: dict) -> dict:
    risks, total_risk = compute_risks(data)
    flags, severity = generate_flags(risks)
    risk_level = classify_risk(total_risk)

    return {
        "flags": flags,
        "severity": severity,
        "risk_level": risk_level,
        "total_risk": round(total_risk, 4),
        "risk_breakdown": {k: round(v, 4) for k, v in risks.items()},
    }
