"""
Financial Statement Forensic Risk Engine — v5 (RL Optimized)
============================================================
Shifted from pure variance math to forensic accounting ratios.
Optimized for RL Environment observation states to minimize false positives.
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

# ===========================================================================
# 1. Primitives
# ===========================================================================

def _clean(series) -> np.ndarray:
    """Convert to float, replace NaN/Inf with 0."""
    arr = np.array(series, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(np.clip(value, lo, hi))

# ===========================================================================
# 2. Forensic Risk Dimensions
# ===========================================================================

def earnings_quality_risk(data: dict) -> float:
    """
    Accrual Bloat: Are profits backed by actual cash?
    High paper profit but declining/flat cash is a massive red flag for manipulation.
    """
    profit = _clean(data.get("profit", []))
    cash   = _clean(data["cash"])
    assets = _clean(data["assets"])
    
    if len(profit) < 2 or len(cash) < 2:
        return 0.0
        
    # Approximate operating cash flow via change in cash 
    delta_cash = cash[-1] - cash[-2]
    latest_profit = profit[-1]
    latest_assets = max(assets[-1], 1.0)
    
    # Positive accrual ratio means profit is higher than actual cash generated
    accrual_ratio = (latest_profit - delta_cash) / latest_assets
    
    # Healthy companies hover around 0.0. Fraudsters spike > 0.10.
    return _clamp(accrual_ratio / 0.15) if accrual_ratio > 0 else 0.0

def channel_stuffing_risk(data: dict) -> float:
    """
    Receivables vs Revenue: Are they booking fake sales?
    A sudden spike in receivables relative to sales indicates uncollected, possibly fake, revenue.
    """
    if "receivables" not in data:
        return 0.0
        
    rect = _clean(data["receivables"])
    rev  = _clean(data["revenue"])
    
    if len(rect) < 2 or len(rev) < 2:
        return 0.0
        
    latest_rect_ratio = rect[-1] / max(rev[-1], 1.0)
    prev_rect_ratio   = rect[-2] / max(rev[-2], 1.0)
    
    spike = latest_rect_ratio - prev_rect_ratio
    
    # A > 5% jump in receivables relative to sales is suspicious
    return _clamp(spike / 0.10) if spike > 0 else 0.0

def leverage_risk(data: dict) -> float:
    """Absolute debt burden relative to assets."""
    debt   = _clean(data["debt"])
    assets = _clean(data["assets"])
    
    if len(debt) == 0:
        return 0.0
        
    latest_debt   = debt[-1]
    latest_assets = max(assets[-1], 1.0)
    ratio = latest_debt / latest_assets
    
    # > 50% long term debt-to-assets starts getting dangerous
    return _clamp((ratio - 0.40) / 0.40) if ratio > 0.40 else 0.0

def liquidity_risk(data: dict) -> float:
    """Cash exhaustion risk."""
    cash   = _clean(data["cash"])
    assets = _clean(data["assets"])
    
    if len(cash) < 2:
        return 0.0
        
    latest_cash   = cash[-1]
    latest_assets = max(assets[-1], 1.0)
    cash_ratio    = latest_cash / latest_assets
    
    # If cash drops below 2% of total assets, liquidity is highly stressed
    return _clamp(1.0 - (cash_ratio / 0.05)) if cash_ratio < 0.05 else 0.0

def profitability_risk(data: dict) -> float:
    """Severe margin collapse."""
    profit = _clean(data.get("profit", []))
    assets = _clean(data["assets"])
    
    if len(profit) < 2:
        return 0.0
        
    roa = profit[-1] / max(assets[-1], 1.0)
    
    # Only penalize deeply negative ROA, not normal business fluctuations
    return _clamp(abs(roa) / 0.20) if roa < -0.05 else 0.0

# ===========================================================================
# 3. Weights & Composites
# ===========================================================================

_WEIGHTS: Dict[str, float] = {
    "earnings_quality":   0.35,  # Heavily weighted for fraud detection
    "channel_stuffing":   0.25,  # Key indicator of fake revenue
    "leverage":           0.15,
    "liquidity":          0.15,
    "profitability":      0.10,
}

def compute_risks(data: dict) -> Tuple[Dict[str, float], float]:
    risks = {
        "earnings_quality":   earnings_quality_risk(data),
        "channel_stuffing":   channel_stuffing_risk(data),
        "leverage":           leverage_risk(data),
        "liquidity":          liquidity_risk(data),
        "profitability":      profitability_risk(data),
    }
    total = sum(_WEIGHTS[k] * v for k, v in risks.items())
    return risks, _clamp(total)

# ===========================================================================
# 4. Strict Flag Generation (Optimized for RL Ground Truth)
# ===========================================================================

_FLAG_RULES: List[Tuple[str, float, str, str]] = [
    ("earnings_quality",  0.70, "high_accrual_bloat", 
     "Profits are dangerously disconnected from actual cash flows."),
    ("channel_stuffing",  0.65, "receivables_spike", 
     "Receivables are growing much faster than revenue; possible fake sales."),
    ("leverage",          0.75, "critical_debt_load", 
     "Long-term debt exceeds safe asset coverage levels."),
    ("liquidity",         0.80, "severe_cash_depletion", 
     "Cash reserves have fallen to critically low levels relative to assets."),
    ("profitability",     0.70, "deep_margin_collapse", 
     "Company is sustaining severe, unsustainable operational losses."),
]

def generate_flags(risks: Dict[str, float]) -> Tuple[List[str], Dict[str, Dict]]:
    flags: List[str] = []
    severity: Dict[str, Dict] = {}
    for dim, threshold, flag_name, description in _FLAG_RULES:
        score = risks.get(dim, 0.0)
        if score >= threshold:
            flags.append(flag_name)
            severity[flag_name] = {
                "score": round(score, 4),
                "description": description,
                "dimension": dim,
            }
    return flags, severity

def classify_risk(total_risk: float) -> str:
    if total_risk < 0.30: return "LOW"
    elif total_risk < 0.55: return "MEDIUM"
    elif total_risk < 0.75: return "HIGH"
    else: return "CRITICAL"

# ===========================================================================
# 5. API
# ===========================================================================

def evaluate_company(data: dict) -> dict:
    risks, total_risk = compute_risks(data)
    flags, severity   = generate_flags(risks)
    risk_level        = classify_risk(total_risk)
    
    return {
        "flags":          flags,
        "severity":       severity,
        "risk_level":     risk_level,
        "total_risk":     round(total_risk, 4),
        "risk_breakdown": {k: round(v, 4) for k, v in risks.items()},
    }