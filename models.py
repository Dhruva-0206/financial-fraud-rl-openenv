# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Financial Fraud Detection RL Environment.

The agent observes forensic risk scores computed from financial statements
and decides whether to HOLD (continue monitoring) or FLAG (report fraud).
"""

from typing import List

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class RiskPredictionAction(Action):
    """Agent action: decide to hold or flag the current company as fraudulent."""

    action_type: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = HOLD (looks clean, keep monitoring), 1 = FLAG (fraud detected)",
    )


class RiskPredictionObservation(Observation):
    """
    Observation from the forensic risk engine over a 4-year financial window.

    All risk scores are in [0.0, 1.0], where higher = more suspicious.
    """

    # --- Forensic risk dimensions from risk_engine_v5 ---
    earnings_quality_risk: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Accrual bloat: profits disconnected from actual cash flow",
    )
    channel_stuffing_risk: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Receivables spike: possible fake/inflated sales revenue",
    )
    leverage_risk: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Debt burden: long-term debt relative to total assets",
    )
    liquidity_risk: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Cash exhaustion: cash reserves near critically low levels",
    )
    profitability_risk: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Margin collapse: sustained deeply negative return on assets",
    )
    total_risk: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Weighted composite of all five risk dimensions",
    )

    # --- Risk engine summary ---
    risk_level: str = Field(
        default="LOW",
        description="Categorical risk: LOW / MEDIUM / HIGH / CRITICAL",
    )
    flags: List[str] = Field(
        default_factory=list,
        description="Active forensic flags triggered by threshold breaches",
    )

    # --- Episode context ---
    gvkey: str = Field(default="", description="Company identifier (COMPUSTAT gvkey)")
    fiscal_year: int = Field(default=0, description="Most recent fiscal year in the window")
    is_fraud_company: bool = Field(
        default=False,
        description="Whether the company has any misstatement in its full history",
    )
    step_number: int = Field(default=0, description="Current step within this episode")

    # --- Task + grading context ---
    task_id: str = Field(
        default="medium",
        description="Task identifier used for grading (easy, medium, hard)",
    )
    task_difficulty: str = Field(
        default="medium",
        description="Task difficulty label",
    )
    grader_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized grader reward for this step in [0, 1]",
    )
    raw_reward: float = Field(
        default=0.0,
        description="Underlying environment reward before grader normalization",
    )
