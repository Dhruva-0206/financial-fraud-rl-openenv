# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Financial Fraud Detection Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import RiskPredictionAction, RiskPredictionObservation
except ImportError:
    from models import RiskPredictionAction, RiskPredictionObservation


class RiskPredictionEnv(
    EnvClient[RiskPredictionAction, RiskPredictionObservation, State]
):
    """
    Client for the Financial Fraud Detection RL Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance gets its own dedicated episode session.

    Example:
        >>> with RiskPredictionEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Total risk: {result.observation.total_risk:.2f}")
        ...
        ...     # Agent decides to hold (0) or flag (1)
        ...     action = RiskPredictionAction(action_type=0)
        ...     result = client.step(action)
        ...     print(f"Reward: {result.reward}, Done: {result.done}")

    Example with Docker:
        >>> client = RiskPredictionEnv.from_docker_image("risk_prediction-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(RiskPredictionAction(action_type=1))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: RiskPredictionAction) -> Dict:
        """Convert RiskPredictionAction to JSON payload for the step message."""
        return {"action_type": action.action_type}

    def _parse_result(self, payload: Dict) -> StepResult[RiskPredictionObservation]:
        """Parse server response into StepResult[RiskPredictionObservation]."""
        obs_data = payload.get("observation", {})
        observation = RiskPredictionObservation(
            # Risk dimensions
            earnings_quality_risk=obs_data.get("earnings_quality_risk", 0.0),
            channel_stuffing_risk=obs_data.get("channel_stuffing_risk", 0.0),
            leverage_risk=obs_data.get("leverage_risk", 0.0),
            liquidity_risk=obs_data.get("liquidity_risk", 0.0),
            profitability_risk=obs_data.get("profitability_risk", 0.0),
            total_risk=obs_data.get("total_risk", 0.0),
            # Risk engine summary
            risk_level=obs_data.get("risk_level", "LOW"),
            flags=obs_data.get("flags", []),
            # Episode context
            gvkey=obs_data.get("gvkey", ""),
            fiscal_year=obs_data.get("fiscal_year", 0),
            is_fraud_company=obs_data.get("is_fraud_company", False),
            step_number=obs_data.get("step_number", 0),
            task_id=obs_data.get("task_id", "medium"),
            task_difficulty=obs_data.get("task_difficulty", "medium"),
            grader_score=obs_data.get("grader_score", 0.0),
            raw_reward=obs_data.get("raw_reward", obs_data.get("reward", 0.0) or 0.0),
            # Gym signals
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
