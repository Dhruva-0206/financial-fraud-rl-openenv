# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Risk Prediction Environment.

This module creates an HTTP server that exposes the RiskPredictionEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from typing import Dict, List

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import RiskPredictionAction, RiskPredictionObservation
    from ..task_graders import TASK_DEFINITIONS
    from .risk_prediction_environment import RiskPredictionEnvironment
except (ModuleNotFoundError, ImportError):
    from models import RiskPredictionAction, RiskPredictionObservation
    from task_graders import TASK_DEFINITIONS
    from server.risk_prediction_environment import RiskPredictionEnvironment


# Create the app with web interface and README integration
app = create_app(
    RiskPredictionEnvironment,
    RiskPredictionAction,
    RiskPredictionObservation,
    env_name="risk_prediction",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


_TASK_GRADER_REGISTRY: Dict[str, Dict[str, str]] = {
    "small": {
        "grader_id": "small_grader",
        "entrypoint": "task_graders:grade_small",
    },
    "medium": {
        "grader_id": "medium_grader",
        "entrypoint": "task_graders:grade_medium",
    },
    "hard": {
        "grader_id": "hard_grader",
        "entrypoint": "task_graders:grade_hard",
    },
}


def _build_task_payload() -> List[Dict]:
    payload: List[Dict] = []
    for task_id, task in TASK_DEFINITIONS.items():
        grader = _TASK_GRADER_REGISTRY.get(task_id, {})
        payload.append(
            {
                "id": task.task_id,
                "task_id": task.task_id,
                "difficulty": task.difficulty,
                "description": task.description,
                "reward_range": [0.0, 1.0],
                "grader_id": grader.get("grader_id", f"{task.task_id}_grader"),
                "grader": grader.get("entrypoint", ""),
            }
        )
    return payload


@app.get(
    "/tasks",
    tags=["Environment Info"],
    summary="List available tasks",
)
def list_tasks() -> List[Dict]:
    """Return a validator-friendly task registry with explicit grader mappings."""
    return _build_task_payload()


@app.get(
    "/graders",
    tags=["Environment Info"],
    summary="List available graders",
)
def list_graders() -> List[Dict]:
    """Return grader registry and the task each grader is linked to."""
    return [
        {
            "id": spec["grader_id"],
            "task": task_id,
            "entrypoint": spec["entrypoint"],
        }
        for task_id, spec in _TASK_GRADER_REGISTRY.items()
    ]


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m risk_prediction.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn risk_prediction.server.app:app --workers 4
    """
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
