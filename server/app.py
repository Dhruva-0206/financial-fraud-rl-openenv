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

from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import RiskPredictionAction, RiskPredictionObservation
    from .risk_prediction_environment import RiskPredictionEnvironment
except (ModuleNotFoundError, ImportError):
    from models import RiskPredictionAction, RiskPredictionObservation
    from server.risk_prediction_environment import RiskPredictionEnvironment


# Create the app with web interface and README integration
app = create_app(
    RiskPredictionEnvironment,
    RiskPredictionAction,
    RiskPredictionObservation,
    env_name="risk_prediction",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


_OPENENV_YAML_PATH = Path(__file__).resolve().parents[1] / "openenv.yaml"


def _build_task_payload() -> List[Dict[str, Any]]:
    if yaml is None or not _OPENENV_YAML_PATH.exists():
        return []

    try:
        with _OPENENV_YAML_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return []

    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return []

    payload: List[Dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue

        grader = task.get("grader")
        if not isinstance(grader, dict):
            grader = {}

        payload.append(
            {
                "id": str(task.get("id", "")).strip(),
                "difficulty": str(task.get("difficulty", "")).strip(),
                "max_steps": int(task.get("max_steps", 0) or 0),
                "grader": {
                    "type": str(grader.get("type", "")).strip(),
                    "prompt_template": str(grader.get("prompt_template", "")).strip(),
                },
            }
        )

    return payload


@app.get(
    "/tasks",
    tags=["Environment Info"],
    summary="List available tasks",
)
def list_tasks() -> List[Dict]:
    """Return tasks discovered from openenv.yaml instead of hardcoded mappings."""
    return _build_task_payload()


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
