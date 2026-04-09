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
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
import yaml
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    from openenv.core.env_server.http_server import create_app
except Exception:  # pragma: no cover
    create_app = None  # type: ignore[assignment]

try:
    from ..models import RiskPredictionAction, RiskPredictionObservation
    from .risk_prediction_environment import RiskPredictionEnvironment
except (ModuleNotFoundError, ImportError):
    from models import RiskPredictionAction, RiskPredictionObservation
    from server.risk_prediction_environment import RiskPredictionEnvironment


def _build_degraded_app(detail: str) -> FastAPI:
    degraded = FastAPI(title="Risk Prediction (degraded)")

    @degraded.get("/health")
    def _health() -> Dict[str, str]:
        return {"status": "degraded", "detail": detail}

    @degraded.post("/reset")
    def _reset_unavailable() -> None:
        raise HTTPException(status_code=503, detail="environment_unavailable")

    @degraded.post("/step")
    def _step_unavailable() -> None:
        raise HTTPException(status_code=503, detail="environment_unavailable")

    return degraded


def _create_openenv_app() -> FastAPI:
    """Build the OpenEnv FastAPI app with cross-version create_app compatibility."""
    if create_app is None:
        raise RuntimeError("openenv_http_server_unavailable")

    call_variants = [
        {
            "args": (
                RiskPredictionEnvironment,
                RiskPredictionAction,
                RiskPredictionObservation,
            ),
            "kwargs": {
                "env_name": "risk_prediction",
                "max_concurrent_envs": 1,
            },
        },
        {
            "args": (
                RiskPredictionEnvironment,
                RiskPredictionAction,
                RiskPredictionObservation,
            ),
            "kwargs": {"env_name": "risk_prediction"},
        },
        {
            "args": (
                RiskPredictionEnvironment,
                RiskPredictionAction,
                RiskPredictionObservation,
            ),
            "kwargs": {},
        },
    ]

    last_type_error: Optional[TypeError] = None
    for variant in call_variants:
        try:
            return create_app(*variant["args"], **variant["kwargs"])
        except TypeError as exc:
            last_type_error = exc

    if last_type_error is not None:
        raise last_type_error
    raise RuntimeError("openenv_create_app_failed")


# Create the app with web interface and README integration
_APP_STARTUP_ERROR: Optional[str] = None
try:
    app = _create_openenv_app()
except Exception as exc:  # pragma: no cover
    _APP_STARTUP_ERROR = f"{type(exc).__name__}: {exc}"
    app = _build_degraded_app(_APP_STARTUP_ERROR)


_OPENENV_YAML_PATH = Path(__file__).resolve().parents[1] / "openenv.yaml"


def _build_task_payload() -> List[Dict[str, Any]]:
    if not _OPENENV_YAML_PATH.exists():
        raise FileNotFoundError(f"Missing OpenEnv config: {_OPENENV_YAML_PATH}")

    with _OPENENV_YAML_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    tasks = data.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("openenv.yaml must define a non-empty tasks list")

    payload: List[Dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("Each task entry must be a mapping")

        task_id = str(task.get("id", "")).strip()
        difficulty = str(task.get("difficulty", "")).strip()
        grader = task.get("grader")

        if not task_id or not difficulty:
            raise ValueError("Each task must include non-empty id and difficulty")
        if grader is None:
            raise ValueError(f"Task '{task_id}' is missing grader")

        payload.append(
            {
                "id": task_id,
                "difficulty": difficulty,
                "max_steps": int(task.get("max_steps", 0) or 0),
                "grader": grader,
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


# ---------------------------------------------------------------------------
# Grader endpoints
# ---------------------------------------------------------------------------

_GRADER_PROMPTS = {
    "easy": (
        "You are grading EASY fraud-detection trajectories. "
        "Reward strong early fraud signals and consistent risk-aware behavior. "
        "Return exactly one numeric score between 0.0 and 1.0 and nothing else."
    ),
    "medium": (
        "You are grading MEDIUM difficulty trajectories. "
        "Balance precision and recall, penalize unnecessary flags, and reward accurate "
        "escalation under mixed risk evidence. "
        "Return exactly one numeric score between 0.0 and 1.0 and nothing else."
    ),
    "hard": (
        "You are grading HARD production-style trajectories. "
        "Require robust multi-step reasoning, penalize both misses and false alarms, "
        "and reward only highly reliable fraud judgments. "
        "Return exactly one numeric score between 0.0 and 1.0 and nothing else."
    ),
}


def _llm_grade(difficulty: str) -> float:
    if OpenAI is None:
        raise RuntimeError("openai_client_unavailable")

    prompt = _GRADER_PROMPTS[difficulty]
    api_key = os.environ.get("API_KEY")
    api_base = os.environ.get("API_BASE_URL")
    model = os.environ.get("MODEL_NAME")
    if not api_key or not api_base or not model:
        raise RuntimeError("missing_grader_env_vars")

    client = OpenAI(base_url=api_base, api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Grade the most recent trajectory for this difficulty level."},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    return float((completion.choices[0].message.content or "").strip())


def _task_difficulty_map() -> Dict[str, str]:
    payload = _build_task_payload()
    return {
        str(task["id"]).strip(): str(task["difficulty"]).strip().lower()
        for task in payload
    }


@app.get("/grade/{task_id}", tags=["Graders"], summary="Grade task trajectory")
def grade_task(task_id: str):
    difficulty_map = _task_difficulty_map()
    difficulty = difficulty_map.get(task_id)
    if difficulty is None:
        raise HTTPException(status_code=404, detail=f"Unknown task_id: {task_id}")
    if difficulty not in _GRADER_PROMPTS:
        raise HTTPException(status_code=422, detail=f"Unsupported task difficulty: {difficulty}")

    try:
        score = max(0.01, min(0.99, _llm_grade(difficulty)))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"grader_unavailable: {type(exc).__name__}") from exc

    return {"score": score, "reward": score}


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m risk_prediction.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn risk_prediction.server.app:app --workers 4
    """
    import os
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=7860)


if __name__ == '__main__':
    main()
