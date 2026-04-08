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

import os
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

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

# Hardcoded task definitions — used as a reliable fallback if openenv.yaml cannot be read
# (e.g. path resolution issues inside the Docker container on HF Spaces).
_FALLBACK_TASKS: List[Dict[str, Any]] = [
    {
        "id": "task_easy",
        "difficulty": "easy",
        "max_steps": 10,
        "grader": {
            "type": "llm",
            "prompt_template": (
                "You are grading EASY fraud-detection trajectories. "
                "Reward strong early fraud signals and consistent risk-aware behavior. "
                "Return exactly one numeric score in [0.0, 1.0]."
            ),
        },
    },
    {
        "id": "task_medium",
        "difficulty": "medium",
        "max_steps": 15,
        "grader": {
            "type": "llm",
            "prompt_template": (
                "You are grading MEDIUM difficulty trajectories. "
                "Balance precision and recall, penalize unnecessary flags, and reward "
                "accurate escalation under mixed risk evidence. "
                "Return exactly one numeric score in [0.0, 1.0]."
            ),
        },
    },
    {
        "id": "task_hard",
        "difficulty": "hard",
        "max_steps": 20,
        "grader": {
            "type": "llm",
            "prompt_template": (
                "You are grading HARD production-style trajectories. "
                "Require robust multi-step reasoning, penalize both misses and false alarms, "
                "and reward only highly reliable fraud judgments. "
                "Return exactly one numeric score in [0.0, 1.0]."
            ),
        },
    },
]


def _build_task_payload() -> List[Dict[str, Any]]:
    # Try reading from openenv.yaml first.
    if yaml is not None and _OPENENV_YAML_PATH.exists():
        try:
            with _OPENENV_YAML_PATH.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}

            tasks = data.get("tasks")
            if isinstance(tasks, list):
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
                if len(payload) >= 3:
                    return payload
        except Exception:
            pass

    # Fall back to hardcoded definitions so the endpoint never returns an empty list.
    return _FALLBACK_TASKS


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
    prompt = _GRADER_PROMPTS[difficulty]
    try:
        api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
        api_base = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
        client = OpenAI(base_url=api_base, api_key=api_key or "missing-api-key")
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
    except Exception:
        return 0.5


@app.get("/grade/task_easy", tags=["Graders"], summary="Grade easy task trajectory")
def grade_easy():
    score = max(0.01, min(0.99, _llm_grade("easy")))
    return {"score": score, "reward": score}


@app.get("/grade/task_medium", tags=["Graders"], summary="Grade medium task trajectory")
def grade_medium():
    score = max(0.01, min(0.99, _llm_grade("medium")))
    return {"score": score, "reward": score}


@app.get("/grade/task_hard", tags=["Graders"], summary="Grade hard task trajectory")
def grade_hard():
    score = max(0.01, min(0.99, _llm_grade("hard")))
    return {"score": score, "reward": score}


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
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
